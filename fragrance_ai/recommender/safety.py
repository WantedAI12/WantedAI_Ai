"""Fail-closed candidate screening and formula safety gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import date, datetime

from fragrance_ai.rules.ifra_rules import ProductCategory, check_compliance

from .catalog import IngredientCatalog, normalize_name
from .models import Ingredient, RecipeConstraints, RecipeLine, SafetyReport, ScentBrief
from .promotion_activation import formulation_scope_allows
from .registry_activation import REGISTRY_CONDITIONAL_DATA_SOURCE
from .supplier import SupplierRegistry


VALIDATION_LEVELS = {"prototype", "qualified", "commercial"}
PRODUCT_CATEGORY_MAP = {
    "eau_de_parfum": ProductCategory.EAU_DE_PARFUM,
    "eau_de_toilette": ProductCategory.EAU_DE_TOILETTE,
    "eau_de_cologne": ProductCategory.EAU_DE_COLOGNE,
    "face_cream": ProductCategory.FACE_CREAM,
    "face_toner": ProductCategory.FACE_TONER,
    "mouthwash": ProductCategory.MOUTHWASH,
    "shampoo": ProductCategory.SHAMPOO,
    "body_wash": ProductCategory.BODY_WASH,
    "candle": ProductCategory.CANDLE,
    "room_spray": ProductCategory.ROOM_SPRAY,
    "diffuser": ProductCategory.DIFFUSER,
}
RINSE_OFF_CATEGORIES = {"shampoo", "body_wash"}
NON_SKIN_CATEGORIES = {"candle", "room_spray", "diffuser"}


def _promotion_approval_expired(ingredient: Ingredient, as_of: date) -> bool:
    if not ingredient.approval_expires_at:
        return False
    value = ingredient.approval_expires_at
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return as_of >= datetime.fromisoformat(value).date()


class CandidateSafetyScreen:
    """Allow affordable, available, lower-risk, reviewed materials only."""

    def screen(
        self,
        catalog: IngredientCatalog,
        brief: ScentBrief,
        supplier_registry: SupplierRegistry | None = None,
        as_of: date | None = None,
    ) -> tuple[list[Ingredient], dict[str, int]]:
        rejected: Counter[str] = Counter()
        accepted: list[Ingredient] = []
        constraints = brief.constraints
        as_of = as_of or date.today()

        if constraints.validation_level not in VALIDATION_LEVELS:
            raise ValueError(
                f"validation_level must be one of {sorted(VALIDATION_LEVELS)}"
            )
        explicit_bans = {normalize_name(name) for name in constraints.explicit_bans}
        registry = supplier_registry or SupplierRegistry()

        for ingredient in catalog.ingredients:
            if (
                constraints.experimental_disable_safety
                and constraints.enable_registry_trace_candidates
                and constraints.validation_level == "prototype"
            ):
                names = {normalize_name(name) for name in ingredient.all_names()}
                names.add(normalize_name(ingredient.ingredient_id))
                if names & explicit_bans:
                    rejected["explicitly_excluded"] += 1
                    continue
                accepted.append(
                    replace(
                        ingredient,
                        formulation_ready=True,
                        blocked=False,
                        blocked_reason=None,
                        max_concentrate_percent=100.0,
                    )
                )
                continue
            if ingredient.blocked:
                rejected["blocked_or_prohibited"] += 1
                continue
            if not ingredient.formulation_ready:
                rejected["reference_only"] += 1
                continue
            if (
                ingredient.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
                and not constraints.enable_registry_trace_candidates
            ):
                rejected["registry_conditional_not_requested"] += 1
                continue
            if (
                ingredient.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE
                and constraints.validation_level != "prototype"
            ):
                rejected["registry_conditional_prototype_only"] += 1
                continue
            if ingredient.approved_formulation_scopes:
                if not formulation_scope_allows(
                    ingredient.approved_formulation_scopes,
                    constraints.target_region,
                    constraints.product_category,
                ):
                    rejected["signed_promotion_scope"] += 1
                    continue
                if _promotion_approval_expired(ingredient, as_of):
                    rejected["signed_promotion_expired"] += 1
                    continue
            if ingredient.risk_tier > constraints.max_risk_tier:
                rejected["risk_tier"] += 1
                continue
            if ingredient.price_per_kg > constraints.max_ingredient_price_per_kg:
                rejected["too_expensive"] += 1
                continue
            if ingredient.availability < constraints.min_availability:
                rejected["hard_to_source"] += 1
                continue
            if ingredient.rarity == "rare" and not constraints.allow_rare:
                rejected["rare"] += 1
                continue
            names = {normalize_name(name) for name in ingredient.all_names()}
            names.add(normalize_name(ingredient.ingredient_id))
            if names & explicit_bans:
                rejected["explicitly_excluded"] += 1
                continue

            if constraints.validation_level in {"qualified", "commercial"}:
                assessment = registry.best_assessment(ingredient, constraints, as_of)
                if not assessment.qualified or assessment.offer is None:
                    if assessment.offer is None:
                        rejected["missing_supplier_offer"] += 1
                    else:
                        rejected["supplier_or_document_gate"] += 1
                    continue
                ingredient = registry.overlay(ingredient, assessment.offer)
            accepted.append(ingredient)

        return accepted, dict(sorted(rejected.items()))


class FormulaSafetyGate:
    ACTIVE_IFRA_AMENDMENT = "51"
    ACTIVE_IFRA_LABEL = (
        "Supplier certificate gate: IFRA 51st Amendment; local screen: "
        "Amendment 50 embedded partial subset"
    )
    EMBEDDED_IFRA_SCOPE_WARNING = (
        "Local IFRA screening covers only an explicit Amendment 50 subset. "
        "Any unlisted material/category is unknown to this package, not unrestricted."
    )
    REVIEWED_ON = date(2026, 7, 11)
    REVIEW_DUE = date(2026, 11, 30)
    EU_LEAVE_ON_LABEL_THRESHOLD_PERCENT = 0.001
    EU_RINSE_OFF_LABEL_THRESHOLD_PERCENT = 0.01

    def __init__(self, supplier_registry: SupplierRegistry | None = None):
        self.supplier_registry = supplier_registry or SupplierRegistry()

    @staticmethod
    def _audit_id(
        lines: list[RecipeLine], constraints: RecipeConstraints, as_of: date
    ) -> str:
        payload = {
            "as_of": as_of.isoformat(),
            "validation_level": constraints.validation_level,
            "region": constraints.target_region,
            "category": constraints.product_category,
            "lines": sorted(
                (line.ingredient_id, round(line.concentrate_percent, 4))
                for line in lines
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        return "audit:" + digest[:20]

    def evaluate(
        self,
        lines: list[RecipeLine],
        ingredients_by_id: dict[str, Ingredient],
        constraints: RecipeConstraints,
        as_of: date | None = None,
    ) -> SafetyReport:
        as_of = as_of or date.today()
        violations: list[str] = []
        warnings: list[str] = []
        declarations: set[str] = set()
        potential_allergens: set[str] = set()
        missing_documents: set[str] = set()
        evidence_materials = 0
        allergen_materials = 0
        registry_conditionals: list[str] = []

        if constraints.validation_level not in VALIDATION_LEVELS:
            violations.append("지원하지 않는 검증 단계입니다.")
        if as_of > self.REVIEW_DUE:
            violations.append(
                "안전·규제 데이터 검토 기한이 지났습니다. 최신 IFRA 개정과 지역 규정을 갱신하세요."
            )
        if constraints.product_category not in PRODUCT_CATEGORY_MAP:
            violations.append(
                f"지원하지 않는 제품 카테고리: {constraints.product_category}"
            )
        if not 0 < constraints.product_concentration_percent <= 30:
            violations.append(
                "피부용 향 제품의 향료 농도는 현재 게이트에서 0~30%만 허용합니다."
            )
        if constraints.finished_batch_mass_g <= 0:
            violations.append("완제품 배치 질량은 양수여야 합니다.")
        if abs(sum(line.concentrate_percent for line in lines) - 100.0) > 0.05:
            violations.append("향료 농축액 배합비 합계가 100%가 아닙니다.")

        category = PRODUCT_CATEGORY_MAP.get(
            constraints.product_category, ProductCategory.EAU_DE_PARFUM
        )
        finished_recipe = {"ingredients": []}
        allergen_totals: Counter[str] = Counter()

        for line in lines:
            ingredient = ingredients_by_id[line.ingredient_id]
            if ingredient.data_source == REGISTRY_CONDITIONAL_DATA_SOURCE:
                registry_conditionals.append(ingredient.name)
                missing_documents.add(
                    f"{ingredient.name}: independently signed safety and supplier dossier"
                )
            if ingredient.blocked or not ingredient.formulation_ready:
                violations.append(f"{ingredient.name}: 제조 후보로 허용되지 않는 원료")
            if ingredient.approved_formulation_scopes:
                if not formulation_scope_allows(
                    ingredient.approved_formulation_scopes,
                    constraints.target_region,
                    constraints.product_category,
                ):
                    violations.append(
                        f"{ingredient.name}: 서명된 시장·제품군 범위 불일치"
                    )
                if _promotion_approval_expired(ingredient, as_of):
                    violations.append(f"{ingredient.name}: 서명된 원료 승격 증거 만료")
            if ingredient.risk_tier > constraints.max_risk_tier:
                violations.append(f"{ingredient.name}: 허용 위험 등급 초과")
            active_concentrate_percent = (
                line.concentrate_percent * ingredient.active_strength_percent / 100.0
            )
            if active_concentrate_percent > ingredient.max_concentrate_percent + 1e-6:
                violations.append(
                    f"{ingredient.name}: 내부 활성물질 상한 "
                    f"{ingredient.max_concentrate_percent:.3f}% 초과"
                )
            if ingredient.price_per_kg > constraints.max_ingredient_price_per_kg:
                violations.append(f"{ingredient.name}: 개별 원료 가격 상한 초과")
            if ingredient.availability < constraints.min_availability:
                violations.append(f"{ingredient.name}: 조달 가능성 기준 미달")

            finished_recipe["ingredients"].append(
                {
                    "name": ingredient.name,
                    "concentration": line.finished_product_percent,
                }
            )
            potential_allergens.update(ingredient.eu_allergens)

            assessment = self.supplier_registry.best_assessment(
                ingredient, constraints, as_of
            )
            if assessment.qualified and assessment.offer is not None:
                evidence_materials += 1
                allergen_materials += 1
                for allergen, fraction in assessment.offer.allergen_fractions.items():
                    allergen_totals[allergen] += (
                        line.finished_product_percent * fraction
                    )
            else:
                missing_documents.update(
                    f"{ingredient.name}: {item}"
                    for item in assessment.missing_documents
                )
                if assessment.offer is None:
                    missing_documents.add(f"{ingredient.name}: supplier quotation")

        if registry_conditionals:
            warnings.append(
                "조건부 산업 레지스트리 실험 후보 사용: "
                + ", ".join(sorted(registry_conditionals))
                + ". 공개 냄새 기술과 구조 스크리닝만 연결된 R&D 가설이며 "
                "공급사·독성·규제 승인 원료가 아닙니다."
            )

        ifra_result = check_compliance(
            finished_recipe,
            category,
            constraints.product_concentration_percent,
        )
        if not ifra_result["ifra"]["compliant"]:
            for detail in ifra_result["ifra"].get("details", []):
                violations.append(
                    f"내장 IFRA 부분 규칙 위반: {detail.get('ingredient', 'unknown')}"
                )

        uncovered_ifra = (
            ifra_result["ifra"].get("coverage", {}).get("uncovered_ingredients", [])
        )
        if uncovered_ifra:
            warnings.append(
                "Local IFRA subset has no rule for: " + ", ".join(uncovered_ifra)
            )

        if (
            constraints.target_region.upper() == "EU"
            and constraints.product_category not in NON_SKIN_CATEGORIES
        ):
            threshold = (
                self.EU_RINSE_OFF_LABEL_THRESHOLD_PERCENT
                if constraints.product_category in RINSE_OFF_CATEGORIES
                else self.EU_LEAVE_ON_LABEL_THRESHOLD_PERCENT
            )
            declarations.update(
                name for name, value in allergen_totals.items() if value >= threshold
            )
        elif constraints.validation_level in {"qualified", "commercial"}:
            violations.append(
                f"{constraints.target_region} 지역의 완전한 자동 규제 규칙셋이 없어 외부 검토가 필요합니다."
            )

        material_count = max(1, len(lines))
        evidence_coverage = evidence_materials / material_count * 100.0
        allergen_complete = allergen_materials == len(lines) and bool(lines)
        qualified_level = constraints.validation_level in {"qualified", "commercial"}
        if qualified_level and evidence_materials != len(lines):
            violations.append(
                "모든 사용 원료의 최신 공급사 문서 팩이 확인되지 않았습니다."
            )
        if qualified_level and not allergen_complete:
            violations.append("전 원료의 정량 알레르겐 조성이 확인되지 않았습니다.")

        warnings.append(self.EMBEDDED_IFRA_SCOPE_WARNING)
        warnings.extend(
            [
                "내장 IFRA 데이터는 전체 Standards Library가 아닌 검증용 부분집합입니다.",
                "prototype 결과는 공급사 IFRA 적합성 증명서·SDS·COA·정량 알레르겐 명세를 대체하지 않습니다.",
                "표시 알레르겐 목록은 정량 공급사 자료가 모두 있을 때만 완전합니다.",
            ]
        )
        passed = not violations
        internal_evidence_complete = qualified_level and passed and allergen_complete
        # The bundled rule set is explicitly partial. Full regional regulatory
        # completion must be signed by an external rule pack/assessor and is
        # therefore never inferred by this internal gate.
        regulatory_complete = False
        return SafetyReport(
            internal_gate_passed=passed,
            status="prototype_partial_screen" if passed else "blocked",
            active_ifra_amendment=self.ACTIVE_IFRA_LABEL,
            standards_checked_on=self.REVIEWED_ON.isoformat(),
            standards_review_due=self.REVIEW_DUE.isoformat(),
            regulatory_data_complete=regulatory_complete,
            manufacturing_ready=False,
            violations=violations,
            warnings=warnings,
            eu_label_declarations=sorted(declarations),
            potential_eu_allergens=sorted(potential_allergens),
            allergen_quantification_complete=allergen_complete,
            evidence_coverage_percent=round(evidence_coverage, 4),
            missing_documents=sorted(missing_documents),
            target_region=constraints.target_region.upper(),
            product_category=constraints.product_category,
            validation_level=constraints.validation_level,
            audit_id=self._audit_id(lines, constraints, as_of),
            internal_evidence_complete=internal_evidence_complete,
        )
