"""Weight-first lab batching with separate, fail-closed readiness gates."""

from __future__ import annotations

from datetime import date

from .manufacturing_profiles import ManufacturingProfileRegistry
from .models import (
    Ingredient,
    ManufacturingLine,
    ManufacturingPlan,
    RecipeConstraints,
    RecipeLine,
)


REQUIRED_STABILITY_TESTS: tuple[str, ...] = (
    "ambient_25c_12_weeks",
    "accelerated_40c_12_weeks",
    "refrigerated_5c_12_weeks",
    "freeze_thaw_3_cycles",
    "light_exposure",
    "packaging_compatibility",
    "pilot_batch_reproducibility",
)


class ManufacturingPlanner:
    """Calculate a batch without confusing arithmetic with readiness."""

    def __init__(
        self,
        profile_registry: ManufacturingProfileRegistry | None = None,
    ) -> None:
        self.profile_registry = profile_registry or ManufacturingProfileRegistry()

    def build(
        self,
        lines: list[RecipeLine],
        ingredients_by_id: dict[str, Ingredient],
        constraints: RecipeConstraints,
        stability_passed: bool = False,
        as_of: date | None = None,
    ) -> ManufacturingPlan:
        if constraints.finished_batch_mass_g <= 0:
            raise ValueError("finished_batch_mass_g must be positive")
        if not 0 < constraints.product_concentration_percent <= 100:
            raise ValueError("product concentration must be in (0, 100]")

        finished_mass = constraints.finished_batch_mass_g
        concentrate_mass = finished_mass * constraints.product_concentration_percent / 100.0
        base_mass = finished_mass - concentrate_mass
        manufacturing_lines: list[ManufacturingLine] = []
        warnings: list[str] = []

        for line in lines:
            ingredient = ingredients_by_id[line.ingredient_id]
            supplied_mass = concentrate_mass * line.concentrate_percent / 100.0
            active_mass = supplied_mass * ingredient.active_strength_percent / 100.0
            carrier_mass = supplied_mass - active_mass
            estimated_volume = (
                supplied_mass / ingredient.density_g_ml
                if ingredient.density_g_ml is not None
                else None
            )
            tolerance = max(0.001, supplied_mass * 0.005)
            manufacturing_lines.append(
                ManufacturingLine(
                    ingredient_id=ingredient.ingredient_id,
                    name=ingredient.name,
                    as_supplied_mass_g=round(supplied_mass, 5),
                    active_material_mass_g=round(active_mass, 5),
                    carrier_mass_g=round(carrier_mass, 5),
                    estimated_volume_ml=(
                        round(estimated_volume, 5) if estimated_volume is not None else None
                    ),
                    active_strength_percent=ingredient.active_strength_percent,
                    recommended_weighing_tolerance_g=round(tolerance, 5),
                )
            )
            if ingredient.density_g_ml is None:
                warnings.append(
                    f"{ingredient.name}: 검증된 밀도가 없어 부피를 계산하지 않았습니다."
                )
            if ingredient.oxidation_risk == "unknown":
                warnings.append(f"{ingredient.name}: 산화 위험 데이터가 확인되지 않았습니다.")
            if ingredient.discoloration_risk == "unknown":
                warnings.append(f"{ingredient.name}: 변색 위험 데이터가 확인되지 않았습니다.")

        total_supplied = sum(item.as_supplied_mass_g for item in manufacturing_lines)
        minimum_mass = min((item.as_supplied_mass_g for item in manufacturing_lines), default=0.0)
        if minimum_mass and minimum_mass < 0.1:
            readability = 0.0001
        elif minimum_mass and minimum_mass < 1:
            readability = 0.001
        else:
            readability = 0.01

        process_steps = [
            "PPE, 환기, 점화원 통제 및 원료별 SDS 작업 조건을 확인합니다.",
            "교정 상태가 확인된 저울로 원료를 질량 기준 계량합니다.",
            "저충격 베이스 원료부터 혼합하고 고체·고점도 원료를 순차 투입합니다.",
            "균질하고 발열이 관리된 상태로 숙성하며 로트와 실제 계량값을 기록합니다.",
            "확정된 제품 베이스를 투입한 뒤 요구되는 안정성 시험을 수행합니다.",
            "출하 전 안정성, 포장 적합성, 규제 및 관능 승인 결과를 확인합니다.",
        ]

        readiness = self.profile_registry.assess(
            lines=lines,
            ingredients_by_id=ingredients_by_id,
            constraints=constraints,
            as_of=as_of or date.today(),
            stability_passed=stability_passed,
        )
        return ManufacturingPlan(
            basis="mass_fraction_as_supplied",
            finished_batch_mass_g=round(finished_mass, 5),
            fragrance_concentrate_mass_g=round(concentrate_mass, 5),
            product_base_mass_g=round(base_mass, 5),
            total_as_supplied_material_mass_g=round(total_supplied, 5),
            recommended_balance_readability_g=readability,
            lines=manufacturing_lines,
            process_steps=process_steps,
            required_stability_tests=list(REQUIRED_STABILITY_TESTS),
            stability_status="passed" if stability_passed else "not_tested",
            ready_for_lab_trial=readiness.ready_for_lab_trial,
            warnings=sorted(set(warnings)),
            ready_for_manufacture=readiness.ready_for_manufacture,
            readiness_blockers=list(readiness.blockers),
            product_base_profile_version=readiness.base_profile_version,
            packaging_profile_version=readiness.packaging_profile_version,
        )
