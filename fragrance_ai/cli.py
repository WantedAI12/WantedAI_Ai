"""Command-line entry point for recipe candidates and the optional legacy demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .domain.fragrance_chemistry import FragranceChemistry
from .recommender import NaturalLanguagePerfumeryAI, RecipeConstraints
from .recommender.artifact_trust import EvidenceTrustRoot
from .recommender.data_hub import NonHumanDataHub
from .recommender.odor_profiles import OdorProfileStore
from .recommender.quality import QualityEvidenceStore
from .recommender.release import CommercialReleaseStore
from .recommender.science import ScientificPropertyStore
from .recommender.sensory import CalibrationArtifact, SensoryEvaluationStore
from .recommender.supplier import SupplierRegistry
from .rules.ifra_rules import ProductCategory, check_compliance


def _json_object(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def run_demo(population: int = 8, generations: int = 2) -> dict[str, Any]:
    try:
        from .ai.unified_ai_system import UnifiedAIConfig, UnifiedFragranceAI
    except ImportError as error:
        raise RuntimeError(
            "legacy AI demo dependencies are missing; "
            "install perfumery-ai-core[legacy-ai]"
        ) from error
    config = UnifiedAIConfig(
        dl_embedding_dim=32,
        dl_num_layers=1,
        dl_num_heads=4,
        dl_max_length=24,
        moga_population_size=population,
        moga_generations=generations,
        device="cpu",
        seed=42,
    )
    ai = UnifiedFragranceAI(config)
    generated = ai.generate_with_dl([1, 2])
    pareto = ai.optimize_with_moga()
    evolved = ai.evolve_with_rl([5.0] * 20, 4.0)
    chemistry = FragranceChemistry.evaluate_fragrance_complete(
        [("bergamot", 20.0), ("lemon", 5.0)],
        [("rose", 20.0), ("jasmine", 15.0)],
        [("sandalwood", 20.0), ("vanilla", 20.0)],
    )
    compliance = check_compliance(
        {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 2.0},
                {"name": "Rose Absolute", "concentration": 1.0},
                {"name": "Sandalwood", "concentration": 97.0},
            ]
        },
        ProductCategory.EAU_DE_PARFUM,
    )
    return {
        "deep_learning": {"generated_note_count": len(generated["notes"][0])},
        "moga": {"pareto_solution_count": len(pareto)},
        "reinforcement_learning": {
            "action": evolved["action_taken"],
            "normalized_total": round(sum(evolved["evolved_formula"]), 6),
        },
        "chemistry": chemistry,
        "ifra": {
            "overall_compliant": compliance["overall_compliant"],
            "embedded_limits_compliant": compliance["ifra"].get(
                "embedded_limits_compliant",
                compliance["ifra"].get("compliant", False),
            ),
            "violation_count": compliance["ifra"]["count"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="자연어 기반 안전 제약 조향 R&D 후보 생성기"
    )
    parser.add_argument("--brief", help="원하는 향을 한국어 또는 영어 자연어로 설명")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--target-similarity", type=float, default=90.0)
    parser.add_argument(
        "--max-price", type=float, default=300.0, help="개별 원료 kg당 가격 상한"
    )
    parser.add_argument(
        "--max-formula-cost",
        type=float,
        default=180.0,
        help="농축액 kg당 예상 원가 상한",
    )
    parser.add_argument("--volume", type=float, default=50.0, help="완제품 용량(ml)")
    parser.add_argument(
        "--batch-mass", type=float, default=50.0, help="완제품 배치 질량(g)"
    )
    parser.add_argument(
        "--product-concentration",
        type=float,
        default=15.0,
        help="완제품 향료 농도(퍼센트)",
    )
    parser.add_argument(
        "--validation-level",
        choices=("prototype", "qualified", "commercial"),
        default="prototype",
    )
    parser.add_argument("--region", default="EU", help="목표 판매 지역 코드")
    parser.add_argument("--product-category", default="eau_de_parfum")
    parser.add_argument("--supplier-csv", help="검증할 공급사 원료 CSV")
    parser.add_argument("--sensory-db", help="관능평가 증거 인덱스 SQLite DB")
    parser.add_argument("--quality-db", help="안정성·파일럿 증거 인덱스 SQLite DB")
    parser.add_argument(
        "--calibration", help="서명 검증 데이터로 생성된 관능 보정 JSON"
    )
    parser.add_argument("--odor-db", help="원료별 실측 관능 프로필 SQLite DB")
    parser.add_argument("--scientific-db", help="분자·증기압·후각역치 물성 SQLite DB")
    parser.add_argument("--release-db", help="외부 규제 서명·보고서 인덱스 SQLite DB")
    parser.add_argument("--data-hub", help="비인간 데이터 근거·참조 SQLite DB")
    parser.add_argument(
        "--evidence-trust-root",
        help="품질·관능 증거용 독립 Ed25519 signer allowlist JSON",
    )
    parser.add_argument(
        "--release-trust-root",
        help="규제 출시 승인용 독립 Ed25519 signer allowlist JSON",
    )
    parser.add_argument("--product-base-id", default="")
    parser.add_argument("--packaging-id", default="")
    parser.add_argument("--rule-pack-version", default="")
    parser.add_argument("--data-version", default="")
    parser.add_argument("--model-version", default="")
    parser.add_argument(
        "--supplier-evidence-json",
        help="원료별 supplier/SKU/lot 및 실제 문서 경로 JSON",
    )
    parser.add_argument("--reference-target-id", default="")
    parser.add_argument("--min-panelists", type=int, default=12)
    parser.add_argument("--min-experts", type=int, default=3)
    parser.add_argument("--simulation-draws", type=int, default=200)
    parser.add_argument(
        "--no-simulation-gate",
        action="store_true",
        help="Deprecated compatibility flag; synthetic diagnostics never approve by default",
    )
    parser.add_argument(
        "--require-evidenced-simulation-gate",
        action="store_true",
        help="검증된 정량 기준 향이 있을 때만 비인간 비교 게이트를 필수화",
    )
    parser.add_argument(
        "--allow-moderate-risk",
        action="store_true",
        help="중간 위험 등급 2 원료 허용(기본값은 등급 1 이하)",
    )
    args = parser.parse_args()

    if not args.brief:
        print(
            json.dumps(
                run_demo(args.population, args.generations),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    evidence_root = (
        EvidenceTrustRoot.from_json_file(args.evidence_trust_root)
        if args.evidence_trust_root
        else EvidenceTrustRoot()
    )
    supplier_evidence = _json_object(args.supplier_evidence_json)
    constraints = RecipeConstraints(
        target_similarity=args.target_similarity,
        max_ingredient_price_per_kg=args.max_price,
        max_formula_cost_per_kg=args.max_formula_cost,
        finished_volume_ml=args.volume,
        finished_batch_mass_g=args.batch_mass,
        product_concentration_percent=args.product_concentration,
        max_risk_tier=2 if args.allow_moderate_risk else 1,
        validation_level=args.validation_level,
        target_region=args.region.upper(),
        product_category=args.product_category,
        min_panelists=args.min_panelists,
        min_expert_panelists=args.min_experts,
        simulation_draws=args.simulation_draws,
        require_simulation_pass=(
            args.require_evidenced_simulation_gate and not args.no_simulation_gate
        ),
        commercial_product_base_id=args.product_base_id,
        commercial_packaging_id=args.packaging_id,
        commercial_rule_pack_version=args.rule_pack_version,
        commercial_data_version=args.data_version,
        commercial_model_version=args.model_version,
        commercial_supplier_evidence=supplier_evidence,
        reference_target_id=args.reference_target_id,
    )
    supplier_registry = (
        SupplierRegistry.from_csv(args.supplier_csv)
        if args.supplier_csv
        else SupplierRegistry.load_builtin()
    )
    sensory_store = (
        SensoryEvaluationStore(args.sensory_db, trusted_signers=evidence_root)
        if args.sensory_db
        else None
    )
    quality_store = (
        QualityEvidenceStore(args.quality_db, trusted_signers=evidence_root)
        if args.quality_db
        else None
    )
    calibration = (
        CalibrationArtifact.load(Path(args.calibration)) if args.calibration else None
    )
    odor_store = OdorProfileStore(args.odor_db) if args.odor_db else None
    scientific_store = (
        ScientificPropertyStore(args.scientific_db) if args.scientific_db else None
    )
    release_store = None
    if args.release_db:
        release_policy = _json_object(args.release_trust_root)
        release_store = CommercialReleaseStore(
            args.release_db,
            trusted_signers=release_policy.get("signers", release_policy),
        )
    data_hub = NonHumanDataHub(args.data_hub) if args.data_hub else None
    result = NaturalLanguagePerfumeryAI(
        supplier_registry=supplier_registry,
        sensory_store=sensory_store,
        quality_store=quality_store,
        calibration=calibration,
        odor_store=odor_store,
        scientific_store=scientific_store,
        release_store=release_store,
        data_hub=data_hub,
    ).create_recipe(args.brief, constraints)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.status not in {
        "prototype_ready",
        "lab_validated",
        "commercial_evidence_ready",
        "manufacturing_ready",
    }:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
