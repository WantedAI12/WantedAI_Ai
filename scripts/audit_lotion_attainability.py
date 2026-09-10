"""Isolate catalogue coverage in a synthetic equal-transport counterfactual.

Never creates a deployable parameter dataset or exports a manufacturing formula.
The supplied fixture coefficients are invented TEST values, not lotion evidence.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion, prepare_lotion_optimization
from fragrance_ai.recommender.registry_activation import load_runtime_catalog
from tests.test_lotion import data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only-full", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    from deploy.modal_app import RUNTIME_CATALOG, RUNTIME_CATALOG_SHA256, WHEEL_SHA256, REGISTRY_SHA256
    runtime, _, _ = load_runtime_catalog(RUNTIME_CATALOG, expected_sha256=RUNTIME_CATALOG_SHA256,
        expected_wheel_sha256=WHEEL_SHA256, expected_registry_sha256=REGISTRY_SHA256)
    builtin = IngredientCatalog.load_builtin()
    initial = [item for item in builtin.ingredients if item.formulation_ready and not item.blocked and item.risk_tier == 1]
    simulation = data()
    simulation.update(transport_mode="bidirectional_air", coefficient_scope="dilute_fixed_base",
        coefficient_scope_reference="SYNTHETIC TEST ONLY: identical transport to isolate catalogue coverage",
        times_minutes=[0., 15., 60.])
    template = dict(simulation["materials"][0])
    request = {"simulation": simulation, "brief": "clean fresh citrus woody scent", "minimum_air_concentration_mg_m3": 1e-8}
    rows = []
    for label, catalog, risk, registry in [
        ("original_core_caps", builtin, 1, "core"),
        ("diagnostic_core_caps_relaxed_NOT_a_safe_recipe", IngredientCatalog([replace(item, max_concentrate_percent=100.) for item in builtin.ingredients]), 1, "core"),
        ("full_eligible_catalog_with_SYNTHETIC_transport", runtime, 2, "conditional_research"),
    ]:
        if args.only_full and label != "full_eligible_catalog_with_SYNTHETIC_transport":
            continue
        request.update(max_risk_tier=risk, registry_pool=registry)
        request["simulation"]["materials"] = [dict(template, ingredient_id=item.ingredient_id, concentrate_percent=100./len(initial)) for item in initial]
        parsed = LotionOptimizationRequest.model_validate(request)
        # Determine the eligible pool with existing policy; do not activate or
        # write invented coefficients into the runtime/source registry.
        from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
        from fragrance_ai.recommender.models import RecipeConstraints
        from fragrance_ai.recommender.safety import CandidateSafetyScreen
        constraints = RecipeConstraints(product_category="body_lotion", max_risk_tier=risk,
            enable_registry_trace_candidates=registry == "conditional_research", min_availability=.75)
        brief = NaturalLanguageBriefParser(catalog).parse(request["brief"], constraints)
        candidates, _ = CandidateSafetyScreen().screen(catalog, brief)
        candidates = [item for item in candidates if item.active_strength_percent == 100. and sum(item.vector()) > 0]
        request["simulation"]["materials"] = [dict(template, ingredient_id=item.ingredient_id,
            concentrate_percent=100./len(candidates)) for item in candidates]
        parsed = LotionOptimizationRequest.model_validate(request)
        start = time.perf_counter()
        result = optimize_lotion(parsed, catalog)
        row = {"case": label, "candidate_count": len(candidates), "status": result["status"],
            "score": result.get("score"), "solver_calls": result["solver_calls"],
            "search_incomplete": result.get("search_incomplete", result["status"] == "search_incomplete"),
            "attainability": result.get("attainability"),
            "seconds": time.perf_counter()-start,
            "manufacturing_formula_exported": False, "measured_transport": False}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"schema": "lotion-attainability-audit-1", "claim": "synthetic_counterfactual_not_lotion_accuracy",
        "brief": request["brief"], "runtime_catalog_sha256": RUNTIME_CATALOG_SHA256,
        "catalog_bound_wheel_sha256": WHEEL_SHA256, "rows": rows,
        "executed_source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ["fragrance_ai/recommender/lotion_optimizer.py", "fragrance_ai/recommender/lotion.py", "fragrance_ai/recommender/lotion_transport.py"]},
        "universal_95_proven": False, "measured_lotion_validation_performed": False,
        "warning": "Relaxed caps and synthetic transport exist only inside this diagnostic process, not the service."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
