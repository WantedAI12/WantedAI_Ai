"""Audit the fixed request space and compare real baseline/full-pool generation.

Schema audit: every uniform 1/2/3-axis target on the existing 19 dimensions.
Runtime audit: every 1/2-axis combination in Korean and English, plus explicit
phase and previous V7 requests. This finite suite is not 'all possible text'.
No gold outcomes, external APIs, scoring changes or target redefinitions.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
import hashlib
from itertools import combinations
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 9, 5)
KO = ("시트러스", "프레시", "클린", "그린", "아쿠아틱", "플로럴", "장미", "화이트플로럴", "프루티",
      "스파이시", "아로마틱", "우디", "앰버", "머스크", "구르망", "파우더리", "스모키", "레더", "얼씨")
EXTRAS = (
    "clean fresh citrus woody", "floral fruity woody", "green aromatic woody", "gourmand vanilla woody",
    "soft clean musk", "smoky earthy woody", "opening citrus, drydown woody musk",
    "opening no sweetness, drydown woody musk",
    "opening woody, drydown citrus", "opening floral, drydown musk", "opening green, drydown amber",
    "opening amber, drydown green", "opening musk, drydown floral", "opening citrus, heart floral, drydown woody",
    "첫향은 시트러스, 잔향은 우디 머스크", "첫향은 우디, 잔향은 시트러스", "첫향은 플로럴, 잔향은 머스크",
    "첫향은 그린, 잔향은 앰버", "달지 않은 우디", "머스크 없이 시트러스",
)
ENGINES = None
LIMITS = None
SINGLE_PASS = False
ADAPTIVE_COMPARISON = False
DOSE_COMPARISON = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def guard(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("outbound network forbidden in request-space evaluation")
    if event == "open" and isinstance(args[0], str) and "actualvalue" in args[0].casefold():
        raise RuntimeError("public sensory holdout outcomes are forbidden")


def request_cases():
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    cases = []
    for size in (1, 2):
        for ids in combinations(range(len(SCENT_DIMENSIONS)), size):
            for language in ("en", "ko"):
                text = ", ".join(SCENT_DIMENSIONS[i].replace("_", " ") if language == "en" else KO[i] for i in ids)
                cases.append({"id": f"{language}-{'-'.join(map(str, ids))}", "brief": text + (" scent" if language == "en" else " 향"),
                              "group": f"{language}_{size}_axis", "expected_dimensions": [SCENT_DIMENSIONS[i] for i in ids]})
    cases.extend({"id": f"extra-{i}", "brief": text, "group": "phase_or_regression"} for i, text in enumerate(EXTRAS))
    return cases


def load_catalog(manifest_path):
    if not manifest_path:
        return None
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog
    from fragrance_ai.recommender.runtime import _data_snapshot
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    data = manifest["runtime_catalog"]
    # Offline search experiments deliberately use new consumer code with the
    # same frozen input data. Production RuntimeAIFactory remains code-bound.
    actual = _data_snapshot()
    if any(actual.get(name) != value for name, value in manifest["runtime_data_sha256"].items()):
        raise ValueError("frozen inference data changed during the experiment")
    catalog, _, _ = load_runtime_catalog(path.parent / data["path"], expected_sha256=data["sha256"],
        expected_wheel_sha256=data["wheel_sha256"], expected_registry_sha256=data["registry_sha256"])
    return catalog


def initialize_worker(package_root, output, runtime_manifest=None, include_conditionals=False, single_pass=False, adaptive_comparison=False, dose_comparison=False, target_similarity=90., strict=False, max_ingredients=12):
    global ENGINES, LIMITS, SINGLE_PASS, ADAPTIVE_COMPARISON, DOSE_COMPARISON
    sys.path.insert(0, package_root)
    sys.addaudithook(guard)
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import HistoricalReferenceCorpus
    catalog = load_catalog(runtime_manifest)
    LIMITS = RecipeConstraints(max_risk_tier=2, enable_registry_trace_candidates=True, target_similarity=target_similarity) if include_conditionals else RecipeConstraints(target_similarity=target_similarity)
    LIMITS.max_ingredients = max_ingredients
    SINGLE_PASS = single_pass
    ADAPTIVE_COMPARISON = adaptive_comparison
    DOSE_COMPARISON = dose_comparison
    modes = (("full_pool", True),) if single_pass else (("v7_control", False), ("full_pool", True))
    ENGINES = {
        mode: NaturalLanguagePerfumeryAI(
            corpus=HistoricalReferenceCorpus(Path(output) / "absent-reference.db"),
            enable_full_pool_search=True if adaptive_comparison or dose_comparison else enabled,
            enable_adaptive_pyramid=True if dose_comparison else (enabled if adaptive_comparison else False),
            enable_dose_refinement=enabled if dose_comparison else False, catalog=catalog,
            require_full_profile_match=strict, minimum_profile_target=target_similarity if strict else None,
        ) for mode, enabled in modes
    }
    atexit.register(lambda: [engine.close() for engine in ENGINES.values()])


def run_case(case):
    payloads, elapsed = {}, {}
    for mode, engine in ENGINES.items():
        started = time.perf_counter()
        try:
            payloads[mode] = engine.create_recipe(case["brief"], replace(LIMITS), as_of=AS_OF).to_dict()
        except ValueError as error:
            payloads[mode] = {"status": "invalid_request", "error_type": type(error).__name__, "error": str(error)}
        elapsed[mode] = time.perf_counter() - started
    b = payloads["full_pool"]
    a = payloads.get("v7_control")
    pool = b.get("full_profile_assessment", {}).get("search", {}).get("full_pool_search", {})
    adaptive = pool.get("adaptive_pyramid", {})
    dose = pool.get("dose_refinement", {})
    control = dose if DOSE_COMPARISON else (adaptive if ADAPTIVE_COMPARISON else pool)
    before_key = "baseline_v9_score" if DOSE_COMPARISON else ("baseline_v8_score" if ADAPTIVE_COMPARISON else "baseline_v7_score")
    before = control.get(before_key) if SINGLE_PASS else a.get("calculated_profile_similarity")
    after = b.get("calculated_profile_similarity")
    if b["status"] == "invalid_request" or (a is not None and a["status"] == "invalid_request"):
        if case.get("expected_dimensions"):
            raise AssertionError(f"canonical request was not parsed: {case['brief']}")
        if a is not None and a != b:
            raise AssertionError("search mode changed parser behavior")
    else:
        for payload in payloads.values():
            from fragrance_ai.recommender.odor_integrity import registry_odor_rejection
            by_id = {item.ingredient_id: item for item in ENGINES["full_pool"].catalog.ingredients}
            for field in ("recipe", "closest_candidate"):
                for line in payload.get(field, []):
                    if registry_odor_rejection(by_id[line["ingredient_id"]]):
                        raise AssertionError("unverified odor input leaked into a formula")
            if payload["brief"]["constraints"]["target_similarity"] != LIMITS.target_similarity:
                raise AssertionError("target was changed")
            if payload["actual_olfactory_similarity_score"] is not None or payload["human_similarity_90_claim_authorized"]:
                raise AssertionError("model-only search manufactured human evidence")
            if payload["recipe"] and (not payload["safety"]["internal_gate_passed"]
                                       or abs(sum(line["concentrate_percent"] for line in payload["recipe"]) - 100) > .001):
                raise AssertionError("unsafe or unbalanced formula")
        if before is not None and (after is None or after + 1e-8 < before):
            raise AssertionError(f"full profile regressed for {case['brief']}: {before} -> {after}")
        if a is not None and not a["recipe"] and b["recipe"] and (after is None or after + 1e-8 < LIMITS.target_similarity):
            raise AssertionError("previous rejection promoted without complete target pass")
        expected = set(case.get("expected_dimensions", []))
        if expected:
            allowed = expected | ({"floral"} if expected & {"rose", "white_floral"} else set())
            actual = set(b["brief"]["desired_dimensions"])
            if not expected.issubset(actual) or not actual.issubset(allowed):
                raise AssertionError(f"requested scent meaning changed: {case['brief']} -> {sorted(actual)}")
    row = {
        **case, "before_score": before, "after_score": after,
        "gain_points": None if before is None or after is None else after - before,
        "before_status": a["status"] if a is not None else None, "after_status": b["status"],
        "before_formula_id": a.get("formula_id") if a is not None else control.get(before_key.replace("_score", "_formula_id")), "after_formula_id": b.get("formula_id"),
        "comparison_basis": ("actual_v9_stage_vs_dose_refinement" if DOSE_COMPARISON else ("actual_v8_stage_vs_adaptive" if ADAPTIVE_COMPARISON else "actual_inference_baseline_stage_vs_final")) if SINGLE_PASS else "separate_generator_calls_same_parsed_intent",
        "original_pyramid": dose.get("original_pyramid") if DOSE_COMPARISON else (adaptive.get("original_inferred_pyramid") if ADAPTIVE_COMPARISON else None),
        "selected_pyramid": dose.get("selected_pyramid") if DOSE_COMPARISON else (adaptive.get("selected_pyramid") if ADAPTIVE_COMPARISON else None),
        "recipe_returned": bool(b.get("recipe")), "full_profile_target_met": b.get("full_profile_target_met", False),
        "ingredient_count": len(b.get("recipe", [])), "seconds": elapsed,
    }
    return row, payloads


def schema_audit(runtime_manifest=None, include_conditionals=False, target_similarity=90., max_ingredients=12):
    import numpy as np
    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.global_profile_search import _fractional_lp, profile_upper_bound
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    with NaturalLanguagePerfumeryAI(enable_full_pool_search=False, catalog=load_catalog(runtime_manifest)) as ai:
        limits = RecipeConstraints(max_risk_tier=2, enable_registry_trace_candidates=True, target_similarity=target_similarity) if include_conditionals else RecipeConstraints(target_similarity=target_similarity)
        limits.max_ingredients = max_ingredients
        base = ai.parser.parse("woody", limits)
        candidates, _ = ai.screen.screen(ai.catalog, base, ai.supplier_registry, as_of=AS_OF)
        rows = []
        for size in (1, 2, 3):
            for dimensions in combinations(SCENT_DIMENSIONS, size):
                target = {name: 1 / size for name in dimensions}
                request = replace(base, target_profile=target, desired_dimensions=list(dimensions))
                bound = profile_upper_bound(candidates, target)
                relaxed = _fractional_lp(candidates, request)
                numerical_upper = None if relaxed.relaxed_overlap_score is None else min(
                    100., relaxed.relaxed_overlap_score + bound["legacy_render_allowance_points"],
                )
                rows.append({"dimensions": dimensions, "convex_upper": bound["upper_score"],
                             "lp_status": relaxed.status, "lp_overlap": relaxed.relaxed_overlap_score,
                             "lp_render_adjusted_upper": numerical_upper})
    return {
        "scope": "all uniform 1/2/3-axis structured targets, fixed standard pyramid; not all free text or mixtures",
        "candidate_count": len(candidates), "candidate_ids": [item.ingredient_id for item in candidates],
        "pyramid_ratios": base.pyramid_ratios, "max_cost": base.constraints.max_formula_cost_per_kg,
        "target_unchanged": target_similarity, "case_count": len(rows),
        "analytic_bound_below_target": sum(row["convex_upper"] < target_similarity for row in rows),
        "analytic_bound_below_90": sum(row["convex_upper"] < 90 for row in rows),
        "numerical_lp_bound_below_90": sum(row["lp_render_adjusted_upper"] is not None and row["lp_render_adjusted_upper"] < 90 for row in rows),
        "maximum_numerical_lp_upper": float(np.max([row["lp_render_adjusted_upper"] for row in rows if row["lp_render_adjusted_upper"] is not None])),
        "lp_scope": "overlap/avoidance relaxation; ignores cosine, cardinality, detailed safety and time physics; numerical not formal certificate",
        "actual_human_accuracy_claim": False, "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2, 3, 4))
    parser.add_argument("--target-similarity", type=float, default=90.)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-ingredients", type=int, default=12)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--installed-package", type=Path)
    parser.add_argument("--runtime-catalog-manifest", type=Path)
    parser.add_argument("--include-conditionals", action="store_true")
    parser.add_argument("--skip-schema", action="store_true")
    parser.add_argument("--single-pass", action="store_true", help="use the actual baseline-stage score already evaluated inside each inference")
    parser.add_argument("--adaptive-comparison", action="store_true", help="compare V8's completed stage against adaptive note allocation")
    parser.add_argument("--dose-comparison", action="store_true", help="compare V9's completed stage against actual-dose joint-time refinement")
    parser.add_argument("--case-ids", nargs="+", help="explicit subset of the unchanged case IDs for focused development checks")
    args = parser.parse_args()
    if not 3 <= args.max_ingredients <= 50000:
        parser.error("max ingredients must be in [3, 50000]")
    if not 0 < args.target_similarity <= 100:
        parser.error("target similarity must be finite and in (0, 100]")
    if args.output.exists():
        parser.error("use a new output directory")
    if args.include_conditionals and not args.runtime_catalog_manifest:
        parser.error("conditional mode requires an explicit hash-bound runtime catalog")
    if args.adaptive_comparison and args.dose_comparison:
        parser.error("choose exactly one stage comparison")
    code_root = (args.installed_package or ROOT).resolve()
    sys.path.insert(0, str(code_root))
    import fragrance_ai
    if not Path(fragrance_ai.__file__).resolve().is_relative_to(code_root):
        raise RuntimeError("wrong package imported")
    cases = request_cases()
    if args.case_ids is not None:
        if len(set(args.case_ids)) != len(args.case_ids) or set(args.case_ids) - {case["id"] for case in cases}:
            parser.error("case IDs must be unique existing requests")
        cases = [case for case in cases if case["id"] in args.case_ids]
    if args.max_cases is not None:
        if not 1 <= args.max_cases <= len(cases):
            parser.error("invalid case count")
        cases = cases[:args.max_cases]
    args.output.mkdir(parents=True)
    snapshot_paths = (
        "fragrance_ai/recommender/profile_match.py", "fragrance_ai/recommender/brief_parser.py",
        "fragrance_ai/recommender/models.py", "fragrance_ai/data/safe_ingredient_catalog.json",
        "fragrance_ai/recommender/global_profile_search.py", "fragrance_ai/recommender/optimizer.py", "fragrance_ai/recommender/service.py",
        "fragrance_ai/recommender/adaptive_pyramid.py",
        "fragrance_ai/recommender/dose_refinement.py", "fragrance_ai/recommender/science.py",
        "fragrance_ai/recommender/odor_integrity.py", "fragrance_ai/recommender/registry_activation.py",
        "fragrance_ai/recommender/industrial_catalog.py", "fragrance_ai/recommender/safety.py",
        "fragrance_ai/recommender/runtime.py",
    )
    snapshot = {name: digest(code_root / name) for name in snapshot_paths}
    protocol = {"target_unchanged": args.target_similarity, "strict": args.strict, "max_ingredients": args.max_ingredients, "as_of": AS_OF.isoformat(), "cases": cases, "input_code_sha256": snapshot,
                "workers": args.workers, "script_sha256": digest(__file__), "no_human_holdout_outcomes": True,
                "imported_package": str(Path(fragrance_ai.__file__).resolve()),
                "catalog_scope": "conditional_registry_prototype" if args.include_conditionals else "default_reviewed",
                "comparison_basis": ("actual_v9_stage_vs_dose_refinement" if args.dose_comparison else ("actual_v8_stage_vs_adaptive" if args.adaptive_comparison else "actual_inference_baseline_stage_vs_final")) if args.single_pass else "separate_generator_calls_same_parsed_intent",
                "experimental_disable_safety": False,
                "runtime_catalog_manifest_sha256": digest(args.runtime_catalog_manifest) if args.runtime_catalog_manifest else None}
    protocol["catalog_binding_mode"] = "frozen_data_with_recorded_experimental_consumer_source"
    write_json(args.output / "protocol.json", protocol)
    (args.output / "evaluate_request_space_snapshot.py").write_bytes(Path(__file__).read_bytes())
    runtime_manifest = str(args.runtime_catalog_manifest.resolve()) if args.runtime_catalog_manifest else None
    audit = (schema_audit(runtime_manifest, args.include_conditionals, args.target_similarity, args.max_ingredients) if not args.skip_schema else
             {"status": "not_run_by_explicit_flag", "case_count": 0, "numerical_lp_bound_below_90": None})
    write_json(args.output / "schema_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key not in ("cases", "candidate_ids")}), flush=True)
    if args.schema_only:
        return 0
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker,
                             initargs=(str(code_root), str(args.output.resolve()), runtime_manifest, args.include_conditionals, args.single_pass, args.adaptive_comparison, args.dose_comparison, args.target_similarity, args.strict, args.max_ingredients)) as executor:
        for index, (row, payloads) in enumerate(executor.map(run_case, cases, chunksize=1)):
            rows.append(row)
            write_json(args.output / f"{index:04d}.json", payloads)
            if (index + 1) % 20 == 0 or index + 1 == len(cases):
                print(json.dumps({"completed": index + 1, "total": len(cases),
                                  "improved": sum((item["gain_points"] or 0) > 1e-6 for item in rows),
                                  "target": args.target_similarity,
                                  "full_profile_target_pass": sum(item["full_profile_target_met"] for item in rows)}), flush=True)
    if snapshot != {name: digest(code_root / name) for name in snapshot_paths}:
        raise RuntimeError("code or data changed during the evaluation")
    report = {
        "scope": "finite bilingual request-space integration audit; not all possible text or human accuracy",
        "target_unchanged": args.target_similarity, "strict": args.strict, "max_ingredients": args.max_ingredients, "case_count": len(rows),
        "catalog_scope": protocol["catalog_scope"], "experimental_disable_safety": False,
        "comparison_basis": protocol["comparison_basis"],
        "canonical_meaning_checks_passed": sum(bool(case.get("expected_dimensions")) for case in cases),
        "improved_case_count": sum((row["gain_points"] or 0) > 1e-6 for row in rows),
        "calculable_case_count": sum(row["after_score"] is not None for row in rows),
        "full_profile_90_case_count": sum(row["after_score"] is not None and row["after_score"] >= 90 for row in rows),
        "full_profile_target_case_count": sum(row["full_profile_target_met"] for row in rows),
        "recipe_returned_count": sum(row["recipe_returned"] for row in rows),
        "regressions": 0, "maximum_gain": max((row["gain_points"] or 0) for row in rows),
        "unverified_registry_odor_formula_lines": 0,
        "schema_case_count": audit["case_count"], "schema_numerical_upper_below_90": audit["numerical_lp_bound_below_90"],
        "source_sha256": snapshot, "protocol_sha256": digest(args.output / "protocol.json"),
        "schema_audit_sha256": digest(args.output / "schema_audit.json"), "cases": rows,
    }
    write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
