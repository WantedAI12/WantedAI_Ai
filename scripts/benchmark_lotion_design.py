"""Frozen 400-request software diagnostic, including all failures (not human accuracy)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import os
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CATALOG = None
AUTO_BASE = False
ADAPTIVE_STEPS = 3
PROVIDER = None
BUNDLE = None


def release_audit(result):
    """Read-only V58 diagnostics; never change recipe search or acceptance."""
    simulations = ([result['simulation']] if result.get('simulation') else []) + result.get('scenario_simulations', [])
    releases = [s.get('learned_release') for s in simulations]
    attached = [r for r in releases if r is not None]
    comparisons = [r['same_request_solver_comparison'] for r in attached if 'same_request_solver_comparison' in r]
    return {'final_scenario_count': len(simulations), 'attached_count': len(attached),
        'statuses': dict(Counter(r.get('status') for r in attached)),
        'abstention_reasons': dict(Counter(r.get('reason') for r in attached if r.get('status') == 'abstained')),
        'checkpoint_sha256s': sorted({r['checkpoint_sha256'] for r in attached}),
        'all_final_scenarios_attached': bool(simulations) and len(attached) == len(simulations),
        'maximum_state_fraction_error_vs_solver': max((c['maximum_state_fraction_error'] for c in comparisons), default=None),
        'mean_scenario_state_fraction_mae_vs_solver': (sum(c['state_fraction_mae'] for c in comparisons)/len(comparisons)) if comparisons else None,
        'mass_balance_max_abs_error_mg_cm2': max((r.get('diagnostics', {}).get('mass_balance_max_abs_error_mg_cm2', 0.) for r in attached), default=None),
        'nonmonotone_scenario_count': sum(r.get('diagnostics', {}).get('cumulative_sinks_monotone') is False for r in attached),
        'negative_mass_scenario_count': sum(r.get('diagnostics', {}).get('nonnegative') is False for r in attached)}


def initialize(auto_base=False, catalog_manifest=None, adaptive_steps=3, catalog_sha=None, model_mode='local'):
    global CATALOG, AUTO_BASE, ADAPTIVE_STEPS, PROVIDER, BUNDLE
    AUTO_BASE = auto_base
    ADAPTIVE_STEPS = adaptive_steps
    from fragrance_ai.recommender.runtime import MANIFEST_ENV, MANIFEST_HASH_ENV, load_verified_catalog_bundle
    from fragrance_ai.recommender.local_runtime import configured_pair, local_lotion_provider
    from fragrance_ai.recommender.perception_runtime import configured_perception
    path, digest = ((catalog_manifest, catalog_sha) if catalog_manifest is not None or catalog_sha is not None
                    else configured_pair(MANIFEST_ENV, MANIFEST_HASH_ENV, 'catalog'))
    BUNDLE = load_verified_catalog_bundle(path, digest)
    CATALOG = BUNDLE['catalog']
    if model_mode == 'none':
        PROVIDER = None
    elif model_mode in ('local', 'component', 'atlas'):
        component = configured_perception('body_lotion')
        PROVIDER = local_lotion_provider(component, reference=None if model_mode == 'local' else model_mode)
        if PROVIDER is None:
            raise ValueError('selected learned model is not configured; use --model-mode none only for a named ablation')
    else:
        raise ValueError('unknown benchmark model mode')


def run(case):
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    start = time.perf_counter()
    try:
        r = estimate_lotion_recipe(LotionEstimateRequest(brief=case["brief"], registry_pool="conditional_research", max_risk_tier=2,
            base_design={'adaptive_oil_refinement_steps': ADAPTIVE_STEPS} if AUTO_BASE else None), CATALOG,
            perception_guidance=PROVIDER, use_configured_perception=False)
        design = r.get("base_design", {})
        return {**case, **{key: r.get(key) for key in ("status", "score", "baseline_score", "profile_target_met", "search_incomplete", "attainability")},
                "fixed_reference_score": design.get("fixed_reference_score", r.get("score")),
                "fixed_reference_passed95": design.get("fixed_reference_passed95", bool(r.get("profile_target_met"))),
                "selected_oil_base_percent": design.get("selected_oil_base_percent", 10.),
                "base_variants_evaluated": len(design.get("evaluated_variants", [])) or 1,
                "adaptive_oil_trials": design.get('adaptive_oil_trials', []),
                "profile_coverage": design.get('profile_coverage'),
                "additional_variants_skipped": design.get('additional_variants_skipped'),
                "basis_cache_status": r.get('basis_cache_status'),
                "transport_simulation_calls": r.get('transport_simulation_calls'),
                "solver_conditioning_calls": r.get('solver_conditioning_calls'),
                "conic_recovery": r.get('conic_recovery'), "accord_refinement": r.get('accord_refinement'),
                "perception_model": {key: (r.get('perception_model') or {}).get(key) for key in (
                    'configured','component_model_sha256','component_model_version','status','optimization_guidance_evaluated',
                    'optimization_guidance_applied','unmodeled_target_axes','strict_score_modified')},
                "learned_optimization_status": (r.get('learned_optimization') or {}).get('status'),
                "physical_evidence_connection": r['estimation'].get('physical_evidence_connection'),
                "physical_coverage": r['estimation'].get('coverage'),
                "learned_release_audit": release_audit(r),
                "recipe_emitted": bool(r.get('recipe')), "closest_candidate_count": len(r.get('closest_candidate', [])),
                "timepoint_assessments": r.get('timepoint_assessments', []),
                "intent": r.get('preparation', {}).get('intent'),
                "candidate_count": r["estimation"]["candidate_count"], "seconds": time.perf_counter()-start}
    except Exception as error:
        return {**case, "status": "error", "score": None, "profile_target_met": False,
                "error": type(error).__name__ + ": " + str(error), "seconds": time.perf_counter()-start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1,9))
    p.add_argument("--auto-base", action="store_true", help="opt in to oil/water base co-design; fixed baseline is retained")
    p.add_argument("--case-id", action="append", help="run an explicitly identified diagnostic subset; default is all 400")
    p.add_argument('--catalog-manifest', type=Path, help='evaluate a separate hash-bound candidate without changing deployment configuration')
    p.add_argument('--catalog-manifest-sha256', help='trusted digest required together with an explicit manifest')
    p.add_argument('--model-mode', choices=('local','component','atlas','none'), default='local',
                   help='local uses the same pinned model as API/SDK; none is an explicit transport-only ablation')
    p.add_argument('--adaptive-oil-steps', type=int, default=3, choices=range(7))
    args = p.parse_args()
    if bool(args.catalog_manifest) != bool(args.catalog_manifest_sha256):
        p.error('explicit catalog manifest and trusted SHA256 must be supplied together')
    if args.output.exists():
        p.error("choose a new output directory")
    from scripts.evaluate_request_space import request_cases
    cases = request_cases()
    if args.case_id:
        unknown = set(args.case_id) - {case['id'] for case in cases}
        if unknown:
            p.error('unknown case IDs: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['id'] in set(args.case_id)]
    initialize(args.auto_base, args.catalog_manifest, args.adaptive_oil_steps, args.catalog_manifest_sha256, args.model_mode)
    args.output.mkdir(parents=True)
    source_names = ["fragrance_ai/recommender/lotion_estimation.py", "fragrance_ai/recommender/lotion_optimizer.py",
                    "fragrance_ai/recommender/lotion.py", "fragrance_ai/recommender/formulation_science.py",
                    "fragrance_ai/platform/lotion_inputs.py", "fragrance_ai/platform/formulation_inputs.py",
                    "fragrance_ai/platform/lotion_reference.py", "fragrance_ai/recommender/lotion_coverage.py",
                    "fragrance_ai/recommender/lotion_numerics.py", "fragrance_ai/recommender/lotion_basis_cache.py",
                    "fragrance_ai/recommender/brief_parser.py", "fragrance_ai/recommender/odor_integrity.py",
                    "fragrance_ai/recommender/models.py", "scripts/evaluate_request_space.py",
                    "scripts/benchmark_lotion_design.py", "fragrance_ai/recommender/lotion_learned_search.py",
                    "fragrance_ai/recommender/lotion_perception.py", "fragrance_ai/recommender/lotion_evaluation.py", "fragrance_ai/recommender/perception_guidance.py",
                    "fragrance_ai/recommender/perception_runtime.py", "fragrance_ai/recommender/lotion_incumbent.py",
                    "fragrance_ai/recommender/odor_descriptors.py", "fragrance_ai/data/odor_descriptor_projections.json"]
    source_names += ['fragrance_ai/recommender/lotion_conic.py', 'fragrance_ai/recommender/accord_trials.py',
                     'fragrance_ai/recommender/lotion_profile_balance.py']
    from fragrance_ai.recommender.runtime import _source_snapshot, _data_snapshot
    from fragrance_ai.recommender.perception_runtime import model_contract, assert_provider_current
    from fragrance_ai.recommender.local_runtime import local_snapshot
    from fragrance_ai.recommender.lotion_surrogate import configured_lotion_surrogate
    release_model = configured_lotion_surrogate()
    hashes = {**_source_snapshot(), **_data_snapshot(),
              **{name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_names}}
    initial_local_snapshot = local_snapshot()
    artifact = BUNDLE['binding']
    RUNTIME_CATALOG_SHA256, WHEEL_SHA256, REGISTRY_SHA256 = artifact['sha256'], artifact['wheel_sha256'], artifact['registry_sha256']
    from fragrance_ai.recommender.odor_descriptors import LANGUAGE_REPRESENTATION_VERSION
    manifest = {"cases": cases, "target": 95, "source_sha256": hashes, "workers": args.workers,
                "product": "body_lotion", "evaluation_version": "body-lotion-transport-profile/v2",
                "language_representation_version": LANGUAGE_REPRESENTATION_VERSION,
                "base_variant_incumbent_transfer": True,
                "auto_base_design": args.auto_base, "oil_base_percent_design_space": [10,20,30] if args.auto_base else [10],
                "adaptive_oil_refinement_steps": args.adaptive_oil_steps if args.auto_base else 0,
                "adaptive_envelope": [10,30] if args.auto_base else None,
                "learned_model_enabled": PROVIDER is not None,
                "model_mode": args.model_mode, "perception_model": model_contract(PROVIDER),
                "trained_lotion_release": release_model.contract() if release_model is not None else None,
                "catalog_manifest_sha256": BUNDLE['manifest_sha256'],
                "local_profile_sha256": initial_local_snapshot[0],
                "same_primary_95_gate_with_secondary_learned_search": True,
                "runtime_catalog_sha256": RUNTIME_CATALOG_SHA256,
                "runtime_bound_wheel_sha256": WHEEL_SHA256, "registry_sha256": REGISTRY_SHA256,
                "case_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "pool": "conditional_research_risk2_existing_screen_and_caps",
                "scope": "selected_frozen_request_subset_not_full400" if args.case_id else
                    "400_frozen_requests_not_all_possible_language_requests_or_human_accuracy"}
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    start = time.perf_counter()
    with (args.output/"results.jsonl").open("w", encoding="utf-8") as f, ProcessPoolExecutor(args.workers, initializer=initialize,
            initargs=(args.auto_base, str(args.catalog_manifest.resolve()) if args.catalog_manifest else None, args.adaptive_oil_steps,
                      args.catalog_manifest_sha256, args.model_mode)) as executor:
        futures = [executor.submit(run, case) for case in cases]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            f.flush()
            if len(rows) % 20 == 0:
                print(json.dumps({"completed": len(rows), "total": len(cases), "passed95": sum(bool(r.get("profile_target_met")) for r in rows),
                                  "elapsed_seconds": round(time.perf_counter()-start, 1)}), flush=True)
    assert_provider_current(PROVIDER)
    if release_model is not None:
        release_model.assert_current()
    unchanged = (all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name,sha in hashes.items())
                 and local_snapshot() == initial_local_snapshot
                 and hashlib.sha256(BUNDLE['manifest_path'].read_bytes()).hexdigest() == BUNDLE['manifest_sha256']
                 and hashlib.sha256(BUNDLE['catalog_path'].read_bytes()).hexdigest() == RUNTIME_CATALOG_SHA256)
    summary = {"count": len(rows), "passed95": sum(bool(r.get("profile_target_met")) for r in rows),
        "product": "body_lotion", "cross_product_scores_comparable": False,
        "errors": sum(r["status"] == "error" for r in rows), "seconds": time.perf_counter()-start,
        "source_unchanged": unchanged, "universal_95_proven": False, "human_accuracy_measured": False,
        "learned_model_enabled": PROVIDER is not None, "perception_model": model_contract(PROVIDER),
        "failed_request_ids": [r["id"] for r in rows if not r.get("profile_target_met")]}
    summary["pass_rate_percent"] = summary["passed95"] / len(rows) * 100
    summary["fixed_reference_passed95_in_same_run"] = sum(bool(r.get("fixed_reference_passed95")) for r in rows)
    summary["additional_passes_from_base_design"] = sum(bool(r.get("profile_target_met")) and not r.get("fixed_reference_passed95") for r in rows)
    summary["lost_fixed_reference_passes"] = sum(bool(r.get("fixed_reference_passed95")) and not r.get("profile_target_met") for r in rows)
    excluded_ids = [r['id'] for r in rows if (r.get('profile_coverage') or {}).get('target_excluded')]
    summary['profile_upper_excluded_ids'] = excluded_ids
    summary['frozen_profile_max_possible_passes_upper'] = len(rows)-len(excluded_ids)
    summary['upper_bound_scope'] = 'current_frozen_catalog_profile_pool_not_future_models_or_real_human_olfaction'
    summary['frozen_400_complete_pass'] = len(rows) == 400 and summary['passed95'] == 400 and summary['errors'] == 0 and unchanged
    audits = [r['learned_release_audit'] for r in rows if 'learned_release_audit' in r]
    summary['learned_release_audit'] = {
        'requests_with_audit': len(audits),
        'requests_with_all_scenarios_attached': sum(a['all_final_scenarios_attached'] for a in audits),
        'final_scenarios': sum(a['final_scenario_count'] for a in audits),
        'attached_scenarios': sum(a['attached_count'] for a in audits),
        'statuses': dict(sum((Counter(a['statuses']) for a in audits), Counter())),
        'abstention_reasons': dict(sum((Counter(a['abstention_reasons']) for a in audits), Counter())),
        'nonmonotone_scenarios': sum(a['nonmonotone_scenario_count'] for a in audits),
        'negative_mass_scenarios': sum(a['negative_mass_scenario_count'] for a in audits),
        'checkpoint_sha256s': sorted({sha for a in audits for sha in a['checkpoint_sha256s']})}
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
