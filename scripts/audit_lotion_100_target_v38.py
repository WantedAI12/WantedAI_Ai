"""Explain the unchanged 95-point target using current profile support bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-manifest', type=Path, help='audit the exact candidate catalog used by the benchmark')
    args = p.parse_args()
    if args.output.exists():
        p.error('preserve existing evidence')
    from scripts.benchmark_lotion_design import initialize
    from scripts import benchmark_lotion_design as bench
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    initialize(catalog_manifest=args.catalog_manifest)
    inputs, _, provenance = build_estimated_lotion_inputs(LotionEstimateRequest(brief='floral fruity woody',
        registry_pool='conditional_research', max_risk_tier=2), bench.CATALOG)
    ids = {row.ingredient_id for row in inputs.simulation.materials}
    pool = [item for item in bench.CATALOG.ingredients if item.ingredient_id in ids]
    matrix = np.array([item.vector() for item in pool])
    matrix /= matrix.sum(axis=1)[:, None]
    pure_bounds = []
    for j, name in enumerate(SCENT_DIMENSIONS):
        limit = float(matrix[:, j].max())
        pure_bounds.append({'axis': name, 'maximum_pure_axis_overlap_percent': 100*limit,
            'target95_excluded': limit < .95-1e-10,
            'maximal_profile_material_ids': [item.ingredient_id for item, value in zip(pool, matrix[:, j])
                                           if abs(value-limit) < 1e-10][:5]})
    from scripts.compare_lotion_benchmarks import read_run
    manifest, summary, rows = read_run(args.benchmark)
    from deploy.modal_app import RUNTIME_CATALOG_SHA256
    if args.catalog_manifest:
        RUNTIME_CATALOG_SHA256 = json.loads(args.catalog_manifest.read_text(encoding='utf-8'))['runtime_catalog']['sha256']
    if manifest['runtime_catalog_sha256'] != RUNTIME_CATALOG_SHA256:
        raise ValueError('axis audit and benchmark must use the same frozen catalog')
    excluded = [row for row in rows.values() if (row.get('profile_coverage') or {}).get('target_excluded')]
    remaining = [row for row in rows.values() if not row.get('profile_target_met') and row not in excluded]
    report = {'schema': 'lotion-100-target-audit/v1', 'target_score': 95,
        'count': len(rows), 'passed95': summary['passed95'], 'errors': summary['errors'],
        'profile_upper_excluded_count': len(excluded),
        'maximum_possible_pass_count_with_same_profile_pool_upper': len(rows)-len(excluded),
        'remaining_nonexcluded_failures': len(remaining),
        'scope': 'fixed_catalog_19_axis_convex_profile_proxy_not_all_future_models_or_human_accuracy',
        'why': 'Every transported profile is a nonnegative normalized combination of ingredient profiles. A pure-axis target cannot exceed the maximum ingredient fraction on that axis. Mixed-axis bounds use the separately recomputed clipped dual inequality in each benchmark result.',
        'all_400_target_achieved': len(rows) == 400 and summary['passed95'] == 400 and summary['errors'] == 0,
        'screened_transport_pool_count': len(pool), 'parameter_evidence_kind': provenance['evidence_kind'],
        'pure_axis_bounds': pure_bounds,
        'excluded_request_ids': [row['id'] for row in excluded],
        'nonexcluded_failed_request_ids': [row['id'] for row in remaining],
        'runtime_catalog_sha256': manifest['runtime_catalog_sha256'],
        'benchmark_evidence_sha256': {name: hashlib.sha256((args.benchmark/name).read_bytes()).hexdigest()
            for name in ('manifest.json', 'summary.json', 'results.jsonl')}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('excluded_request_ids', 'nonexcluded_failed_request_ids', 'benchmark_evidence_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
