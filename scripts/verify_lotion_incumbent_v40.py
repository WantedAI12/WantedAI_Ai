"""Paired ablation of incumbent transfer on the previous failed request."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender import lotion_estimation as module
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    catalog = read_catalog(ROOT/'dist/odor-concepts-v39/catalog/catalog_manifest.json')
    original = module._estimate_single_base
    request = module.LotionEstimateRequest(brief='clean fresh citrus woody', registry_pool='conditional_research',
        max_risk_tier=2, base_design={})
    report = {'scope': 'single_previous_failure_paired_same_profiles_targets_physics_thresholds', 'rows': []}
    for enabled in (False, True):
        def run(*a, **kw):
            if not enabled:
                kw.pop('_incumbent_recipe', None)
            return original(*a, **kw)
        module._estimate_single_base = run
        start = time.perf_counter()
        try:
            r = module.estimate_lotion_recipe(request, catalog)
        finally:
            module._estimate_single_base = original
        row = {'incumbent_transfer_enabled': enabled, 'seconds': time.perf_counter()-start,
               **{k:r.get(k) for k in ('score','profile_target_met','base_design','search_incomplete')}}
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    report['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'fragrance_ai/recommender/lotion_estimation.py',
                  ROOT/'fragrance_ai/recommender/lotion_optimizer.py', ROOT/'fragrance_ai/recommender/lotion_incumbent.py')}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
