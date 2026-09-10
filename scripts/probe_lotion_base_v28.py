"""Research-only base-ratio hypothesis; not a validated supplier formula."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oil', type=float, required=True, choices=[5., 20., 30.])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new output required')
    from scripts import benchmark_lotion_design as bench
    from fragrance_ai.recommender import lotion_estimation as module
    bench.initialize()
    reference = module.lotion_reference
    def altered(identifier):
        value = copy.deepcopy(reference(identifier))
        context = value['application_context']
        context['formula_reference'] += f'; unqualified engineering trial oil={args.oil}%'
        for c in context['base_components']:
            if c['role'] == 'oil': c['mass_percent'] = args.oil
            elif c['role'] == 'water': c['mass_percent'] = 94.-args.oil
        return value
    module.lotion_reference = altered
    rows = []
    for brief in ('white floral, earthy scent', 'white floral, musky scent', 'fresh, white floral scent'):
        start = time.perf_counter()
        result = module.estimate_lotion_recipe(module.LotionEstimateRequest(brief=brief,
            registry_pool='conditional_research', max_risk_tier=2), bench.CATALOG)
        row = {'brief': brief, 'oil_base_percent': args.oil, 'score': result.get('score'),
            'passed95': result.get('profile_target_met'), 'seconds': time.perf_counter()-start}
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.write_text(json.dumps({'rows': rows, 'scope': 'hypothesis_test_not_manufacturing_qualification'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
