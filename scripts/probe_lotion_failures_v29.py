"""Bounded alternative-ratio diagnostic with unchanged score and material constraints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(task):
    from scripts import benchmark_lotion_design as bench
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, _estimate_single_base
    case, oil, support = task
    start = time.perf_counter()
    try:
        catalog, parser = bench.CATALOG, None
        if support:
            from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
            from fragrance_ai.recommender.models import RecipeConstraints
            from fragrance_ai.recommender.catalog import IngredientCatalog
            parser = NaturalLanguageBriefParser(catalog)
            intent = parser.parse(case['brief'], RecipeConstraints(product_category='body_lotion', product_concentration_percent=.5,
                max_risk_tier=2, enable_registry_trace_candidates=True))
            axes = set(k for t in [intent.target_profile, *intent.phase_target_profiles.values()] for k,v in t.items() if v>0)
            catalog = IngredientCatalog([i for i in catalog.ingredients if sum(i.profile.get(a,0) for a in axes) >= support*sum(i.profile.values())])
        result = _estimate_single_base(LotionEstimateRequest(brief=case['brief'], registry_pool='conditional_research', max_risk_tier=2),
            catalog, parser, oil_base_percent=oil, target_only=True)
        return {'id':case['id'], 'brief':case['brief'], 'oil':oil, 'previous_score':case['score'],
            'score':result.get('score'), 'passed95':bool(result.get('profile_target_met')),
            'incomplete':result.get('search_incomplete'), 'candidate_count':result['estimation']['candidate_count'],
            'support':support, 'seconds':time.perf_counter()-start}
    except Exception as error:
        return {'id':case['id'], 'oil':oil, 'error':str(error), 'passed95':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--oil', type=float, action='append', required=True)
    p.add_argument('--case-id', action='append')
    p.add_argument('--audit', type=Path)
    p.add_argument('--support', type=float, default=0.)
    args = p.parse_args()
    if args.output.exists(): p.error('new output required')
    rows = [json.loads(line) for line in (ROOT/'benchmarks/lotion_design_v28_auto_full400/results.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = [r for r in rows if not r['profile_target_met'] and (not args.case_id or r['id'] in args.case_id)]
    if args.audit:
        possible = {r['id'] for r in json.loads(args.audit.read_text(encoding='utf-8'))['rows'] if r['reason'] != 'fixed_profile_coverage_limit'}
        cases = [r for r in cases if r['id'] in possible and not r['id'].startswith('ko-')]
    from scripts.benchmark_lotion_design import initialize
    results = []
    with ProcessPoolExecutor(4, initializer=initialize) as executor:
        for future in as_completed([executor.submit(run,(c,o,args.support)) for c in cases for o in args.oil]):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps({'scope':'targeted_diagnostic_not_full_pass_rate', 'rows':results}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
