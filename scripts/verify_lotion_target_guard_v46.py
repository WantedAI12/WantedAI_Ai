"""One affected real API request plus full-suite routing coverage, not pass-rate proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from scripts.evaluate_request_space import request_cases
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.models import RecipeConstraints
    from fragrance_ai.recommender.lotion_evaluation import LOTION_PROJECTION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new evidence file')
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'fragrance_ai').rglob('*.py')}
    catalog = read_catalog(ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    brief_parser = NaturalLanguageBriefParser(catalog)
    routing = []
    for case in request_cases():
        brief = brief_parser.parse(case['brief'],RecipeConstraints(product_category='body_lotion',
            enable_registry_trace_candidates=True,max_risk_tier=2))
        full,partial,missing = True,True,set()
        for phase in ('opening','heart','drydown'):
            target = brief.phase_target_profiles.get(phase,brief.target_profile)
            target = target if sum(target.values()) > 0 else brief.target_profile
            unsupported = {k for k,v in target.items() if v > 0 and k not in LOTION_PROJECTION}
            avoided = set(brief.avoided_dimensions)|set(brief.phase_avoided_dimensions.get(phase,[]))
            negative_missing = avoided-set(LOTION_PROJECTION)
            missing |= unsupported|negative_missing
            full &= not bool(unsupported|negative_missing)
            partial &= not negative_missing and any(v > 0 and k in LOTION_PROJECTION for k,v in target.items())
        routing.append({'id':case['id'],'previous_full_target_eligible':bool(full),
            'current_full_or_partial_eligible':bool(full or partial),'unmodeled_axes':sorted(missing)})
    payload = {'brief':'clean fresh citrus woody','registry_pool':'conditional_research','max_risk_tier':2}
    with TestClient(create_app()) as client:
        start = time.perf_counter()
        response = client.post('/v1/applications/body-lotion/design',json=payload)
        assert response.status_code == 200,response.text
        result,seconds = response.json(),time.perf_counter()-start
        report = result['learned_optimization']
        assert report['partial_target_guidance'] and not report['all_target_axes_modeled']
        assert report['unsupported_target_axes'] == ['clean','fresh']
        assert report['solver_calls'] > 0
        assert result['score']+1e-8 >= report['baseline_strict_score']
        assert not report['acceptance_threshold_modified']
        assert report['timepoint_tradeoffs']['lost_target_timepoints'] == 0
        repeat = client.post('/v1/applications/body-lotion/design',json=payload)
        assert repeat.status_code == 200 and repeat.json() == result
        assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items())
    summary = {'scope':'400_request_model_routing_analysis_and_one_actual_API_call_not_400_recipe_scores',
        'routing_cases':len(routing),'previous_full_target_eligible':sum(r['previous_full_target_eligible'] for r in routing),
        'current_full_or_partial_eligible':sum(r['current_full_or_partial_eligible'] for r in routing),
        'new_partial_routes':sum(r['current_full_or_partial_eligible'] and not r['previous_full_target_eligible'] for r in routing),
        'seconds':seconds,'recipe_changed':report['recipe_changed'],'strict_score':result['score'],
        'partial_affinity_before':report['baseline_affinity'],'partial_affinity_after':report['selected_affinity'],
        'strict_passed95':result['profile_target_met'],'model_target_coverage':report['modeled_target_mass_fractions'],
        'recipe_400_pass_rate_measured':False,'deployed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({**summary,'routing':routing,'source_sha256':hashes,'result':result},
        ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
