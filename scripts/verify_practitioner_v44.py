"""Actual local ASGI integration, with product-bound V4 checkpoint and catalog."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new evidence filename')
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'fragrance_ai').rglob('*.py')}
    payload = {'brief':'citrus woody scent', 'registry_pool':'conditional_research', 'max_risk_tier':2,
               'dose_trials':{'concentrations_percent':[.25,.5]}}
    with TestClient(create_app()) as client:
        start = time.perf_counter()
        response = client.post('/v1/applications/body-lotion/design', json=payload)
        assert response.status_code == 200, response.text
        result, seconds = response.json(), time.perf_counter()-start
        dose = result['dose_design']
        assert len(dose['evaluated_variants']) == 2 and not dose['variant_errors']
        assert result['score']+1e-8 >= dose['fixed_dose_score']
        assert result['estimation']['application_context']['fragrance_concentration_percent'] == dose['selected_dose_percent']
        assert sum(x['concentrate_percent'] for x in result['candidate_recipe']) > 99.99999
        assert abs(sum(x['finished_product_percent'] for x in result['candidate_recipe'])-dose['selected_dose_percent']) < 1e-8
        assert abs(result['score']-min(x['score'] for x in result['timepoint_assessments'])) < 1e-8
        repeat = client.post('/v1/applications/body-lotion/design', json=payload)
        assert repeat.status_code == 200 and repeat.json() == result
        assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
        invalid = client.post('/v1/applications/body-lotion/design', json={**payload,
            'dose_trials':{'concentrations_percent':[True]}})
        assert invalid.status_code == 422
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items())
    evidence = {'scope':'one_actual_local_ASGI_dose_ladder_not_deployed_or_human_accuracy',
        'seconds':seconds, 'source_sha256':hashes, 'source_unchanged':True, 'request':payload,
        'http_status':200, 'cache_hit_verified':True, 'invalid_dose_status':422, 'result':result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'seconds':seconds,'score':result['score'],'dose_design':dose}),flush=True)


if __name__ == '__main__':
    main()
