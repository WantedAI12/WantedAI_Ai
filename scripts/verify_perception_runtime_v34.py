"""Actual local API routing with the new trained model; no deployment/push."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from fragrance_ai.recommender.perception_runtime import PATH_ENV, HASH_ENV
    output = ROOT/'benchmarks/perception_runtime_v34_integration.json'
    if output.exists():
        raise ValueError('preserve existing evidence; choose a separate run')
    os.environ[PATH_ENV] = str(ROOT/'.benchmarks/perception_runtime_v34/manifest.json')
    os.environ[HASH_ENV] = 'a0be5c314fbac46e175d718a5e3ed8ed920e07b0e8ef8500ac8a1f1f810fbea1'
    os.environ['PERFUMERY_AI_ENV'] = 'development'
    from deploy.modal_app import create_web_app, REGISTRY
    rows = []
    formula = {'brief': 'floral fruity woody', 'target_similarity': 95., 'max_ingredients': 12}
    with TestClient(create_web_app(str(REGISTRY))) as client:
        capabilities = client.get('/v1/ai/capabilities').json()
        assert capabilities['perception_model']['configured']
        def request(path, body, nested=False):
            start = time.perf_counter()
            response = client.post(path, json=body)
            value = response.json()
            if response.status_code != 200:
                raise AssertionError((path, response.status_code, value))
            payload = value['candidates'][0]['result'] if nested else value
            guidance = payload.get('perception_guidance') or {}
            rows.append({'path': path, 'seconds': time.perf_counter()-start,
                         'cache': response.headers.get('X-Perfumery-Cache'),
                         'checkpoint': guidance.get('component_model_sha256'),
                         'model_application': guidance.get('model_application'),
                         'status': guidance.get('status'), 'strict_score': payload.get('calculated_profile_similarity'),
                         'strict_target_met': payload.get('full_profile_target_met'),
                         'operation': guidance.get('operation'), 'recipe': payload.get('recipe'),
                         'closest_candidate': payload.get('closest_candidate')})
            assert guidance['component_model_sha256'] == capabilities['perception_model']['component_model_sha256']
            print(json.dumps({k:v for k,v in rows[-1].items() if k not in ('recipe', 'closest_candidate')}, ensure_ascii=False), flush=True)
            return value, payload
        _, base = request('/v1/formulas', formula)
        request('/v1/formulas', formula)
        assert rows[-1]['cache'] == 'hit'
        request('/v1/formulas/evaluate', {'formula': formula}, True)
        request('/v1/formulas/revise', {'formula': formula, 'instruction': 'more woody'}, True)
        lines = base.get('recipe') or base.get('closest_candidate')
        assert lines
        fixed = [{'ingredient_id': row['ingredient_id'], 'concentrate_percent': row['concentrate_percent']} for row in lines]
        _, assessed = request('/v1/formulas/reassess', {'formula': formula, 'lines': fixed}, True)
        assert {r['ingredient_id']: r['concentrate_percent'] for r in assessed['closest_candidate']} == {
            r['ingredient_id']: r['concentrate_percent'] for r in fixed}
        assert rows[-1]['operation'] == 'fixed_formula_reassessment'
        # Cached results must not conceal an operator checkpoint switch.
        os.environ[HASH_ENV] = '0'*64
        drift = client.post('/v1/formulas', json=formula)
        assert drift.status_code == 422
    output.write_text(json.dumps({'scope': 'actual configured research model through local full-catalog API; not all scents or deployment',
        'capabilities': capabilities['perception_model'], 'requests': rows, 'cached_drift_rejected': True,
        'production_deployed': False}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
