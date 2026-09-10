"""Local paired lotion design runs using the actual V3 checkpoint."""
import json
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/lotion_learned_v36_final_integration.json')
    parser.add_argument('--expanded', action='store_true')
    parser.add_argument('--brief', action='append')
    args = parser.parse_args()
    path = args.output
    if path.exists():
        raise ValueError('existing evidence must be preserved')
    os.environ['PERFUMERY_AI_ENV'] = 'development'
    from scripts.serve_perception_runtime_v34 import create_app
    from fastapi.testclient import TestClient
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, _estimate_lotion_recipe
    app = create_app()
    catalog = IngredientCatalog.load_builtin()
    if args.expanded:
        from deploy.modal_app import RUNTIME_CATALOG, RUNTIME_CATALOG_SHA256, WHEEL_SHA256, REGISTRY_SHA256
        from fragrance_ai.recommender.registry_activation import load_runtime_catalog
        catalog = load_runtime_catalog(RUNTIME_CATALOG, expected_sha256=RUNTIME_CATALOG_SHA256,
            expected_wheel_sha256=WHEEL_SHA256, expected_registry_sha256=REGISTRY_SHA256)[0]
    rows = []
    with TestClient(app) as client:
        briefs = args.brief or (('floral fruity woody',) if args.expanded else ('citrus woody', 'floral fruity woody', 'green aromatic', 'floral woody', 'gourmand woody', 'fresh clean'))
        for brief in briefs:
            request = LotionEstimateRequest(brief=brief, registry_pool='conditional_research' if args.expanded else 'core',
                                           max_risk_tier=2 if args.expanded else 1)
            baseline = _estimate_lotion_recipe(request, catalog)
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json=request.model_dump(mode='json'))
            elapsed = time.perf_counter()-start
            assert response.status_code == 200, response.text
            result = response.json()
            assert result.get('learned_optimization') is not None, 'learned search was not wired into design'
            if result['learned_optimization']['recipe_changed']:
                assert result['learned_optimization']['fresh_transport_verified']
            assert result['score']+1e-7 >= baseline['score']
            assert result['estimated_concentrate_cost_per_kg'] <= request.max_formula_cost_per_kg+1e-7
            assert all(a['score']+1e-7 >= b['score'] for a,b in zip(result['timepoint_assessments'], baseline['timepoint_assessments']))
            cached = client.post('/v1/applications/body-lotion/design', json=request.model_dump(mode='json'))
            assert cached.status_code == 200 and cached.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            assert cached.json() == result
            summary = {'brief': brief, 'seconds': elapsed, 'baseline_score': baseline['score'], 'score': result['score'],
                       'baseline_cost': baseline['estimated_concentrate_cost_per_kg'],
                       'cost': result['estimated_concentrate_cost_per_kg'], 'learned': result.get('learned_optimization')}
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            rows.append({**summary, 'response': result, 'baseline': baseline})
    path.write_text(json.dumps({'scope': 'paired local API integration cases; not a general accuracy benchmark',
        'registry_pool': 'conditional_research' if args.expanded else 'core',
        'production_deployed': False, 'requests': rows}, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
