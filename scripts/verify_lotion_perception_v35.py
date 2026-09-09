"""Run real configured checkpoint through local lotion API; no deployment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/lotion_perception_v35_integration.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve existing evidence; supply a new output filename')
    os.environ['PERFUMERY_AI_ENV'] = 'development'
    from fastapi.testclient import TestClient
    from scripts.serve_perception_runtime_v34 import create_app
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
    app = create_app()
    catalog = IngredientCatalog.load_builtin()
    design = LotionEstimateRequest(brief='floral fruity woody', search_goal='reach_target')
    inputs, _, _ = build_estimated_lotion_inputs(design, catalog)
    baseline = optimize_lotion(inputs, catalog, _transport_only=True)
    requests = []
    with TestClient(app) as client:
        from fragrance_ai.recommender.perception_runtime import configured_perception
        from fragrance_ai.recommender.models import RecipeConstraints
        from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
        session = configured_perception('body_lotion').begin(NaturalLanguageBriefParser(catalog).parse('woody', RecipeConstraints(product_category='body_lotion')))
        unsupported_ids = [item.ingredient_id for item in catalog.ingredients if not session.supports(item)]
        supported_design = design.model_copy(update={'excluded_ingredient_ids': unsupported_ids})
        for operation, body in [('simulate', inputs.simulation.model_dump(mode='json')),
                                ('optimize', inputs.model_dump(mode='json')),
                                ('design', design.model_dump(mode='json')),
                                ('design', supported_design.model_dump(mode='json'))]:
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/'+operation, json=body)
            seconds = time.perf_counter()-start
            assert response.status_code == 200, response.text
            payload = response.json()
            contract = payload['perception_model']
            assert contract['component_model_sha256'] == '688ec8eae6af5fbb24a6953c6ad09f19ede77fd382f22eda84cefef8f4f29aca'
            assert not contract['strict_score_modified'] and not contract['optimization_guidance_applied']
            if body == supported_design.model_dump(mode='json'):
                assert contract['evaluated_timepoint_count'] > 0 and not contract['unmapped_ingredient_ids']
            assert payload['human_similarity_percent'] is None and not payload['manufacturing_approved']
            if operation == 'optimize':
                assert payload['score'] == baseline['score']
                assert payload['recipe'] == baseline['recipe'] and payload['closest_candidate'] == baseline['closest_candidate']
            start = time.perf_counter()
            cached = client.post('/v1/applications/body-lotion/'+operation, json=body)
            cached_seconds = time.perf_counter()-start
            assert cached.status_code == 200 and cached.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            assert cached.json() == payload
            requests.append({'operation': operation, 'request': body, 'seconds': seconds, 'cached_seconds': cached_seconds,
                             'http_status': response.status_code, 'response': payload})
            print(json.dumps({'operation': operation, 'seconds': seconds, 'cached_seconds': cached_seconds,
                              'status': contract['status'], 'model_calls': contract['component_model_calls'],
                              'evaluated_points': contract['evaluated_timepoint_count'],
                              'partial_points': contract['partial_timepoint_count'],
                              'unmapped_ids': contract['unmapped_ingredient_ids'], 'strict_score': payload.get('score')},
                             ensure_ascii=False), flush=True)
        from fragrance_ai.recommender.perception_runtime import LOTION_HASH_ENV
        previous = os.environ[LOTION_HASH_ENV]
        try:
            os.environ[LOTION_HASH_ENV] = '0'*64
            drift = client.post('/v1/applications/body-lotion/design', json=design.model_dump(mode='json'))
            assert drift.status_code == 422
        finally:
            os.environ[LOTION_HASH_ENV] = previous
    args.output.write_text(json.dumps({'scope': 'local actual-checkpoint integration, not quality gain or deployment',
        'optimization_score_and_weights_unchanged': True, 'cached_model_drift_rejected': True,
        'production_deployed': False, 'requests': requests}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
