"""Actual local ASGI/CPU routing with two independently bound V4 instances."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.perception_runtime import configured_perception, HASH_ENV, LOTION_HASH_ENV
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('preserve prior evidence; choose a new output')
    app = create_app()
    perfume, lotion = configured_perception(), configured_perception('body_lotion')
    assert perfume is not lotion
    assert perfume.model is not lotion.model and perfume.lotion_shape_cache is not lotion.lotion_shape_cache
    perfume_request = {'brief': 'floral fruity woody', 'target_similarity': 95., 'max_ingredients': 12}
    lotion_request = {'brief': 'floral fruity woody', 'registry_pool': 'conditional_research', 'max_risk_tier': 2}
    rows = {}
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        capabilities = client.get('/v1/ai/capabilities').json()
        for product, path, request, cache_header in (
            ('perfume', '/v1/formulas', perfume_request, 'X-Perfumery-Cache'),
            ('body_lotion', '/v1/applications/body-lotion/design', lotion_request, 'X-Perfumery-Lotion-Cache'),
        ):
            start = time.perf_counter()
            response = client.post(path, json=request)
            assert response.status_code == 200, response.text
            result = response.json()
            seconds = time.perf_counter()-start
            cached = client.post(path, json=request)
            assert cached.json() == result and cached.headers[cache_header] == 'hit'
            contract = result['deployment']['product_model'] if product == 'perfume' else result['product_model']
            assert contract['product'] == product
            assert not contract['cross_product_scores_comparable']
            if product == 'perfume':
                assert result['perception_guidance']['component_model_sha256'] == perfume.component_model_sha256
                assert result['score_contract']['product_model']['product'] == 'perfume'
            else:
                assert result['perception_model']['component_model_sha256'] == lotion.component_model_sha256
                assert result['perception_model']['product'] == 'body_lotion'
                assert result['simulation']['product_model']['prediction_model'] == 'finite_dose_ow_film_transport'
            rows[product] = {'seconds': seconds, 'cache_repeat': 'hit', 'result': result}
            print(json.dumps({'product': product, 'http': response.status_code, 'seconds': seconds,
                              'score': result.get('score', result.get('calculated_profile_similarity'))}), flush=True)
        # Changing a lotion configuration must invalidate lotion responses but
        # must neither invalidate nor disable the already-bound perfume cache.
        saved = os.environ[LOTION_HASH_ENV]
        try:
            os.environ[LOTION_HASH_ENV] = '0'*64
            wrong = client.post('/v1/applications/body-lotion/design', json=lotion_request)
            independent = client.post('/v1/formulas', json=perfume_request)
            assert wrong.status_code == 422
            assert independent.status_code == 200 and independent.headers['X-Perfumery-Cache'] == 'hit'
        finally:
            os.environ[LOTION_HASH_ENV] = saved
        saved = os.environ[HASH_ENV]
        try:
            os.environ[HASH_ENV] = '0'*64
            wrong = client.post('/v1/formulas', json=perfume_request)
            independent = client.post('/v1/applications/body-lotion/design', json=lotion_request)
            assert wrong.status_code == 422
            assert independent.status_code == 200 and independent.headers['X-Perfumery-Lotion-Cache'] == 'hit'
        finally:
            os.environ[HASH_ENV] = saved
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'scope': 'actual_local_API_product_routing_not_accuracy_benchmark',
        'independent_provider_instances': True, 'independent_caches': True,
        'bidirectional_configuration_drift_isolation': True,
        'shared_component_checkpoint_not_separately_matrix_trained': True,
        'capabilities': capabilities, 'rows': rows, 'deployed': False,
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
