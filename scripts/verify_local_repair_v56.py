"""Exercise the actual pinned local API, SDK and owned CPU language process.

This writes diagnostic evidence, never deploys and never approves manufacture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve evidence: select a new output file')
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.runtime import RuntimeAIFactory, _source_snapshot, _data_snapshot
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    from fragrance_ai.recommender.models import ScentBrief, RecipeConstraints
    from fragrance_ai.recommender.perception_runtime import configured_perception

    source, data, profile = _source_snapshot(), _data_snapshot(), local_profile()
    factory = RuntimeAIFactory.from_environment()
    app = create_app()
    language = app.state.local_language_backend
    report = {'schema': 'local-repair-verification/v1', 'deployed': False,
              'scope': 'actual local software integration; not human accuracy or manufacturing approval',
              'catalog_materials': len(factory.catalog.ingredients),
              'runtime_contract': factory.runtime_contract, 'source_sha256': source,
              'data_sha256': data, 'local_profile_sha256': profile['profile_sha256']}
    owned_process = None
    try:
        with TestClient(app) as client:
            for path in ('/health', '/v1/ai/capabilities', '/v1/catalog'):
                response = client.get(path)
                assert response.status_code == 200, (path, response.text[:1000])
                if path != '/v1/catalog':
                    report[path] = response.json()
            caps = report['/v1/ai/capabilities']
            assert caps['features']['explicit_stock_mixture_prediction']
            assert caps['features']['quantized_language_backend']
            assert profile['atlas'][1] in json.dumps(caps['product_models']['body_lotion'])
            assert caps['language_model']['external_llm_api_calls'] == 0

            start = time.perf_counter()
            response = client.post('/v1/ai/assistant', json={
                'message': '머스크는 넣지 말고 장미향이 나는 바디로션으로 해줘'})
            assert response.status_code == 200, response.text
            reply = response.json()
            report['assistant'] = {'seconds_including_cold_start': time.perf_counter()-start,
                                   'response': reply}
            print(json.dumps({'assistant_source': reply['source'], 'intent': reply['intent_proposal']},
                             ensure_ascii=False), flush=True)
            assert reply['source'] == 'quantized_language_model', reply
            assert 'rose' in reply['intent_proposal']['desired'], reply
            assert 'musky' in reply['intent_proposal']['avoided'], reply
            assert reply['intent_proposal']['product'] == 'body_lotion', reply
            assert not reply['formula_generated'] and not reply['scientific_score_generated']
            owned_process = language._process
            assert owned_process is not None and owned_process.poll() is None

            request = LotionEstimateRequest(brief='clean scent lotion', target_profile={'clean': 1.},
                registry_pool='conditional_research', max_risk_tier=2)
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json=request.model_dump(mode='json'))
            assert response.status_code == 200, response.text[:2000]
            lotion = response.json()
            report['lotion_api'] = {'seconds': time.perf_counter()-start, 'response': lotion}
            assert profile['atlas'][1] in json.dumps(lotion['product_model'])
            assert lotion.get('learned_optimization') is not None
            cached = client.post('/v1/applications/body-lotion/design', json=request.model_dump(mode='json'))
            assert cached.status_code == 200 and cached.json() == lotion
            assert cached.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            direct = estimate_lotion_recipe(request, factory.catalog)
            assert abs(direct['score']-lotion['score']) < 1e-8
            assert direct['candidate_recipe'] == lotion['candidate_recipe']
            assert bool(direct['profile_target_met']) == bool(lotion['profile_target_met'])
            if not lotion['profile_target_met']:
                assert not lotion.get('recipe')
            report['lotion_api_sdk_exact_recipe_agreement'] = True
            report['lotion_cache_hit'] = True
            print(json.dumps({'lotion_score': lotion['score'], 'status': lotion['status'],
                'coverage': lotion['estimation']['coverage'], 'api_sdk_agree': True}), flush=True)

            session = configured_perception().begin(ScentBrief('', {}, [], [], [], [], 'medium', {}, RecipeConstraints()))
            selected, ids = [], set()
            for item in factory.catalog.ingredients:
                if not session.supports(item):
                    continue
                cid, _ = session._prepare(item)
                if cid is not None and cid not in ids:
                    selected.append(item)
                    ids.add(cid)
                    if len(selected) == 2:
                        break
            assert len(selected) == 2
            body = {'basis': 'relative_volume', 'components': [
                {'ingredient_id': item.ingredient_id, 'stock_dilution': dose,
                 'relative_volume': 1., 'solvent': 'pg'}
                for item, dose in zip(selected, (.1, .01))]}
            response = client.post('/v1/formulations/stock-mixture/predict', json=body)
            assert response.status_code == 200, response.text
            stock = response.json()
            assert profile['stock_mixture'][1] in json.dumps(stock)
            assert stock['predicted_rata_profile']
            cached = client.post('/v1/formulations/stock-mixture/predict', json=body)
            assert cached.status_code == 200 and cached.json() == stock
            assert cached.headers['X-Perfumery-Stock-Cache'] == 'hit'
            report['stock_mixture'] = {'request': body, 'response': stock, 'cache_hit': True}
        assert owned_process.poll() is not None
        assert language._process is None and language._closed
        report['owned_language_process_closed'] = True
        factory.assert_current_snapshot()
        assert source == _source_snapshot() and data == _data_snapshot()
        assert profile['profile_sha256'] == local_profile()['profile_sha256']
        report['source_data_profile_unchanged'] = True
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        language.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': report.get('status'), 'output': str(args.output),
                          'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
