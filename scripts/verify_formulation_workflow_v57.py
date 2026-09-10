"""One local live-model integration check; no deployment or new sensory benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wheel', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('preserve existing evidence; choose a new output file')
    from fastapi.testclient import TestClient
    from fragrance_ai import StockAliquot
    from fragrance_ai.recommender.runtime import RuntimeAIFactory
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.formulation_workflow import formulation_workflow, knowledge_contract
    from fragrance_ai.platform.lotion_reference import lotion_reference
    from scripts.serve_product_runtime_v42 import create_app

    profile = local_profile()
    expected = {'perfume': '2dfbcb2d19640a6e42051dcfaf32c2aff0fdc59fa2238bbe2f8718180a1ec23d',
        'body_lotion': '3eb057bce1f06bd9487b1fc34366d440f926381ce3bc2d640ecf70616bd9b164',
        'atlas': '4c01417cd4b9cbd37ea2ba88ca0a62784e5ead60bec9638b1e7bbfc7af557591',
        'stock_mixture': 'fff5e2abb24f7df113a916fd469193a620c065e13144791437fb9d0752fb7a26'}
    for key, sha in expected.items():
        assert profile[key][1] == sha and digest(Path(profile[key][0])) == sha
    wheel = args.wheel
    with zipfile.ZipFile(wheel) as archive:
        name = 'fragrance_ai/data/formulation_process_knowledge_v1.json'
        assert hashlib.sha256(archive.read(name)).hexdigest() == knowledge_contract()['sha256']
    factory = RuntimeAIFactory.from_environment()
    app = create_app()
    report = {'schema_version': 'formulation-workflow-local-verification/1', 'status': 'failed_or_interrupted',
        'deployed': False, 'weights_unchanged': expected, 'knowledge': knowledge_contract(),
        'wheel_sha256': digest(wheel), 'local_profile_sha256': profile['profile_sha256'],
        'material_count': len(factory.catalog.ingredients), 'checks': {},
        'new_sensory_observations': 0, 'accuracy_improvement_measured': False, 'recipe400_remeasured': False}
    try:
        with TestClient(app) as client:
            caps = client.get('/v1/ai/capabilities')
            assert caps.status_code == 200 and caps.json()['formulation_knowledge']['source_count'] == 6
            assert caps.json()['stock_mixture_model']['integrated_v54']
            report['checks']['capabilities'] = caps.json()
            base = lotion_reference()['application_context']
            base['fragrance_concentration_percent'] = .5
            body = {'product_type': 'body_lotion', 'application_context': base, 'process': {'batch_mass_g': 400}}
            start = time.perf_counter()
            response = client.post('/v1/formulation-workflows/plan', json=body)
            elapsed = time.perf_counter() - start
            assert response.status_code == 200 and response.json() == formulation_workflow(body)
            report['checks']['workflow'] = {'request': body, 'response': response.json(),
                'api_sdk_identical': True, 'local_seconds': elapsed}
            start = time.perf_counter()
            answer = client.post('/v1/ai/assistant', json={'message': '향수 조향 방법 알려줘'})
            elapsed = time.perf_counter() - start
            assert answer.status_code == 200 and answer.json()['llm_calls_maximum'] == 0
            assert app.state.local_language_backend._process is None
            report['checks']['grounded_assistant'] = {'response': answer.json(), 'local_seconds': elapsed,
                'llama_process_started': False}
            print('Source-grounded assistant and workflow API/SDK checks passed.', flush=True)
            start = time.perf_counter()
            answer = client.post('/v1/ai/assistant', json={'message': '달지 않은 우디 향수'})
            elapsed = time.perf_counter() - start
            report['checks']['actual_local_llm'] = {'response': answer.json(), 'local_seconds': elapsed}
            assert answer.status_code == 200 and answer.json()['source'] in (
                'quantized_language_model', 'deterministic_grounding_after_model_mismatch'), answer.text
            assert answer.json()['intent_proposal']['desired'] == ['woody']
            assert answer.json()['intent_proposal']['avoided'] == ['gourmand']
            print('Actual quantized local LLM odor-intent path passed.', flush=True)
            prepared = client.post('/v1/briefs/prepare', json={'formula': {'brief': 'citrus woody scent'}})
            assert prepared.status_code == 200 and prepared.json()['status'] == 'ready'
            assert prepared.json()['formulation_workflow']['product_type'] == 'perfume'
            report['checks']['prepare'] = prepared.json()
            request = {'brief': 'citrus woody scent', 'process': {'batch_mass_g': 400}}
            start = time.perf_counter()
            design = client.post('/v1/applications/body-lotion/design', json=request)
            elapsed = time.perf_counter() - start
            assert design.status_code == 200, design.text
            value = design.json()
            direct = formulation_workflow({'product_type': 'body_lotion',
                'application_context': value['estimation']['application_context'], 'process': request['process']})
            assert value['formulation_workflow'] == direct
            assert abs(direct['batch']['total_mass_g'] - 400) < 1e-8
            assert value['perception_model'] and not value['manufacturing_approved']
            assert bool(value['recipe']) == bool(value['profile_target_met'])
            report['checks']['actual_lotion_design'] = {'request': request, 'response': value, 'local_seconds': elapsed,
                'workflow_bound_to_actual_selected_context': True}
            print('Actual lotion-model design and selected-context process integration passed.', flush=True)
            stock = app.state.local_stock_mixture_predictor
            live = [item for item in factory.catalog.ingredients if item.ingredient_id in stock.provider.structures
                    and '.' not in stock.provider.structures[item.ingredient_id][0]][:2]
            assert len(live) == 2
            stocks = {'basis': 'relative_volume', 'components': [
                {'ingredient_id': item.ingredient_id, 'stock_dilution': dose, 'relative_volume': volume, 'solvent': 'pg'}
                for item, dose, volume in zip(live, (.1, .01), (2., 1.))]}
            stock_response = client.post('/v1/formulations/stock-mixture/predict', json=stocks)
            assert stock_response.status_code == 200, stock_response.text
            direct_stock = stock.predict([StockAliquot(item, dose, volume, 'pg') for item, dose, volume in zip(live, (.1, .01), (2., 1.))])
            actual = stock_response.json()['predicted_rata_profile']
            assert all(abs(actual[k] - v) < 1e-12 for k, v in direct_stock['predicted_rata_profile'].items())
            report['checks']['v54_stock_api_sdk'] = {'request': stocks, 'response': stock_response.json(), 'equal_to_1e_12': True}
            factory.assert_current_snapshot()
            report['status'] = 'passed'
    finally:
        app.state.local_language_backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': list(report['checks']), 'material_count': report['material_count'],
                      'wheel_sha256': report['wheel_sha256']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
