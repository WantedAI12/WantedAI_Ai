"""Run the local pinned models and new API contract, without registering test evidence."""
import argparse
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
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.local_runtime import local_profile
    profile = local_profile()
    app = create_app(enable_language=False)
    summary = {'scope': 'local_asgi_pinned_runtime_not_remote_deployment',
               'profile_sha256': profile['profile_sha256'], 'odor_backbone_sha256': profile['odor_backbone'][1],
               'catalog_manifest_sha256': profile['catalog'][1], 'external_language_model_enabled': False,
               'registered_test_evidence': False, 'results': [], 'human_similarity_percent': None}
    with TestClient(app) as client:
        caps = client.get('/v1/ai/capabilities')
        assert caps.status_code == 200, caps.text
        save('capabilities.json', caps.json())
        contract = caps.json()['integration_contract']
        assert contract['operations']['compare_saved']['runtime_available']
        assert contract['operations']['conditioned_product_prediction']['runtime_available']
        assert caps.json()['unified_product_model']['shared_odor_backbone_sha256'] == profile['odor_backbone'][1]
        schema = client.get('/openapi.json')
        assert schema.status_code == 200, schema.text
        save('openapi.json', schema.json())
        for key in ('clarify', 'compare_saved', 'revise_saved_intent', 'audit_report'):
            assert contract['operations'][key]['path'] in schema.json()['paths']
        request = {'request': {'formula': {'brief': 'green woody scent'}}, 'evidence_policy': {
            'finished_batch_mass_g': 1000., 'maximum_lead_time_days': 10, 'maximum_purchase_cost_usd': 100.}}
        prepared = client.post('/v2/briefs/prepare', json=request)
        assert prepared.status_code == 200 and prepared.json()['status'] == 'needs_input', prepared.text
        assert len(prepared.json()['missing_fields']) == 4
        save('prepare-missing.json', prepared.json())
        clarified = client.post('/v2/briefs/clarify', json={**request,
            'prepared_result_id': prepared.json()['result_id'], 'answers': {
                'request.formula.product_category': 'eau_de_parfum', 'request.formula.target_region': 'EU',
                'request.formula.product_concentration_percent': 15., 'request.formula.max_formula_cost_per_kg': 180.}})
        assert clarified.status_code == 200 and clarified.json()['prepared']['status'] == 'ready', clarified.text
        save('clarification.json', clarified.json())
        summary['rd_clarification'] = {'http_status': 200, 'status': 'ready', 'missing_fields_resolved': 4,
                                       'new_inference_count': clarified.json()['new_inference_count']}
        if not contract['evidence']['bundle_configured']:
            blocked = client.post('/v2/formulas/evaluate', json={**clarified.json()['request'],
                'confirmed_review_id': clarified.json()['prepared']['review_id']})
            assert blocked.status_code == 422
            assert blocked.json()['detail']['reason'] == 'registered_regulatory_and_supply_evidence_missing'
            summary['unregistered_evidence_gate'] = {'http_status': 422, 'preserved': True}
        # The same two requests used in the V66 local smoke. Scores may remain
        # below target; this verifies numerical/API compatibility, not accuracy.
        old = json.loads((ROOT / '.benchmarks/main_backbone_v66/api-01/api-report.json').read_text(encoding='utf-8'))
        for index, previous in enumerate(old['results']):
            path, body = previous['path'], previous['request']
            start = time.perf_counter()
            response = client.post(path, json=body)
            seconds = time.perf_counter() - start
            assert response.status_code == 200, response.text
            value = response.json()
            save(f'recipe-{index}.json', value)
            repeated = client.post(path, json=body)
            assert repeated.status_code == 200 and repeated.json() == value
            score = value.get('calculated_profile_similarity', value.get('score'))
            assert abs(score - previous['score']) < 1e-8 and value['status'] == previous['status']
            result = {'path': path, 'http_status': response.status_code, 'seconds': seconds,
                      'status': value['status'], 'score': score, 'score_equal_to_v66': True, 'repeat_equal': True,
                      'cache': repeated.headers.get('X-Perfumery-Cache') or repeated.headers.get('X-Perfumery-Lotion-Cache')}
            summary['results'].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    save('report.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
