"""Replay saved model outputs through the new local API; never call Modal."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = read(args.preparation)
    assert sha(meta['wheel']) == meta['wheel_sha256']
    sys.path[:0] = [meta['installed'], str(ROOT)]
    os.environ.update(PERFUMERY_AI_LOCAL_PROFILE=meta['profile'], PERFUMERY_AI_ENV='research',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import fragrance_ai
    from fragrance_ai.platform.backend_wire_contract import recipe_delivery_response
    assert Path(fragrance_ai.__file__).resolve().is_relative_to(Path(meta['installed']))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    source = ROOT/'output/ai-qa-modal-v23-20260916'
    fixtures = sorted(source.glob('retired_*/retired_*.response.json'))
    assert len(fixtures) == 10
    fixtures += [ROOT/'output/backend-v89-latency-deployed-20260916'/name
                 for name in ('perfume_design.response.json', 'body_lotion_design.response.json')]
    for index, path in enumerate(fixtures):
        original = read(path)
        product = 'body_lotion' if 'body_lotion' in path.name else 'perfume'
        result = recipe_delivery_response(original, product=product)
        key = 'score' if product == 'body_lotion' else 'calculated_profile_similarity'
        assert result['target_match_score'] == original[key]
        assert all(result[k] == v for k, v in original.items() if k not in ('recipe', 'status'))
        if result.get('status') != original.get('status'):
            assert result['assessment_status'] == original['status']
        if result['recipe_delivery']['source'] == 'closest_candidate':
            assert result['recipe'] == original['closest_candidate']
        name = f'replayed-{index:02}.json'
        write(args.output/name, result)
        rows.append({'source': str(path.relative_to(ROOT)), 'source_sha256': sha(path),
            'product': product, 'score': result['target_match_score'], 'target_met': result['target_match_met'],
            'delivery': result['recipe_delivery'], 'material_count': len(result.get('recipe') or []),
            'output_file': name, 'new_model_inference': False})

    from fastapi.testclient import TestClient
    from deploy.target_runtime_v87 import create_release_app
    from fragrance_ai import NaturalLanguagePerfumeryAI
    from fragrance_ai.platform import ai_extensions
    perfume_dir = source/'retired_08'
    perfume = read(perfume_dir/'retired_perfume_extra_13.response.json')
    perfume_request = read(perfume_dir/'retired_perfume_extra_13.request.json')
    lotion_dir = ROOT/'output/ai-qa-modal-v24-latency-20260916/run-01/retired_04'
    lotion = read(lotion_dir/'retired_body_lotion_extra_6.response.json')
    lotion_request = read(lotion_dir/'retired_body_lotion_extra_6.request.json')
    calls, replays = [], {'perfume': 0, 'body_lotion': 0}

    def perfume_engine(*a, **kw):
        replays['perfume'] += 1
        return SimpleNamespace(to_dict=lambda: deepcopy(perfume))

    def lotion_engine(*a, **kw):
        replays['body_lotion'] += 1
        return deepcopy(lotion)

    def call(client, name, path, body, *, stream=False, expected=200):
        response = client.post(path, json=body)
        filename = name + ('.response.sse' if stream else '.response.json')
        (args.output/filename).write_bytes(response.content)
        write(args.output/(name+'.request.json'), body)
        calls.append({'name': name, 'path': path, 'http_status': response.status_code,
                      'headers': dict(response.headers), 'response_file': filename,
                      'response_sha256': hashlib.sha256(response.content).hexdigest()})
        assert response.status_code == expected, response.text[:1000]
        return response.text if stream else response.json()

    with patch.object(NaturalLanguagePerfumeryAI, 'create_recipe', perfume_engine), \
            patch.object(ai_extensions, 'estimate_lotion_recipe', lotion_engine):
        app = create_release_app(registry_path=str(ROOT/'benchmarks/industrial_ingredient_registry_v1.db'))
        with TestClient(app) as client:
            assert client.get('/health').json()['wheel_sha256'] == meta['wheel_sha256']
            response = call(client, 'perfume_json', '/v1/formulas', perfume_request)
            stream = call(client, 'perfume_sse', '/v1/formulas/stream', perfume_request, stream=True)
            frames = [frame for frame in stream.split('\n\n') if 'event: result\n' in frame]
            assert len(frames) == 1
            decoded = json.loads(next(line[6:] for line in frames[0].splitlines() if line.startswith('data: ')))
            assert decoded == response
            assert response['recipe'] == perfume['closest_candidate']
            assert response['target_match_score'] == perfume['calculated_profile_similarity']
            assert not response['target_match_met'] and response['model_accuracy_percent'] is None
            body = call(client, 'lotion_json', '/v1/applications/body-lotion/design', lotion_request)
            from fragrance_ai.platform.response_transport import unpack_lotion_response, encoded, MAX_BYTES
            assert len(encoded(body)) <= MAX_BYTES
            body = unpack_lotion_response(body)
            assert body['recipe'] == lotion['closest_candidate']
            assert body['target_match_score'] == lotion['score'] and not body['profile_target_met']
            extended = {'formula': {**perfume_request, 'product_category': 'eau_de_parfum', 'target_region': 'EU',
                'product_concentration_percent': 15., 'max_formula_cost_per_kg': 180.}}
            evaluated = call(client, 'evaluate_json', '/v1/formulas/evaluate', extended)
            candidate = evaluated['candidates'][0]
            assert candidate['status'] == 'candidate_only' and candidate['result']['recipe']
            assert not candidate['selection_gates']['profile_target_met']
            request = {'request': extended, 'evidence_policy': {'finished_batch_mass_g': 1000.,
                'maximum_lead_time_days': 10, 'maximum_purchase_cost_usd': 100.}}
            reviewed = call(client, 'rd_prepare', '/v2/briefs/prepare', request)
            blocked = call(client, 'rd_missing_evidence', '/v2/formulas/evaluate',
                {**request, 'confirmed_review_id': reviewed['review_id']}, expected=422)
            assert blocked['detail']['reason'] == 'registered_regulatory_and_supply_evidence_missing'
            write(args.output/'openapi.json', client.get('/openapi.json').json())
    summary = {'passed': True, 'execution': 'isolated_local_ASGI_with_saved_model_output_replay',
        'wheel_sha256': meta['wheel_sha256'], 'model_sha256': meta['model_sha256'],
        'target_reference_sha256': meta['target_reference_sha256'], 'source_outputs': rows, 'api_calls': calls,
        'saved_model_output_replays': replays, 'new_model_inference_performed': False,
        'all_original_scores_preserved': True, 'model_accuracy_inferred_from_match': False,
        'json_sse_result_equal': True, 'rd_missing_evidence_still_422': True,
        'deployed': False, 'public_api_called': False}
    write(args.output/'verification.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('source_outputs', 'api_calls')}, ensure_ascii=False))
    print(json.dumps({'saved_responses': len(rows), 'recipes_returned': sum(row['delivery']['returned'] for row in rows),
                      'blocked': [row for row in rows if not row['delivery']['returned']], 'api_calls': len(calls)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
