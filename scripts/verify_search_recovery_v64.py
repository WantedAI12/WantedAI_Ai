"""Real pinned model parity and local API smoke, with raw results retained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    from fragrance_ai.recommender.fine_odor_model import configured_fine_odor
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import load_verified_catalog_bundle
    profile = local_profile()
    catalog = load_verified_catalog_bundle(*profile['catalog'])
    model = configured_fine_odor()
    source = ROOT/'.benchmarks/odor_calibration_v62/run-03/holdout.npz'
    with np.load(source, allow_pickle=False) as data:
        start = time.perf_counter()
        prediction = model.predict(data['graphs'].tolist())
        error = float(np.max(np.abs(prediction-data['after'])))
        parity = {'rows': len(prediction), 'maximum_absolute_difference': error,
            'seconds': time.perf_counter()-start, 'holdout_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        assert error <= 1e-12, parity
    eligible = [item for item in catalog['catalog'].ingredients if item.formulation_ready and not item.blocked]
    calls = []
    original = model.assert_current
    def counted():
        calls.append(1)
        original()
    model.assert_current = counted
    try:
        start = time.perf_counter()
        first, evidence = model.materials(eligible)
        batch_seconds = time.perf_counter()-start
        first_guards = len(calls)
        second, second_evidence = model.materials(eligible)
        np.testing.assert_array_equal(first, second)
        assert evidence == second_evidence and first_guards == 2
    finally:
        model.assert_current = original
    save('model-parity.json', {'scope': 'existing_714_holdout_parity_not_new_blind_accuracy',
        'holdout': parity, 'catalog_batch_rows': len(eligible), 'catalog_batch_seconds': batch_seconds,
        'batch_full_hash_checks': first_guards, 'repeat_equal': True,
        'model_sha256': model.sha256, 'profile_sha256': profile['profile_sha256']})
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    app = create_app(enable_language=False)
    results = []
    requests = [('/v1/formulas', {'brief': '피오니와 청사과 향', 'max_risk_tier': 2,
                                 'enable_registry_trace_candidates': True}),
                ('/v1/applications/body-lotion/design', {'brief': 'woody scent',
                    'registry_pool': 'conditional_research', 'max_risk_tier': 2})]
    with TestClient(app) as client:
        for index, (path, body) in enumerate(requests):
            start = time.perf_counter()
            first = client.post(path, json=body)
            seconds = time.perf_counter()-start
            assert first.status_code == 200, first.text
            value = first.json()
            save(f'response-{index}.json', value)
            repeat = client.post(path, json=body)
            assert repeat.status_code == 200 and repeat.json() == value
            results.append({'path': path, 'request': body, 'http_status': first.status_code,
                'seconds': seconds, 'repeat_equal': True,
                'cache': repeat.headers.get('X-Perfumery-Cache') or repeat.headers.get('X-Perfumery-Lotion-Cache'),
                'status': value['status'], 'score': value.get('calculated_profile_similarity', value.get('score'))})
    save('api-report.json', {'scope': 'local_asgi_not_remote_deployment', 'results': results,
        'external_language_model_enabled': False, 'profile_sha256': profile['profile_sha256'],
        'human_similarity_percent': None})
    print(json.dumps({'parity': parity, 'api': results}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
