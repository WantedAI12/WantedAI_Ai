"""Actual pinned local ASGI path: reference objective, legacy mode and cache."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new report path required')
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.local_runtime import local_profile
    app = create_app()
    report = {'scope': 'local_actual_pinned_asgi_not_deployment', 'checks': {}, 'human_similarity_percent': None}
    try:
        with TestClient(app) as client:
            capabilities = client.get('/v1/ai/capabilities')
            assert capabilities.status_code == 200, capabilities.text
            cap = capabilities.json()
            assert cap['features']['body_lotion_observed_reference_design']
            assert cap['product_models']['body_lotion']['observed_target_reference']['reference_sha256'] == local_profile()['lotion_target_reference'][1]
            report['checks']['capabilities'] = cap['product_models']['body_lotion']
            request = {'brief': 'lemon cedar scent', 'registry_pool': 'conditional_research', 'max_risk_tier': 2}
            start = time.perf_counter()
            first = client.post('/v1/applications/body-lotion/design', json=request)
            assert first.status_code == 200, first.text
            value = first.json()
            assert value['perceptual_evaluation']['version'] == 'lotion-observed-reference/v1'
            assert value['perception_model']['learned_profiles_drive_primary_objective']
            assert value['human_similarity_percent'] is None
            assert all(set(t['concepts']) == {'lemon','cedar'} for t in value['perceptual_evaluation']['targets'])
            assert bool(value['recipe']) == bool(value['profile_target_met'])
            report['checks']['automatic_new_objective'] = {'seconds': time.perf_counter()-start,
                'score': value['score'], 'status': value['status'], 'candidate_count': len(value['candidate_recipe']),
                'perceptual_evaluation': value['perceptual_evaluation']}
            repeat = client.post('/v1/applications/body-lotion/design', json=request)
            assert repeat.status_code == 200 and repeat.json() == value
            assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            report['checks']['same_request_cache'] = True
            legacy = client.post('/v1/applications/body-lotion/design', json={**request, 'evaluation_mode': 'legacy_profile'})
            assert legacy.status_code == 200, legacy.text
            assert 'perceptual_evaluation' not in legacy.json()
            assert legacy.headers['X-Perfumery-Lotion-Cache'] != 'hit'
            report['checks']['separate_legacy_cache_and_contract'] = {'score': legacy.json()['score']}
            unsupported = client.post('/v1/applications/body-lotion/design', json={**request, 'brief': 'aquatic scent'})
            assert unsupported.status_code == 200, unsupported.text
            assert unsupported.json()['status'] == 'insufficient_observed_target_coverage'
            assert unsupported.json()['score'] is None and not unsupported.json()['profile_target_met']
            report['checks']['unsupported_not_silently_substituted'] = True
        report['status'] = 'passed'
    finally:
        backend = getattr(app.state, 'local_language_backend', None)
        if backend is not None:
            backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report.get('status'), 'checks': list(report['checks'])}))


if __name__ == '__main__':
    main()
