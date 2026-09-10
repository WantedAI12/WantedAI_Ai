"""Exercise the two affected requests through the rebound local API."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app, CATALOG_MANIFEST, CATALOG_MANIFEST_SHA256
    from scripts.build_odor_integrity_catalog import verify_wheel_sources
    from fragrance_ai.recommender.runtime import load_configured_catalog, _material_digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new evidence directory required')
    args.output.mkdir(parents=True)
    wheel = CATALOG_MANIFEST.parent.parent/'wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
    source = verify_wheel_sources(wheel)
    before_path = ROOT/'benchmarks/lotion_solver_v52_baseline.json'
    after_path = ROOT/'benchmarks/lotion_solver_v52_after.json'
    before = {r['brief']: r for r in json.loads(before_path.read_text(encoding='utf-8'))['rows']}
    after = {r['brief']: r for r in json.loads(after_path.read_text(encoding='utf-8'))['rows']}
    app = create_app()
    catalog, manifest_sha = load_configured_catalog()
    assert manifest_sha == CATALOG_MANIFEST_SHA256
    catalog_sha = _material_digest(catalog)
    assert catalog_sha == '8c962601b6567d56bd1043733b2b2eaccf9deb5d35e4aba70101271fe65bf139'
    rows = []
    with TestClient(app) as client:
        response = client.get('/v1/catalog')
        assert response.status_code == 200, response.text
        binding = response.json()['runtime_binding']
        assert binding['catalog_manifest_sha256'] == manifest_sha
        assert binding['material_snapshot_sha256'] == catalog_sha
        assert binding['wheel_sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()
        for i, brief in enumerate(before):
            payload = {'brief': brief, 'registry_pool': 'conditional_research', 'max_risk_tier': 2}
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json=payload)
            assert response.status_code == 200, response.text
            elapsed = time.perf_counter()-start
            result = response.json()
            assert result['catalog_snapshot'] == binding
            assert not result['search_incomplete']
            assert result['score']+1e-7 >= after[brief]['score']
            assert result['preparation']['effective_target'] == 95.
            assert result['profile_target_met'] == (result['score']+1e-8 >= 95.)
            if not result['profile_target_met']:
                assert result['recipe'] == [] and result['closest_candidate']
            assert result['human_similarity_percent'] is None
            repeat = client.post('/v1/applications/body-lotion/design', json=payload)
            assert repeat.status_code == 200 and repeat.json() == result
            assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            record = {'brief': brief, 'baseline_score': before[brief]['score'], 'score': result['score'],
                'baseline_search_incomplete': before[brief]['search_incomplete'],
                'search_incomplete': result['search_incomplete'], 'profile_target_met': result['profile_target_met'],
                'solver_calls': result['solver_calls'], 'solver_conditioning_calls': result['solver_conditioning_calls'],
                'seconds': elapsed, 'response_cache_verified': True}
            rows.append(record)
            (args.output/f'request-{i}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps(record, ensure_ascii=False), flush=True)
    assert verify_wheel_sources(wheel) == source
    report = {'scope': 'two_observed_numerical_failures_at_fixed_10_percent_oil_not_full400',
        'rows': rows, 'catalog_binding': binding, 'all_material_fields_preserved': True,
        'new_95_passes': sum(r['profile_target_met'] for r in rows),
        'verified_wheel_runtime_files': len(source), 'source_unchanged': True,
        'baseline_sha256': hashlib.sha256(before_path.read_bytes()).hexdigest(),
        'after_sha256': hashlib.sha256(after_path.read_bytes()).hexdigest(),
        'full400_repeated': False, 'deployed': False}
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
