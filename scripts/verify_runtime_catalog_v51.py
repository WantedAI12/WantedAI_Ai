"""Current API/SDK material identity and real affected recipes, not 400 pass-rate proof."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app, CATALOG_MANIFEST, CATALOG_MANIFEST_SHA256
    from scripts.verify_odor_concepts_v39 import read_catalog
    from scripts.build_odor_integrity_catalog import verify_wheel_sources
    from fragrance_ai.recommender.runtime import load_configured_catalog, RuntimeAIFactory, _material_digest
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
    from fragrance_ai.recommender.lotion_atlas import AtlasLotionGuidance
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new evidence directory')
    args.output.mkdir(parents=True)
    source_files = list((ROOT/'fragrance_ai').rglob('*.py')) + [ROOT/'deploy/modal_app.py', ROOT/'scripts/serve_product_runtime_v42.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    app = create_app(lotion_reference='atlas')
    catalog, manifest_sha = load_configured_catalog()
    old = read_catalog(ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json')
    assert old.ingredients == catalog.ingredients
    assert manifest_sha == CATALOG_MANIFEST_SHA256
    material_sha = _material_digest(catalog)
    factory = RuntimeAIFactory.from_environment()
    assert factory.catalog.ingredients == catalog.ingredients
    assert factory.runtime_contract['catalog_manifest_sha256'] == manifest_sha
    assert factory.runtime_contract['material_snapshot_sha256'] == material_sha
    wheel = CATALOG_MANIFEST.parent.parent/'wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
    verified_files = verify_wheel_sources(wheel)
    binding = json.loads(CATALOG_MANIFEST.read_text(encoding='utf-8'))['runtime_catalog']
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == binding['wheel_sha256']
    results = []
    with TestClient(app) as client:
        health = client.get('/health')
        assert health.status_code == 200 and health.json()['wheel_sha256'] == binding['wheel_sha256']
        inventory = client.get('/v1/catalog')
        assert inventory.status_code == 200
        actual_binding = inventory.json()['runtime_binding']
        assert actual_binding['catalog_manifest_sha256'] == manifest_sha
        assert actual_binding['material_snapshot_sha256'] == material_sha
        assert actual_binding['catalog_sha256'] == binding['sha256']
        assert inventory.json()['connected_catalog_rows'] == len(catalog.ingredients)
        for identifier, brief in (('ko-2', '클린 향'), ('ko-0-11', '시트러스, 우디 향')):
            payload = {'brief': brief, 'registry_pool': 'conditional_research', 'max_risk_tier': 2, 'base_design': {}}
            started = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json=payload)
            assert response.status_code == 200, response.text
            seconds = time.perf_counter()-started
            result = response.json()
            assert result['catalog_snapshot'] == actual_binding
            assert result['perception_model']['component_model_sha256'] == 'f136d3d9dae4a30e9b6fb5f910111e5cb02aedf9d9c99db7cb79b667361c0554'
            prior = json.loads((ROOT/'benchmarks/lotion_atlas_v50_actual'/(identifier+'.json')).read_text(encoding='utf-8'))
            assert result['preparation']['candidate_ids'] == prior['preparation']['candidate_ids']
            assert result['preparation']['evaluation_targets'] == prior['preparation']['evaluation_targets']
            np.testing.assert_allclose(result['score'], prior['score'], atol=1e-7, rtol=0)
            assert result['profile_target_met'] == prior['profile_target_met']
            assert result['human_similarity_percent'] is None
            repeat = client.post('/v1/applications/body-lotion/design', json=payload)
            assert repeat.status_code == 200 and repeat.json() == result
            assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            row = {'id': identifier, 'brief': brief, 'score': result['score'],
                'strict_passed95': result['profile_target_met'], 'seconds': seconds,
                'candidate_count': result['estimation']['candidate_count'],
                'same_candidate_ids_and_targets_as_V40': True, 'matches_prior_V40_SDK_score': True,
                'response_cache_verified': True}
            results.append(row)
            (args.output/(identifier+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps(row, ensure_ascii=False), flush=True)
    # One current direct-SDK call verifies the new shared loader as well, not
    # only agreement with a saved result from the preceding model integration.
    model = AtlasProfilePredictor(ROOT/'.benchmarks/atlas_profiles_v50/run-01/model.json',
        sha256='f136d3d9dae4a30e9b6fb5f910111e5cb02aedf9d9c99db7cb79b667361c0554', experimental=True)
    provider = AtlasLotionGuidance(model, configured_perception('body_lotion').structures, experimental=True)
    sdk = estimate_lotion_recipe(LotionEstimateRequest(brief='클린 향', registry_pool='conditional_research',
        max_risk_tier=2, base_design={}), catalog, perception_guidance=provider)
    np.testing.assert_allclose(sdk['score'], results[0]['score'], atol=1e-7, rtol=0)
    # The default local launcher must also use the bound catalog, independently of choosing
    # the optional Atlas reference model.
    with TestClient(create_app()) as client:
        response = client.get('/v1/catalog')
        assert response.status_code == 200 and response.json()['runtime_binding'] == actual_binding
        assert client.get('/v1/ai/capabilities').json()['product_models']['body_lotion']['component_model']['component_model_sha256'] != model.sha256
    factory.assert_current_snapshot()
    unchanged = all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    assert unchanged
    previous_api = json.loads((ROOT/'benchmarks/lotion_atlas_v50_actual/api.json').read_text(encoding='utf-8'))
    summary = {'scope': 'full_material_identity_and_two_affected_API_recipes_not_full400_scores',
        'catalog_rows': len(catalog.ingredients), 'active_materials': sum(i.formulation_ready and not i.blocked for i in catalog.ingredients),
        'all_V40_material_fields_preserved': True, 'material_snapshot_sha256': material_sha,
        'API_SDK_factory_same_catalog': True, 'worker_job_executed': False,
        'catalog_binding': actual_binding, 'current_SDK_clean_score': sdk['score'],
        'previous_V32_API_clean_score': previous_api['score'], 'new_API_clean_score': results[0]['score'],
        'default_and_Atlas_launcher_use_same_catalog': True, 'rows': results,
        'wheel': str(wheel.relative_to(ROOT)), 'verified_wheel_runtime_files': len(verified_files),
        'source_unchanged': unchanged, 'source_sha256': hashes,
        'model_retrained': False, '400_pass_rate_measured': False, 'deployed': False}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'rows')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
