"""Real Atlas checkpoint -> fixed V40 catalog -> existing recipe/API paths.

Four declared integration cases, not a new 400-case pass-rate measurement.
The API retains its existing pinned catalog, recorded separately from V40.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE_IDS = ('ko-2', 'ko-0-11', 'ko-6-17', 'ko-11-13')


def main():
    import numpy as np
    from fastapi.testclient import TestClient
    from scripts.evaluate_request_space import request_cases
    from scripts.serve_product_runtime_v42 import create_app
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
    from fragrance_ai.recommender.lotion_atlas import AtlasLotionGuidance, ATLAS_PROJECTION
    from fragrance_ai.recommender.lotion_evaluation import LOTION_PROJECTION
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, estimate_lotion_recipe
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.models import RecipeConstraints
    from fragrance_ai.recommender.perception_runtime import configured_perception
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new evidence directory')
    args.output.mkdir(parents=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'fragrance_ai').rglob('*.py')}
    app = create_app(lotion_reference='atlas')
    checkpoint = ROOT/'.benchmarks/atlas_profiles_v50/run-01/model.json'
    model = AtlasProfilePredictor(checkpoint,
        sha256='f136d3d9dae4a30e9b6fb5f910111e5cb02aedf9d9c99db7cb79b667361c0554', experimental=True)
    predictions = model.predict(['CCO', 'OCC', 'CCN'])
    assert len(model.endpoints) == 146
    assert set(predictions) == {'applicability', 'use'}
    for values in predictions.values():
        assert values.shape == (3, 146) and np.isfinite(values).all() and np.all(values >= 0)
        np.testing.assert_allclose(values[0], values[1], rtol=0, atol=0)
    for invalid in ('CCO.CCN', 'not a graph'):
        try:
            model.predict([invalid])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid molecular identity was accepted')
    provider = AtlasLotionGuidance(model, configured_perception('body_lotion').structures, experimental=True)
    catalog_path = ROOT/'dist/lotion-incumbent-v40/catalog/catalog_manifest.json'
    catalog = read_catalog(catalog_path)
    brief_parser = NaturalLanguageBriefParser(catalog)
    cases = request_cases()
    routing = []
    for case in cases:
        brief = brief_parser.parse(case['brief'], RecipeConstraints(product_category='body_lotion',
            enable_registry_trace_candidates=True, max_risk_tier=2))
        row = {'id': case['id']}
        for label, projection in (('component', LOTION_PROJECTION), ('atlas', ATLAS_PROJECTION)):
            full, partial = True, True
            for phase in ('opening', 'heart', 'drydown'):
                target = brief.phase_target_profiles.get(phase, brief.target_profile)
                target = target if sum(target.values()) > 0 else brief.target_profile
                absent = {k for k, v in target.items() if v > 0} - set(projection)
                avoided = set(brief.avoided_dimensions)|set(brief.phase_avoided_dimensions.get(phase, []))
                full &= not bool(absent | (avoided-set(projection)))
                partial &= not (avoided-set(projection)) and any(v > 0 and k in projection for k, v in target.items())
            row[label] = {'full_target_routable': bool(full), 'full_or_partial_routable': bool(full or partial)}
        routing.append(row)
    results = []
    for case in cases:
        if case['id'] not in CASE_IDS:
            continue
        started = time.perf_counter()
        request = LotionEstimateRequest(brief=case['brief'], registry_pool='conditional_research',
                                       max_risk_tier=2, base_design={})
        result = estimate_lotion_recipe(request, catalog, parser=brief_parser, perception_guidance=provider)
        report = result['learned_optimization']
        assert report['all_target_axes_modeled'] and not report['acceptance_threshold_modified']
        assert result['score']+1e-8 >= report['baseline_strict_score']
        assert len(result['perception_model']['reference_profile_scenarios']) == 2
        assert not result['perception_model']['gas_to_stock_conversion_used']
        if report['recipe_changed']:
            assert report['fresh_transport_verified']
        summary = {**case, 'seconds': time.perf_counter()-started, 'score': result['score'],
            'strict_passed95': result['profile_target_met'], 'recipe_changed': report['recipe_changed'],
            'baseline_affinity': report['baseline_affinity'], 'selected_affinity': report['selected_affinity'],
            'profile_balance': report.get('profile_balance'), 'solver_calls': report['solver_calls'],
            'candidate_count': result['estimation']['candidate_count'],
            'component_calls': result['perception_model']['component_model_calls'],
            'forward_batches': result['perception_model']['model_forward_batches']}
        results.append(summary)
        (args.output/(case['id']+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({k: v for k, v in summary.items() if k != 'profile_balance'}, ensure_ascii=False), flush=True)
    with TestClient(app) as client:
        capabilities = client.get('/v1/ai/capabilities')
        assert capabilities.status_code == 200
        products = capabilities.json()['product_models']
        assert products['body_lotion']['component_model']['component_model_sha256'] == model.sha256
        assert products['perfume']['component_model']['component_model_sha256'] != model.sha256
        payload = {'brief': '클린 향', 'registry_pool': 'conditional_research', 'max_risk_tier': 2}
        started = time.perf_counter()
        response = client.post('/v1/applications/body-lotion/design', json=payload)
        assert response.status_code == 200, response.text
        seconds = time.perf_counter()-started
        result = response.json()
        assert result['learned_optimization']['all_target_axes_modeled']
        repeat = client.post('/v1/applications/body-lotion/design', json=payload)
        assert repeat.status_code == 200 and repeat.json() == result
        assert repeat.headers['X-Perfumery-Lotion-Cache'] == 'hit'
        from deploy.modal_app import RUNTIME_CATALOG, RUNTIME_CATALOG_SHA256
        api = {'seconds': seconds, 'score': result['score'], 'recipe_changed': result['learned_optimization']['recipe_changed'],
               'cache_reuse_verified': True, 'catalog_path': str(RUNTIME_CATALOG), 'catalog_sha256': RUNTIME_CATALOG_SHA256,
               'catalog_is_V40': False, 'product_lanes_independent': True, 'result': result}
        (args.output/'api.json').write_text(json.dumps(api, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    assert unchanged
    summary = {'scope': 'four_declared_V40_recipe_integrations_and_existing_API_not_400_scores',
        'endpoints': len(model.endpoints), 'projection_axes': len(ATLAS_PROJECTION), 'model_sha256': model.sha256,
        'catalog_manifest_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        'source_sha256': hashes, 'source_unchanged': unchanged, 'rows': results,
        'routing': routing, 'routing_summary': {label: {key: sum(r[label][key] for r in routing)
            for key in ('full_target_routable', 'full_or_partial_routable')} for label in ('component', 'atlas')},
        'api_contract_and_cache_verified': True, 'recipe_400_pass_rate_measured': False,
        'manufacturing_approved': False, 'human_accuracy_measured': False, 'deployed': False}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('source_sha256', 'routing', 'rows')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
