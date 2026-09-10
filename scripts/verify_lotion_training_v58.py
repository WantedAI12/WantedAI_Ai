"""One local verification of the actual trained checkpoint and current APIs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import RuntimeAIFactory, _material_digest
    from fragrance_ai.recommender.lotion_surrogate import (configured_lotion_surrogate, predict_release,
        request_features, MASS_FIELDS)
    from fragrance_ai.recommender.lotion import _simulate_lotion_transport
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
    from fragrance_ai import StockAliquot
    profile = local_profile()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    for role, binding in source['parent_models'].items():
        path, pinned = profile[role]
        assert pinned == binding['sha256'] == sha(path), role
    model = configured_lotion_surrogate()
    assert model and model.manifest['source_sha256'] == sha(args.source)
    evaluation_path = model.path.parent/model.manifest['evaluation']['path']
    assert sha(evaluation_path) == model.manifest['evaluation']['sha256']
    report = {'schema': 'lotion-trained-local-verification/v1', 'status': 'failed_or_interrupted',
        'deployed': False, 'parent_models_unchanged': source['parent_models'],
        'model': model.contract(), 'training_report_sha256': sha(evaluation_path),
        'checks': {}, 'recipe400_remeasured': False, 'human_similarity_percent': None}
    app = None
    try:
        factory = RuntimeAIFactory.from_environment()
        assert _material_digest(factory.catalog) == source['material_snapshot_sha256']
        report['material_count'] = len(factory.catalog.ingredients)
        report['material_fields_unchanged'] = True
        full = LotionSimulationRequest.model_validate(source['reference_request'])
        # All currently usable coefficient materials, not just a demonstration
        # formula. Measure raw transport, excluding component-model load time.
        start = time.perf_counter()
        exact = _simulate_lotion_transport(full, factory.catalog)
        exact_seconds = time.perf_counter()-start
        start = time.perf_counter()
        neural = predict_release(full, factory.catalog, model=model)
        neural_seconds = time.perf_counter()-start
        assert neural['status'] in ('research_prediction', 'research_prediction_flagged'), neural
        assert len(neural['temporal_profile']) == len(exact['temporal_profile'])
        differences, air_errors = [], []
        for a, b in zip(exact['temporal_profile'], neural['temporal_profile']):
            assert len(a['materials']) == len(b['materials']) == source['coefficient_pool_count']
            for left, right in zip(a['materials'], b['materials']):
                assert left['ingredient_id'] == right['ingredient_id']
                differences.extend((left[key]-right[key])/left['initial_mg_cm2'] for key in MASS_FIELDS)
                if left['air_concentration_mg_m3'] > 1e-8:
                    air_errors.append(abs(right['air_concentration_mg_m3']/left['air_concentration_mg_m3']-1))
        report['checks']['full_coefficient_pool_release'] = {'material_count': source['coefficient_pool_count'],
            'returned_times_minutes': full.times_minutes, 'all_materials_processed': True,
            'numerical_simulator_seconds': exact_seconds, 'trained_cpu_seconds': neural_seconds,
            'speed_ratio_single_local_run': exact_seconds/neural_seconds,
            'state_fraction_rmse_vs_existing_simulator': float(np.sqrt(np.mean(np.square(differences)))),
            'state_fraction_mae_vs_existing_simulator': float(np.mean(np.abs(differences))),
            'max_state_fraction_error_vs_existing_simulator': float(np.max(np.abs(differences))),
            'median_air_relative_error_above_1e_8_mg_m3': float(np.median(air_errors)),
            'p95_air_relative_error_above_1e_8_mg_m3': float(np.quantile(air_errors, .95)),
            'diagnostics': neural['diagnostics']}
        assert report['checks']['full_coefficient_pool_release']['state_fraction_rmse_vs_existing_simulator'] < .005
        assert neural['diagnostics']['mass_balance_max_abs_error_mg_cm2'] < 1e-12
        print(json.dumps({'event': 'all_coefficient_materials_checked', **report['checks']['full_coefficient_pool_release']}), flush=True)
        # A representative request proves the API is invoking real weights and
        # V54 reference shapes, not a stub. It is not a universal recipe test.
        small, _, _ = build_estimated_lotion_inputs(LotionEstimateRequest(brief='citrus woody scent'), factory.catalog)
        app = create_app()
        with TestClient(app) as client:
            caps = client.get('/v1/ai/capabilities')
            assert caps.status_code == 200, caps.text
            assert caps.json()['features']['body_lotion_trained_release_surrogate']
            assert caps.json()['product_models']['body_lotion']['trained_release_model']['checkpoint_sha256'] == model.sha256
            assert caps.json()['stock_mixture_model']['integrated_v54']
            report['checks']['capabilities'] = caps.json()['product_models']
            request_body = small.simulation.model_dump(mode='json')
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/predict-release', json=request_body)
            fast_seconds = time.perf_counter()-start
            assert response.status_code == 200, response.text
            value = response.json()
            assert value['checkpoint_sha256'] == model.sha256
            assert value['perception_model']['component_model_sha256'] == profile['atlas'][1]
            assert value['perception_model']['applied']
            assert value['temporal_profile'] == predict_release(small.simulation, factory.catalog, model=model)['temporal_profile'] or all(
                a['materials'] == b['materials'] for a,b in zip(value['temporal_profile'],
                predict_release(small.simulation, factory.catalog, model=model)['temporal_profile']))
            report['checks']['real_trained_api'] = {'local_seconds': fast_seconds, 'request': request_body, 'response': value}
            again = client.post('/v1/applications/body-lotion/predict-release', json=request_body)
            assert again.status_code == 200 and again.json() == value
            assert again.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            report['checks']['trained_api_cache'] = {'identical': True, 'cache': 'hit'}
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json={'brief': 'citrus woody scent'})
            design_seconds = time.perf_counter()-start
            assert response.status_code == 200, response.text
            value = response.json()
            simulations = [value['simulation'], *value.get('scenario_simulations', [])]
            chosen = {row['ingredient_id'] for row in (value['recipe'] or value['closest_candidate'])}
            for simulation in simulations:
                learned = simulation['learned_release']
                assert learned['checkpoint_sha256'] == model.sha256
                assert learned['status'] in ('research_prediction', 'research_prediction_flagged')
                assert {m['ingredient_id'] for m in learned['temporal_profile'][0]['materials']} == chosen
                assert learned['perception_model']['component_model_sha256'] == profile['atlas'][1]
            assert bool(value['recipe']) == bool(value['profile_target_met'])
            assert not value['manufacturing_approved'] and value['human_similarity_percent'] is None
            old = json.loads((ROOT/'.benchmarks/formulation_workflow_v57/live-api-02.json').read_text(encoding='utf-8'))
            previous = old['checks']['actual_lotion_design']['response']
            assert abs(value['score']-previous['score']) < 1e-10
            assert value['closest_candidate'] == previous['closest_candidate']
            report['checks']['actual_design'] = {'local_seconds': design_seconds, 'response': value,
                'legacy_recipe_and_strict_score_unchanged': True, 'all_final_scenarios_use_new_checkpoint': True}
            stock = app.state.local_stock_mixture_predictor
            live = [item for item in factory.catalog.ingredients if item.ingredient_id in stock.provider.structures
                    and '.' not in stock.provider.structures[item.ingredient_id][0]][:2]
            payload = {'basis': 'relative_volume', 'components': [
                {'ingredient_id': item.ingredient_id, 'stock_dilution': dose, 'relative_volume': volume, 'solvent': 'pg'}
                for item, dose, volume in zip(live, (.1, .01), (2., 1.))]}
            response = client.post('/v1/formulations/stock-mixture/predict', json=payload)
            assert response.status_code == 200, response.text
            direct = stock.predict([StockAliquot(item, dose, volume, 'pg')
                for item, dose, volume in zip(live, (.1, .01), (2., 1.))])
            assert all(abs(response.json()['predicted_rata_profile'][k]-v) < 1e-12
                       for k,v in direct['predicted_rata_profile'].items())
            report['checks']['v54_perfume_stock_preserved'] = {'api_sdk_agree_to_1e_12': True,
                'manifest_sha256': profile['stock_mixture'][1]}
            assert app.state.local_language_backend._process is None
            report['checks']['no_llm_or_external_calls'] = {'llm_process_started': False, 'lotion_inference_runtime': 'numpy_cpu'}
        factory.assert_current_snapshot()
        report['status'] = 'passed'
    finally:
        if app is not None:
            app.state.local_language_backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': list(report['checks']),
        'model_sha256': model.sha256, 'report_sha256': sha(args.output)}), flush=True)


if __name__ == '__main__':
    main()
