"""Actual local pinned API/model integration; process inputs are labelled fixtures."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('new verification directory required')
    a.output.mkdir(parents=True)
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
    from fragrance_ai.platform.unified_product_inputs import UnifiedProductContext, context_id
    from fragrance_ai.recommender.lotion_surrogate import request_features
    from fragrance_ai.recommender.local_runtime import local_profile
    source = json.loads((ROOT/'.benchmarks/lotion_training_v58/source.json').read_text(encoding='utf-8'))
    lotion = copy.deepcopy(source['reference_request'])
    lotion['materials'] = lotion['materials'][:3]
    for row, fraction in zip(lotion['materials'], (40., 35., 25.)):
        row['concentrate_percent'] = fraction
    lotion['times_minutes'] = [0.,15.,60.,240.,480.]
    request = LotionSimulationRequest(**lotion)
    raw = request_features(request)[:3]
    raw[:, :6] /= request.times_minutes[1]
    report = {'scope': 'actual_local_pinned_asgi_not_deployment', 'checks': {}, 'human_similarity_percent': None}
    def save(name, value):
        (a.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    app = create_app(enable_language=False)
    with TestClient(app) as client:
        cap = client.get('/v1/ai/capabilities')
        assert cap.status_code == 200, cap.text
        cap = cap.json()
        pin = local_profile()['unified_product'][1]
        assert cap['unified_product_model']['checkpoint_sha256'] == pin
        assert cap['product_models']['body_lotion']['trained_release_model']['checkpoint_sha256'] == pin
        report['checks']['shared_model_and_legacy_lotion_binding'] = pin
        save('capabilities.json', cap)
        results = {}
        for product in ('perfume','body_lotion','body_wash'):
            wash = product == 'body_wash'
            schedule = ([{'stage_id':'wash', 'duration_minutes':.5, 'formulation_reference':'V60 API functional wash fixture'},
                         {'stage_id':'after_rinse', 'duration_minutes':29.5, 'formulation_reference':'V60 API functional post-rinse fixture'}]
                        if wash else [{'stage_id':'application', 'duration_minutes':480.,
                                       'formulation_reference': 'V60 integration fixture; not measured product calibration'}])
            context = UnifiedProductContext(product_type=product,
                application_mass_mg_cm2=request.application_context.application_mass_mg_cm2,
                fragrance_concentration_percent=request.application_context.fragrance_concentration_percent,
                temperature_c=25., relative_humidity_percent=50., headspace_height_cm=request.headspace_height_cm, stages=schedule)
            components = [{'ingredient_id':m.ingredient_id, 'concentrate_percent':m.concentrate_percent,
                           'odor_threshold_mg_m3':m.odor_threshold_mg_m3} for m in request.materials]
            stages = []
            for stage_context in context.stages:
                coefficients = []
                for m, row in zip(request.materials, raw):
                    # Functional conditioning test only. No new empirical
                    # ethanol, surfactant or deposition coefficients claimed.
                    factor = 2. if product == 'perfume' else .5 if wash else 1.
                    coefficients.append({'ingredient_id':m.ingredient_id,
                        'evaporation_per_min':float(row[0]*factor), 'uptake_per_min':float(row[1]),
                        'hydrolysis_per_min':float(row[2]), 'air_return_per_min':float(row[3]),
                        'ventilation_per_min':float(row[4]), 'capacity_decay_per_min':float(row[5]),
                        'evaporating_capacity_fraction':float(row[6]), 'nonreactive_capacity_fraction':float(row[7]),
                        'source_kind':'simulated', 'source_reference':'V60 API integration fixture; functional inputs, not measured product parameters'})
                stage = {'stage_id':stage_context.stage_id, 'coefficients':coefficients}
                if wash and not stages:
                    stage.update(rinse_retained_film_fractions={m.ingredient_id:f for m,f in zip(request.materials,(.05,.12,.2))},
                                 rinse_source_reference='V60 functional mass-balance event fixture, not measured deposition')
                stages.append(stage)
            payload = {'context':context.model_dump(mode='json'), 'parameter_context_id':context_id(context),
                'components':components, 'stages':stages,
                'times_minutes':[0.,.1,.5,1.,5.,30.] if wash else [0.,15.,60.,240.,480.], 'brief':'lemon cedar scent'}
            start = time.perf_counter()
            response = client.post('/v1/applications/unified/predict', json=payload)
            seconds = time.perf_counter()-start
            assert response.status_code == 200, response.text
            result = response.json(); results[product] = result
            assert result['model']['checkpoint_sha256'] == pin and result['diagnostics']['cumulative_sinks_monotone']
            assert not result['missing_odor_profile_ids'] and len(result['odor_endpoints']) == 146
            assert result['reference_assessment']['status'] == 'diagnostic'
            assert result['human_similarity_percent'] is None and not result['manufacturing_approved']
            repeat = client.post('/v1/applications/unified/predict', json=payload)
            assert repeat.status_code == 200 and repeat.json() == result
            assert repeat.headers['X-Perfumery-Unified-Cache'] == 'hit'
            save(product+'-request.json', payload); save(product+'-response.json', result)
            report['checks'][product] = {'seconds':seconds, 'cache':'hit', 'diagnostics':result['diagnostics'],
                'assessment_status':result['reference_assessment']['status'],
                'maximum_returned_air_mg_m3':max(r['total_air_concentration_mg_m3'] for r in result['temporal_profile'])}
        assert results['body_wash']['process_events'][0]['washed_off_mg_cm2'] > 0
        legacy = client.post('/v1/applications/body-lotion/predict-release', json=lotion)
        assert legacy.status_code == 200, legacy.text
        old_api = legacy.json()
        assert old_api['checkpoint_sha256'] == pin and old_api['diagnostics']['cumulative_sinks_monotone']
        report['checks']['existing_lotion_url_uses_v60'] = True
        save('legacy-lotion-request.json', lotion); save('legacy-lotion-response.json', old_api)
        # Same physical lotion coefficients/dose must match the common route.
        import numpy as np
        for old_row, new_row in zip(old_api['temporal_profile'], results['body_lotion']['temporal_profile']):
            np.testing.assert_allclose(old_row['total_air_concentration_mg_m3'], new_row['total_air_concentration_mg_m3'], rtol=1e-9, atol=1e-12)
        report['checks']['legacy_and_common_lotion_numerical_parity'] = True
        old = client.post('/v1/briefs/prepare', json={'formula':{'brief':'citrus woody scent'}})
        assert old.status_code == 200, old.text
        report['checks']['existing_perfume_intent_api'] = True
        report['status'] = 'passed'
    report['local_profile_sha256'] = local_profile()['profile_sha256']
    save('report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
