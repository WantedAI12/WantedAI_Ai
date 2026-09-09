"""Verify frozen export, train-only bank and real local API wiring, no fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from fragrance_ai.recommender.fine_odor_model import FineOdorModel,configured_fine_odor,structure_features
    from fragrance_ai.recommender.odor_expression import expression_utility
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.safety import CandidateSafetyScreen
    from fragrance_ai.recommender.models import RecipeConstraints
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output
    if out.exists():raise ValueError('new output directory required')
    out.mkdir(parents=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    profile=local_profile()
    if profile is None or 'odor_calibration' not in profile:
        raise ValueError('real pinned local correction required')
    catalog,_=load_configured_catalog();model=configured_fine_odor();correction=model.calibration
    if correction is None or correction.sha256!=profile['odor_calibration'][1]:
        raise ValueError('selected correction is not connected')
    base=FineOdorModel(model.path,model.sha256,identity_provider=model.identity_provider)
    split=json.loads(model.split_path.read_text(encoding='utf-8'))['records']
    train=[r for r in split if r['split']==0]
    target=np.zeros_like(correction.bank[1])
    for i,r in enumerate(train):target[i,base.annotations[r['graph']]]=1.
    np.testing.assert_array_equal(correction.bank[1],target)
    np.testing.assert_array_equal(correction.bank[0],structure_features([r['graph'] for r in train])[:,:1024])
    with np.load(correction.path.parent/'holdout.npz',allow_pickle=False) as saved:
        graphs=saved['graphs'].tolist()
        assert graphs==[r['graph'] for r in split if r['split']==2]
        start=time.perf_counter();predicted=model.predict(graphs);elapsed=time.perf_counter()-start
        maximum=float(np.max(np.abs(predicted-saved['after'])))
        if maximum>1e-7:raise AssertionError(('frozen/runtime export differs',maximum))
    report={'scope':'local_annotation_correction_export_and_api_integration_not_human_recipe_validation',
        'profile_sha256':profile['profile_sha256'],'model':model.contract(),
        'checks':{'training_bank_exact_parent_train_features_and_labels':True,
            'frozen_export_max_abs_difference':maximum,'batch_714_cpu_seconds':elapsed},
        'recipe_score_offset':0.,'recipe_threshold_changed':False,'deployed':False}
    parser=NaturalLanguageBriefParser(catalog)
    brief=parser.parse('피오니와 청사과 향',RecipeConstraints(enable_registry_trace_candidates=True,max_risk_tier=2))
    pool,_=CandidateSafetyScreen().screen(catalog,brief)
    start=time.perf_counter();corrected,evidence=model.materials(pool);seconds=time.perf_counter()-start
    original,_=base.materials(pool)
    delta=np.max(np.abs(original-corrected),axis=1)
    before_utility,_=expression_utility(pool,brief,model=base)
    after_utility,details=expression_utility(pool,brief,model=model)
    assert np.any(delta>1e-8) and np.any(np.abs(after_utility-before_utility)>1e-8)
    assert details['prediction_correction_sha256']==correction.sha256
    report['checks']['full_eligible_catalog']={'materials':len(pool),'changed_annotation_predictions':int((delta>1e-8).sum()),
        'evidence_counts':dict(Counter(r['status'] for r in evidence)),'cold_cpu_seconds':seconds,
        'maximum_prediction_correction':float(delta.max()),
        'candidate_search_preference_changed':True,'safety_or_material_caps_changed':False}
    save('catalog-correction.json',report['checks']['full_eligible_catalog'])
    with TestClient(create_app(enable_language=False)) as client:
        def post(path,payload):
            start=time.perf_counter();response=client.post(path,json=payload);elapsed=time.perf_counter()-start
            if response.status_code!=200:raise AssertionError((path,response.status_code,response.text[:1200]))
            return response.json(),elapsed
        cap=client.get('/v1/ai/capabilities');assert cap.status_code==200
        assert cap.json()['odor_expression']['prediction_model']['prediction_correction']['sha256']==correction.sha256
        save('capabilities.json',cap.json())
        value,seconds=post('/v1/odor-expressions/predict',{'ingredient_ids':['dihydromyrcenol','linalyl_acetate','hexyl_acetate'],
            'brief':'green apple and peach'})
        assert value['model']['prediction_correction']['sha256']==correction.sha256
        save('material-expression.json',value);report['checks']['material_api_seconds']=seconds
        for product in ('perfume','body_lotion','body_wash'):
            payload=json.loads((ROOT/f'.benchmarks/unified_product_v60/api-01/{product}-request.json').read_text(encoding='utf-8'))
            value,seconds=post('/v1/applications/unified/predict',payload)
            assert value['model']['fine_odor_expression']['prediction_correction']['sha256']==correction.sha256
            old=json.loads((ROOT/f'.benchmarks/odor_expression_v61/api-02/{product}-response.json').read_text(encoding='utf-8'))
            assert len(value['temporal_profile'])==len(old['temporal_profile']) and len(value['odor_endpoints'])==146
            changed=False
            for a,b in zip(old['temporal_profile'],value['temporal_profile']):
                assert a['total_air_concentration_mg_m3']==b['total_air_concentration_mg_m3']
                assert a['full_odor_reference_profiles']==b['full_odor_reference_profiles']
                changed |= a['fine_odor_expression']!=b['fine_odor_expression']
            assert changed
            repeated,_=post('/v1/applications/unified/predict',payload);assert repeated==value
            save(product+'-response.json',value)
            report['checks'][product]={'correction_connected':True,'fine_predictions_changed':True,
                'original_transport_and_146_axes_unchanged':True,'cache_repeat_equal':True,'seconds':seconds}
        save('integration-before-recipe.json',report)
        print('Pinned model, full catalog and three product APIs verified; checking recipe integration.',flush=True)
        value,seconds=post('/v1/formulas',{'brief':'피오니와 청사과 향',
            'enable_registry_trace_candidates':True,'max_risk_tier':2})
        save('recipe-response.json',value)
        assert value['score_contract']['fine_odor_expression']['prediction_correction_sha256']==correction.sha256
        previous=json.loads((ROOT/'.benchmarks/odor_expression_v61/api-02/recipe-response.json').read_text(encoding='utf-8'))
        report['checks']['recipe']={'status':value['status'],'seconds':seconds,
            'previous_coarse_score':previous.get('calculated_profile_similarity'),
            'current_coarse_score':value.get('calculated_profile_similarity'),
            'full_profile_target_met':value.get('full_profile_target_met'),
            'candidate_lines':len(value.get('recipe') or value.get('closest_candidate') or []),
            'correction_connected':True,'single_request_diagnostic_not_all_recipe_benchmark':True}
    report['status']='passed_local_integration';save('report.json',report)
    print(json.dumps({'status':report['status'],'checks':report['checks']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
