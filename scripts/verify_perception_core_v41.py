"""Actual V3/V4 checkpoints through the same local API and recipe conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from fastapi.testclient import TestClient
    from deploy.modal_app import create_web_app, REGISTRY
    from fragrance_ai.recommender.perception_guidance import PerceptionGuidance
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, _estimate_lotion_recipe
    from scripts.verify_odor_concepts_v39 import read_catalog
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    cases=('floral fruity woody','green aromatic woody','gourmand woody','clean fresh citrus woody')
    catalog=read_catalog(ROOT/'dist/body-lotion-v32/profile-extension/catalog_manifest.json')
    rows={text:{} for text in cases}
    for version in ('v3','v4'):
        model=ROOT/f'.benchmarks/perception_core_{version}/run-01/model.json'
        sha=hashlib.sha256(model.read_bytes()).hexdigest()
        provider=PerceptionGuidance(ROOT/'.benchmarks/conditional_profiles_v2/final-01/models.json',REGISTRY,
            solvent='pg',experimental=True,component_model_path=model,component_model_sha256=sha)
        app=create_web_app(str(REGISTRY),lotion_perception_guidance=provider)
        with TestClient(app) as client:
            for text in cases:
                request=LotionEstimateRequest(brief=text,registry_pool='conditional_research',max_risk_tier=2)
                start=time.perf_counter()
                response=client.post('/v1/applications/body-lotion/design',json=request.model_dump(mode='json'))
                assert response.status_code==200,response.text
                result=response.json()
                assert result['perception_model']['component_model_sha256']==sha
                cached=client.post('/v1/applications/body-lotion/design',json=request.model_dump(mode='json'))
                assert cached.json()==result and cached.headers['X-Perfumery-Lotion-Cache']=='hit'
                learned=result['learned_optimization']
                if learned['recipe_changed']:
                    assert learned['fresh_transport_verified']
                row={'strict_score':result['score'],'passed95':result['profile_target_met'],
                     'learned':learned,'seconds':time.perf_counter()-start,
                     'component_model_sha256':sha,'perception_model':result['perception_model'],
                     'recipe':result.get('recipe') or result.get('closest_candidate'),
                     'simulation':result['simulation'],'scenario_simulations':result.get('scenario_simulations',[])}
                rows[text][version]=row
                print(json.dumps({'version':version,'brief':text,'strict_score':row['strict_score'],
                    'learned_status':learned['status'],'recipe_changed':learned['recipe_changed']},ensure_ascii=False),flush=True)
    changes=[]
    for text, pair in rows.items():
        a,b=pair['v3'],pair['v4']
        changes.append({'brief':text,'recipe_changed_between_models':a['recipe']!=b['recipe'],
            'v3_strict_score':a['strict_score'],'v4_strict_score':b['strict_score'],
            'warning':'affinities from different checkpoints are not an accuracy comparison'})
    args.output.write_text(json.dumps({'scope':'four_predeclared_local_API_integration_cases_not_accuracy_or_full400',
        'component_quality_evidence':'.benchmarks/perception_core_v4/run-01/report.json',
        'learned_model_enabled':True,'catalog_held_fixed':'V32','api_url_or_key_changed':False,
        'production_deployed':False,'comparison':changes,'rows':rows},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
