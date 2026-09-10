"""Actual frozen-source SDK parity and local API acceptance for integrated V54."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('preserve existing verification evidence')
    from scripts import train_conditional_profiles_v2 as old
    from scripts.train_mixture_core_v48 import AXES
    from fragrance_ai import StockAliquot,StockControlAliquot,StockMixturePredictor
    from fragrance_ai.recommender.models import Ingredient
    from fragrance_ai.recommender.perception_guidance import PerceptionGuidance
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.runtime import RuntimeAIFactory,_source_snapshot,_data_snapshot
    from fragrance_ai.research.conditional_profiles import feature_matrix,predict_conditional
    from fragrance_ai.research.fine_odor_features import append_features
    from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
    from fragrance_ai.research.mixture_profiles import mixture_features
    from fragrance_ai.research.unified_mixture import atlas_mixture_features,predict_unified
    from fragrance_ai.research.perception_validation import profile_errors
    from scripts.serve_product_runtime_v42 import create_app
    from fastapi.testclient import TestClient

    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    protocol = json.loads((args.experiment/'protocol.json').read_text(encoding='utf-8'))
    evaluation = json.loads((args.experiment/'report.json').read_text(encoding='utf-8'))
    model_path = args.experiment/'model.json'
    artifact = json.loads(model_path.read_text(encoding='utf-8'))
    if bindings != protocol['inputs'] or old.v1.sha(model_path) != evaluation['model_sha256']:
        raise ValueError('experiment provenance mismatch')
    if not evaluation['local_development_accepted']: raise ValueError('unaccepted experiment')
    local = local_profile()
    assert local['stock_mixture'] == (str(model_path.resolve()),evaluation['model_sha256'])
    source_sha,data_sha = _source_snapshot(),_data_snapshot()
    _,_,stimuli,_ = old.extended_observations(args.source)
    ids = protocol['ids']
    bank = old.molecular_bank(args.source,ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    component_path = ROOT/'.benchmarks/perception_core_v5/run-01/model.json'
    component = json.loads(component_path.read_text(encoding='utf-8'))
    component_sha = old.v1.sha(component_path)
    provider = PerceptionGuidance(ROOT/'.benchmarks/conditional_profiles_v2/final-01/models.json',
        ROOT/'benchmarks/industrial_ingredient_registry_v1.db',solvent='pg',experimental=True,
        component_model_path=component_path,component_model_sha256=component_sha)
    original = AtlasProfilePredictor(args.experiment/'frozen_v54/model.json',
        sha256=artifact['atlas_sha256'],experimental=True)
    sdk = StockMixturePredictor(provider,model_path,sha256=evaluation['model_sha256'],
        atlas_predictor=original,experimental=True)
    keys = sorted({k for i in ids for k in stimuli[i]})
    stock_x = append_features(feature_matrix(keys,bank,'molecular'),
        [bank.get(k[0],{}).get('canonical_smiles','') for k in keys],component['fine_odor_features'])
    stock_values,_ = predict_conditional(component['component_model'],stock_x,keys)
    stock_lookup = dict(zip(keys,stock_values))
    graphs = sorted({bank[k[0]]['canonical_smiles'] for k in keys if k[0] in bank and '.' not in bank[k[0]]['canonical_smiles']})
    high,low = original.predict(graphs,reference_level='high'),original.predict(graphs,reference_level='low')
    embeddings = dict(zip(graphs,np.concatenate([high['applicability'],high['use'],low['applicability'],low['use']],axis=1)))
    for level,expected in (('high',high),('low',low)):
        actual = sdk.predict_molecular(graphs,reference_level=level)
        for head in expected: np.testing.assert_array_equal(expected[head],actual[head])
    # Input adapter for the exact public assay stocks, not a claim that every
    # one is a formulatable material in the live catalog. No bank is modified.
    materials = {}
    for key in keys:
        cid = key[0]
        if cid not in bank: continue
        identifier = 'verification-stock-'+cid
        materials[cid] = Ingredient(identifier,identifier,(),None,'top',{'citrus':1.},
            10.,1.,'standard',0,1.,100.,True)
        provider.structures[identifier] = (bank[cid]['canonical_smiles'],None)
    deltas,timings,partials = [],[],0
    for i in ids:
        ks = stimuli[i]
        gg = [bank.get(k[0],{}).get('canonical_smiles','unresolved:'+k[0]) for k in ks]
        supported = [g in embeddings for g in gg]
        x = mixture_features([stock_lookup[k] for k in ks],ks)
        z = atlas_mixture_features([embeddings.get(g,np.zeros(584)) for g in gg],gg,
                [float(k[1]) for k in ks],[k[2] for k in ks],supported=supported)
        expected = predict_unified(artifact['model'],x[None],z[None])[0]
        start = time.perf_counter()
        result = sdk.predict([StockAliquot(materials[k[0]],float(k[1]),1.,k[2]) for k in ks if k[0] in materials],
            controls=[StockControlAliquot(k[0],float(k[1]),1.,k[2]) for k in ks if k[0] not in materials])
        timings.append(time.perf_counter()-start)
        actual = np.array(list(result['predicted_rata_profile'].values()))
        delta = float(np.max(np.abs(actual-expected)))
        deltas.append(delta)
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0,err_msg=i)
        if result['atlas_supported_volume_fraction'] < 1.-1e-12: partials += 1
    assert partials == len(protocol['unsupported_atlas_retained_with_coverage_mask'])
    frozen = json.loads((args.experiment/'predictions.json').read_text(encoding='utf-8'))
    outcomes = {r['stimulus']:old.v1.vector(r) for r in old.v1.rows(args.source/old.INPUTS['training'][0])}
    y = np.asarray([outcomes[i] for i in ids])[:,AXES]
    base = profile_errors(np.asarray(frozen['v48_recomputed'])[:,AXES],y)
    new = profile_errors(np.asarray(frozen['unified'])[:,AXES],y)
    per_case = {metric:{'improved':int(np.sum(new[metric] < base[metric]-1e-12)),
                       'regressed':int(np.sum(new[metric] > base[metric]+1e-12)),
                       'unchanged':int(np.sum(np.abs(new[metric]-base[metric]) <= 1e-12))}
                for metric in ('mae','cosine_distance')}
    factory = RuntimeAIFactory.from_environment()
    app = create_app()
    actual_sdk = app.state.local_stock_mixture_predictor
    report = {'schema':'v54-integrated-local-verification/v1','deployed':False,
        'model_sha256':evaluation['model_sha256'],'local_profile_sha256':local['profile_sha256'],
        'source_sha256':source_sha,'data_sha256':data_sha,'sdk_assay_replay_count':len(ids),
        'sdk_training_max_absolute_delta':max(deltas),'retained_partial_atlas_cases':partials,
        'original_v54_identical_graphs':len(graphs),'original_v54_identical_heads_and_levels':4,
        'sdk_seconds':{'p50':float(np.quantile(timings,.5)),'p95':float(np.quantile(timings,.95)),
                       'max':max(timings)},'development_per_case_vs_v48':per_case,
        'local_catalog_count':len(factory.catalog.ingredients),'comparisons':evaluation['comparisons'],
        'human_accuracy_claim':False,'recipe400_remeasured':False}
    try:
        with TestClient(app) as client:
            caps = client.get('/v1/ai/capabilities')
            assert caps.status_code == 200,caps.text
            assert caps.json()['stock_mixture_model']['integrated_v54']
            assert caps.json()['stock_mixture_model']['model_sha256'] == evaluation['model_sha256']
            assert caps.json()['features']['quantized_language_backend']
            report['capabilities'] = caps.json()
            live = []
            for item in factory.catalog.ingredients:
                if item.ingredient_id not in actual_sdk.provider.structures: continue
                if '.' in actual_sdk.provider.structures[item.ingredient_id][0]: continue
                live.append(item)
                if len(live) == 2: break
            assert len(live) == 2
            body = {'basis':'relative_volume','components':[{'ingredient_id':item.ingredient_id,
                'stock_dilution':dose,'relative_volume':volume,'solvent':'pg'}
                for item,dose,volume in zip(live,(.1,.01),(2.,1.))]}
            start = time.perf_counter()
            response = client.post('/v1/formulations/stock-mixture/predict',json=body)
            elapsed = time.perf_counter()-start
            assert response.status_code == 200,response.text
            payload = response.json()
            direct = actual_sdk.predict([StockAliquot(item,dose,volume,'pg') for item,dose,volume in zip(live,(.1,.01),(2.,1.))])
            np.testing.assert_allclose(list(payload['predicted_rata_profile'].values()),
                list(direct['predicted_rata_profile'].values()),atol=1e-12,rtol=0)
            start = time.perf_counter()
            cached = client.post('/v1/formulations/stock-mixture/predict',json=body)
            cached_seconds = time.perf_counter()-start
            assert cached.status_code == 200 and cached.json() == payload
            assert cached.headers['X-Perfumery-Stock-Cache'] == 'hit'
            mass_body = {'basis':'supplied_mass','components':[
                {'ingredient_id':item.ingredient_id,'stock_dilution':dose,'supplied_mass_g':mass,
                 'stock_density_g_ml':density,'solvent':'pg'}
                for item,dose,mass,density in zip(live,(.1,.01),(4.,1.),(2.,1.))]}
            mass_response = client.post('/v1/formulations/stock-mixture/predict',json=mass_body)
            assert mass_response.status_code == 200,mass_response.text
            np.testing.assert_allclose(list(mass_response.json()['predicted_rata_profile'].values()),
                list(payload['predicted_rata_profile'].values()),atol=1e-12,rtol=0)
            assert not payload['recipe_acceptance_modified'] and payload['human_similarity_percent'] is None
            report['api'] = {'status':200,'request':body,'response':payload,'sdk_exact':True,
                'mass_volume_equal':True,'cache_hit':True,'first_call_seconds':elapsed,'cache_seconds':cached_seconds}
        factory.assert_current_snapshot()
        assert source_sha == _source_snapshot() and data_sha == _data_snapshot()
        assert local['profile_sha256'] == local_profile()['profile_sha256']
        assert not denied
        report['outcome_read_attempts'] = denied
        report['status'] = 'passed'
    finally:
        app.state.local_language_backend.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','model_sha256','sdk_assay_replay_count',
        'sdk_training_max_absolute_delta','sdk_seconds','development_per_case_vs_v48','retained_partial_atlas_cases')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
