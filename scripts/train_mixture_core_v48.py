"""Training-only composition-disjoint mixture learning using fixed V5 stocks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_conditional_profiles_v2 as old
from fragrance_ai.research.conditional_profiles import feature_matrix,predict_conditional
from fragrance_ai.research.fine_odor_features import append_features,validate_features
from fragrance_ai.research.mixture_profiles import mixture_features,fit_mixture_model,predict_mixture_model
from fragrance_ai.research.perception_validation import group_folds,profile_errors,paired_profile_comparison,summarize_profiles

OPTIONS = [(alpha,bandwidth,scaling) for alpha in (.01,.1,1.) for bandwidth in (.25,1.,4.)
           for scaling in ('per_feature','shared_sensory')]
AXES = [old.v1.ENDPOINTS.index(name) for name in old.v1.EVALUATED_ENDPOINTS]


def choose(x,y,groups):
    present = np.isfinite(x).all(1)
    folds = group_folds(groups,3)
    choices = []
    for alpha,bandwidth,scaling in OPTIONS:
        prediction = np.full(y.shape,np.nan)
        for fold in range(3):
            train,test = (folds != fold)&present,(folds == fold)&present
            model = fit_mixture_model(x[train],y[train],alpha,bandwidth,scaling)
            prediction[test] = predict_mixture_model(model,x[test])
        loss = float(np.nan_to_num(profile_errors(prediction[:,AXES],y[:,AXES])['cosine_distance'],nan=1.).mean())
        choices.append({'alpha':alpha,'bandwidth':bandwidth,'scaling':scaling,'loss':loss})
    selected = min(choices,key=lambda row:row['loss'])
    return fit_mixture_model(x[present],y[present],selected['alpha'],selected['bandwidth'],selected['scaling']),{'selected':selected,'options':choices}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--component',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('new output directory required')
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    component = json.loads(args.component.read_text(encoding='utf-8'))
    if component['schema'] != 'perception-core-candidate/v5' or component['inputs'] != bindings:
        raise ValueError('fixed V5 training provenance mismatch')
    validate_features(component['fine_odor_features'])
    keys,_,stimuli,_ = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source,ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    protected = {old.v1.formula_key(stimuli[row['stimulus']])
        for name in ('target_ids','final_test_ids') for row in old.v1.rows(args.source/old.INPUTS[name][0])}
    raw = old.v1.rows(args.source/old.INPUTS['training'][0])
    training = [row for row in raw if old.v1.formula_key(stimuli[row['stimulus']]) not in protected]
    ids = [row['stimulus'] for row in training]
    groups = [old.v1.formula_key(stimuli[i]) for i in ids]
    stock_keys = sorted({key for stimulus in ids for key in stimuli[stimulus]})
    x = append_features(feature_matrix(stock_keys,bank,'molecular'),
        [bank.get(k[0],{}).get('canonical_smiles','') for k in stock_keys],component['fine_odor_features'])
    predicted,status = predict_conditional(component['component_model'],x,stock_keys)
    lookup = dict(zip(stock_keys,predicted))
    status = dict(zip(stock_keys,status))
    old_x,mean,meta = old.blend_arrays(ids,stimuli,lookup,status)
    y = np.array([old.v1.vector(row) for row in training])
    features = np.array([mixture_features(np.array([lookup[k] for k in stimuli[i]]),stimuli[i])
        if all(np.isfinite(lookup[k]).all() for k in stimuli[i]) else np.full(3*y.shape[1]+4,np.nan) for i in ids])
    folds = group_folds(groups,5)
    protocol = {'scope':'known_stock_components_unseen_composition_development_CV_not_lotion_or_human_accuracy',
        'inputs':bindings,'component_sha256':old.v1.sha(args.component),'options':OPTIONS,
        'training_mixtures':len(training),'training_compositions':len(set(groups)),
        'protected_composition_rows_removed':len(raw)-len(training),'feature_width':features.shape[1],
        'evaluated_endpoints':list(old.v1.EVALUATED_ENDPOINTS),'outer_folds':folds.tolist(),'composition_groups':groups,
        'old_feature_baseline':'same_V5_stock_predictions_plus_nested_legacy_ridge_mixture_calibration',
        'code_sha256':{str(path.relative_to(ROOT)):old.v1.sha(path) for path in
            (Path(__file__),ROOT/'fragrance_ai/research/mixture_profiles.py')}}
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'protocol.json',protocol)
    predictions = {name:np.full(y.shape,np.nan) for name in ('legacy_blend','nonlinear')}
    selections = []
    started = time.perf_counter()
    present = np.isfinite(features).all(1)
    for fold in range(5):
        train,test = folds != fold,folds == fold
        assert not set(np.asarray(groups)[train])&set(np.asarray(groups)[test])
        base,base_choice = old.fit_blend(old_x[train],y[train],mean[train],np.asarray(groups)[train].tolist())
        predictions['legacy_blend'][test] = old.predict_blend(base,old_x[test],mean[test])
        model,selection = choose(features[train],y[train],np.asarray(groups)[train].tolist())
        predictions['nonlinear'][test&present] = predict_mixture_model(model,features[test&present])
        selections.append({'fold':fold,'legacy':base_choice,'nonlinear':selection})
        print('Completed nonlinear-mixture composition fold '+str(fold+1)+'/5',flush=True)
    model,selection = choose(features,y,groups)
    comparison = paired_profile_comparison(predictions['legacy_blend'][:,AXES],predictions['nonlinear'][:,AXES],y[:,AXES],groups)
    against_mean = paired_profile_comparison(mean[:,AXES],predictions['nonlinear'][:,AXES],y[:,AXES],groups)
    artifact = {'schema':'stock-mixture-candidate/v1','model':model,'endpoints':list(old.v1.ENDPOINTS),
        'component_sha256':protocol['component_sha256'],'inputs':bindings,
        'application_domain':'relative_aliquots_of_explicit_stock_conditions',
        'lotion_headspace_calibrated':False,'runtime_promotion_allowed':False,'data_redistribution_authorized':False}
    old.v1.write_new(args.output/'model.json',artifact)
    replay = json.loads((args.output/'model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(predict_mixture_model(model,features[present]),
        predict_mixture_model(replay['model'],features[present]),atol=1e-12,rtol=0)
    assert all(old.v1.sha(ROOT/name) == digest for name,digest in protocol['code_sha256'].items())
    report = {**{k:v for k,v in protocol.items() if k not in ('inputs','outer_folds','composition_groups','code_sha256')},
        'predicted_mixtures':int(present.sum()),'comparison_vs_same_stock_legacy_blend':comparison,
        'comparison_vs_component_mean':against_mean,'model_sha256':old.v1.sha(args.output/'model.json'),
        'results':{name:summarize_profiles(pred[:,AXES],y[:,AXES],groups) for name,pred in predictions.items()},
        'selections':selections,'selected':selection,'outcome_read_attempts':denied,'seconds':time.perf_counter()-started,
        'recipe400_measured':False,'actual_human_accuracy_95_authorized':False}
    old.v1.write_new(args.output/'report.json',report)
    old.v1.write_new(args.output/'predictions.json',{'ids':ids,**{name:np.where(np.isfinite(pred),pred,-1.).tolist()
        for name,pred in predictions.items()}})
    print(json.dumps({k:v for k,v in report.items() if k in ('training_mixtures','training_compositions','predicted_mixtures',
        'comparison_vs_same_stock_legacy_blend','comparison_vs_component_mean','seconds')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
