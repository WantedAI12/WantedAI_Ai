"""Bounded development comparison of cross-descriptor mixture interactions."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import train_mixture_core_v48 as base
from fragrance_ai.research.mixture_profiles import mixture_features,predict_mixture_model


def choose_balanced(x,y,groups):
    """Same nested tuning budget; fixed block geometry, no fold-label fitting."""
    from fragrance_ai.research.mixture_profiles import fit_mixture_model
    folds = base.group_folds(groups,3)
    choices = []
    for alpha in (.01,.1,1.):
        for bandwidth in (.25,1.,4.):
            prediction = np.full(y.shape,np.nan)
            for fold in range(3):
                train,test = folds != fold,folds == fold
                model = fit_mixture_model(x[train],y[train],alpha,bandwidth,'balanced_sensory')
                prediction[test] = predict_mixture_model(model,x[test])
            loss = float(np.nan_to_num(base.profile_errors(prediction[:,base.AXES],y[:,base.AXES])['cosine_distance'],nan=1.).mean())
            choices.append({'alpha':alpha,'bandwidth':bandwidth,'loss':loss})
    selected = min(choices,key=lambda row:row['loss'])
    return fit_mixture_model(x,y,selected['alpha'],selected['bandwidth'],'balanced_sensory'),{'selected':selected,'options':choices}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--component',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--balanced-blocks',action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('new output directory required')
    old = base.old
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    component = json.loads(args.component.read_text(encoding='utf-8'))
    if component['schema'] != 'perception-core-candidate/v5' or component['inputs'] != bindings:
        raise ValueError('fixed V5 provenance mismatch')
    base.validate_features(component['fine_odor_features'])
    _,_,stimuli,_ = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source,ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    protected = {old.v1.formula_key(stimuli[r['stimulus']]) for name in ('target_ids','final_test_ids')
                 for r in old.v1.rows(args.source/old.INPUTS[name][0])}
    raw = old.v1.rows(args.source/old.INPUTS['training'][0])
    training = [r for r in raw if old.v1.formula_key(stimuli[r['stimulus']]) not in protected]
    ids = [r['stimulus'] for r in training]
    groups = [old.v1.formula_key(stimuli[i]) for i in ids]
    keys = sorted({k for i in ids for k in stimuli[i]})
    x = base.append_features(base.feature_matrix(keys,bank,'molecular'),
        [bank.get(k[0],{}).get('canonical_smiles','') for k in keys],component['fine_odor_features'])
    prediction,_ = base.predict_conditional(component['component_model'],x,keys)
    lookup = dict(zip(keys,prediction))
    y = np.array([old.v1.vector(r) for r in training])
    features = {name:np.array([mixture_features(np.array([lookup[k] for k in stimuli[i]]),stimuli[i],
                    cross_moments=cross) for i in ids]) for name,cross in (('v48',False),('cross',True))}
    if not all(np.isfinite(value).all() for value in features.values()):
        raise ValueError('missing predictions must not be removed from this comparison')
    folds = base.group_folds(groups,5)
    protocol = {'scope':'reused_development_composition_CV_not_new_blind_or_lotion_validation',
        'inputs':bindings,'component_sha256':old.v1.sha(args.component),'options':base.OPTIONS,
        'training_mixtures':len(ids),'training_compositions':len(set(groups)),
        'protected_composition_rows_removed':len(raw)-len(training),
        'feature_widths':{k:v.shape[1] for k,v in features.items()},
        'outer_folds':folds.tolist(),'composition_groups':groups,'evaluated_endpoints':list(old.v1.EVALUATED_ENDPOINTS),
        'code_sha256':{str(path.relative_to(ROOT)):old.v1.sha(path) for path in
            (Path(__file__),Path(base.__file__),ROOT/'fragrance_ai/research/mixture_profiles.py')},
        'selection_rule':'candidate must improve both MAE and cosine distance on the same development rows'}
    protocol['cross_geometry'] = 'equal_sensory_block_weight' if args.balanced_blocks else 'equal_coordinate_weight'
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'protocol.json',protocol)
    predicted = {name:np.full(y.shape,np.nan) for name in features}
    selections = []
    started = time.perf_counter()
    for fold in range(5):
        train,test = folds != fold,folds == fold
        assert not set(np.asarray(groups)[train])&set(np.asarray(groups)[test])
        selection = {'fold':fold}
        for name,feature in features.items():
            choose = choose_balanced if name == 'cross' and args.balanced_blocks else base.choose
            model,choice = choose(feature[train],y[train],np.asarray(groups)[train].tolist())
            predicted[name][test] = predict_mixture_model(model,feature[test])
            selection[name] = choice
        selections.append(selection)
        print('Completed paired interaction fold '+str(fold+1)+'/5',flush=True)
    comparison = base.paired_profile_comparison(predicted['v48'][:,base.AXES],predicted['cross'][:,base.AXES],y[:,base.AXES],groups)
    improved = comparison['baseline_mae_minus_candidate_mae'] > 0 and comparison['baseline_cosine_distance_minus_candidate'] > 0
    choose = choose_balanced if args.balanced_blocks else base.choose
    model,selection = choose(features['cross'],y,groups)
    artifact = {'schema':'stock-mixture-candidate/v1','model':model,'endpoints':list(old.v1.ENDPOINTS),
        'component_sha256':protocol['component_sha256'],'inputs':bindings,
        'application_domain':'relative_aliquots_of_explicit_stock_conditions',
        'lotion_headspace_calibrated':False,'runtime_promotion_allowed':False,'data_redistribution_authorized':False}
    old.v1.write_new(args.output/'model.json',artifact)
    assert all(old.v1.sha(ROOT/name) == sha for name,sha in protocol['code_sha256'].items())
    report = {**{k:v for k,v in protocol.items() if k not in ('inputs','outer_folds','composition_groups','code_sha256')},
        'model_sha256':old.v1.sha(args.output/'model.json'),'paired_comparison':comparison,
        'candidate_improved_both_metrics':bool(improved),'predicted_mixtures':len(ids),
        'results':{k:base.summarize_profiles(v[:,base.AXES],y[:,base.AXES],groups) for k,v in predicted.items()},
        'selections':selections,'selected':selection,'seconds':time.perf_counter()-started,
        'outcome_read_attempts':denied,'recipe400_measured':False,'actual_human_accuracy_95_authorized':False}
    old.v1.write_new(args.output/'report.json',report)
    old.v1.write_new(args.output/'predictions.json',{'ids':ids,**{k:v.tolist() for k,v in predicted.items()}})
    print(json.dumps({k:report[k] for k in ('paired_comparison','candidate_improved_both_metrics','seconds')},ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
