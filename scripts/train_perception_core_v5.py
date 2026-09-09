"""V5 fine-source features; same molecule-disjoint development folds as V4."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_perception_core_v3 as prior
from scripts.verify_odor_concepts_v39 import read_catalog
from fragrance_ai.research.conditional_profiles import feature_matrix, SCHEMA
from fragrance_ai.research.fine_odor_features import catalog_features, append_features
from fragrance_ai.research.kernel_profiles import fit_kernel_v5, predict_component_regressor
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison

OPTIONS = [(alpha,weight) for alpha in (.1,1.,10.) for weight in (.0,.25,.5,.75)]


def choose(x,y,groups):
    folds, results = group_folds(groups,3), []
    for alpha,weight in OPTIONS:
        p = np.full(y.shape,np.nan)
        for fold in range(3):
            train = folds != fold
            m = fit_kernel_v5(x[train],y[train],alpha,fine_weight=weight)
            p[~train] = predict_component_regressor(m,x[~train])
        loss = float(np.nan_to_num(profile_errors(p[:,prior.AXES],y[:,prior.AXES])['cosine_distance'],nan=1.).mean())
        results.append({'alpha':alpha,'fine_weight':weight,'loss':loss})
    selected = min(results,key=lambda r:r['loss'])
    return fit_kernel_v5(x,y,selected['alpha'],fine_weight=selected['fine_weight']), selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--catalog-manifest',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output directory')
    old = prior.old
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    keys,y,_,audit = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source,ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    baseline_model = json.loads((args.baseline/'model.json').read_text(encoding='utf-8'))
    baseline_report = json.loads((args.baseline/'report.json').read_text(encoding='utf-8'))
    if baseline_model['schema'] != 'perception-core-candidate/v4' or baseline_model['inputs'] != bindings:
        raise ValueError('baseline data provenance differs')
    catalog = read_catalog(args.catalog_manifest)
    features = catalog_features(catalog)
    x = append_features(feature_matrix(keys,bank,'molecular'),
        [bank.get(k[0],{}).get('canonical_smiles','') for k in keys], features)
    present, groups = np.isfinite(x).all(1), [k[0] for k in keys]
    folds = group_folds(groups,5)
    if baseline_report['outer_group_ids'] != groups or baseline_report['outer_fold_ids'] != folds.tolist():
        raise ValueError('baseline case order or fold assignments differ')
    baseline = np.asarray(json.loads((args.baseline/'predictions.json').read_text(encoding='utf-8'))['v4'],float)
    baseline[baseline < 0] = np.nan
    if baseline.shape != y.shape or not np.array_equal(np.isfinite(baseline).all(1),present):
        raise ValueError('baseline prediction coverage changed')
    predictions, selections = np.full(y.shape,np.nan), []
    started = time.perf_counter()
    for fold in range(5):
        train,test = present & (folds != fold),present & (folds == fold)
        assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
        model,selected = choose(x[train],y[train],np.asarray(groups)[train].tolist())
        predictions[test] = predict_component_regressor(model,x[test])
        selections.append({'fold':fold,**selected})
        print('Completed V5 molecule-disjoint fold '+str(fold+1)+'/5',flush=True)
    comparison = paired_profile_comparison(baseline[:,prior.AXES],predictions[:,prior.AXES],y[:,prior.AXES],groups)
    selected_model,selected = choose(x[present],y[present],np.asarray(groups)[present].tolist())
    fitted = dict(zip(np.flatnonzero(present),predict_component_regressor(selected_model,x[present])))
    component = {'schema':SCHEMA,'regressor':selected_model,'feature_width':x.shape[1],
        'feature_family':'source_bound_fine_v1','profile_width':y.shape[1],'anchored':True,
        'extrapolation_decay_decades':1.,'actual_human_accuracy_90_authorized':False,
        'anchors':[{'key':list(k),'profile':v.tolist(),'base_prediction':fitted[i].tolist() if i in fitted else None}
                   for i,(k,v) in enumerate(zip(keys,y))]}
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'model.json',{'schema':'perception-core-candidate/v5','component_model':component,
        'fine_odor_features':features,'profile_dimensions':list(old.v1.ENDPOINTS),'inputs':bindings,
        'runtime_promotion_allowed':False,'data_redistribution_authorized':False})
    report = {'scope':'same nested molecule-disjoint development population as V4; not fresh blind test or lotion accuracy',
        'comparison':comparison,'all_conditions':len(keys),'predicted_conditions':int(present.sum()),
        'conditions_with_fine_features':int((present & (x[:,-1] > 0)).sum()),
        'fine_vocabulary_count':len(features['vocabulary']),'fine_structure_count':len(features['by_structure']),
        'features_do_not_change_catalog_profiles_or_95_gate':True,
        'catalog_manifest_sha256':old.v1.sha(args.catalog_manifest), 'fine_features_sha256':features['content_sha256'],
        'baseline_sha256':{name:old.v1.sha(args.baseline/name) for name in ('model.json','report.json','predictions.json')},
        'model_sha256':old.v1.sha(args.output/'model.json'),'data_audit':audit,'selections':selections,
        'selected':selected,'options_predeclared':OPTIONS,'outcome_read_attempts':denied,
        'outer_group_ids':groups,'outer_fold_ids':folds.tolist(),'seconds':time.perf_counter()-started,
        'candidate_improves_both_point_errors':comparison['baseline_mae_minus_candidate_mae'] > 0
            and comparison['baseline_cosine_distance_minus_candidate'] > 0,
        'recipe_400_pass_rate_measured':False,'human_accuracy_95_authorized':False}
    old.v1.write_new(args.output/'report.json',report)
    old.v1.write_new(args.output/'predictions.json',{'v5':np.where(np.isfinite(predictions),predictions,-1).tolist()})
    print(json.dumps({k:v for k,v in report.items() if k not in ('selections','outer_group_ids','outer_fold_ids','options_predeclared','baseline_sha256','data_audit')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
