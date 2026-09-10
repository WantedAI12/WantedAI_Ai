"""Fixed nested molecule-disjoint V3/V4 comparison on training data only."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_perception_core_v3 as previous
from fragrance_ai.research.kernel_profiles import fit_kernel_v4, predict_component_regressor
from fragrance_ai.research.conditional_profiles import feature_matrix, SCHEMA
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison

OPTIONS = previous.OPTIONS + [('v4', alpha, fp, native, dose)
    for alpha in (.1, 1., 10.) for fp, native in ((.25,.25), (.5,.25), (.75,0.), (1.,0.)) for dose in (None,2.)]


def fit(option, x, y):
    if option[0] != 'v4':
        return previous.fit(option, x, y)
    return fit_kernel_v4(x, y, option[1], fingerprint_weight=option[2], native_weight=option[3], dose_bandwidth=option[4])


def choose(x, y, groups, options):
    folds, candidates = group_folds(groups, 3), []
    for option in options:
        predictions = np.full(y.shape, np.nan)
        for fold in range(3):
            train = folds != fold
            predictions[~train] = predict_component_regressor(fit(option, x[train], y[train]), x[~train])
        loss = float(np.nan_to_num(profile_errors(predictions[:, previous.AXES], y[:, previous.AXES])['cosine_distance'], nan=1.).mean())
        candidates.append({'option': option, 'loss': loss})
    selected = min(candidates, key=lambda row: row['loss'])
    return fit(selected['option'], x, y), selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output directory')
    old = previous.old
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    keys, y, _, audit = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source, ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    x = feature_matrix(keys, bank, 'molecular')
    present = np.isfinite(x).all(1)
    groups = [key[0] for key in keys]
    folds = group_folds(groups, 5)
    predictions = {name: np.full(y.shape, np.nan) for name in ('v3', 'v4')}
    selections = []
    start = time.perf_counter()
    for fold in range(5):
        train, test = present & (folds != fold), present & (folds == fold)
        assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
        for name, options in (('v3', previous.OPTIONS), ('v4', OPTIONS)):
            model, selected = choose(x[train], y[train], np.asarray(groups)[train].tolist(), options)
            predictions[name][test] = predict_component_regressor(model, x[test])
            selections.append({'fold':fold, 'method':name, **selected})
        print('Completed outer fold '+str(fold+1)+'/5', flush=True)
    comparison = paired_profile_comparison(predictions['v3'][:, previous.AXES], predictions['v4'][:, previous.AXES], y[:, previous.AXES], groups)
    regressor, selected = choose(x[present], y[present], np.asarray(groups)[present].tolist(), OPTIONS)
    fitted = dict(zip(np.flatnonzero(present), predict_component_regressor(regressor, x[present])))
    anchors = [{'key': list(key), 'profile': row.tolist(), 'base_prediction': fitted[i].tolist() if i in fitted else None}
               for i, (key, row) in enumerate(zip(keys, y))]
    component = {'schema':SCHEMA, 'regressor':regressor, 'anchors':anchors, 'anchored':True,
        'feature_width':x.shape[1], 'profile_width':y.shape[1], 'extrapolation_decay_decades':1.,
        'actual_human_accuracy_90_authorized':False}
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'model.json', {'schema':'perception-core-candidate/v4', 'component_model':component,
        'profile_dimensions':list(old.v1.ENDPOINTS), 'runtime_promotion_allowed':False,
        'data_redistribution_authorized':False, 'inputs':bindings})
    report = {'scope':'fixed nested molecule-disjoint training-development CV; not independent blind test or lotion validation',
        'comparison':comparison, 'all_conditions':len(keys), 'predicted_conditions':int(present.sum()),
        'molecule_groups':len(set(groups)), 'selections':selections, 'selected':selected, 'data_audit':audit,
        'outcome_read_attempts':denied, 'seconds':time.perf_counter()-start,
        'model_sha256':old.v1.sha(args.output/'model.json'), 'runtime_promotion_allowed':False,
        'outer_group_ids':groups, 'outer_fold_ids':folds.tolist(), 'options_predeclared':OPTIONS}
    old.v1.write_new(args.output/'report.json', report)
    old.v1.write_new(args.output/'predictions.json', {name:np.where(np.isfinite(v),v,-1).tolist() for name,v in predictions.items()})
    print(json.dumps({k:v for k,v in report.items() if k not in ('selections','outer_group_ids','outer_fold_ids','options_predeclared')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
