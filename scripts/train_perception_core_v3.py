"""Train a CPU component-model candidate; sealed test outcomes are never opened."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_conditional_profiles_v2 as old
from fragrance_ai.research.conditional_profiles import feature_matrix, SCHEMA
from fragrance_ai.research.kernel_profiles import fit_kernel, predict_component_regressor
from fragrance_ai.research.perception_validation import fit_ridge, group_folds, profile_errors, paired_profile_comparison

OPTIONS = [('ridge', a) for a in (10., 100., 1000.)] + [('kernel', a) for a in (.01, .1, 1., 10.)]
AXES = [old.v1.ENDPOINTS.index(n) for n in old.v1.EVALUATED_ENDPOINTS]


def fit(option, x, y):
    return (fit_ridge if option[0] == 'ridge' else fit_kernel)(x, y, option[1])


def choose(x, y, groups, options):
    folds, candidates = group_folds(groups, 3), []
    for option in options:
        p = np.full(y.shape, np.nan)
        for fold in range(3):
            train = folds != fold
            p[~train] = predict_component_regressor(fit(option, x[train], y[train]), x[~train])
        loss = float(np.nan_to_num(profile_errors(p[:, AXES], y[:, AXES])['cosine_distance'], nan=1.).mean())
        candidates.append({'option': option, 'loss': loss})
    selected = min(candidates, key=lambda row: row['loss'])
    return fit(selected['option'], x, y), selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('choose a new output directory')
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    keys, y, _, audit = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source, ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    x = feature_matrix(keys, bank, 'molecular')
    present = np.isfinite(x).all(1)
    indices = np.flatnonzero(present)
    groups = [key[0] for key in keys]
    folds = group_folds(groups, 5)
    predictions = {name: np.full(y.shape, np.nan) for name in ('ridge', 'candidate')}
    selections = []
    start = time.perf_counter()
    for fold in range(5):
        train, test = present & (folds != fold), present & (folds == fold)
        for name, options in [('ridge', OPTIONS[:3]), ('candidate', OPTIONS)]:
            model, selected = choose(x[train], y[train], [g for g, keep in zip(groups, train) if keep], options)
            predictions[name][test] = predict_component_regressor(model, x[test])
            selections.append({'fold': fold, 'method': name, **selected})
        print(f'Completed molecule-disjoint outer fold {fold+1}/5', flush=True)
    comparison = paired_profile_comparison(predictions['ridge'][:, AXES], predictions['candidate'][:, AXES], y[:, AXES], groups)
    regressor, selected = choose(x[present], y[present], [groups[i] for i in indices], OPTIONS)
    fitted = predict_component_regressor(regressor, x[present])
    by_index = dict(zip(indices, fitted))
    anchors = [{'key': list(key), 'profile': row.tolist(), 'base_prediction': by_index[i].tolist() if i in by_index else None}
               for i, (key, row) in enumerate(zip(keys, y))]
    model = {'schema': SCHEMA, 'regressor': regressor, 'anchors': anchors, 'anchored': True,
             'feature_width': x.shape[1], 'profile_width': y.shape[1], 'extrapolation_decay_decades': 1.,
             'actual_human_accuracy_90_authorized': False}
    artifact = {'schema': 'perception-core-candidate/v3', 'component_model': model,
                'profile_dimensions': list(old.v1.ENDPOINTS), 'runtime_promotion_allowed': False,
                'data_redistribution_authorized': False, 'inputs': bindings}
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'model.json', artifact)
    report = {'scope': 'nested molecule-disjoint training-development CV, not new blind test or lotion validation',
              'all_conditions': len(keys), 'predicted_conditions': int(present.sum()), 'molecule_groups': len(set(groups)),
              'data_audit': audit, 'comparison': comparison, 'selections': selections, 'selected': selected,
              'outcome_read_attempts': denied, 'seconds': time.perf_counter()-start,
              'model_sha256': old.v1.sha(args.output/'model.json'), 'runtime_promotion_allowed': False}
    old.v1.write_new(args.output/'report.json', report)
    old.v1.write_new(args.output/'predictions.json', {name: np.where(np.isfinite(values), values, -1).tolist() for name, values in predictions.items()})
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
