"""One output-scale comparison on the frozen V5 molecular development split."""
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
from fragrance_ai.research.conditional_profiles import feature_matrix, SCHEMA
from fragrance_ai.research.fine_odor_features import append_features
from fragrance_ai.research.kernel_profiles import fit_kernel_v5, predict_component_regressor
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison

OPTIONS = [(alpha, weight, transform) for alpha in (.1, 1., 10.) for weight in (.0, .25, .5, .75)
           for transform in ('identity', 'sqrt', 'log1p')]
BASELINE_SHA = '33075f5af60d7579ebb2c3df8aeada146c782db73e576f7d4f41c1b9ef4732d9'


def fit(x, y, alpha, weight, transform):
    if transform not in ('identity', 'sqrt', 'log1p') or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('invalid quantitative component target')
    target = np.sqrt(y) if transform == 'sqrt' else np.log1p(y) if transform == 'log1p' else y
    model = fit_kernel_v5(x, target, alpha, fine_weight=weight)
    if transform != 'identity':
        model.update(kind='molecular-kernel-v6', target_transform=transform)
    return model


def predict(model, x):
    transform = model.get('target_transform', 'identity')
    if (transform not in ('identity', 'sqrt', 'log1p') or
            model['kind'] not in ('molecular-kernel-v5', 'molecular-kernel-v6') or
            (model['kind'] == 'molecular-kernel-v5' and transform != 'identity')):
        raise ValueError('invalid transformed component model')
    raw = predict_component_regressor({**model, 'kind': 'molecular-kernel-v5'}, x)
    with np.errstate(over='raise', invalid='raise'):
        return raw**2 if transform == 'sqrt' else np.expm1(raw) if transform == 'log1p' else raw


def choose(x, y, groups):
    folds, records = group_folds(groups, 3), []
    for alpha, weight, transform in OPTIONS:
        output = np.full(y.shape, np.nan)
        for fold in range(3):
            train = folds != fold
            output[~train] = predict(fit(x[train], y[train], alpha, weight, transform), x[~train])
        loss = float(np.nan_to_num(profile_errors(output[:, prior.AXES], y[:, prior.AXES])['cosine_distance'], nan=1.).mean())
        records.append({'alpha': alpha, 'fine_weight': weight, 'target_transform': transform, 'loss': loss})
    best = min(records, key=lambda r: r['loss'])
    return fit(x, y, best['alpha'], best['fine_weight'], best['target_transform']), {'selected': best, 'options': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new experiment directory required')
    old = prior.old
    denied = old.install_training_only_guard(args.source.resolve())
    bindings = old.verify_inputs(args.source)
    keys, y, _, audit = old.extended_observations(args.source)
    bank = old.molecular_bank(args.source, ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    baseline_dir = ROOT/'.benchmarks/perception_core_v5/run-01'
    raw = (baseline_dir/'model.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError('fixed V5 checkpoint mismatch')
    baseline_model = json.loads(raw)
    if baseline_model['inputs'] != bindings:
        raise ValueError('frozen training sources changed')
    fine = baseline_model['fine_odor_features']
    x = append_features(feature_matrix(keys, bank, 'molecular'),
        [bank.get(k[0], {}).get('canonical_smiles', '') for k in keys], fine)
    present, groups = np.isfinite(x).all(1), [k[0] for k in keys]
    folds = group_folds(groups, 5)
    original_report = json.loads((baseline_dir/'report.json').read_text(encoding='utf-8'))
    if groups != original_report['outer_group_ids'] or folds.tolist() != original_report['outer_fold_ids']:
        raise ValueError('frozen molecular split changed')
    baseline = np.array(json.loads((baseline_dir/'predictions.json').read_text(encoding='utf-8'))['v5'], float)
    baseline[baseline < 0] = np.nan
    assert np.array_equal(np.isfinite(baseline).all(1), present)
    paths = [Path(__file__), ROOT/'fragrance_ai/research/kernel_profiles.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    protocol = {'scope': 'existing_molecule_disjoint_development_not_fresh_blind_or_recipe95',
        'options': OPTIONS, 'outer_fold_ids': folds.tolist(), 'outer_group_ids': groups,
        'inputs': bindings, 'baseline_sha256': BASELINE_SHA, 'code_sha256': hashes,
        'all_conditions': len(keys), 'predicted_conditions': int(present.sum()),
        'evaluated_axes': list(prior.AXES), 'protected_outcomes_excluded': True}
    args.output.mkdir(parents=True)
    old.v1.write_new(args.output/'protocol.json', protocol)
    start = time.perf_counter()
    output, choices = np.full(y.shape, np.nan), []
    for fold in range(5):
        train, test = present & (folds != fold), present & (folds == fold)
        assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
        model, choice = choose(x[train], y[train], np.asarray(groups)[train].tolist())
        output[test] = predict(model, x[test])
        choices.append({'fold': fold, **choice})
        print(f'component output-scale: molecule fold {fold+1}/5', flush=True)
    comparison = paired_profile_comparison(baseline[:, prior.AXES], output[:, prior.AXES], y[:, prior.AXES], groups)
    model, choice = choose(x[present], y[present], np.asarray(groups)[present].tolist())
    fitted = dict(zip(np.flatnonzero(present), predict(model, x[present])))
    component = {**baseline_model['component_model'], 'regressor': model,
        'anchors': [{'key': list(k), 'profile': row.tolist(), 'base_prediction': fitted[i].tolist() if i in fitted else None}
                    for i, (k, row) in enumerate(zip(keys, y))]}
    artifact = {**baseline_model, 'component_model': component, 'parent_checkpoint_sha256': BASELINE_SHA,
                'candidate_version': 'component-output-transform/v6'}
    old.v1.write_new(args.output/'model.json', artifact)
    replay = json.loads((args.output/'model.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(predict(model, x[present]), predict(replay['component_model']['regressor'], x[present]), rtol=0, atol=1e-12)
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    report = {'comparison': comparison, 'selections': choices, 'selected': choice, 'data_audit': audit,
        'model_sha256': old.v1.sha(args.output/'model.json'), 'outcome_read_attempts': denied,
        'seconds': time.perf_counter()-start, 'source_unchanged': True, 'runtime_promoted': False,
        'recipe400_measured': False, 'all_conditions': len(keys), 'predicted_conditions': int(present.sum())}
    old.v1.write_new(args.output/'report.json', report)
    old.v1.write_new(args.output/'predictions.json', {'candidate': np.where(np.isfinite(output), output, -1.).tolist()})
    print(json.dumps({'comparison': comparison, 'selected': choice['selected'], 'seconds': report['seconds'], 'denied_reads': denied}), flush=True)


if __name__ == '__main__':
    main()
