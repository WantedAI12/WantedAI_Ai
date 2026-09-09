"""Train-only output-scale selection for full quantitative odor profiles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fragrance_ai.research.atlas_profiles import load_atlas, atlas_features, fit_atlas, predict_atlas, SCHEMA
from fragrance_ai.research.perception_validation import group_folds, profile_errors, paired_profile_comparison, summarize_profiles

BASELINE_SHA = 'f136d3d9dae4a30e9b6fb5f910111e5cb02aedf9d9c99db7cb79b667361c0554'
OPTIONS = [(alpha, weight, transform) for alpha in (.1, 1., 10.) for weight in (0., .25, .5, .75)
           for transform in ('identity', 'sqrt', 'log1p')]


def fit(x, y, alpha, weight, transform):
    if transform not in ('identity', 'sqrt', 'log1p'):
        raise ValueError('unknown quantitative target transform')
    y = np.asarray(y, float)
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError('finite nonnegative quantitative observations required')
    target = np.sqrt(y) if transform == 'sqrt' else np.log1p(y) if transform == 'log1p' else y
    model = fit_atlas(x, target, alpha, weight)
    if transform != 'identity':
        # Old runtimes must reject transformed coefficients, not interpret them
        # as raw measurements. The feature and output endpoint contracts stay.
        model.update(kind='atlas-quantitative-profiles/v2', target_transform=transform)
    return model


def predict(model, x):
    transform = model.get('target_transform', 'identity')
    if (transform not in ('identity', 'sqrt', 'log1p') or
            (model['kind'] == SCHEMA and transform != 'identity') or
            model['kind'] not in (SCHEMA, 'atlas-quantitative-profiles/v2')):
        raise ValueError('invalid quantitative target transform contract')
    # One implementation for training round trips and the real CPU runtime.
    return predict_atlas(model, x)


def choose(x, y, groups):
    folds = group_folds(groups, 3)
    records = []
    for alpha, weight, transform in OPTIONS:
        prediction = np.full(y.shape, np.nan)
        for fold in range(3):
            train, test = folds != fold, folds == fold
            model = fit(x[train], y[train], alpha, weight, transform)
            prediction[test] = predict(model, x[test])
        errors = profile_errors(prediction, y)
        records.append({'alpha': alpha, 'fine_weight': weight, 'target_transform': transform,
                        'loss': float(np.nan_to_num(errors['cosine_distance'], nan=1.).mean())})
    selected = min(records, key=lambda r: r['loss'])
    return fit(x, y, selected['alpha'], selected['fine_weight'], selected['target_transform']), {
        'selected': selected, 'options': records}


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new experiment directory required')
    baseline_dir = ROOT/'.benchmarks/atlas_profiles_v50/run-01'
    raw = (baseline_dir/'model.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError('fixed baseline checkpoint drift')
    baseline = json.loads(raw)
    source = ROOT/'.benchmarks/atlas_profiles_v50/source'
    rows, endpoints, source_manifest = load_atlas(source)
    if baseline['source'] != source_manifest or baseline['endpoints'] != endpoints:
        raise ValueError('baseline source/endpoint mismatch')
    old_protocol = json.loads((baseline_dir/'protocol.json').read_text(encoding='utf-8'))
    old_prediction = json.loads((baseline_dir/'predictions.json').read_text(encoding='utf-8'))
    if old_prediction['ids'] != [r['id'] for r in rows]:
        raise ValueError('baseline prediction identity mismatch')
    groups = [r['graph'] or 'unresolved:'+r['id'] for r in rows]
    x = atlas_features([r['graph'] for r in rows], [r['level'] for r in rows],
                       baseline['native_profiles'], baseline['fine_features'])
    present = np.isfinite(x).all(1)
    folds = group_folds(groups, 5)
    if old_protocol['groups'] != groups or old_protocol['outer_folds'] != folds.tolist():
        raise ValueError('frozen development partition mismatch')
    code_paths = [Path(__file__), ROOT/'fragrance_ai/research/atlas_profiles.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    protocol = {'scope': 'quantitative_applicability_and_use_not_intensity_or_recipe95',
        'all_stimuli': len(rows), 'predictable_stimuli': int(present.sum()), 'endpoints': endpoints,
        'groups': groups, 'outer_folds': folds.tolist(), 'inner_folds': 3, 'options': OPTIONS,
        'old_predictions_sha256': hashlib.sha256((baseline_dir/'predictions.json').read_bytes()).hexdigest(),
        'baseline_checkpoint_sha256': BASELINE_SHA, 'source': source_manifest, 'code_sha256': hashes,
        'recipe400_read_for_training': False, 'new_blind_evaluation': False,
        'unresolved_stimuli_retained_in_denominator': True}
    args.output.mkdir(parents=True)
    write(args.output/'protocol.json', protocol)
    start = time.perf_counter()
    models, reports, predictions, selections = {}, {}, {}, {}
    for measurement in ('applicability', 'use'):
        y = np.array([r[measurement] for r in rows])
        old = np.array(old_prediction[measurement]['atlas'], float)
        old[old < 0] = np.nan
        candidate = np.full(y.shape, np.nan)
        selections[measurement] = []
        for fold in range(5):
            train, test = present & (folds != fold), present & (folds == fold)
            assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
            model, choice = choose(x[train], y[train], np.asarray(groups)[train].tolist())
            candidate[test] = predict(model, x[test])
            selections[measurement].append({'fold': fold, **choice})
            print(f'{measurement}: molecule fold {fold+1}/5', flush=True)
        models[measurement], selection = choose(x[present], y[present], np.asarray(groups)[present].tolist())
        predictions[measurement] = np.where(np.isfinite(candidate), candidate, -1.).tolist()
        reports[measurement] = {'comparison': paired_profile_comparison(old, candidate, y, groups),
            'baseline': summarize_profiles(old, y, groups), 'candidate': summarize_profiles(candidate, y, groups),
            'selected': selection}
    artifact = {**baseline, 'models': models, 'parent_checkpoint_sha256': BASELINE_SHA,
                'quantitative_target_model_version': 'output-transformed-atlas/v2'}
    write(args.output/'model.json', artifact)
    restored = json.loads((args.output/'model.json').read_text(encoding='utf-8'))
    for measurement in models:
        np.testing.assert_allclose(predict(models[measurement], x[present]),
            predict(restored['models'][measurement], x[present]), rtol=0, atol=1e-12)
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    report = {'scope': protocol['scope'], 'all_stimuli': len(rows), 'predictable_stimuli': int(present.sum()),
        'measurements': reports, 'selections': selections, 'seconds': time.perf_counter()-start,
        'source_unchanged': True, 'recipe400_measured': False, 'runtime_promoted': False,
        'model_sha256': hashlib.sha256((args.output/'model.json').read_bytes()).hexdigest()}
    write(args.output/'report.json', report)
    write(args.output/'predictions.json', {'ids': [r['id'] for r in rows], **predictions})
    print(json.dumps({'comparisons': {k: v['comparison'] for k, v in reports.items()},
        'selected': {k: v['selected']['selected'] for k, v in reports.items()}, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
