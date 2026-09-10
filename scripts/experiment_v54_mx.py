"""Copy V54, test MX ablations locally, and keep all failures and old artifacts.

No deployment, final-test outcomes, synthetic perceptual targets, or recipe400
tuning. The nominal-stock supervised experiment and release numerical checks
are separate evidence lanes. This is repeated development CV, NOT fresh blind.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_conditional_profiles_v2 as old
from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
from fragrance_ai.research.physsim_mx import (
    VERSION, DOMAIN, MODES, mx_features, prepare_kernel, fit_mx, predict_mx,
    ReleaseEnvironment, release_trajectory)
from fragrance_ai.research.perception_validation import (
    group_folds, profile_errors, paired_profile_comparison, summarize_profiles)

V54 = ROOT/'.benchmarks/quantitative_profiles_v54/run-01'
V54_SHA = '4c01417cd4b9cbd37ea2ba88ca0a62784e5ead60bec9638b1e7bbfc7af557591'
V48 = ROOT/'.benchmarks/mixture_core_v48/run-03'
V48_HASHES = {
    'predictions.json': 'af9593d2e9380c56c36c6ece1424145b11be540d18e7698c61ff1b8a516b8148',
    'protocol.json': '14f2e78f705dbbe64a8f97618513113129e1920c1d9bd818bebc57270ac57d0f',
    'report.json': '41aba6f11016cf4dd2c446fa4a0f94ce353b2c68456328adaa08e0ebe531d622'}
ALPHAS = (.1, 1., 10.)
BANDWIDTHS = (.25, 1., 4.)
AXES = [old.v1.ENDPOINTS.index(name) for name in old.v1.EVALUATED_ENDPOINTS]


def choose(x, y, groups, mode):
    """Nine equal-budget settings, inner outcomes only; cache Gram matrices."""
    present = np.isfinite(x).all(1)
    folds = group_folds(groups, 3)
    predictions = {(alpha,bw): np.full(y.shape, np.nan) for alpha in ALPHAS for bw in BANDWIDTHS}
    for fold in range(3):
        train, test = present & (folds != fold), present & (folds == fold)
        intercept = y[train].mean(0)
        for bw in BANDWIDTHS:
            _, _, _, kernel, query = prepare_kernel(x[train], x[test], bw)
            for alpha in ALPHAS:
                weights = np.linalg.solve(kernel + alpha*np.eye(train.sum()), y[train]-intercept)
                value = np.maximum(0., query @ weights+intercept)
                value[np.all(x[test] == 0, axis=1)] = 0.
                predictions[alpha,bw][test] = value
    choices = []
    for (alpha,bw), prediction in predictions.items():
        loss = float(np.nan_to_num(profile_errors(prediction[:,AXES], y[:,AXES])['cosine_distance'], nan=1.).mean())
        choices.append({'alpha': alpha, 'bandwidth': bw, 'loss': loss})
    selected = min(choices, key=lambda row: (row['loss'], row['alpha'], row['bandwidth']))
    model = fit_mx(x[present], y[present], selected['alpha'], selected['bandwidth'], mode=mode)
    return model, {'selected': selected, 'options': choices, 'inner_group_folds': 3}


def release_diagnostic():
    """Numerical scenario, not invented experimental release observations."""
    times = [0., 1., 15., 60., 240.]
    initial = [1e-6, 2e-6, 3e-6]
    scenarios = {}
    # Deliberately explicit hypothetical per-compound/formulation inputs.
    # Not presented as measured coefficients of an actual commercial lotion.
    for name, partition, transfer in (
            ('reference_solvent_scenario', [.01,.001,.0001], 1e-6),
            ('retentive_emulsion_scenario', [.001,.0002,.00005], 2e-7)):
        env = ReleaseEnvironment(1e-6, .001, transfer, 1e-5)
        scenarios[name] = release_trajectory(initial, partition, times, env)
    return {'scope': 'numerical_physics_only_not_a_fit_to_release_or_sensory_measurements',
            'initial_moles': initial, 'scenarios': scenarios,
            'worst_mass_balance_relative_error': max(v['mass_balance_relative_error'] for v in scenarios.values()),
            'human_observations_created': 0, 'release_training_observations': 0,
            'reason_no_joint_perceptual_training': 'no_paired_product_headspace_and_sensory_labels_in_this_experiment'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source.resolve()
    if output.exists():
        parser.error('new experiment directory required; never overwrite a previous run')
    if not output.is_relative_to(ROOT/'.benchmarks'):
        parser.error('keep experimental artifacts inside local .benchmarks')
    if old.v1.sha(V54/'model.json') != V54_SHA:
        raise ValueError('V54 checkpoint drift before cloning')
    for name, digest in V48_HASHES.items():
        if old.v1.sha(V48/name) != digest:
            raise ValueError('frozen V48 comparator drift: '+name)
    denied = old.install_training_only_guard(source)
    bindings = old.verify_inputs(source)
    _, _, stimuli, _ = old.extended_observations(source)
    bank = old.molecular_bank(source, ROOT/'.benchmarks/human_mixture_profiles_v1/models.json')
    protected = {old.v1.formula_key(stimuli[row['stimulus']]) for name in ('target_ids','final_test_ids')
                 for row in old.v1.rows(source/old.INPUTS[name][0])}
    raw = old.v1.rows(source/old.INPUTS['training'][0])
    training = [r for r in raw if old.v1.formula_key(stimuli[r['stimulus']]) not in protected]
    ids = [r['stimulus'] for r in training]
    groups = [old.v1.formula_key(stimuli[i]) for i in ids]
    folds = group_folds(groups, 5)
    legacy = json.loads((V48/'predictions.json').read_text(encoding='utf-8'))
    legacy_protocol = json.loads((V48/'protocol.json').read_text(encoding='utf-8'))
    if (ids != legacy['ids'] or groups != legacy_protocol['composition_groups']
            or folds.tolist() != legacy_protocol['outer_folds'] or bindings != legacy_protocol['inputs']):
        raise ValueError('same-case/same-fold V48 comparison could not be established')
    # Freeze both the unchanged parent and the current portable loader source.
    output.mkdir(parents=True)
    parent = output/'frozen_v54'; parent.mkdir()
    for name in ('model.json','report.json','protocol.json','predictions.json'):
        shutil.copy2(V54/name, parent/name)
    code_paths = [Path(__file__), ROOT/'fragrance_ai/research/physsim_mx.py',
                  ROOT/'fragrance_ai/research/atlas_profiles.py', ROOT/'fragrance_ai/research/conditional_profiles.py',
                  ROOT/'fragrance_ai/research/perception_validation.py', ROOT/'fragrance_ai/research/fine_odor_features.py',
                  ROOT/'scripts/train_conditional_profiles_v2.py', ROOT/'scripts/benchmark_human_mixture_profiles.py']
    hashes = {p.relative_to(ROOT).as_posix(): old.v1.sha(p) for p in code_paths}
    for path in code_paths:
        target = output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = {'schema': VERSION, 'parent_checkpoint_sha256': V54_SHA, 'parent_modified': False,
        'source': bindings, 'code_sha256': hashes, 'v48_comparator_sha256': V48_HASHES,
        'ids': ids, 'composition_groups': groups, 'outer_folds': folds.tolist(),
        'inner_folds': 3, 'equal_budget_per_ablation': 9, 'alphas': ALPHAS, 'bandwidths': BANDWIDTHS,
        'modes': MODES, 'supervised_input_domain': DOMAIN, 'atlas_feature_blocks': ['applicability_high','use_high','applicability_low','use_low'],
        'ordinal_reference_to_numeric_concentration_conversion': False,
        'endpoints': old.v1.ENDPOINTS, 'evaluated_endpoints': old.v1.EVALUATED_ENDPOINTS,
        'protected_composition_rows_removed': len(raw)-len(training), 'denominator': len(ids),
        'task': 'known_molecular_components_unseen_nominal_stock_compositions',
        'model_selection_criterion': 'training_only_inner_group_OOF_49_axis_cosine_distance',
        'external_target_or_final_outcomes_read': False, 'recipe400_read_for_training': False,
        'new_blind_evaluation': False, 'unknown_molecule_generalization_measured': False,
        'historical_v48_has_18_hyperparameter_settings_not_retuned': True,
        'release_fit_enabled': False, 'synthetic_human_labels': 0,
        'default_runtime_changed_by_experiment': False, 'runtime_promoted': False}
    old.v1.write_new(output/'protocol.json', protocol)
    start = time.perf_counter()
    atlas = AtlasProfilePredictor(parent/'model.json', sha256=V54_SHA, experimental=True)
    stock_keys = sorted({key for i in ids for key in stimuli[i]})
    # The frozen Atlas encoder supports one molecular graph, not a virtual
    # molecule made from an ionic/fragmented material. Preserve affected rows
    # as unsupported in the denominator instead of failing the whole batch.
    graphs = sorted({bank[key[0]]['canonical_smiles'] for key in stock_keys if key[0] in bank
                     and '.' not in bank[key[0]]['canonical_smiles']})
    high, low = atlas.predict(graphs, reference_level='high'), atlas.predict(graphs, reference_level='low')
    embeddings = np.concatenate([high['applicability'], high['use'], low['applicability'], low['use']], axis=1)
    lookup = dict(zip(graphs, embeddings))
    widths = {mode: len(mx_features(embeddings[:1], ['graph'], [.01], ['pg'], mode=mode)) for mode in MODES}
    features = {mode: [] for mode in MODES}
    missing_ids = []
    for i in ids:
        keys = stimuli[i]
        molecular_ids = [bank.get(k[0], {}).get('canonical_smiles') for k in keys]
        if any(k not in lookup for k in molecular_ids):
            missing_ids.append(i)
            for mode in MODES:
                features[mode].append(np.full(widths[mode], np.nan))
            continue
        for mode in MODES:
            features[mode].append(mx_features([lookup[k] for k in molecular_ids], molecular_ids,
                [float(k[1]) for k in keys], [k[2] for k in keys], mode=mode))
    features = {k: np.asarray(v) for k,v in features.items()}
    present = np.isfinite(features['semantic']).all(1)
    y = np.asarray([old.v1.vector(r) for r in training])
    predictions = {mode: np.full(y.shape, np.nan) for mode in MODES}
    predictions['nested_selected_mx'] = np.full(y.shape, np.nan)
    selections = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[test])
        choice = {}
        for mode in MODES:
            model, audit = choose(features[mode][train], y[train], np.asarray(groups)[train].tolist(), mode)
            predictions[mode][test & present] = predict_mx(model, features[mode][test & present])
            choice[mode] = audit
        winner = min(MODES, key=lambda mode: choice[mode]['selected']['loss'])
        predictions['nested_selected_mx'][test] = predictions[winner][test]
        selections.append({'fold': fold, 'selected_mode': winner, 'choices': choice})
        print(json.dumps({'completed_outer_folds': fold+1, 'total_folds': 5,
                          'training_selected_mode': winner, 'seconds': round(time.perf_counter()-start, 2)}), flush=True)
    models, final_choices = {}, {}
    for mode in MODES:
        models[mode], final_choices[mode] = choose(features[mode], y, groups, mode)
    selected_mode = min(MODES, key=lambda mode: final_choices[mode]['selected']['loss'])
    artifact = {'schema': VERSION, 'models': models, 'selected_mode': selected_mode,
        'parent': {'path': 'frozen_v54/model.json', 'sha256': V54_SHA}, 'feature_reference': .01,
        'endpoints': list(old.v1.ENDPOINTS), 'input_domain': DOMAIN, 'source': bindings,
        'reference_blocks': protocol['atlas_feature_blocks'], 'runtime_promotion_allowed': False,
        'data_redistribution_authorized': False, 'release_calibrated': False,
        'gas_to_human_perception_calibrated': False}
    old.v1.write_new(output/'model.json', artifact)
    restored = json.loads((output/'model.json').read_text(encoding='utf-8'))
    for mode in MODES:
        np.testing.assert_allclose(predict_mx(models[mode], features[mode][present]),
            predict_mx(restored['models'][mode], features[mode][present]), rtol=0, atol=1e-12)
    old_prediction = np.asarray(legacy['nonlinear'], float)
    old_prediction[old_prediction < 0] = np.nan
    comparisons = {mode: paired_profile_comparison(old_prediction[:,AXES], pred[:,AXES], y[:,AXES], groups)
                   for mode, pred in predictions.items()}
    ablations = {f'{a}_to_{b}': paired_profile_comparison(predictions[a][:,AXES], predictions[b][:,AXES], y[:,AXES], groups)
                 for a,b in (('semantic','concentration'), ('concentration','interaction'), ('semantic','nested_selected_mx'))}
    report = {'scope': 'repeated_composition_disjoint_development_CV_not_product_or_recipe_accuracy',
        'observations': len(ids), 'composition_groups': len(set(groups)), 'predictable_observations': int(present.sum()),
        'unresolved_ids_retained': missing_ids, 'feature_widths': widths,
        'results': {mode: summarize_profiles(pred[:,AXES], y[:,AXES], groups) for mode,pred in predictions.items()},
        'historical_v48': summarize_profiles(old_prediction[:,AXES], y[:,AXES], groups),
        'comparisons_vs_v48': comparisons, 'ablations': ablations,
        'selections': selections, 'final_choices': final_choices, 'selected_mode': selected_mode,
        'model_sha256': old.v1.sha(output/'model.json'), 'clone_sha256': old.v1.sha(parent/'model.json'),
        'parent_unchanged': old.v1.sha(V54/'model.json') == V54_SHA,
        'sources_unchanged': all(old.v1.sha(ROOT/name) == digest for name,digest in hashes.items())
            and bindings == old.verify_inputs(source),
        'forbidden_outcome_read_attempts': denied, 'release_diagnostic': release_diagnostic(),
        'recipe400_evaluated': False, 'human_recipe_similarity_measured': False,
        'default_runtime_promoted': False, 'seconds': time.perf_counter()-start}
    if not report['parent_unchanged'] or not report['sources_unchanged']:
        raise ValueError('experiment input changed during evaluation')
    old.v1.write_new(output/'predictions.json', {'ids': ids, **{mode: np.where(np.isfinite(pred), pred, -1.).tolist()
        for mode,pred in predictions.items()}})
    old.v1.write_new(output/'report.json', report)
    print(json.dumps({'observations': len(ids), 'predictable': int(present.sum()), 'selected_mode': selected_mode,
        'comparisons_vs_v48': comparisons, 'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
