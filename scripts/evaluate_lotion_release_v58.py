"""Post-training evaluation ONLY: fresh held-out-molecule lotion mixtures.

No fitting, promotion, deployment or changes to runtime policy. Reference
labels are synthetic transport, not new human or physical observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def generate(source, *, seed, count):
    # Retain EXACTLY the original held-out molecular identities. Changing the
    # split seed would incorrectly allow originally trained molecules here.
    heldout = [i for i,m in enumerate(source['materials']) if
        int(hashlib.sha256(('580901'+m['identity_group']).encode()).hexdigest()[:8], 16)%100 >= 85]
    if count < len(heldout):
        raise ValueError('every original held-out molecule must enter this evaluation')
    rng = np.random.default_rng(seed)
    times = np.array([1., 2., 5., 10., 20., 45., 90., 150., 210., 300., 390., 480.])
    groups, raw, parameters = [], [], []
    offset = 0
    for case in range(count):
        n = int(rng.integers(2, 21))
        first = heldout[case % len(heldout)]
        indices = np.r_[first, rng.choice([i for i in heldout if i != first], n-1, replace=False)]
        oil = rng.uniform(5., 30.)
        dose = 10**rng.uniform(-1., 1.)
        fragrance = 10**rng.uniform(-2., np.log10(3.))
        base_dose = dose*(1-fragrance/100.)
        water, fixed, lipid = base_dose*(94-oil)/1e5, base_dose*(3/1.26+3)/1e5, base_dose*oil/95000.
        drying, retained = .015*2**rng.uniform(-1., 1.), rng.uniform(.05, .3)
        headspace, ventilation = 10**rng.uniform(-.3, .3), 10**rng.uniform(-.5, .5)
        weights = rng.dirichlet(np.full(n, .7))
        mass = dose*fragrance/100.*weights
        parent = rng.uniform(.6, 1., n)
        p = [source['materials'][int(i)]['parameters'] for i in indices]
        partition = np.array([x['lipid_water_partition'] for x in p])*10**rng.uniform(-.5, .5, n)
        gas = .1*2**rng.uniform(-1., 1., n)
        air = np.array([x['air_water_partition'] for x in p])*10**rng.uniform(-.5, .5, n)
        permeability = np.array([x['skin_permeability_cm_min'] for x in p])*10**rng.uniform(-.3, .3, n)
        permeability[rng.random(n) < .15] = 0.
        hydrolysis = np.where(rng.random(n) < .25, 10**rng.uniform(-5., -1.5, n), 0.)
        capacity = fixed+water+partition*lipid
        rates = np.column_stack((gas*air/capacity, permeability/capacity, hydrolysis,
            gas/headspace, np.full(n, ventilation), np.full(n, drying)))
        fractions = np.column_stack((water*(1-retained)/capacity, partition*lipid/capacity))
        x = np.concatenate([np.column_stack((rates*t, fractions)) for t in times])
        raw.append(x)
        groups.extend([case]*len(x))
        parameters.append({'id': f'fresh-{case:04d}', 'offset': offset, 'rows': len(x),
            'material_indices': indices.tolist(), 'initial_mg_cm2': mass.tolist(),
            'initial_parent_fraction': parent.tolist(), 'minutes': times.tolist(),
            'oil_base_percent': oil, 'fragrance_concentration_percent': fragrance,
            'application_mass_mg_cm2': dose, 'headspace_height_cm': headspace,
            'water_loss_per_min': drying, 'retained_water_fraction': retained,
            'air_exchange_per_min': ventilation, 'temperature_c': 25.,
            'material_count': n, 'label_kind': 'synthetic_transport'})
        offset += len(x)
    return np.concatenate(raw), np.asarray(groups, dtype=np.int32), parameters, heldout


def independent_check(raw, targets, prediction):
    from scipy.integrate import solve_ivp
    # Include worst neural errors as well as a pre-spaced sample. This checks
    # that hard-error labels are real ODE outcomes, not self-confirming artifacts.
    errors = np.max(np.abs(prediction-targets), axis=1)
    chosen = np.unique(np.r_[np.argsort(errors)[-16:], np.linspace(0, len(raw)-1, 16, dtype=int)])
    rows = []
    for index in chosen:
        e, u, h, r, v, k, w, lipid = raw[index]
        def rhs(t, y):
            c = 1-w+w*np.exp(-k*t)
            evaporation, uptake, reaction = e/c, u/c, h*(c-lipid)/c
            f, a = y[:2]
            return [-(evaporation+uptake+reaction)*f+r*a, evaporation*f-(r+v)*a,
                    uptake*f, v*a, reaction*f]
        solved = solve_ivp(rhs, (0., 1.), [1., 0., 0., 0., 0.], method='Radau', rtol=1e-10, atol=1e-13)
        if not solved.success:
            raise ValueError('independent reference integration failed')
        y = solved.y[:, -1]
        rows.append({'index': int(index), 'teacher_error': float(np.max(np.abs(y-targets[index]))),
            'neural_error': float(np.max(np.abs(y-prediction[index]))),
            'reference': y.tolist(), 'prediction': prediction[index].tolist()})
    return {'integrator': 'scipy.Radau', 'count': len(rows),
        'maximum_teacher_error': max(r['teacher_error'] for r in rows), 'cases': rows}


def runtime_boundaries(source):
    from fragrance_ai.platform.application_context import assess_application_context, ApplicationContext
    from fragrance_ai.platform.lotion_inputs import LotionSimulationRequest
    from fragrance_ai.recommender.runtime import RuntimeAIFactory
    from fragrance_ai.recommender.lotion_surrogate import predict_release
    from copy import deepcopy
    factory = RuntimeAIFactory.from_environment()
    value = deepcopy(source['reference_request'])
    value['materials'] = [dict(value['materials'][0], concentrate_percent=100.)]
    changes = [('temperature_20', 'temperature_c', 20.), ('temperature_30', 'temperature_c', 30.),
        ('early_time', 'times_minutes', [0., .1, 15.]), ('late_time', 'times_minutes', [0., 15., 600.]),
        ('low_retention', 'retained_water_fraction', .02), ('high_retention', 'retained_water_fraction', .5),
        ('low_headspace', 'headspace_height_cm', .1), ('high_headspace', 'headspace_height_cm', 3.),
        ('low_ventilation', 'air_exchange_per_min', .1), ('high_ventilation', 'air_exchange_per_min', 10.),
        ('low_drying', 'water_loss_per_min', .003), ('high_drying', 'water_loss_per_min', .1),
        ('open_sink', 'transport_mode', 'open_sink')]
    results = []
    valid = predict_release(LotionSimulationRequest.model_validate(value), factory.catalog)
    results.append({'case': 'in_domain', 'expected': 'prediction', 'status': valid['status'],
                    'passed': valid['status'] in ('research_prediction', 'research_prediction_flagged')})
    for name, key, new in changes:
        body = deepcopy(value)
        if key == 'temperature_c':
            body['application_context'][key] = new
            context = ApplicationContext.model_validate(body['application_context'])
            body['parameter_context_id'] = assess_application_context(context)['context_id']
        else:
            body[key] = new
        predicted = predict_release(LotionSimulationRequest.model_validate(body), factory.catalog)
        results.append({'case': name, 'expected': 'abstained', 'status': predicted['status'],
                        'reason': predicted.get('reason'), 'passed': predicted['status'] == 'abstained'})
    factory.assert_current_snapshot()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mixtures', type=int, default=600)
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--resume-summary', action='store_true', help='reuse completed raw predictions after a reporting interruption')
    args = p.parse_args()
    if args.output.exists() and not args.resume_summary:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume_summary)
    from fragrance_ai.recommender.lotion_surrogate import configured_lotion_surrogate, physical_baseline, STATE_NAMES
    from fragrance_ai.recommender.local_runtime import local_profile
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    from scripts.train_lotion_surrogate_v58 import teacher, metrics
    model = configured_lotion_surrogate()
    profile = local_profile()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    assert digest(args.source) == model.manifest['source_sha256']
    initial_bindings = {role: digest(profile[role][0])
                       for role in ('perfume', 'atlas', 'body_lotion', 'stock_mixture', 'lotion_release')}
    manifest = {'schema': 'lotion-post-training-fresh-evaluation/v1', 'training_executed': False,
        'label_kind': 'synthetic_transport', 'no_new_human_observations': True,
        'checkpoint_sha256': model.sha256, 'source_sha256': digest(args.source),
        'script_sha256': digest(__file__), 'seed': args.seed, 'new_mixtures': args.mixtures,
        'selected_molecules': 'only_original_553_held_out_molecular_identities',
        'quality_gates_predeclared': {'state_fraction_mae_max': .005,
            'p95_max_state_fraction_error_max': .03, 'headspace_log10_rmse_max': .15,
            'mass_balance_error_max': 1e-10, 'negative_mass_count_max': 0,
            'nonmonotone_trajectory_count_max': 0}}
    if args.resume_summary:
        saved = json.loads((args.output/'manifest.json').read_text(encoding='utf-8'))
        for key in ('checkpoint_sha256','source_sha256','seed','new_mixtures','quality_gates_predeclared'):
            if saved[key] != manifest[key]:
                raise ValueError('cannot resume a different evaluation')
        if (args.output/'summary.json').exists():
            raise FileExistsError('completed evaluation must not be overwritten')
    else:
        write(args.output/'manifest.json', manifest)
    started = time.perf_counter()
    raw, groups, cases, heldout = generate(source, seed=args.seed, count=args.mixtures)
    assert len(heldout) == 553
    if args.resume_summary:
        with np.load(args.output/'predictions.npz', allow_pickle=False) as data:
            assert np.array_equal(data['raw'], raw) and np.array_equal(data['mixture_group'], groups)
            predicted, target, baseline = data['predicted'], data['reference'], data['baseline']
        assert json.loads((args.output/'cases.json').read_text(encoding='utf-8')) == cases
        independent = json.loads((args.output/'independent_ode.json').read_text(encoding='utf-8'))
        neural_seconds = teacher_seconds = None
    else:
        write(args.output/'cases.json', cases)
        start = time.perf_counter()
        predicted = model.predict(raw)
        neural_seconds = time.perf_counter()-start
        target = np.empty_like(predicted)
        start = time.perf_counter()
        for i in range(0, len(raw), 4096):
            target[i:i+4096] = teacher(raw[i:i+4096], steps=1024)
        teacher_seconds = time.perf_counter()-start
        baseline = physical_baseline(raw)
        independent = independent_check(raw, target, predicted)
        write(args.output/'independent_ode.json', independent)
        np.savez_compressed(args.output/'predictions.npz', raw=raw, mixture_group=groups,
                            predicted=predicted, reference=target, baseline=baseline)
    detail, score_deltas, monotonic_failures = [], [], []
    profiles = np.array([[m['profile'].get(axis, 0.) for axis in SCENT_DIMENSIONS] for m in source['materials']])
    for case in cases:
        i, length, n = case['offset'], case['rows'], case['material_count']
        y = predicted[i:i+length].reshape(-1, n, 5)
        reference = target[i:i+length].reshape(-1, n, 5)
        change = np.diff(y[:, :, 2:], axis=0)
        bad = bool(np.any(change < -1e-5))
        if bad:
            monotonic_failures.append({'case': case['id'], 'largest_cumulative_sink_decrease_fraction': float(-change.min())})
        mass = np.array(case['initial_mg_cm2'])*np.array(case['initial_parent_fraction'])
        ids = case['material_indices']
        thresholds = np.array([source['materials'][j]['parameters']['odor_threshold_mg_m3'] for j in ids])
        weighted = y[:, :, 1]*mass/thresholds
        true_weighted = reference[:, :, 1]*mass/thresholds
        scent_profile = weighted@profiles[ids]
        true_profile = true_weighted@profiles[ids]
        scent_profile /= np.maximum(scent_profile.sum(axis=1, keepdims=True), 1e-300)
        true_profile /= np.maximum(true_profile.sum(axis=1, keepdims=True), 1e-300)
        tv = .5*np.abs(scent_profile-true_profile).sum(axis=1)
        cosine = (scent_profile*true_profile).sum(axis=1)/np.maximum(np.linalg.norm(scent_profile,axis=1)*np.linalg.norm(true_profile,axis=1), 1e-300)
        score_deltas.extend(tv.tolist())
        error = np.abs(y-reference)
        detail.append({'case': case['id'], 'material_count': n, 'state_fraction_mae': float(error.mean()),
            'max_state_fraction_error': float(error.max()), 'cumulative_sinks_monotone': not bad,
            'max_profile_total_variation': float(tv.max()), 'minimum_profile_cosine': float(cosine.min())})
    boundary = runtime_boundaries(source)
    m = metrics(predicted, target)
    gates = {'state_fraction_mae': m['state_fraction_mae'] <= .005,
        'p95_state_error': m['p95_max_state_fraction_error'] <= .03,
        'headspace_log_error': m['headspace_log10_rmse_above_1e_10_fraction'] <= .15,
        'mass_conservation': m['mass_balance_max_error'] <= 1e-10,
        'nonnegative': m['nonnegative'], 'all_cumulative_sink_trajectories_monotone': not monotonic_failures,
        'independent_teacher_accuracy': independent['maximum_teacher_error'] < 1e-4,
        'runtime_abstention_boundaries': all(r['passed'] for r in boundary)}
    current = local_profile()
    unchanged = all(digest(current[role][0]) == sha for role, sha in initial_bindings.items())
    model.assert_current()
    assert unchanged and current['profile_sha256'] == profile['profile_sha256']
    bootstrap_rng = np.random.default_rng(args.seed+1)
    per_case = np.array([d['state_fraction_mae'] for d in detail])
    bootstrap = per_case[bootstrap_rng.integers(0, len(per_case), (2000, len(per_case)))].mean(axis=1)
    summary = {**manifest, 'status': 'evaluation_completed', 'all_quality_gates_passed': all(gates.values()),
        'resumed_summary_from_unchanged_raw_artifacts': args.resume_summary,
        'rows': len(raw), 'mixtures': len(cases), 'heldout_molecular_identities': len(heldout),
        'all_original_heldout_molecules_covered': set(heldout) == {i for c in cases for i in c['material_indices']},
        'baseline': metrics(baseline, target), 'trained': m,
        'unweighted_mixture_mean_state_fraction_mae_95_bootstrap_interval': np.quantile(bootstrap,[.025,.975]).tolist(),
        'interval_scope': 'synthetic_mixture_groups_not_human_uncertainty',
        'profile_propagation': {'mean_total_variation': float(np.mean(score_deltas)),
            'p95_total_variation': float(np.quantile(score_deltas,.95)), 'max_total_variation': float(np.max(score_deltas)),
            'scope': 'catalog_OAV_profile_vs_same_synthetic_solver_not_user_scent_agreement'},
        'negative_mass_entries': int((predicted < 0).sum()), 'nonmonotone_mixtures': len(monotonic_failures),
        'monotonicity_failures': monotonic_failures, 'runtime_boundaries': boundary,
        'quality_gates': gates, 'checkpoint_and_local_profile_unchanged': unchanged,
        'raw_batched_cpu_neural_seconds': neural_seconds, 'reference_1024_step_seconds': teacher_seconds,
        'timing_is_concurrent_local_evaluation_not_sla': True, 'total_seconds': time.perf_counter()-started,
        'human_similarity_percent': None, 'prediction_artifact_sha256': digest(args.output/'predictions.npz')}
    write(args.output/'case_results.json', detail)
    write(args.output/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status','mixtures','rows','heldout_molecular_identities',
        'trained','profile_propagation','nonmonotone_mixtures','all_quality_gates_passed','total_seconds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
