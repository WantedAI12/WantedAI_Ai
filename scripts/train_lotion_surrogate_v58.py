"""Actually train a lotion-only neural surrogate on randomized synthetic O/W data.

No human labels or recipe benchmark targets are used. Molecular identity and
oil/water formulation families are split BEFORE label generation. Validation
selects epochs; the sealed test split is evaluated exactly once at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from fragrance_ai.recommender.lotion_surrogate import (
    VERSION, FEATURE_NAMES, STATE_NAMES, LOWER, UPPER, EPS,
    features, state_mask, physical_baseline)
from fragrance_ai.recommender.lotion_transport import bidirectional_step


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def teacher(raw, steps=256):
    """Fine-grid independent normalized implementation of time-varying transport.

    Exact positive constant-rate transitions on a quadratic time mesh. Refinement
    and a separate SciPy Radau integration check guard the generated labels.
    """
    e, u, h, r, v, k, w, lipid = np.asarray(raw, dtype=float).T
    f, a = np.ones(len(raw)), np.zeros(len(raw))
    skin, vent, degraded = np.zeros((3, len(raw)))
    edges = np.linspace(0., 1., steps+1)**2
    for left, right in zip(edges, edges[1:]):
        midpoint = (left+right)/2
        capacity = 1-w+w*np.exp(-k*midpoint)
        uptake = u/capacity
        reaction = h*np.maximum(0., capacity-lipid)/capacity
        sink = uptake+reaction
        f, a, loss, exhaust = bidirectional_step(f, a, e/capacity, sink, r, v, right-left)
        share = np.divide(uptake, sink, out=np.zeros_like(sink), where=sink > 0)
        skin += loss*share
        degraded += loss*(1-share)
        vent += exhaust
    result = np.column_stack((f, a, skin, vent, degraded))
    if not np.isfinite(result).all() or np.min(result) < -1e-9:
        raise ValueError('invalid synthetic transport labels')
    error = float(np.max(np.abs(result.sum(axis=1)-1)))
    if error > 1e-7:
        raise ValueError('teacher violates mass conservation')
    return np.maximum(result, 0)/result.sum(axis=1, keepdims=True)


def generate(source, contexts, seed):
    rng = np.random.default_rng(seed)
    rows, splits, ids, groups, formulation_ids, times = [], [], [], [], [], []
    oil_choices = ([5., 8., 10., 12., 15., 20., 25., 30.], [7., 18., 28.], [6., 11., 23., 27.])
    for index, material in enumerate(source['materials']):
        identity = material['identity_group']
        n = int(hashlib.sha256((str(seed)+identity).encode()).hexdigest()[:8], 16) % 100
        split = 0 if n < 70 else 1 if n < 85 else 2
        p = material['parameters']
        for c in range(contexts):
            oil = float(rng.choice(oil_choices[split]))
            dose = 10**rng.uniform(-1., 1.)
            fragrance = 10**rng.uniform(-2., np.log10(3.))
            base_dose = dose*(1-fragrance/100.)
            water = base_dose*(94-oil)/100000.
            fixed = base_dose*(3/1.26+3)/100000.
            lipid_volume = base_dose*oil/95000.
            # Coefficient uncertainty is varied explicitly, never a fabricated
            # temperature/experimental annotation. These remain 25C assumptions.
            partition = p['lipid_water_partition']*10**rng.uniform(-.5, .5)
            gas = .1*2**rng.uniform(-1., 1.)
            air = p['air_water_partition']*10**rng.uniform(-.5, .5)
            permeability = p['skin_permeability_cm_min']*10**rng.uniform(-.3, .3)
            if rng.random() < .15:
                permeability = 0.
            hydrolysis = 10**rng.uniform(-5, -1.5) if rng.random() < .25 else 0.
            drying = .015*2**rng.uniform(-1., 1.)
            if rng.random() < .05:
                drying = 0.
            retained = rng.uniform(.05, .3)
            headspace = 10**rng.uniform(-.3, .3)
            ventilation = 10**rng.uniform(-.5, .5)
            capacity = fixed+water+partition*lipid_volume
            rates = np.array([gas*air/capacity, permeability/capacity, hydrolysis,
                              gas/headspace, ventilation, drying])
            fractions = [water*(1-retained)/capacity, partition*lipid_volume/capacity]
            # Include API times and additional intermediate times. A complete
            # trajectory belongs to one split; no random row split is allowed.
            ts = sorted([15., 60., 240., 480., *10**rng.uniform(0., np.log10(480.), 2)])
            for t in ts:
                rows.append(np.r_[rates*t, fractions])
                splits.append(split)
                ids.append(index)
                groups.append(identity)
                formulation_ids.append(index*contexts+c)
                times.append(t)
    x = np.asarray(rows)
    if not np.isfinite(x).all() or np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError('generated data exceed prespecified numerical domain')
    return {'raw': x, 'split': np.array(splits, np.int8), 'material_index': np.array(ids, np.int32),
        'identity_group': np.array(groups), 'formulation_id': np.array(formulation_ids, np.int32),
        'minutes': np.array(times)}, oil_choices


def validate_teacher(raw):
    from scipy.integrate import solve_ivp
    # Label QA uses training data only, independent from model holdouts.
    base = physical_baseline(raw)
    ref = teacher(raw, steps=512)
    differences = np.max(np.abs(ref-base), axis=1)
    chosen = np.unique(np.r_[np.argsort(differences)[-12:], np.linspace(0, len(raw)-1, 12, dtype=int)])
    errors, refinement = [], []
    for x in raw[chosen]:
        e, u, h, r, v, k, w, lipid = x
        def rhs(t, y):
            capacity = 1-w+w*np.exp(-k*t)
            evaporation, uptake = e/capacity, u/capacity
            reaction = h*(capacity-lipid)/capacity
            f, a = y[:2]
            return [-(evaporation+uptake+reaction)*f+r*a,
                evaporation*f-(r+v)*a, uptake*f, v*a, reaction*f]
        ode = solve_ivp(rhs, (0., 1.), [1., 0., 0., 0., 0.], method='Radau', rtol=1e-9, atol=1e-12)
        if not ode.success:
            raise ValueError('independent ODE label check failed')
        coarse, fine = teacher(x[None, :], 256)[0], teacher(x[None, :], 512)[0]
        errors.append(float(np.max(np.abs(fine-ode.y[:, -1]))))
        refinement.append(float(np.max(np.abs(fine-coarse))))
    report = {'independent_integrator': 'scipy.solve_ivp.Radau', 'cases': len(chosen),
        'maximum_state_fraction_error': max(errors), 'maximum_256_to_512_change': max(refinement)}
    if max(errors) > 1e-4 or max(refinement) > 1e-4:
        raise ValueError('synthetic teacher precision gate failed: '+str(report))
    return report


def metrics(prediction, target):
    diff = np.abs(prediction-target)
    live = target[:, 1] > 1e-10
    log = np.log10(np.maximum(prediction[live, 1], 1e-14)/target[live, 1])
    return {'state_fraction_mae': float(diff.mean()), 'state_fraction_rmse': float(np.sqrt(np.mean(diff**2))),
        'p95_max_state_fraction_error': float(np.quantile(diff.max(axis=1), .95)),
        'max_state_fraction_error': float(diff.max()),
        'headspace_log10_rmse_above_1e_10_fraction': float(np.sqrt(np.mean(log**2))),
        'headspace_evaluated_rows': int(live.sum()),
        'headspace_fraction_mae': float(diff[:, 1].mean()),
        'mass_balance_max_error': float(np.max(np.abs(prediction.sum(axis=1)-1))),
        'nonnegative': bool(np.all(prediction >= 0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts-per-material', type=int, default=12)
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--seed', type=int, default=580901)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if args.contexts_per_material < 12 or args.epochs < 1:
        raise ValueError('at least twelve whole formulation contexts per material required')
    if args.output.exists():
        raise FileExistsError('preserve previous runs; choose a new output directory')
    args.output.mkdir(parents=True)
    started = time.time()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA training is unavailable; no silent CPU substitution')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    data, oils = generate(source, args.contexts_per_material, args.seed)
    x, split = data['raw'], data['split']
    identity_sets = [set(data['identity_group'][split == k]) for k in range(3)]
    formula_sets = [set(data['formulation_id'][split == k]) for k in range(3)]
    assert all(not identity_sets[a]&identity_sets[b] and not formula_sets[a]&formula_sets[b]
               for a, b in ((0, 1), (0, 2), (1, 2)))
    split_report = {name: {'rows': int((split == k).sum()), 'molecular_groups': len(identity_sets[k]),
        'whole_formulation_contexts': len(formula_sets[k]), 'oil_base_percent_families': oils[k]}
        for k, name in enumerate(('train', 'validation', 'test'))}
    write_json(args.output/'run_started.json', {'source_sha256': sha(args.source), 'seed': args.seed,
        'splits': split_report, 'training_executed': False, 'requested_epochs': args.epochs,
        'parents': source['parent_models'], 'label_kind': 'synthetic_transport'})
    print(json.dumps({'event': 'data_design', 'splits': split_report, 'device': str(device)}), flush=True)
    audit = validate_teacher(x[split == 0][::31])
    write_json(args.output/'teacher_audit.json', audit)
    print(json.dumps({'event': 'independent_teacher_check', **audit}), flush=True)
    targets = np.empty((len(x), 5))
    for i in range(0, len(x), 4096):
        targets[i:i+4096] = teacher(x[i:i+4096], steps=512)
        if i % (4096*10) == 0:
            print(json.dumps({'event': 'synthetic_labels', 'completed': min(i+4096, len(x)), 'total': len(x)}), flush=True)
    np.savez_compressed(args.output/'dataset.npz', **data, targets=targets)
    train, val, test = (np.flatnonzero(split == i) for i in range(3))
    f = features(x)
    mean, scale = f[train].mean(axis=0), np.maximum(f[train].std(axis=0), 1e-6)
    baseline = physical_baseline(x)
    masks = state_mask(x)
    tensors = [torch.tensor(a, dtype=torch.float32, device=device) for a in
               ((f-mean)/scale, np.log(np.maximum(baseline, EPS)), targets, masks)]
    tx, tb, ty, tm = tensors
    dimensions = [8, 128, 128, 64, 5]
    net = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5)).to(device)
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    optimizer = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=3e-5)
    def predict_tensor(indices):
        return torch.softmax((tb[indices]+net(tx[indices])).masked_fill(tm[indices] == 0, -1e30), dim=1)
    def loss(indices):
        predicted = predict_tensor(indices)
        y = ty[indices]
        mass = ((torch.sqrt(predicted+1e-10)-torch.sqrt(y+1e-10))**2).sum(dim=1)
        log = ((torch.log10(predicted+1e-10)-torch.log10(y+1e-10))**2).mean(dim=1)
        return (mass+.02*log).mean()
    def output(indices):
        with torch.no_grad():
            return np.concatenate([predict_tensor(indices[i:i+8192]).cpu().numpy()
                                   for i in range(0, len(indices), 8192)]).astype(float)
    rng = np.random.default_rng(args.seed+1)
    best, best_epoch, history = float('inf'), 0, []
    checkpoint = args.output/'training_checkpoint.pt'
    initial_val = metrics(baseline[val], targets[val])
    train_started = time.time()
    for epoch in range(1, args.epochs+1):
        order = rng.permutation(train)
        net.train()
        epoch_loss = 0.
        for i in range(0, len(order), 4096):
            idx = order[i:i+4096]
            optimizer.zero_grad(set_to_none=True)
            value = loss(idx)
            if not torch.isfinite(value):
                raise ValueError('nonfinite training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            epoch_loss += float(value.detach())*len(idx)
        scheduler.step()
        net.eval()
        validation = output(val)
        m = metrics(validation, targets[val])
        criterion = m['state_fraction_rmse'] + .01*m['headspace_log10_rmse_above_1e_10_fraction']
        record = {'epoch': epoch, 'training_loss': epoch_loss/len(train), 'selection_loss': criterion, 'validation': m}
        history.append(record)
        if criterion < best:
            best, best_epoch = criterion, epoch
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'seed': args.seed, 'mean': mean.tolist(), 'scale': scale.tolist(),
                'source_sha256': sha(args.source), 'dataset_sha256': sha(args.output/'dataset.npz')}, checkpoint)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({'event': 'training', **record, 'best_epoch': best_epoch,
                'elapsed_seconds': round(time.time()-train_started, 1)}), flush=True)
            write_json(args.output/'learning_curve.json', history)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model'])
    net.eval()
    arrays = {'mean': mean, 'scale': scale}
    for i, layer in enumerate(m for m in net if isinstance(m, nn.Linear)):
        arrays[f'weight_{i}'] = layer.weight.detach().cpu().numpy().T.copy()
        arrays[f'bias_{i}'] = layer.bias.detach().cpu().numpy().copy()
    np.savez_compressed(args.output/'weights.npz', **arrays)
    # No tuning/retraining follows this first and only sealed-test evaluation.
    final_val = metrics(output(val), targets[val])
    predicted_test = output(test)
    test_metrics = metrics(predicted_test, targets[test])
    baseline_test = metrics(baseline[test], targets[test])
    gates = {
        'state_rmse_improves_at_least_20_percent': test_metrics['state_fraction_rmse'] <= .8*baseline_test['state_fraction_rmse'],
        'state_fraction_mae_under_0_005': test_metrics['state_fraction_mae'] <= .005,
        'p95_max_state_fraction_error_under_0_03': test_metrics['p95_max_state_fraction_error'] <= .03,
        'headspace_log_rmse_under_0_15_dex': test_metrics['headspace_log10_rmse_above_1e_10_fraction'] <= .15,
        'headspace_log_error_not_regressed': test_metrics['headspace_log10_rmse_above_1e_10_fraction'] <= baseline_test['headspace_log10_rmse_above_1e_10_fraction'],
        'nonnegative': test_metrics['nonnegative'], 'mass_conservation': test_metrics['mass_balance_max_error'] < 1e-6}
    report = {'schema': 'lotion-synthetic-training-report/v1', 'training_executed': True,
        'label_kind': 'synthetic_transport', 'measured_lotion_observations': 0,
        'source_sha256': sha(args.source), 'dataset_sha256': sha(args.output/'dataset.npz'),
        'splits': split_report, 'split_disjoint_molecular_identities': True,
        'split_disjoint_whole_formulations': True, 'split_disjoint_oil_families': True,
        'teacher_audit': audit, 'training_epochs_completed': args.epochs, 'selected_epoch': best_epoch,
        'baseline_validation': initial_val, 'trained_validation': final_val,
        'baseline_sealed_test': baseline_test, 'trained_sealed_test': test_metrics,
        'promotion_gates': gates, 'accepted_for_local_surrogate_inference': all(gates.values()),
        'test_set_used_for_epoch_selection': False, 'recipe400_used_for_training': False,
        'training_seconds': time.time()-train_started, 'total_seconds': time.time()-started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'device': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'},
        'accuracy_claim': 'synthetic_transport_emulation_not_actual_odor_similarity'}
    write_json(args.output/'evaluation.json', report)
    write_json(args.output/'learning_curve.json', history)
    manifest = {'schema': VERSION, 'training_executed': True, 'label_kind': 'synthetic_transport',
        'measured_lotion_observations': 0, 'accepted_for_local_surrogate_inference': all(gates.values()),
        'feature_names': list(FEATURE_NAMES), 'state_names': list(STATE_NAMES),
        'architecture': {'dimensions': dimensions, 'activation': 'relu', 'output': 'physical_baseline_residual_masked_softmax'},
        'weights': {'path': 'weights.npz', 'sha256': sha(args.output/'weights.npz')},
        'training_checkpoint': {'path': checkpoint.name, 'sha256': sha(checkpoint)},
        'evaluation': {'path': 'evaluation.json', 'sha256': sha(args.output/'evaluation.json')},
        'parent_models': source['parent_models'], 'source_sha256': sha(args.source),
        'training_dataset_sha256': sha(args.output/'dataset.npz'), 'selected_epoch': best_epoch,
        'domain': {'lower': LOWER.tolist(), 'upper': UPPER.tolist(), 'transport_mode': 'bidirectional_air',
            'temperature_c': 25., 'phase_type': 'oil_in_water', 'condition_ranges_are_engineering_assumptions': True},
        'material_input_count': len(source['materials']), 'synthetic_rows': len(x),
        'human_similarity_percent': None, 'recipe_acceptance_score_modified': False}
    write_json(args.output/'model.json', manifest)
    if all(gates.values()):
        from fragrance_ai.recommender.lotion_surrogate import LotionReleaseSurrogate
        loaded = LotionReleaseSurrogate(args.output/'model.json', sha256=sha(args.output/'model.json'))
        parity = float(np.max(np.abs(loaded.predict(x[test[:1024]])-predicted_test[:1024])))
        if parity > 2e-6:
            raise ValueError('CPU export differs from trained model: '+str(parity))
        write_json(args.output/'cpu_export_check.json', {'rows': min(1024, len(test)), 'max_abs_difference': parity,
            'torch_required_at_runtime': False, 'passed': True})
    print(json.dumps({'event': 'completed', 'accepted': all(gates.values()), 'selected_epoch': best_epoch,
        'baseline_test': baseline_test, 'trained_test': test_metrics, 'gates': gates,
        'manifest_sha256': sha(args.output/'model.json')}), flush=True)


if __name__ == '__main__':
    main()
