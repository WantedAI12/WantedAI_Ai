"""Train one absorbing transition operator across three product regimes.

All 3,795 source material identities are retained. Product regimes are explicit
engineering assumptions, not additional measured formulation data. Splits use
the V58 identity key so V58 held-out molecules remain held out here as well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fragrance_ai.recommender.unified_transport import (
    VERSION, PRODUCTS, FEATURE_NAMES, STATE_NAMES, EPS, baseline_kernel,
    constant_kernel, correction_gate, features, kernel_mask, validate_raw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def teacher(raw, steps=256):
    e, u, h, r, v, k, w, lipid = np.asarray(raw, float).T
    states = np.zeros((len(raw), 2, 5))
    states[:, 0, 0] = states[:, 1, 1] = 1.
    edges = np.linspace(0., 1., steps+1)**2
    for left, right in zip(edges[:-1], edges[1:]):
        capacity = 1.-w+w*np.exp(-k*(left+right)/2)
        transition = constant_kernel(e/capacity, u/capacity,
            h*np.maximum(0., 1.-lipid/capacity), r, v, right-left)
        flow = np.einsum('nbi,nij->nbj', states[:, :, :2], transition)
        states[:, :, :2] = flow[:, :, :2]
        states[:, :, 2:] += flow[:, :, 2:]
    if np.max(np.abs(states.sum(axis=-1)-1)) > 1e-8 or np.min(states) < 0:
        raise ValueError('nonconservative numerical training labels')
    return states


def generate(source, seed, contexts):
    rng = np.random.default_rng(seed)
    rows, identities, splits, products, groups = [], [], [], [], []
    for index, material in enumerate(source['materials']):
        identity = material['identity_group']
        fold = int(hashlib.sha256(('580901'+identity).encode()).hexdigest()[:8], 16) % 100
        split = 0 if fold < 70 else 1 if fold < 85 else 2
        p = material['parameters']
        for product_index, product in enumerate(PRODUCTS):
            for c in range(contexts):
                dose = 10**rng.uniform(-1, 1)
                water = dose*10**rng.uniform(-3.4, -3.)
                fixed = water*rng.uniform(.03, .3)
                # A phase-capacity stress envelope only. These numbers are NOT
                # fitted ethanol activity or measured surfactant partition data.
                phase = water*10**rng.uniform(-3, -.3)
                partition = p['lipid_water_partition']*10**rng.uniform(-.5, .5)
                if product == 'perfume':
                    phase *= .01
                    decay = 10**rng.uniform(-2, -.4)
                elif product == 'body_lotion':
                    decay = .015*2**rng.uniform(-1, 1)
                else:
                    phase *= rng.uniform(.25, 2.)
                    decay = 0. if c % 2 == 0 else 10**rng.uniform(-2.5, -1.)
                retained = rng.uniform(.05, .3)
                capacity = fixed+water+phase*partition
                e = .1*p['air_water_partition']/capacity*10**rng.uniform(-.5, .5)
                u = p['skin_permeability_cm_min']/capacity*10**rng.uniform(-.3, .3)
                if rng.random() < .2:
                    u = 0.
                h = 10**rng.uniform(-5, -1.5) if rng.random() < .3 else 0.
                r, v = 10**rng.uniform(-1.3, -.7), 10**rng.uniform(-.5, .5)
                duration = 10**rng.uniform(-1, np.log10(480.))
                rows.append([e*duration, u*duration, h*duration, r*duration, v*duration,
                    decay*duration, water*(1-retained)/capacity, phase*partition/capacity])
                identities.append(identity); splits.append(split); products.append(product_index)
                groups.append((index*3+product_index)*contexts+c)
    raw = validate_raw(np.asarray(rows))
    return {'raw': raw, 'identity': np.array(identities), 'split': np.array(splits, np.int8),
            'product': np.array(products, np.int8), 'context_id': np.array(groups, np.int32)}


def metrics(predicted, target):
    diff = predicted-target
    live = target > 1e-10
    logs = np.log10(np.maximum(predicted[live], EPS)/target[live])
    return {'mae': float(np.mean(np.abs(diff))), 'rmse': float(np.sqrt(np.mean(diff**2))),
        'max_abs_error': float(np.max(np.abs(diff))),
        'log10_rmse_above_1e_10': float(np.sqrt(np.mean(logs**2))),
        'mass_balance_max_error': float(np.max(np.abs(predicted.sum(axis=-1)-1))),
        'nonnegative': bool(np.all(predicted >= 0))}


def criterion(m):
    return m['rmse']+.005*m['log10_rmse_above_1e_10']


def audit_teacher(raw):
    from scipy.integrate import solve_ivp
    coarse, fine = teacher(raw, 256), teacher(raw, 512)
    # Only train identities determine label-precision QA.
    ids = np.unique(np.r_[np.argsort(np.max(np.abs(fine-baseline_kernel(raw)), axis=(1, 2)))[-8:],
                          np.linspace(0, len(raw)-1, 8, dtype=int)])
    maximum = 0.
    for index in ids:
        e, u, h, r, v, k, w, lipid = raw[index]
        for mode in (0, 1):
            def rhs(t, y):
                cap = 1-w+w*np.exp(-k*t)
                evap, uptake, reaction = e/cap, u/cap, h*(1-lipid/cap)
                f, a = y[:2]
                return [-(evap+uptake+reaction)*f+r*a, evap*f-(r+v)*a,
                        uptake*f, v*a, reaction*f]
            y0 = np.zeros(5); y0[mode] = 1.
            solution = solve_ivp(rhs, (0., 1.), y0, method='Radau', rtol=2e-10, atol=1e-12)
            if not solution.success:
                raise ValueError('independent teacher integration failed')
            maximum = max(maximum, float(np.max(np.abs(fine[index, mode]-solution.y[:, -1]))))
    drift = float(np.max(np.abs(fine-coarse)))
    if maximum > 1e-4 or drift > 1e-4:
        raise ValueError('teacher accuracy gate failed')
    return {'independent_solver': 'scipy.solve_ivp.Radau', 'cases': len(ids)*2,
            'maximum_state_error': maximum, 'maximum_256_to_512_change': drift}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--contexts-per-product', type=int, default=4)
    parser.add_argument('--seed', type=int, default=600909)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1 or args.contexts_per_product < 4:
        raise ValueError('new output directory and at least four contexts per product required')
    source = json.loads(args.source.read_text(encoding='utf-8'))
    if len(source['materials']) != source['coefficient_pool_count']:
        raise ValueError('source material pool is incomplete')
    for binding in source['parent_models'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('source parent model hash mismatch')
    args.output.mkdir(parents=True)
    start = time.time()
    import torch
    from torch import nn
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA is unavailable')
    device = torch.device(args.device)
    data = generate(source, args.seed, args.contexts_per_product)
    x, split, product = data['raw'], data['split'], data['product']
    train, val, test = [np.flatnonzero(split == i) for i in range(3)]
    identity_sets = [set(data['identity'][idx]) for idx in (train, val, test)]
    if any(identity_sets[i]&identity_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('identity split leakage')
    splits = {name: {'rows': len(idx), 'identities': len(identity_sets[i]),
              'rows_by_product': {p: int(np.sum(product[idx] == j)) for j, p in enumerate(PRODUCTS)}}
              for i, (name, idx) in enumerate(zip(('train', 'validation', 'test'), (train, val, test)))}
    write(args.output/'started.json', {'source_sha256': sha(args.source), 'splits': splits,
        'training_executed': False, 'products': list(PRODUCTS), 'seed': args.seed})
    print(json.dumps({'event': 'source_ready', 'materials': len(source['materials']), 'splits': splits}), flush=True)
    audit = audit_teacher(x[train[::41]])
    write(args.output/'teacher_audit.json', audit)
    print(json.dumps({'event': 'teacher_audit', **audit}), flush=True)
    target = np.empty((len(x), 2, 5))
    for offset in range(0, len(x), 4096):
        target[offset:offset+4096] = teacher(x[offset:offset+4096], 512)
        print(json.dumps({'event': 'labels', 'completed': min(offset+4096, len(x)), 'total': len(x)}), flush=True)
    np.savez_compressed(args.output/'dataset.npz', **data, target=target)
    dataset_sha = sha(args.output/'dataset.npz')
    base, masks, f = baseline_kernel(x), kernel_mask(x), features(x)
    mean, scale = f[train].mean(0), np.maximum(f[train].std(0), 1e-6)
    arrays = [(f-mean)/scale, np.log(np.maximum(base, EPS)), target, masks,
              correction_gate(x)[:, None, None]]
    tx, tb, ty, tm, tg = [torch.tensor(a, dtype=torch.float32, device=device) for a in arrays]
    dims = [8, 128, 128, 64, 10]
    net = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)).to(device)
    nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=4e-5)
    def predict(indices):
        logits = tb[indices]+tg[indices]*net(tx[indices]).reshape(-1, 2, 5)
        return torch.softmax(logits.masked_fill(tm[indices] == 0, -1e30), dim=-1)
    def output(indices):
        with torch.no_grad():
            return np.concatenate([predict(indices[i:i+4096]).cpu().numpy()
                                   for i in range(0, len(indices), 4096)]).astype(float)
    rng = np.random.default_rng(args.seed+1)
    history, best, selected = [], float('inf'), 0
    checkpoint = args.output/'training_checkpoint.pt'
    for epoch in range(1, args.epochs+1):
        net.train(); order = rng.permutation(train); total_loss = 0.
        for offset in range(0, len(order), 2048):
            idx = order[offset:offset+2048]
            optimizer.zero_grad(set_to_none=True)
            p, y = predict(idx), ty[idx]
            mass = ((torch.sqrt(p+EPS)-torch.sqrt(y+EPS))**2).sum(dim=-1).mean()
            log = ((torch.log10(p+1e-10)-torch.log10(y+1e-10))**2).mean()
            loss = mass+.02*log
            if not torch.isfinite(loss):
                raise ValueError('nonfinite shared training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            total_loss += float(loss.detach())*len(idx)
        schedule.step(); net.eval()
        vm = metrics(output(val), target[val])
        record = {'epoch': epoch, 'train_loss': total_loss/len(train), 'validation': vm}
        history.append(record)
        if criterion(vm) < best:
            best, selected = criterion(vm), epoch
            torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                        'source_sha256': sha(args.source), 'dataset_sha256': dataset_sha}, checkpoint)
        if epoch == 1 or epoch % 20 == 0:
            write(args.output/'learning_curve.json', history)
            print(json.dumps({'event': 'training', **record, 'best_epoch': selected,
                'elapsed_seconds': round(time.time()-start, 1)}), flush=True)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model']); net.eval()
    pv = output(val)
    blend_candidates = []
    for blend in (0., .25, .5, .75, 1.):
        m = metrics((1-blend)*base[val]+blend*pv, target[val])
        blend_candidates.append({'blend': blend, 'metrics': m, 'criterion': criterion(m)})
    blend = min(blend_candidates, key=lambda r:r['criterion'])['blend']
    # Test labels have not affected epoch or blend selection.
    pt = (1-blend)*base[test]+blend*output(test)
    baseline_test, learned_test = metrics(base[test], target[test]), metrics(pt, target[test])
    per_product = {p: {'baseline': metrics(base[test][product[test] == j], target[test][product[test] == j]),
                      'unified': metrics(pt[product[test] == j], target[test][product[test] == j])}
                   for j, p in enumerate(PRODUCTS)}
    gates = {'positive_learned_blend': blend > 0.,
        'rmse_improves_at_least_15_percent': learned_test['rmse'] <= .85*baseline_test['rmse'],
        'no_product_rmse_regression': all(r['unified']['rmse'] <= r['baseline']['rmse'] for r in per_product.values()),
        'mean_absolute_state_error_under_0_001': learned_test['mae'] < .001,
        'log_error_not_regressed': learned_test['log10_rmse_above_1e_10'] <= baseline_test['log10_rmse_above_1e_10'],
        'mass_conservation': learned_test['mass_balance_max_error'] < 1e-6,
        'nonnegative': learned_test['nonnegative']}
    exported = {'mean': mean, 'scale': scale}
    for i, layer in enumerate(m for m in net if isinstance(m, nn.Linear)):
        exported[f'weight_{i}'] = layer.weight.detach().cpu().numpy().T.copy()
        exported[f'bias_{i}'] = layer.bias.detach().cpu().numpy().copy()
    np.savez_compressed(args.output/'weights.npz', **exported)
    report = {'schema': 'unified-product-training-report/v1', 'training_executed': True,
        'label_kind': 'synthetic_transition_operators', 'source_sha256': sha(args.source),
        'dataset_sha256': dataset_sha, 'material_count': len(source['materials']), 'splits': splits,
        'identity_disjoint': True, 'context_disjoint': True, 'split_seed_matches_v58': 580901,
        'epochs_completed': args.epochs, 'selected_epoch': selected, 'validation_blends': blend_candidates,
        'validation_selected_blend': blend, 'test_used_for_selection': False,
        'baseline_test': baseline_test, 'unified_test': learned_test, 'per_product_test': per_product,
        'teacher_audit': audit, 'acceptance_gates': gates, 'accepted': all(gates.values()),
        'recipe400_used_for_training': False, 'measured_product_release_observations': 0,
        'regime_parameters': 'synthetic_engineering_stress_envelopes_not_empirical_product_coefficients',
        'time_seconds': time.time()-start, 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None}
    write(args.output/'evaluation.json', report)
    write(args.output/'learning_curve.json', history)
    manifest = {'schema': VERSION, 'products': list(PRODUCTS), 'feature_names': list(FEATURE_NAMES),
        'state_names': list(STATE_NAMES), 'architecture': {'dimensions': dims, 'activation': 'relu',
            'output': 'capacity_gated_stochastic_transition_residual'},
        'training_executed': True, 'accepted_for_local_inference': all(gates.values()),
        'label_kind': 'synthetic_transition_operators', 'validation_selected_blend': blend,
        'weights': {'path': 'weights.npz', 'sha256': sha(args.output/'weights.npz')},
        'evaluation': {'path': 'evaluation.json', 'sha256': sha(args.output/'evaluation.json')},
        'training_checkpoint': {'path': checkpoint.name, 'sha256': sha(checkpoint)},
        'parent_models': source['parent_models'], 'source_sha256': sha(args.source),
        'dataset_sha256': dataset_sha, 'material_count': len(source['materials']),
        'synthetic_transition_contexts': len(x), 'selected_epoch': selected,
        'measured_product_release_observations': 0, 'human_similarity_percent': None}
    write(args.output/'model.json', manifest)
    if all(gates.values()):
        from fragrance_ai.recommender.unified_transport import UnifiedTransportModel
        model = UnifiedTransportModel(args.output/'model.json', sha256=sha(args.output/'model.json'))
        cpu = model.kernel(x[test[:1024]])
        parity = float(np.max(np.abs(cpu-pt[:1024])))
        if parity > 2e-6:
            raise ValueError('CPU export parity failed')
        write(args.output/'cpu_export_check.json', {'passed': True, 'rows': len(cpu),
            'max_abs_difference': parity, 'torch_required_at_runtime': False})
    print(json.dumps({'event': 'complete', 'accepted': all(gates.values()), 'gates': gates,
        'baseline': baseline_test, 'unified': learned_test, 'manifest_sha256': sha(args.output/'model.json')}), flush=True)


if __name__ == '__main__':
    main()
