"""Evaluate rollout and legacy V58 against the frozen V60 test identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fragrance_ai.recommender.unified_transport import UnifiedTransportModel, trajectory, PRODUCTS
from fragrance_ai.recommender.lotion_surrogate import LotionReleaseSurrogate, LOWER, UPPER


def metrics(p, y):
    diff = p-y
    return {'mae': float(np.abs(diff).mean()), 'rmse': float(np.sqrt((diff**2).mean())),
            'max_abs_error': float(np.abs(diff).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fresh-contexts', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('preserve previous evaluation')
    manifest = a.run/'model.json'
    model = UnifiedTransportModel(manifest, sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    with np.load(a.run/'dataset.npz', allow_pickle=False) as d:
        test = d['split'] == 2
        x, labels, product = d['raw'][test], d['target'][test, 0], d['product'][test]
    if a.fresh_contexts:
        from scripts.train_unified_product_v60 import generate, teacher
        source = json.loads(Path('.benchmarks/lotion_training_v58/source.json').read_text(encoding='utf-8'))
        d = generate(source, 600910, 4)
        test = d['split'] == 2
        x, product = d['raw'][test], d['product'][test]
        labels = teacher(x, 512)[:, 0]
    times = [0., .01, .05, .1, .25, .5, 1.]
    started = time.perf_counter()
    rows, _ = trajectory(x[:, :6], x[:, 6:], times, operator=model.stable_kernel)
    latency = time.perf_counter()-started
    baseline, _ = trajectory(x[:, :6], x[:, 6:], times)
    per_product = {name: {'rows': int(np.sum(product == i)),
        'analytic_32_steps': metrics(baseline[-1, product == i, :5], labels[product == i]),
        'learned_32_steps': metrics(rows[-1, product == i, :5], labels[product == i])}
        for i, name in enumerate(PRODUCTS)}
    legacy_path = Path('.benchmarks/lotion_training_v58/run-01/model.json')
    legacy = LotionReleaseSurrogate(legacy_path, sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest())
    in_domain = (product == 1)&np.all((x >= LOWER)&(x <= UPPER), axis=1)
    old = legacy.predict(x[in_domain])
    old_rows = np.stack([legacy.predict(np.c_[x[in_domain, :6]*t, x[in_domain, 6:]]) for t in times[1:]])
    monotone_old = np.all(np.diff(old_rows[:, :, 2:], axis=0) >= -1e-5, axis=(0, 2))
    report = {'schema': 'unified-rollout-verification/v1', 'test_contexts': len(x), 'per_product': per_product,
        'context_set': 'new_numerical_confirmation_contexts_seed_600910' if a.fresh_contexts else 'previously_evaluated_test_contexts',
        'post_training_operator_policy': 'two_half_step_defect_guard',
        'all_analytic_32': metrics(baseline[-1, :, :5], labels),
        'all_learned_32': metrics(rows[-1, :, :5], labels),
        'lotion_vs_v58': {'matched_test_contexts': int(in_domain.sum()),
            'old_v58': metrics(old, labels[in_domain]), 'new_v60': metrics(rows[-1, in_domain, :5], labels[in_domain]),
            'v58_nonmonotone_contexts': int((~monotone_old).sum()),
            'v60_nonmonotone_contexts': int(np.sum(~np.all(np.diff(rows[:, in_domain, 2:], axis=0) >= -1e-14, axis=(0, 2))))},
        'mass_balance_max_error': float(np.max(np.abs(rows.sum(axis=2)-1))),
        'nonnegative': bool(np.all(rows >= 0)), 'monotone_sinks': bool(np.all(np.diff(rows[:, :, 2:], axis=0) >= -1e-14)),
        'seconds_for_all_test_trajectories': latency,
        'scope': 'numerical_teacher_not_human_similarity', 'checkpoint_sha256': model.sha256}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
