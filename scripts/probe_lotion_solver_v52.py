"""Observe the two distinct unresolved V40 solver requests; do not change solves."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.verify_odor_concepts_v39 import read_catalog
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from fragrance_ai.recommender import lotion_optimizer as module
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, choices=(1,))
    p.add_argument('--brief', choices=('soft clean musk', 'opening woody, drydown citrus'))
    args = p.parse_args()
    if args.output.exists():
        p.error('new evidence file required')
    catalog_path = ROOT/'dist/runtime-catalog-v51/catalog/catalog_manifest.json'
    catalog = read_catalog(catalog_path)
    paths = [Path(__file__), ROOT/'fragrance_ai/recommender/lotion_optimizer.py',
             ROOT/'fragrance_ai/recommender/lotion_numerics.py', ROOT/'fragrance_ai/recommender/lotion_estimation.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original, conditioned = module.linprog, module.conditioned_linprog
    rows = []
    for brief in ((args.brief,) if args.brief else ('soft clean musk', 'opening woody, drydown citrus')):
        request, scenarios, _ = build_estimated_lotion_inputs(LotionEstimateRequest(brief=brief,
            registry_pool='conditional_research', max_risk_tier=2), catalog, oil_base_percent=10.)
        traces = []
        def state():
            frame = inspect.currentframe().f_back
            while frame is not None:
                if frame.f_code.co_name == '_optimize_lotion_transport':
                    return frame.f_locals
                frame = frame.f_back
            return None
        def record(result, s, extra, normalized):
            row = {**extra, 'status': int(result.status), 'message': result.message,
                   'iterations': getattr(result, 'nit', None)}
            if s is not None and normalized and getattr(result, 'x', None) is not None:
                weights = np.maximum(0., result.x[:len(s['pool'])])
                if np.isfinite(weights).all() and weights.sum() > 0:
                    weights /= weights.sum()
                    score, checks = s['assessments'](weights)
                    row.update(actual_score=score, valid=bool(s['valid'](weights)),
                        min_overlap=min(c['overlap_score'] for c in checks),
                        min_cosine=min(c['cosine_score'] for c in checks),
                        min_air=float((s['physical']@weights).min()),
                        formula_cost=float(s['prices']@weights),
                        support=int(np.count_nonzero(weights > 1e-12)))
            traces.append(row)
        def solve(objective, **kwargs):
            if args.threads is not None:
                kwargs['options'] = {**kwargs.get('options', {}), 'threads': args.threads}
            caller = inspect.currentframe().f_back.f_code.co_name
            s = state()
            started = time.perf_counter()
            result = original(objective, **kwargs)
            record(result, s, {'method': kwargs.get('method'), 'caller': caller,
                'seconds': time.perf_counter()-started, 'presolve': kwargs.get('options', {}).get('presolve'),
                'rows': kwargs['A_ub'].shape[0], 'variables': len(objective)}, caller != 'conditioned_linprog')
            return result
        def scaled(*a, **kw):
            s = state()
            result = conditioned(*a, **kw)
            record(result, s, {'method': 'post_coordinate_recovery'}, True)
            return result
        module.linprog, module.conditioned_linprog = solve, scaled
        started = time.perf_counter()
        try:
            result = module.optimize_lotion(request, catalog, transport_scenarios=scenarios,
                                            _transport_only=True, reuse_transport_basis=True)
        finally:
            module.linprog, module.conditioned_linprog = original, conditioned
        row = {'brief': brief, 'oil_percent': 10., 'threads_override': args.threads, 'seconds': time.perf_counter()-started,
            **{k: result.get(k) for k in ('score', 'profile_target_met', 'search_incomplete', 'solver_calls', 'solver_conditioning_calls')},
            'traces': traces, 'result': result}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'result'}, ensure_ascii=False), flush=True)
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'unchanged_solver_two_distinct_prior_incomplete_requests',
        'rows': rows, 'source_sha256': hashes, 'source_unchanged': True,
        'catalog_manifest_sha256': hashlib.sha256(catalog_path.read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
