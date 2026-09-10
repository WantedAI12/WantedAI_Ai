"""Paired solver-only diagnostic; no changes to target, catalog or physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['current', 'ipm', 'dual'], required=True)
    p.add_argument('--case-id', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance', type=float)
    p.add_argument('--small-matrix-value', type=float)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    from scripts.benchmark_lotion_design import initialize, run
    from scripts.evaluate_request_space import request_cases
    import fragrance_ai.recommender.lotion_optimizer as optimizer
    original = optimizer.linprog
    calls = []

    def measured(*a, **kw):
        if args.small_matrix_value is not None:
            kw['options'] = {**kw.get('options', {}), 'small_matrix_value': args.small_matrix_value}
        if args.tolerance is not None:
            kw['options'] = {**kw.get('options', {}),
                'primal_feasibility_tolerance': args.tolerance, 'dual_feasibility_tolerance': args.tolerance}
        if args.mode != 'current':
            kw['method'] = 'highs-ipm' if args.mode == 'ipm' else 'highs-ds'
            kw['options'] = {**kw.get('options', {}), 'presolve': True, 'time_limit': 2.}
        start = time.perf_counter()
        result = original(*a, **kw)
        calls.append({'status': result.status, 'seconds': time.perf_counter()-start,
                      'iterations': result.nit})
        return result

    optimizer.linprog = measured
    cases = [case for case in request_cases() if case['id'] in args.case_id]
    if {c['id'] for c in cases} != set(args.case_id):
        p.error('unknown case ID')
    initialize()
    rows = []
    for case in cases:
        calls.clear()
        row = {**run(case), 'solver_diagnostics': list(calls)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'mode': args.mode, 'rows': rows,
        'source_sha256': hashlib.sha256(Path(optimizer.__file__).read_bytes()).hexdigest(),
        'diagnostic_only': True}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
