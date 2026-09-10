"""Separate fixed-profile coverage limits from transport/search limits; no score changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('new output required')
    from scripts import benchmark_lotion_design as bench
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.models import RecipeConstraints, profile_vector
    from fragrance_ai.recommender.safety import CandidateSafetyScreen
    bench.initialize()
    parser = NaturalLanguageBriefParser(bench.CATALOG)
    previous = [json.loads(line) for line in (ROOT/'benchmarks/lotion_design_v28_auto_full400/results.jsonl').read_text(encoding='utf-8').splitlines()]
    cache, rows = {}, []
    start = time.perf_counter()
    for row in previous:
        if row['profile_target_met']:
            continue
        brief = parser.parse(row['brief'], RecipeConstraints(product_category='body_lotion', product_concentration_percent=.5, max_risk_tier=2,
            enable_registry_trace_candidates=True))
        pool, _ = CandidateSafetyScreen().screen(bench.CATALOG, brief)
        matrix = np.unique(np.array([i.vector() for i in pool if abs(i.active_strength_percent-100)<1e-8 and i.vector().sum()>0]), axis=0)
        targets = [profile_vector(t) for t in [brief.target_profile, *brief.phase_target_profiles.values()] if sum(t.values())>0]
        bounds = []
        for target in targets:
            key = (matrix.tobytes(), target.tobytes())
            if key not in cache:
                n, d = matrix.shape
                # x is any convex mixture; u_j <= min(target_j, predicted_j).
                result = linprog(np.r_[np.zeros(n), -np.ones(d)],
                    A_ub=sparse.hstack([-sparse.csr_matrix(matrix.T), sparse.eye(d)], format='csr'), b_ub=np.zeros(d),
                    A_eq=sparse.csr_matrix(np.r_[np.ones(n), np.zeros(d)][None,:]), b_eq=[1.],
                    bounds=[(0,None)]*n+[(0,float(t)) for t in target], method='highs')
                cache[key] = None if not result.success else min(100., -float(result.fun)*100+1e-6)
            bounds.append(cache[key])
        upper = min(b for b in bounds if b is not None) if any(b is not None for b in bounds) else None
        reason = 'fixed_profile_coverage_limit' if upper is not None and upper < 95 else (
            'search_incomplete_or_transport_constraints' if row.get('search_incomplete') else 'transport_composition_or_search_limit')
        rows.append({**row, 'optimistic_static_overlap_upper': upper, 'reason': reason})
    report = {'scope':'numerical_fixed_profile_convex_hull_diagnostic_not_human_accuracy_or_formal_certificate',
        'relaxations':'ignores transport, composition caps, cost, skin exposure and avoidance; extra physically uncovered screened candidates allowed',
        'failed_requests':len(rows), 'reason_counts':dict(Counter(r['reason'] for r in rows)),
        'seconds':time.perf_counter()-start, 'rows':rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
