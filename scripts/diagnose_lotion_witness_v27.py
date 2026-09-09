"""Check a previous feasible recipe against the new, exact LP constraints."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts import benchmark_lotion_design as bench
    from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
    from fragrance_ai.recommender import lotion_optimizer as current
    bench.initialize()
    request, scenarios, _ = build_estimated_lotion_inputs(LotionEstimateRequest(
        brief='gourmand, smoky scent', registry_pool='conditional_research', max_risk_tier=2), bench.CATALOG)
    path = ROOT/'tmp/build-v26-final-runtime-source/fragrance_ai/recommender/lotion_optimizer.py'
    spec = importlib.util.spec_from_file_location('fragrance_ai.recommender._baseline_v26', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    baseline = old.optimize_lotion(request, bench.CATALOG, transport_scenarios=scenarios)
    lines = baseline['recipe'] or baseline['closest_candidate']
    lookup = {r['ingredient_id']: r['concentrate_percent']/100 for r in lines}
    _, _, pool = current.prepare_lotion_optimization(request, bench.CATALOG)
    weights = np.array([lookup.get(i.ingredient_id, 0.) for i in pool])
    original = current.linprog
    checks = []
    def solve(objective, **kw):
        count = len(objective)-len(pool)
        a = kw['A_ub']
        slack = np.abs(a[:count, :len(pool)] @ weights)
        witness = np.r_[weights, slack]
        result = original(objective, **kw)
        checks.append({'status': result.status,
            'inequality_max_violation': float(np.max(a @ witness-kw['b_ub'])),
            'equality_max_violation': float(np.max(np.abs(kw['A_eq'] @ witness-kw['b_eq']))),
            'bounds_max_violation': float(max(max(lo-v, v-hi if hi is not None else -np.inf)
                for v, (lo,hi) in zip(witness, kw['bounds'])))})
        return result
    current.linprog = solve
    candidate = current.optimize_lotion(request, bench.CATALOG, transport_scenarios=scenarios)
    report = {'baseline_score': baseline['score'], 'candidate_score': candidate['score'],
              'baseline_formula': lines, 'checks': checks, 'research_only': True}
    output = ROOT/'benchmarks/lotion_solver_v27_witness.json'
    if output.exists():
        raise ValueError('refusing to overwrite witness report')
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='baseline_formula'}), flush=True)


if __name__ == '__main__':
    main()
