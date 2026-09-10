"""Conservative coordinate envelope for any nonnegative mixture of fixed profiles."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a new output path")
    from scripts.benchmark_lotion_design import initialize
    import scripts.benchmark_lotion_design as bench
    from scripts.evaluate_request_space import request_cases
    from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS, RecipeConstraints, profile_vector
    initialize()
    catalog = bench.CATALOG
    # Deliberately include even ineligible profiles: a larger hull makes this
    # an optimistic upper bound, never a newly authorised ingredient pool.
    matrix = np.array([m.vector() for m in catalog.ingredients if np.sum(m.vector()) > 0])
    maximum = matrix.max(axis=0)
    parser = NaturalLanguageBriefParser(catalog)
    rows = []
    for case in request_cases():
        try:
            brief = parser.parse(case["brief"], RecipeConstraints(product_category="body_lotion", max_risk_tier=2,
                                 enable_registry_trace_candidates=True))
            # A phase with avoidance only has no positive target vector. It
            # cannot supply an overlap upper bound of zero.
            targets = [t for t in [brief.target_profile] + list(brief.phase_target_profiles.values())
                       if np.sum(profile_vector(t)) > 0]
            if not targets:
                rows.append({"id": case["id"], "error": "no positive target for a profile-envelope bound"})
                continue
            upper = min(float(np.minimum(profile_vector(t), maximum).sum())*100 for t in targets)
            rows.append({"id": case["id"], "brief": case["brief"], "upper_bound_percent": upper,
                         "95_excluded_by_profile_envelope": upper < 95-1e-8})
        except ValueError as error:
            rows.append({"id": case["id"], "error": str(error)})
    report = {"scope": "fixed_catalog_linear_nonnegative_profile_mixtures_not_real_perception",
              "inequality": "p_j <= max_i a_ij; overlap(t,p) <= sum_j min(t_j,max_i a_ij)",
              "catalog_rows": len(catalog.ingredients), "profile_rows": len(matrix),
              "axis_upper_bounds_percent": dict(zip(SCENT_DIMENSIONS, (maximum*100).tolist())),
              "target": 95, "requests": len(rows),
              "excluded_count": sum(r.get("95_excluded_by_profile_envelope", False) for r in rows),
              "rows": rows, "universal_95_possible_under_current_profile_model": not any(r.get("95_excluded_by_profile_envelope") for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
