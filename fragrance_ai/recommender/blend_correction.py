"""Residual corrections in ingredient mass coordinates, never in output scores.

The LP corrects a whole formula against overlap, cosine, exclusions and
background separation together. Its slack is an optimization variable only;
each returned composition is evaluated with the original normalized profiles.
All candidate columns and original physical constraints remain present.
"""

from time import monotonic

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .lotion_numerics import conditioned_linprog, response_variable_scales


def correct_blend(*, shapes, wanted, responses, fixed, rhs, equality, equality_rhs,
                  bounds, initial, target, avoided=None, background=None,
                  maximum_seconds=30., maximum_rounds=80):
    s, q, r, a, b, e, f = [np.asarray(x, float) for x in
                           (shapes, wanted, responses, fixed, rhs, equality, equality_rhs)]
    limits = np.asarray(bounds, float)
    if s.ndim != 3 or r.ndim != 2 or limits.ndim != 2:
        raise ValueError("aligned profile, release and finite mass-bound matrices required")
    h, n, d = s.shape
    t = len(r)
    mask = np.zeros((t, d)) if avoided is None else np.asarray(avoided, float)
    if (q.shape != (h, t, d) or r.shape != (t, n) or limits.shape != (n, 2)
            or a.shape != (len(b), n) or e.shape != (len(f), n) or mask.shape != (t, d)
            or not 0 < target <= 1 or not np.isfinite(maximum_seconds) or maximum_seconds <= 0
            or maximum_rounds < 1
            or any(not np.isfinite(x).all() for x in (s, q, r, a, b, e, f, limits, mask))
            or any(np.any(x < 0) for x in (s, q, r, limits, mask))
            or np.any(limits[:, 0] > limits[:, 1]) or np.any(mask > 1)
            or not np.allclose(s[:, limits[:, 1] > 0].sum(-1), 1., atol=1e-7)
            or not np.allclose(q.sum(-1), 1., atol=1e-7)):
        raise ValueError("finite, normalized profiles and unchanged feasible constraints required")
    unit = q / np.linalg.norm(q, axis=-1, keepdims=True)
    contrast = None
    if background is not None:
        bg = np.asarray(background, float)
        if bg.shape != (h, d) or not np.isfinite(bg).all() or np.any(bg < 0) or np.any(bg.sum(-1) <= 0):
            raise ValueError("valid background profiles required")
        from .reference_discrimination import contrast_direction
        contrast = contrast_direction(q,bg[:,None])
    lo, hi = limits.T
    started = monotonic()
    report = {"version": "ingredient-residual-correction/v88", "all_material_columns": n,
              "descriptor_axes": d, "linear_solves": 0, "cuts": 0,
              "source_targets_unchanged": True, "reported_score_offset": 0.,
              "formal_infeasibility_certificate": False, "accepted_steps": 0}

    def valid(w):
        return (w is not None and np.shape(w) == (n,) and np.isfinite(w).all()
                and np.all(w >= lo - 1e-10) and np.all(w <= hi + 1e-10)
                and np.all(a @ w <= b + 1e-9) and np.all(np.abs(e @ w - f) <= 1e-8)
                and np.all(r @ w > 0))

    def solve(cost, matrix, rhs_values, eq, bounds_values, scales):
        remaining = maximum_seconds - (monotonic() - started)
        if remaining <= 0:
            return None
        report["linear_solves"] += 1
        def bounded_solver(objective, **kwargs):
            return linprog(objective, **{**kwargs, "options": {
                **kwargs.get("options", {}), "time_limit": min(4., remaining),
                "primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}})
        return conditioned_linprog(bounded_solver, cost, A_ub=sparse.csr_matrix(matrix), b_ub=rhs_values,
            A_eq=sparse.csr_matrix(eq), b_eq=f, bounds=bounds_values, scales=scales)

    mass_scales = response_variable_scales(r, 0)
    start = np.asarray(initial, float).copy() if initial is not None else None
    if not valid(start):
        feasible = solve(np.zeros(n), a, b, e, bounds, mass_scales)
        start = feasible.x if feasible is not None and feasible.success and valid(feasible.x) else None
    if start is None:
        return None, {**report, "status": "feasible_incumbent_unavailable"}

    def evaluate(w):
        den = r @ w
        p = np.einsum("tn,n,hnd->htd", r, w, s, optimize=True) / den[None, :, None]
        direction = p / np.linalg.norm(p, axis=-1, keepdims=True)
        overlap = np.minimum(p, q).sum(-1)
        cosine = (direction * unit).sum(-1)
        avoidance = 1 - (p * mask).sum(-1)
        score = float(np.minimum(np.minimum(overlap, cosine), avoidance).min())
        separation = np.inf if contrast is None else float((direction * contrast).sum(-1).min())
        passed = score + 1e-10 >= target and separation > 1e-8
        merit = max(0., target - score, 2e-8 - separation)
        return score, passed, merit, p, direction

    best = start.copy()
    best_score, best_passed, _, _, _ = evaluate(best)
    working = start.copy()
    report["starting_score"] = 100 * best_score
    cuts, times, keys = [], [], set()

    def add(row, time):
        scale = max(float(np.max(np.abs(row))), 1e-300)
        key = (time, np.round(row / scale, 12).tobytes())
        if key not in keys:
            keys.add(key)
            cuts.append(row)
            times.append(time)

    def separate(w):
        score, passed, merit, p, direction = evaluate(w)
        for head in range(h):
            for time in range(t):
                sign = np.sign(p[head, time] - q[head, time])
                add((.5 * (s[head] @ sign - q[head, time] @ sign) - (1 - target)) * r[time], time)
                add((s[head] @ (target * direction[head, time] - unit[head, time])) * r[time], time)
                add((s[head] @ mask[time] - (1 - target)) * r[time], time)
                if contrast is not None:
                    # sum(y) >= ||y|| for nonnegative y: a conservative linear
                    # interior margin implies the unchanged cosine separation.
                    add((2e-8 - s[head] @ contrast[head, time]) * r[time], time)
        return score, passed, merit

    for _ in range(maximum_rounds):
        if best_passed or monotonic() - started >= maximum_seconds:
            break
        _, _, current_merit = separate(working)
        den = np.maximum(r @ working, 1e-30)
        rows = np.asarray(cuts)
        # The correction slack uses each time window's exposure scale. It is
        # never copied to the output score or used to waive a final constraint.
        matrix = np.vstack((np.c_[a, np.zeros(len(b))], np.c_[rows, -den[np.asarray(times)]]))
        # Scale the new slack in the same exposure coordinates as the mass
        # variables; otherwise tiny-release columns disappear beside a unit
        # slack coefficient during numerical presolve.
        exposure_scale = np.max(r * mass_scales, axis=0)
        positive = exposure_scale[exposure_scale > 0]
        slack_scale = max(1e-15, min(1., float(np.median(positive)) / float(den.max())))
        report['residual_coordinate_scale'] = slack_scale
        result = solve(np.r_[np.zeros(n), 1.], matrix, np.r_[b, np.zeros(len(cuts))],
                       np.c_[e, np.zeros(len(f))], [*bounds, (0., None)], np.r_[mass_scales, slack_scale])
        if result is None or not result.success or result.x is None or not valid(result.x[:n]):
            report["last_solver_status"] = None if result is None else int(result.status)
            break
        proposal = result.x[:n]
        previous_cuts = len(cuts)
        separate(proposal)
        next_working, next_merit = working, current_merit
        for fraction in (1., .5, .25, .125):
            trial = working + fraction * (proposal - working)
            if not valid(trial):
                continue
            score, passed, merit, _, _ = evaluate(trial)
            if (passed, score) > (best_passed, best_score):
                best, best_score, best_passed = trial.copy(), score, passed
                report["accepted_steps"] += 1
            if merit < next_merit - 1e-12:
                next_working, next_merit = trial.copy(), merit
        working = next_working
        if len(cuts) == previous_cuts and next_merit >= current_merit - 1e-12:
            report["separation_stalled"] = True
            break
    correction = best - start
    report.update(status="target_met" if best_passed else "best_verified_correction",
                  selected_score=100 * best_score, target_met=bool(best_passed), cuts=len(cuts),
                  changed_materials=int(np.count_nonzero(np.abs(correction) > 1e-10)),
                  absolute_dose_change_percent=float(100 * np.abs(correction).sum()),
                  seconds=monotonic() - started)
    return best, report
