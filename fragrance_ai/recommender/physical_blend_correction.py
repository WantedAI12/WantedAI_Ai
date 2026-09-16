"""Apply source-model release compensation to actual perfume mass fractions."""

from time import monotonic

import numpy as np

from .nonlinear_inverse import profile_loss


def subset_engine(engine, indices):
    from copy import copy
    view=copy(engine)
    view.model=copy(engine.model)
    m,sub=engine.model,view.model
    for name in ('gain','moles','total_moles','transport','vectors'):
        if hasattr(m,name):
            setattr(sub,name,getattr(m,name)[indices])
    sub.coefficients=m.coefficients[:,:,indices]
    sub.interaction=m.interaction[np.ix_(indices,indices)]
    if hasattr(m,'ingredients'):
        sub.ingredients=[m.ingredients[i] for i in indices]
    return view


def release_factors(engine, profiles, weights):
    """Anchor-exact linear proposal factors; off-anchor values need fresh physics.

    Zero-dose columns receive a finite search probe, not invented final signal.
    Sample-wise normalization is retained before averaging the release factors.
    """
    w = np.asarray(weights, float)
    predicted, cache = engine.predict(profiles, w)
    denominator, _, divisor, _, _, total, _ = cache
    probe = np.maximum(w, 1e-5)
    m = engine.model
    power = (probe * m.coefficients / denominator) ** .55
    signal = power / (1 + power) * m.transport / divisor
    factors = (signal / probe / np.maximum(total[..., 0, None], 1e-300)).mean(0)
    response = np.vstack((m.gain, factors))
    amplitude = (signal / probe).mean(0)
    return response, amplitude, predicted


def correct_physical_blend(engine, profiles, targets, time_weights, avoided, initial,
                           lower, upper, prices, budget, *, target_score, bands=(),
                           avoid_ceiling=None, signal_floor=None, point_ceiling=None,
                           persistence_ratio=None, maximum_seconds=24., maximum_rounds=3):
    from .failure_inverse_v87 import point_losses

    p, q, w, lo, hi, cost = [np.asarray(value, float) for value in
                            (profiles, targets, initial, lower, upper, prices)]
    mask = np.broadcast_to(avoided, q.shape)
    if not np.all(mask == mask[0]):
        raise ValueError("head-shared exclusions required for physical compensation")
    fixed, rhs, equal, eq_rhs = [cost], [budget], [np.ones(len(w))], [1.]
    for values, low, high in bands:
        values = np.asarray(values, float)
        if low == high:
            equal.append(values)
            eq_rhs.append(low)
        else:
            fixed.extend((values, -values))
            rhs.extend((high, -low))
    fixed, rhs, equal, eq_rhs = map(np.asarray, (fixed, rhs, equal, eq_rhs))
    fixed_tolerance = np.r_[1e-6, np.full(len(rhs)-1,1e-8)]

    def acceptable(x, pred):
        if (not np.isfinite(x).all() or np.any(x < lo - 1e-10) or np.any(x > hi + 1e-10)
                or np.any(fixed @ x > rhs + fixed_tolerance) or np.any(np.abs(equal @ x - eq_rhs) > 1e-8)):
            return False
        if avoid_ceiling is not None and np.any((pred * mask).sum(-1) > avoid_ceiling + 1e-6):
            return False
        if point_ceiling is not None and np.any(point_losses(pred, q, mask) > point_ceiling + 1e-8):
            return False
        if signal_floor is not None:
            signal, _ = engine.signal_value_gradient(x)
            if np.any(signal < signal_floor * (1 - 1e-9)):
                return False
            if persistence_ratio is not None and signal[-1] < persistence_ratio * signal[0] * (1 - 1e-9):
                return False
        return True

    mass_repaired = False
    initial_mass_error = float(w.sum()-1.)
    if 1e-8 < abs(initial_mass_error) <= 1.01e-5:
        from .mass_projection import repair_mass_residual
        corrected_mass = repair_mass_residual(w[None],lo[None],hi[None])[0]
        corrected_prediction, _ = engine.predict(p,corrected_mass)
        if acceptable(corrected_mass,corrected_prediction):
            w, mass_repaired = corrected_mass, True
    pred, _ = engine.predict(p, w)
    starting = 100 * (1 - profile_loss(pred, q, time_weights, mask)[0])
    report = {"version": "physical-dose-compensation/v88", "starting_score": starting,
              "score_offset": 0., "all_candidate_columns": len(w), "rounds": [],
              "fresh_physics_evaluations": 0, "final_objective_unchanged": True,
              "input_mass_error":initial_mass_error, "roundoff_mass_repaired":mass_repaired}
    if not acceptable(w, pred):
        return None, {**report, "status": "feasible_physical_anchor_required",
            "linear_equality_residual":float(np.abs(equal@w-eq_rhs).max()),
            "linear_inequality_excess":float(np.maximum(fixed@w-rhs,0).max())}
    best, best_score = w.copy(), starting
    started = monotonic()
    for _ in range(maximum_rounds):
        remaining = maximum_seconds - (monotonic() - started)
        if remaining <= 0 or best_score + 1e-8 >= target_score:
            break
        from .failure_inverse_v87 import expanded_reference_support, polish_prepared
        _,gradient,_=engine(p[None],q[None],None,best[None],[0],time_weights=np.asarray(time_weights)[None],
            avoided=mask[None],loss_function=profile_loss,gradient_coordinates='hill_power')
        signal,jac=engine.signal_value_gradient(best)
        required=np.asarray(signal_floor) if signal_floor is not None else np.zeros_like(signal)
        indices=expanded_reference_support(gradient[0],jac,required,best,lo,hi,len(best),
                                          profile_columns=48,signal_columns=24)
        narrowed=subset_engine(engine,indices)
        local,detail=polish_prepared(narrowed,p[:,indices],q,time_weights,mask,best[indices],lo[indices],hi[indices],
            cost[indices],budget,bands=[(np.asarray(v)[indices],low,high) for v,low,high in bands],
            avoid_ceiling=avoid_ceiling,signal_floor=signal_floor,point_ceiling=point_ceiling,
            persistence_ratio=persistence_ratio,maxiter=80,maximum_seconds=max(.001,remaining))
        detail['search_aggregation']='same_nominal_and_time_weighted_objective_as_final'
        detail['all_material_columns_priced']=len(best)
        report['rounds'].append(detail)
        candidate=None
        if local is not None:
            candidate=np.zeros_like(best)
            candidate[indices]=local
        if candidate is None:
            break
        before, anchor = best_score, best.copy()
        for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
            if monotonic() - started >= maximum_seconds:
                break
            trial = anchor + fraction * (candidate - anchor)
            predicted, _ = engine.predict(p, trial)
            report['fresh_physics_evaluations'] += 1
            if not acceptable(trial, predicted):
                continue
            score = 100 * (1 - profile_loss(predicted, q, time_weights, mask)[0])
            if score > best_score + 1e-7:
                best, best_score = trial.copy(), score
            if best_score + 1e-8 >= target_score:
                break
        if best_score <= before + 1e-7:
            break
    report.update(status="target_met" if best_score + 1e-8 >= target_score else "verified_dose_correction",
                  selected_score=best_score, changed_materials=int(np.count_nonzero(np.abs(best - w) > 1e-10)),
                  absolute_dose_change_percent=float(100 * np.abs(best - w).sum()),
                  seconds=monotonic() - started)
    return best, report
