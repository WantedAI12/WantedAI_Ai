"""CPU-only helpers for external human-profile evaluation, not scent certification.

The unit of analysis is a complete stimulus profile, never an individual zero
descriptor or a duplicated rater row. Undefined profiles remain failures in
the all-request denominator. No transformed MAE is called accuracy.
"""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

import numpy as np


def group_folds(groups: list[str], folds: int = 5) -> np.ndarray:
    """Stable, balanced group assignment independent of outcomes and row order."""
    unique = sorted(set(groups), key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    if folds < 2 or len(unique) < folds:
        raise ValueError("not enough distinct groups for the requested folds")
    assignment = {group: index % folds for index, group in enumerate(unique)}
    return np.asarray([assignment[group] for group in groups], dtype=int)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("invalid ridge training data")
    center, scale, intercept = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
    scale[scale < 1e-10] = 1.0
    z = (x - center) / scale
    # A dual solve avoids cubic work in the descriptor dimension for tiny sets.
    if z.shape[1] > len(z):
        coefficients = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), y - intercept)
    else:
        coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return {"center": center.tolist(), "scale": scale.tolist(),
            "intercept": intercept.tolist(), "coefficients": coefficients.tolist(),
            "alpha": float(alpha)}


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    center, scale = np.asarray(model["center"]), np.asarray(model["scale"])
    intercept, weights = np.asarray(model["intercept"]), np.asarray(model["coefficients"])
    if (x.ndim != 2 or center.ndim != 1 or scale.shape != center.shape
            or x.shape[1] != len(center) or weights.shape != (len(center), len(intercept))
            or not all(np.isfinite(v).all() for v in (x, center, scale, intercept, weights))
            or np.any(scale <= 0)):
        raise ValueError("invalid portable model or feature contract")
    return np.maximum(0.0, ((x - center) / scale) @ weights + intercept)


def profile_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if prediction.ndim != 2 or target.shape != prediction.shape or target.shape[1] < 2:
        raise ValueError("profile shape mismatch")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("human outcomes must be finite and nonnegative")
    present = np.isfinite(prediction).all(axis=1) & (prediction >= 0).all(axis=1)
    safe = np.where(np.isfinite(prediction), prediction, 0.0)
    denominator = np.linalg.norm(safe, axis=1) * np.linalg.norm(target, axis=1)
    defined = present & (denominator > 1e-12)
    cosine = np.full(len(target), np.nan)
    cosine[defined] = np.clip(1.0 - np.sum(safe[defined] * target[defined], axis=1) / denominator[defined], 0.0, 2.0)
    a, b = safe - safe.mean(axis=1, keepdims=True), target - target.mean(axis=1, keepdims=True)
    centered_norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    correlated = present & (centered_norm > 1e-12)
    pearson = np.full(len(target), np.nan)
    pearson[correlated] = np.clip(np.sum(a[correlated] * b[correlated], axis=1) / centered_norm[correlated], -1.0, 1.0)
    mae = np.mean(np.abs(safe - target), axis=1)
    mae[~present] = np.nan
    return {"mae": mae, "cosine_distance": cosine, "pearson": pearson}


def wilson_lower(successes: int, total: int, confidence: float = 0.95) -> float:
    """One-sided lower bound; callers must supply independent Bernoulli units."""
    if total <= 0 or not 0 <= successes <= total or not 0.5 < confidence < 1:
        raise ValueError("invalid binomial counts/confidence")
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    return max(0.0, (p + z*z/(2*total) - z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1 + z*z/total))


def cluster_interval(values: np.ndarray, groups: list[str], draws: int = 2000) -> list[float] | None:
    """Equal-weight composition groups; this does not resample unavailable raters."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(groups) or draws < 100:
        raise ValueError("invalid cluster bootstrap input")
    aggregates = []
    for group in sorted(set(groups)):
        selected = values[np.asarray(groups) == group]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            aggregates.append((float(selected.sum()), len(selected)))
    if len(aggregates) < 2:
        return None
    rng = np.random.default_rng(20260905)
    draw = np.asarray(aggregates)[rng.integers(0, len(aggregates), size=(draws, len(aggregates)))].sum(axis=1)
    samples = draw[:, 0] / draw[:, 1]
    return np.quantile(samples, [0.025, 0.975]).tolist()


def summarize_profiles(prediction: np.ndarray, target: np.ndarray, groups: list[str],
                       cosine_tolerance: float = 0.10) -> dict:
    if len(groups) != len(target) or not groups or not 0 < cosine_tolerance < 1:
        raise ValueError("invalid evaluation population/tolerance")
    errors = profile_errors(prediction, target)
    success = np.isfinite(errors["cosine_distance"]) & (errors["cosine_distance"] <= cosine_tolerance)
    result = {name: {"mean": float(np.nanmean(values)) if np.isfinite(values).any() else None,
                     "defined_profiles": int(np.isfinite(values).sum()),
                     "composition_bootstrap_95": cluster_interval(values, groups)}
              for name, values in errors.items()}
    unique = sorted(set(groups))
    # All replicates of a composition must pass; correlated rows are not counted
    # as additional independent evidence. This remains a descriptive bound.
    group_successes = sum(bool(success[np.asarray(groups) == g].all()) for g in unique)
    result["diagnostic_tolerance"] = {
        "criterion": "complete_profile_cosine_distance_lte_predeclared_tolerance",
        "tolerance": cosine_tolerance, "human_repeatability_anchored": False,
        "successes": int(success.sum()), "all_requested_profiles": len(target),
        "pass_rate_percent": 100.0 * float(success.mean()),
        "independent_composition_groups": len(unique), "all_replicates_pass_groups": group_successes,
        "composition_wilson_one_sided_95_lower": wilson_lower(group_successes, len(unique)),
        "bound_scope": "descriptive composition sampling; not participant uncertainty or selection uncertainty",
        "actual_human_accuracy_90_authorized": False,
    }
    return result


def choose_ridge(x: np.ndarray, y: np.ndarray, groups: list[str],
                 alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)) -> tuple[dict, dict]:
    """Tune by training-only group OOF cosine distance; no held-out outcomes."""
    fold_ids = group_folds(groups)
    rows = []
    for alpha in alphas:
        predictions = np.zeros_like(y, dtype=float)
        for fold in range(5):
            train = fold_ids != fold
            predictions[~train] = predict_ridge(fit_ridge(x[train], y[train], alpha), x[~train])
        errors = profile_errors(predictions, y)
        # An undefined vector is worst-case for selection, never silently dropped.
        loss = float(np.mean(np.nan_to_num(errors["cosine_distance"], nan=1.0)))
        rows.append({"alpha": alpha, "oof_cosine_distance": loss,
                     "oof_mae": float(np.mean(errors["mae"]))})
    chosen = min(rows, key=lambda row: (row["oof_cosine_distance"], row["alpha"]))
    return fit_ridge(x, y, chosen["alpha"]), {
        "selection_data": "training_only", "group_count": len(set(groups)),
        "folds": 5, "candidates": rows, "selected_alpha": chosen["alpha"],
        "selected_oof_is_post_selection_diagnostic": True,
    }


def paired_profile_comparison(baseline: np.ndarray, candidate: np.ndarray,
                              target: np.ndarray, groups: list[str]) -> dict:
    """Compare identical stimuli/axes, not means from unequal coverage sets."""
    baseline, candidate, target = (np.asarray(value, dtype=float) for value in (baseline, candidate, target))
    if baseline.shape != candidate.shape or target.shape != candidate.shape or len(groups) != len(target):
        raise ValueError("paired population mismatch")
    common = np.isfinite(candidate).all(axis=1) & np.isfinite(baseline).all(axis=1)
    if not common.any():
        return {"paired_profiles": 0, "paired_cosine_defined_profiles": 0}
    base, cand = profile_errors(baseline[common], target[common]), profile_errors(candidate[common], target[common])
    pair_groups = [group for group, keep in zip(groups, common) if keep]
    cosine_defined = np.isfinite(base["cosine_distance"]) & np.isfinite(cand["cosine_distance"])
    cosine_delta = base["cosine_distance"] - cand["cosine_distance"]
    mae_delta = base["mae"] - cand["mae"]
    return {
        "paired_profiles": int(common.sum()), "paired_cosine_defined_profiles": int(cosine_defined.sum()),
        "baseline_mae_on_common_profiles": float(np.mean(base["mae"])),
        "candidate_mae_on_common_profiles": float(np.mean(cand["mae"])),
        "baseline_cosine_distance_on_common_defined_profiles": float(np.mean(base["cosine_distance"][cosine_defined])) if cosine_defined.any() else None,
        "candidate_cosine_distance_on_common_defined_profiles": float(np.mean(cand["cosine_distance"][cosine_defined])) if cosine_defined.any() else None,
        "baseline_mae_minus_candidate_mae": float(np.mean(mae_delta)),
        "mae_improvement_composition_bootstrap_95": cluster_interval(mae_delta, pair_groups),
        "baseline_cosine_distance_minus_candidate": float(np.mean(cosine_delta[cosine_defined])) if cosine_defined.any() else None,
        "cosine_improvement_composition_bootstrap_95": cluster_interval(cosine_delta, pair_groups),
    }
