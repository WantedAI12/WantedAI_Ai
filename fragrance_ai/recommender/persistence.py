"""Threshold crossing of a model curve, not human detection longevity."""
import math


def model_persistence(points, threshold_percent, required_minutes=None):
    if isinstance(threshold_percent, bool) or not math.isfinite(threshold_percent) or not 0 < threshold_percent <= 100:
        raise ValueError("relative model threshold must be in (0, 100]")
    if required_minutes is not None and (isinstance(required_minutes, bool) or not math.isfinite(required_minutes) or required_minutes <= 0):
        raise ValueError("required model duration must be positive and finite")
    result = {"basis": "relative_model_intensity_not_human_detection", "threshold_percent": threshold_percent,
        "required_minutes": required_minutes, "estimate_minutes": None, "lower_bound_minutes": None,
        "upper_bound_minutes": None, "meets_requirement": None,
        "estimated_requirement_met": None, "human_detection_guaranteed": False,
        "assumption": "piecewise_linear_between_simulated_points", "bound_kind": "model_sample_bracket_not_statistical_confidence_interval"}
    if len(points) < 2:
        return {**result, "status": "unavailable", "reason": "missing_model_timepoints"}
    curve = []
    for row in points:
        t, y = row.get("minutes"), row.get("relative_to_opening_intensity_percent")
        if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0
                or isinstance(y, bool) or not isinstance(y, (int, float)) or not math.isfinite(y) or y < 0):
            return {**result, "status": "unavailable", "reason": "invalid_model_timepoint"}
        curve.append((t, y))
    if curve[0][0] != 0 or any(right[0] <= left[0] for left, right in zip(curve, curve[1:])):
        return {**result, "status": "unavailable", "reason": "requires_time_zero_and_strictly_increasing_times"}
    result["observation_window_minutes"] = curve[-1][0]
    crossing = None
    initial_below = curve[0][1] < threshold_percent
    lower, upper = 0., 0.
    if initial_below:
        crossing = 0.
    else:
        for (left_t, left_y), (right_t, right_y) in zip(curve, curve[1:]):
            if right_y < threshold_percent:
                crossing = left_t + (right_t - left_t) * (left_y - threshold_percent) / (left_y - right_y)
                lower, upper = left_t, right_t
                break
    if crossing is None:
        result.update(status="right_censored", lower_bound_minutes=curve[-1][0],
                      meets_requirement=None if required_minutes is None or required_minutes > curve[-1][0] else True)
    else:
        meets = None if required_minutes is None else True if required_minutes <= lower else False if required_minutes >= upper else None
        result.update(status="below_threshold_at_start" if initial_below else "crossing_estimated",
                      estimate_minutes=crossing, lower_bound_minutes=lower, upper_bound_minutes=upper,
                      meets_requirement=meets,
                      estimated_requirement_met=None if required_minutes is None else crossing + 1e-9 >= required_minutes)
        result["later_reappearance"] = any(t > crossing and y >= threshold_percent for t, y in curve)
    return result
