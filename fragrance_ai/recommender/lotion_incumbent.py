"""Transfer recipe proposals, never old scores, between base-design variants."""
import math

import numpy as np


def incumbent_weights(lines, pool):
    if not lines:
        return None, 'not_supplied'
    if not isinstance(lines, (list, tuple)):
        return None, 'invalid_recipe'
    known = {item.ingredient_id for item in pool}
    weights = {}
    for line in lines:
        if not isinstance(line, dict):
            return None, 'invalid_recipe'
        identifier, percent = line.get('ingredient_id'), line.get('concentrate_percent')
        if not isinstance(identifier, str) or identifier not in known:
            return None, 'ingredient_not_in_current_pool'
        if identifier in weights:
            return None, 'duplicate_ingredient'
        if (isinstance(percent, bool) or not isinstance(percent, (int, float))
                or not math.isfinite(percent) or not 0 < percent <= 100):
            return None, 'invalid_concentration'
        weights[identifier] = percent / 100.
    if not math.isclose(sum(weights.values()), 1., rel_tol=0., abs_tol=1e-8):
        return None, 'concentrations_do_not_sum_to_100'
    return np.array([weights.get(item.ingredient_id, 0.) for item in pool]), 'prepared_for_current_constraint_check'
