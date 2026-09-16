from dataclasses import replace
import pytest

from tests.test_full_pool_search import material, brief
from fragrance_ai.recommender.global_profile_search import optimize_full_pool


def test_fractional_dual_bound_contains_the_primal_and_exposes_missing_mass():
    items = [material('a', 'top', profile={'green': .8, 'woody': .2}),
             material('b', 'heart', profile={'green': .8, 'woody': .2}),
             material('c', 'base', profile={'green': .8, 'woody': .2})]
    request = brief()
    request = replace(request, target_profile={'green': 1.})
    result = optimize_full_pool(items, request, pyramid_bounds={level: (0.,100.) for level in ('top','heart','base')})
    assert result.weights_percent
    assert result.relaxed_overlap_score == pytest.approx(80., abs=1e-6)
    assert result.certified_overlap_upper_score >= result.relaxed_overlap_score-1e-7
    assert result.certified_overlap_upper_score < 80.001
