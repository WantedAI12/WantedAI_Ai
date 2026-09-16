from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from tests.test_dose_refinement import materials
from fragrance_ai.recommender.science import TemporalMixtureSimulator


def test_stock_carrier_cannot_be_reported_as_evaporating_with_the_odorant():
    ingredient=replace(materials()[0],active_strength_percent=1.,carrier='dpg')
    line=SimpleNamespace(ingredient_id=ingredient.ingredient_id,name=ingredient.name,pyramid=ingredient.pyramid,
                         concentrate_percent=100.,finished_product_percent=15.,active_strength_percent=1.)
    engine=TemporalMixtureSimulator()
    prepared=engine._prepare([line],{ingredient.ingredient_id:ingredient},{})
    profile=engine._build_ingredient_temporal_profiles(prepared,np.zeros((1,1)))[0]
    assert profile.points[0].estimated_remaining_finished_product_percent==15.
    assert profile.points[0].active_remaining_finished_product_percent==.15
    assert profile.points[-1].estimated_remaining_finished_product_percent is None
    assert profile.points[-1].estimated_evaporated_concentrate_percent is None
    low,high=profile.points[-1].as_supplied_remaining_finished_product_percent_bounds
    assert 0<=low<=.15 and high-low>14.8
