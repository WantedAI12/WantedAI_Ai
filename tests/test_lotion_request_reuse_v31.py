from datetime import timedelta
from unittest.mock import patch

import pytest
from scipy.optimize import OptimizeResult

from fragrance_ai.recommender import lotion_estimation as estimation, lotion_optimizer as optimizer
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
from tests.test_lotion_v21 import fixture


def test_three_bases_share_only_one_parse_screen_and_property_batch(monkeypatch):
    catalog = IngredientCatalog.load_builtin()
    request = estimation.LotionEstimateRequest(brief='citrus woody scent')
    snapshot = {}
    with patch.object(estimation.NaturalLanguageBriefParser, 'parse', autospec=True,
                      side_effect=estimation.NaturalLanguageBriefParser.parse) as parse, \
         patch.object(estimation.CandidateSafetyScreen, 'screen', autospec=True,
                      side_effect=estimation.CandidateSafetyScreen.screen) as screen, \
         patch.object(estimation.ScientificPropertyStore, 'get_many', autospec=True,
                      side_effect=estimation.ScientificPropertyStore.get_many) as batch:
        contexts = []
        for oil in (10, 20, 30):
            inputs, scenarios, provenance = estimation.build_estimated_lotion_inputs(
                request, catalog, oil_base_percent=oil, _request_snapshot=snapshot)
            prepared, _, _ = optimizer.prepare_lotion_optimization(inputs, catalog, _request_snapshot=snapshot)
            assert prepared['transport_covered_candidate_count'] == provenance['candidate_count']
            contexts.append(inputs.simulation.parameter_context_id)
        assert parse.call_count == screen.call_count == batch.call_count == 1
        assert len(set(contexts)) == 3


@pytest.mark.parametrize('oil', [10, 20, 30])
def test_reused_and_uncached_preparation_are_identical(oil):
    catalog = IngredientCatalog.load_builtin()
    request = estimation.LotionEstimateRequest(brief='citrus woody scent')
    snapshot = {}
    estimation.build_estimated_lotion_inputs(request, catalog, _request_snapshot=snapshot)
    cached = estimation.build_estimated_lotion_inputs(request, catalog, oil_base_percent=oil, _request_snapshot=snapshot)
    fresh = estimation.build_estimated_lotion_inputs(request, catalog, oil_base_percent=oil)
    assert cached == fresh
    assert optimizer.prepare_lotion_optimization(cached[0], catalog, _request_snapshot=snapshot) == \
        optimizer.prepare_lotion_optimization(fresh[0], catalog)
    cached[2]['rejected_candidate_counts']['mutation'] = 1
    again = estimation.build_estimated_lotion_inputs(request, catalog, _request_snapshot=snapshot)
    assert 'mutation' not in again[2]['rejected_candidate_counts']


@pytest.mark.parametrize('change', ['day', 'constraints', 'catalog', 'parser', 'text'])
def test_request_snapshot_invalidation(change):
    catalog = IngredientCatalog.load_builtin()
    request = estimation.LotionEstimateRequest(brief='citrus woody scent')
    snapshot = {}
    estimation.build_estimated_lotion_inputs(request, catalog, _request_snapshot=snapshot)
    parser = None
    if change == 'day':
        snapshot['day'] -= timedelta(days=1)
    elif change == 'constraints':
        request = request.model_copy(update={'min_availability': .9})
    elif change == 'catalog':
        catalog = IngredientCatalog(list(catalog.ingredients))
    elif change == 'parser':
        parser = estimation.NaturalLanguageBriefParser(catalog)
    else:
        request = request.model_copy(update={'brief': 'rose scent'})
    with patch.object(estimation.CandidateSafetyScreen, 'screen', autospec=True,
                      side_effect=estimation.CandidateSafetyScreen.screen) as screen:
        inputs, _, _ = estimation.build_estimated_lotion_inputs(request, catalog, parser, _request_snapshot=snapshot)
        optimizer.prepare_lotion_optimization(inputs, catalog, parser, _request_snapshot=snapshot)
        assert screen.call_count == 1


@pytest.mark.parametrize('brief, error', [
    ('citrus scent concentration 2%', 'concentration'),
    ('citrus scent 100 ml', 'batch/volume'),
    ('citrus woody scent maximum 1 ingredient', 'cardinality'),
])
def test_cached_screen_does_not_bypass_language_constraints(brief, error):
    catalog = IngredientCatalog.load_builtin()
    request = estimation.LotionEstimateRequest(brief=brief)
    snapshot = {}
    inputs, _, _ = estimation.build_estimated_lotion_inputs(request, catalog, _request_snapshot=snapshot)
    with pytest.raises(ValueError, match=error):
        optimizer.prepare_lotion_optimization(inputs, catalog, _request_snapshot=snapshot)


@pytest.mark.parametrize('status', [1, 4])
def test_target_unknown_recovers_in_equivalent_coordinates_below_ratio_trigger(monkeypatch, status):
    value, catalog = fixture()
    value['search_goal'] = 'reach_target'
    value['simulation']['profile_weighting'] = 'odor_activity'
    value['simulation']['materials'][0]['odor_threshold_mg_m3'] = 1e-8
    value['simulation']['materials'][1]['odor_threshold_mg_m3'] = 1.
    original_solver, original_conditioned = optimizer.linprog, optimizer.conditioned_linprog
    calls = []
    def unknown(*args, **kwargs):
        return OptimizeResult(status=status, success=False, x=None)
    def conditioned(_solver, *args, **kwargs):
        calls.append(True)
        return original_conditioned(original_solver, *args, **kwargs)
    monkeypatch.setattr(optimizer, 'linprog', unknown)
    monkeypatch.setattr(optimizer, 'conditioned_linprog', conditioned)
    result = optimizer.optimize_lotion(LotionOptimizationRequest.model_validate(value), catalog)
    assert result['profile_target_met'] and result['score'] >= 95.
    assert 1 <= len(calls) == result['solver_conditioning_calls'] <= 2
    assert all(row['score'] >= 95. for row in result['timepoint_assessments'])
