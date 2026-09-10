"""Synthetic algebra/identity tests; real checkpoint evidence is separate."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest, LotionSimulationRequest
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.lotion import simulate_lotion
from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION, AtlasLotionGuidance
from fragrance_ai.recommender.lotion_optimizer import optimize_lotion
from fragrance_ai.recommender.lotion_perception import make_lotion_shape_predictor
from fragrance_ai.recommender.lotion_profile_balance import prepare_balance, profile_values
from fragrance_ai.recommender.perception_runtime import assert_provider_product
from tests.test_lotion_v21 import fixture


def setup():
    request, old_catalog = fixture()
    request['brief'] = 'clean leathery scent'
    catalog = IngredientCatalog([replace(item, structure_smiles=None,
        profile={'clean': .5, 'leathery': .5}) for item in old_catalog.ingredients])
    endpoints = tuple(dict.fromkeys(name for names in ATLAS_PROJECTION.values() for name in names)) + ('UNMAPPED',)
    calls = []
    def predict(graphs, *, reference_level):
        calls.append((tuple(graphs), reference_level))
        out = np.zeros((len(graphs), len(endpoints)))
        for i, graph in enumerate(graphs):
            out[i, endpoints.index('SOAPY')] = .8 if graph == 'CCO' else .2
            out[i, endpoints.index('LEATHER')] = .1 if graph == 'CCO' else .7
            out[i, endpoints.index('UNMAPPED')] = .1
        return {'applicability': out*30., 'use': out*.2}
    model = SimpleNamespace(endpoints=endpoints, sha256='synthetic-atlas-not-real',
        fine={'vocabulary': []}, native={}, predict=predict)
    structures = {item.ingredient_id: (graph, item.cas_number)
                  for item, graph in zip(catalog.ingredients, ('CCO', 'CCN'))}
    provider = AtlasLotionGuidance(model, structures, experimental=True)
    return request, catalog, provider, calls


def test_ordinal_shapes_keep_heads_and_unknown_mass_separate():
    request, catalog, provider, calls = setup()
    predictor = make_lotion_shape_predictor(provider)
    predictor.prefetch(catalog.ingredients)
    assert predictor.reference_count == 2
    assert len(calls) == 1 and calls[0][1] == 'high'
    shape = predictor.shape(catalog.ingredients[0])
    assert shape[:, provider.endpoints.index('UNMAPPED')] == pytest.approx([.1, .1])
    assert np.sum(shape, axis=1) == pytest.approx([1., 1.])
    assert not shape.flags.writeable
    assert all('stock_dilution' not in row for row in predictor.reference_scenarios)
    rows = [{'target_profile': {'clean': .5, 'leathery': .5}, 'avoided': []}]
    shapes, targets, avoidance = prepare_balance(predictor, catalog.ingredients, rows)
    assert shapes.shape[0] == 2
    values = profile_values(shapes, targets, avoidance, np.ones((1, 2)), np.array([.5, .5]))
    assert values == pytest.approx(np.full((2, 1), .9))


def test_atlas_guidance_changes_actual_recipe_without_replacing_strict_gate():
    request, catalog, provider, _ = setup()
    request = LotionOptimizationRequest.model_validate(request)
    baseline = optimize_lotion(request, catalog, _transport_only=True)
    result = optimize_lotion(request, catalog, perception_guidance=provider)
    report = result['learned_optimization']
    assert report['recipe_changed'] and report['fresh_transport_verified']
    assert report['all_target_axes_modeled']
    assert report['profile_balance']['fresh_transport_verified']
    assert report['profile_balance']['selected_score'] > report['profile_balance']['baseline_score']+10.
    assert result['score'] == pytest.approx(baseline['score'])
    assert result['score'] >= 95.
    assert result['recipe'] != baseline['recipe']
    assert not report['acceptance_threshold_modified']
    contract = result['perception_model']
    assert contract['stock_reference_dilutions'] == []
    assert contract['nominal_stock_reference_dilution'] is None
    assert not contract['gas_to_stock_conversion_used']
    assert contract['nominal_reference_profile']['measurement'] == 'applicability'
    assert result['human_similarity_percent'] is None


def test_missing_identity_keeps_transport_mass_and_abstains():
    request, catalog, provider, _ = setup()
    structures = dict(provider.structures)
    structures.pop(catalog.ingredients[1].ingredient_id)
    provider = AtlasLotionGuidance(provider.model, structures, experimental=True)
    result = simulate_lotion(LotionSimulationRequest.model_validate(request['simulation']), catalog,
                            perception_guidance=provider)
    row = result['temporal_profile'][-1]['learned_perception']
    assert row['status'] == 'abstained_incomplete_component_coverage'
    assert row['predicted_endpoint_fraction'] is None
    assert sum(row['supported_endpoint_contribution'].values()) == pytest.approx(row['covered_transport_weight_percent']/100.)
    assert row['unknown_transport_weight_fraction'] > 0


@pytest.mark.parametrize('change', [{'cas_number': 'wrong'}, {'structure_smiles': 'CCN'}])
def test_conflicting_cas_or_graph_never_inherits_known_material_prediction(change):
    _, catalog, provider, calls = setup()
    item = replace(catalog.ingredients[0], **change)
    assert make_lotion_shape_predictor(provider).shape(item) is None
    assert not calls


def test_graph_deduplication_and_cross_request_cache():
    _, catalog, provider, calls = setup()
    item = catalog.ingredients[0]
    alias = replace(item, ingredient_id='same-graph-new-id')
    structures = {item.ingredient_id: ('CCO', item.cas_number), alias.ingredient_id: ('CCO', item.cas_number)}
    provider = AtlasLotionGuidance(provider.model, structures, experimental=True)
    first = make_lotion_shape_predictor(provider)
    first.prefetch([item, alias])
    assert len(calls) == first.calls == 1
    assert first.shape(item) is first.shape(alias)
    second = make_lotion_shape_predictor(provider)
    second.prefetch([item, alias])
    assert len(calls) == 1 and second.shared_cache_hits == 2


def test_no_cross_product_or_implicit_release_promotion():
    _, _, provider, _ = setup()
    with pytest.raises(ValueError, match='different product'):
        assert_provider_product(provider, 'perfume')
    with pytest.raises(ValueError, match='opt-in'):
        AtlasLotionGuidance(provider.model, provider.structures)


def test_affinity_roundoff_is_not_an_absolute_low_signal_cutoff():
    from fragrance_ai.recommender.lotion_learned_search import _roundoff_difference
    assert _roundoff_difference(np.nextafter(.9, 1.), .9) == 0.
    assert _roundoff_difference(1e-100, 0.) == 1e-100
    assert _roundoff_difference(.9+1e-9, .9) == pytest.approx(1e-9, rel=1e-6)


@pytest.mark.parametrize('bad', ['nan', 'negative', 'missing_head', 'zero'])
def test_invalid_model_predictions_do_not_become_valid_odor_shapes(bad):
    _, catalog, provider, _ = setup()
    original = provider.model.predict
    def predict(*args, **kwargs):
        result = original(*args, **kwargs)
        if bad == 'missing_head':
            result.pop('use')
        else:
            result['applicability'][:] = {'nan': np.nan, 'negative': -1., 'zero': 0.}[bad]
        return result
    provider.model.predict = predict
    predictor = make_lotion_shape_predictor(provider)
    if bad == 'zero':
        assert predictor.shape(catalog.ingredients[0]) is None
    else:
        with pytest.raises(ValueError, match='Atlas'):
            predictor.shape(catalog.ingredients[0])
