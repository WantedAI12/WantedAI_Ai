from dataclasses import replace

import numpy as np
import pytest

from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.intent_controls import apply_intent_controls, effective_phase_target
from fragrance_ai.recommender.lotion_estimation import LotionEstimateRequest, build_estimated_lotion_inputs
from fragrance_ai.recommender.lotion_optimizer import prepare_lotion_optimization
from fragrance_ai.recommender.models import RecipeConstraints, SCENT_DIMENSIONS
from fragrance_ai.recommender.science import TemporalMixtureSimulator


def test_structured_global_and_phase_targets_survive_lotion_preparation():
    catalog = IngredientCatalog.load_builtin()
    request = LotionEstimateRequest(brief='우디 플로럴 로션', target_profile={'woody':3.,'floral':1.},
        phase_target_profiles={'heart':{'floral':1.}})
    prepared_request, _, _ = build_estimated_lotion_inputs(request,catalog)
    prepared, brief, _ = prepare_lotion_optimization(prepared_request,catalog)
    assert brief.target_profile['woody'] == .75
    assert prepared['intent']['representation']['source'] == 'explicit_structured_relative_weights'
    assert all(row['target_profile']['floral'] == 1. for row in prepared['evaluation_targets'] if row['phase'] == 'heart')
    assert prepared['effective_target'] == 95.


def test_changed_structured_target_invalidates_request_cache():
    catalog = IngredientCatalog.load_builtin()
    snapshot = {}
    request = LotionEstimateRequest(brief='우디 플로럴 로션',target_profile={'woody':1.})
    build_estimated_lotion_inputs(request,catalog,_request_snapshot=snapshot)
    updated = request.model_copy(update={'target_profile':{'floral':1.}})
    build_estimated_lotion_inputs(updated,catalog,_request_snapshot=snapshot)
    assert snapshot['brief'].target_profile['floral'] == 1.


def test_prohibition_only_phase_keeps_product_specific_target_contracts():
    catalog = IngredientCatalog.load_builtin()
    parser = NaturalLanguageBriefParser(catalog)
    brief = parser.parse('시트러스 우디', RecipeConstraints())
    brief = replace(brief,phase_target_profiles={'drydown':{}},phase_avoided_dimensions={'drydown':['citrus']})
    target = effective_phase_target(brief,'drydown')
    assert target['citrus'] == 0. and target['woody'] == 1.
    rows = TemporalMixtureSimulator.targets_by_time(brief)
    # Perfume's explicit negative-only phase has no invented positive target.
    # Lotion's retained-base preparation separately resolves the remaining
    # positive target via effective_phase_target above (V67 contract).
    np.testing.assert_array_equal(rows[-1][0], np.zeros(len(SCENT_DIMENSIONS)))
    assert 'citrus' in rows[-1][2]


def test_no_positive_target_is_invented_when_all_are_prohibited():
    brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse('우디 향')
    brief = replace(brief,phase_target_profiles={'drydown':{}},phase_avoided_dimensions={'drydown':['woody']})
    with pytest.raises(ValueError,match='every positive'):
        effective_phase_target(brief,'drydown')


def test_global_override_cannot_remove_explicit_avoidance():
    brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse('머스크 없이 우디 향')
    with pytest.raises(ValueError,match='avoidance'):
        apply_intent_controls(brief,{'target_profile':{'musky':1.}})


def test_sdk_guard_rejects_conflicting_target_before_optimization(monkeypatch):
    from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI
    monkeypatch.setenv('PERFUMERY_AI_LOCAL_PROFILE','disabled')
    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog.load_builtin()) as ai:
        with pytest.raises(ValueError,match='avoidance'):
            ai.create_recipe('머스크 없이 우디 향', RecipeConstraints(max_ingredients=12),
                             target_profile_override={'musky':1.})


def test_sdk_guard_keeps_avoidance_in_nonconflicting_structured_edit(monkeypatch):
    from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI
    monkeypatch.setenv('PERFUMERY_AI_LOCAL_PROFILE','disabled')

    class CapturedBrief(Exception):
        pass

    def inspect(catalog, brief, **kwargs):
        assert brief.avoided_dimensions == ['musky']
        assert brief.target_profile['floral'] == 1.
        assert brief.target_profile_source == 'explicit_structured_relative_weights'
        raise CapturedBrief

    with NaturalLanguagePerfumeryAI(catalog=IngredientCatalog.load_builtin()) as ai:
        monkeypatch.setattr(ai.screen,'screen',inspect)
        with pytest.raises(CapturedBrief):
            ai.create_recipe('머스크 없이 우디 향', RecipeConstraints(max_ingredients=12),
                             target_profile_override={'floral':1.})


@pytest.mark.parametrize('fields', [{'target_profile':{'woody':True}}, {'target_profile':{'not_an_axis':1.}},
    {'phase_target_profiles':{'future':{'woody':1.}}}, {'phase_target_profiles':{'heart':{'woody':0.}}}])
def test_invalid_structured_lotion_controls_are_rejected(fields):
    with pytest.raises(ValueError):
        LotionEstimateRequest(brief='우디 로션',**fields)
