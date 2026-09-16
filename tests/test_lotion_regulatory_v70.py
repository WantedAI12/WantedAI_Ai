from copy import deepcopy
from types import SimpleNamespace

import pytest

from fragrance_ai.recommender.lotion_regulatory import attach_lotion_regulatory
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.platform.rd_evidence import EvidenceAssessment, EvidenceStore
from tests.test_rd_contract import data as data  # noqa: F401


@pytest.mark.parametrize('operation', ['estimated-design-v2', 'optimize'])
@pytest.mark.parametrize('field', ['recipe', 'closest_candidate'])
def test_all_lotion_generation_paths_use_returned_dose_and_preserve_prediction(operation, field):
    catalog = IngredientCatalog.load_builtin()
    value = {field: [{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100., 'finished_product_percent': 2.}],
             'score': 96.25, 'formula_id': 'unchanged', 'profile_target_met': True}
    old = deepcopy(value)
    result = attach_lotion_regulatory(value, {'fragrance_concentration_percent': .5}, catalog, operation)
    assert value == old
    assert {k: v for k, v in result.items() if k != 'regulatory'} == old
    assert result['regulatory']['product_concentration_percent'] == 2.
    assert result['regulatory']['material_count'] == 1
    assert result['regulatory']['target_region'] == 'unspecified'
    assert [row['id'] for row in result['regulatory']['tabs'][:4]] == ['IFRA', 'EU_REACH', 'K_REACH', 'FDA']
    assert not result['regulatory']['base_formulation_screened']


@pytest.mark.parametrize('operation', ['simulate', 'trained-release'])
def test_fixed_lotion_simulation_screens_actual_input_formula(operation):
    result = attach_lotion_regulatory({'prediction': [1., 2.]}, {
        'application_context': {'fragrance_concentration_percent': .8},
        'materials': [{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}]},
        IngredientCatalog.load_builtin(), operation)
    assert result['regulatory']['subject'] == 'input_formula'
    assert result['regulatory']['product_concentration_percent'] == .8


@pytest.mark.parametrize('operation', ['prepare', 'optimize', 'estimated-design-v2'])
def test_preparation_or_failed_search_does_not_present_a_transport_seed_as_a_formula(operation):
    result = attach_lotion_regulatory({'recipe': [], 'closest_candidate': []}, {'simulation': {
        'application_context': {'fragrance_concentration_percent': .8},
        'materials': [{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}]}},
        IngredientCatalog.load_builtin(), operation)
    assert result['regulatory']['subject'] == 'no_formula'
    assert result['regulatory']['status'] == 'not_assessed'


def test_inconsistent_or_zero_lotion_dose_is_rejected():
    catalog = IngredientCatalog.load_builtin()
    lines = [{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 50., 'finished_product_percent': .5},
             {'ingredient_id': 'linalyl_acetate', 'concentrate_percent': 50., 'finished_product_percent': .25}]
    with pytest.raises(ValueError, match='inconsistent'):
        attach_lotion_regulatory({'recipe': lines}, {}, catalog, 'optimize')
    lines[0]['concentrate_percent'] = 0.
    with pytest.raises(ValueError, match='amount'):
        attach_lotion_regulatory({'recipe': lines}, {}, catalog, 'optimize')


def test_registered_approval_cannot_override_published_ifra_conflict(data):
    catalog, factory, _, request, _ = data
    store = factory()
    assert store.assess(EvidenceAssessment.model_validate(request), catalog)['gate_passed']
    store.public_store = SimpleNamespace(assert_current=lambda: None, contract=lambda: {}, screen=lambda *a, **k: {
        'frameworks': [{'id': 'IFRA', 'public_registry_check': {'formula_rule_checks': {
            'checks': [{'rule_id': 'EXPLICIT_TEST_FIXTURE', 'source_limit_exceeded': True}]}}}]})
    result = store.assess(EvidenceAssessment.model_validate(request), catalog)
    assert not result['gate_passed'] and result['status'] == 'blocked'
    assert result['blockers'][-1]['reason'] == 'published_ifra_limit_conflicts_with_recipe'
    assert not next(row for row in result['framework_checks'] if row['id'] == 'IFRA')['registered_review_passed']


def test_rule_only_source_change_is_reported_as_material_change(data):
    catalog, _, _, request, _ = data
    identifier = request['lines'][0]['ingredient_id']
    def screen(*args, version=None, **kwargs):
        return {'snapshot_version': version or 'new', 'materials': [{'ingredient_id': identifier, 'source_observations': []}],
                'frameworks': [{'id': 'IFRA', 'findings': [{'ingredient_id': identifier, 'limit': 2. if version else 1.}]}],
                'gate_passed': False}
    store = EvidenceStore(public_store=SimpleNamespace(screen=screen))
    result = store.change_impact(EvidenceAssessment.model_validate(request), catalog, 'old')
    assert result['affected_material_count'] == 1 and not result['state_changed']
    assert result['changes'][0]['regulatory_sources_changed']
    assert result['changes'][0]['previous'] == result['changes'][0]['current'] == []
