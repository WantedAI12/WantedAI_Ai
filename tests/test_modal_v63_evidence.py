from copy import deepcopy

import pytest

from scripts.assess_modal_v63_evidence import assess_formula_equivalence


def formula():
    return {'recipe': [], 'closest_candidate': [{'ingredient_id': 'a', 'concentrate_percent': 100.}],
        'calculated_profile_similarity': 87., 'full_profile_target_met': False, 'status': 'no_safe_match',
        'brief': {'constraints': {'target_similarity': 95}}, 'safety': {'allowed': False},
        'temporal_profile': [{'minutes': 1, 'score': 87.}], 'ingredient_temporal_profile': [{'concentration': .7}],
        'simulation_draws': 200, 'candidate_variants_evaluated': 32, 'ingredient_sets_evaluated': 15,
        'full_profile_assessment': {'search': {'full_pool_search': {'attempts': [{'score': .1}],
            'dose_refinement': {'attempts': [{'iterations': 10}]}}}},
        'perception_guidance': {'selected': {'nominal_score': .2}, 'exact_formula_evaluations': 29},
        'score_contract': {'fine_odor_expression': {'descriptors': [{'concept_id': 'rose', 'value': .3}]}}}


def test_trace_changes_are_reported_not_called_exact_response_parity():
    old = formula()
    new = deepcopy(old)
    new['candidate_variants_evaluated'] = 31
    new['ingredient_sets_evaluated'] = 14
    new['perception_guidance']['exact_formula_evaluations'] = 28
    new['full_profile_assessment']['search']['full_pool_search']['attempts'] = []
    new['perception_guidance']['selected']['nominal_score'] += 1e-14
    new['score_contract']['fine_odor_expression']['descriptors'][0]['value'] += 1e-8
    result = assess_formula_equivalence(old, new)
    assert result['required_decision_formula_and_temporal_fields_exact']
    assert not result['full_response_exact_except_release_hashes']
    assert result['search_trace_differences_retained']
    assert result['difference_counts']['search_trace'] == 4


@pytest.mark.parametrize('field', ['recipe', 'closest_candidate', 'calculated_profile_similarity',
    'full_profile_target_met', 'status', 'brief', 'safety', 'temporal_profile',
    'ingredient_temporal_profile', 'simulation_draws'])
def test_quality_or_decision_changes_cannot_be_disguised_as_diagnostics(field):
    old = formula()
    new = deepcopy(old)
    new[field] = None
    with pytest.raises(AssertionError):
        assess_formula_equivalence(old, new)


@pytest.mark.parametrize('kind', ['fine_value', 'fine_identity', 'guidance', 'new_field', 'missing_field'])
def test_material_model_output_change_or_schema_change_fails(kind):
    old = formula()
    new = deepcopy(old)
    if kind == 'fine_value':
        new['score_contract']['fine_odor_expression']['descriptors'][0]['value'] += .001
    elif kind == 'fine_identity':
        new['score_contract']['fine_odor_expression']['descriptors'][0]['concept_id'] = 'woody'
    elif kind == 'guidance':
        new['perception_guidance']['selected']['nominal_score'] += .01
    elif kind == 'new_field':
        new['unknown'] = None
    else:
        new.pop('simulation_draws')
    with pytest.raises(AssertionError):
        assess_formula_equivalence(old, new)
