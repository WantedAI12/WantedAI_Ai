"""Support uncertainty is not zero support; partial success is not full success."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fragrance_ai.recommender.odor_space import target_report, target_coverage, unresolved_named_odors
from fragrance_ai.recommender.odor_resolution import validate_resolution, reference_evidence
from scripts.recover_odor_support_v79 import lexical_parts, source_route, lexical_interpretations
from tests import test_odor_space_v77 as fixtures
from tests.test_odor_resolution_v78 import fixture as resolution_fixture

bank = fixtures.bank


def test_known_and_unknown_positive_details_keep_a_partial_objective(bank):
    b = replace(fixtures.brief({'lemon': 1, 'newflower': 1}), target_profile={'citrus': .5, 'floral': .5})
    report = target_report(bank, b, fixtures.rows(b))
    assert report['status'] == 'partial_source_reference' and report['searchable']
    assert report['coverage']['unresolved_requirements'] == ['reference_missing:newflower']
    assert report['targets'][0]['reference_weights'] == {'lemon': 1.}
    assert not report['coverage']['complete']


@pytest.mark.parametrize('missing', ['conflicting_concept:lemon', 'excluded_required_family:citrus',
    'exclusion_endpoint_missing:newflower', 'unresolved_excluded_odor:quasifloral'])
def test_partial_support_does_not_relax_prohibitions(missing):
    assert not target_coverage([{}], [missing])['searchable']


def test_unknown_only_phase_does_not_get_a_fabricated_target(bank):
    b = replace(fixtures.brief({'newflower': 1}), target_profile={'floral': 1})
    report = target_report(bank, b, fixtures.rows(b))
    assert not report['searchable'] and report['targets'] == [None]


def test_empty_warm_start_reports_missing_reference_not_a_phase_contradiction(bank):
    from fragrance_ai.recommender.hierarchical_perfume import target_rows
    b = replace(fixtures.brief({'newflower': 1}), target_profile={})
    report = target_report(bank, b, target_rows(b))
    assert not report['searchable']
    assert 'reference_missing:newflower' in report['unsupported']


def test_idiomatic_alias_does_not_silently_hide_literal_reference(bank):
    routes = lexical_interpretations({'skin': {'kind': 'odor', 'label_en': 'skin'}},
        {'skin': 'skin', 'skin scent': 'musky'}, {'skin': 's', 'musky': 'm'})
    assert routes['skin scent']['literal_concept'] == 'skin'
    b = fixtures.brief({'lemon': 1})
    b.expression_matches = [{'text': 'skin scent', 'polarity': 'want',
        'interpretation_alternatives': routes['skin scent']}]
    report = target_report(bank, b, fixtures.rows(b))
    assert report['searchable'] and not report['coverage']['complete']
    assert report['interpretation_alternatives']


def test_unknown_exclusion_retains_polarity():
    result = unresolved_named_odors('lemon with no quasifloral scent', [{'start': 0, 'end': 5}])
    assert result[0]['polarity'] == 'avoid'


def test_explicit_compound_separators_are_equivalent():
    for label in ('green fruity', 'green_fruity', 'green-fruity'):
        parts = lexical_parts({'id': 'greenfruity', 'label_en': label},
            {'green': 'green', 'fruity': 'fruity'}, {'green': 'g', 'fruity': 'f'})
        assert parts == ['green', 'fruity']
    assert not lexical_parts({'id': 'unknown', 'label_en': 'unknown floral'},
        {'floral': 'floral'}, {'floral': 'f'})


@pytest.mark.parametrize('n', [1, 2])
def test_sparse_source_exemplars_can_be_used_without_calling_them_measured(n):
    annotated = {k: [{'descriptor': 'iris'}] for k in ['CC', 'CCC'][:n]}
    kind, group, reason = source_route('iris', {'annotated_identities': annotated})
    assert kind == 'limited_source_model_reference' and group == annotated and reason is None
    value = resolution_fixture()
    meta = value['concept_metadata']['inferred']
    meta.update(reference_kind=kind, source_graphs_used=list(annotated), distinct_identity_groups=n,
        population_validated=False, human_error_interval=None, between_identity_dispersion=None)
    value['resolution_extension']['new_reference_metadata']['inferred'] = deepcopy(meta)
    assert validate_resolution(value)
    assert reference_evidence(meta)['limited_support']
    assert not reference_evidence(meta)['measured_reference']


def test_one_source_is_not_a_zero_width_error_interval():
    value = resolution_fixture()
    meta = value['concept_metadata']['inferred']
    meta.update(reference_kind='limited_source_model_reference', source_graphs_used=['CC'],
        distinct_identity_groups=1, population_validated=False, human_error_interval=None,
        between_identity_dispersion=0.)
    value['resolution_extension']['new_reference_metadata']['inferred'] = deepcopy(meta)
    with pytest.raises(ValueError, match='dispersion'):
        validate_resolution(value)


def test_stereochemical_alternatives_are_not_discarded_or_silently_identical():
    named = {'CC(O)F': [], 'C[C@H](O)F': []}
    kind, group, _ = source_route('example', {'named_identities': named})
    assert kind == 'named_structure_alternatives_model_reference' and group == named
    assert source_route('example', {'named_identities': {'CCO': [], 'CCN': []}})[0] is None


def test_benzoin_descriptor_not_homonymous_chemical_and_absence_not_positive_odor():
    record = {'named_identities': {'CC': []}, 'annotated_identities': {'CCC': []}}
    kind, group, _ = source_route('benzoin', record)
    assert kind == 'limited_source_model_reference' and list(group) == ['CCC']
    assert source_route('odorless', record)[0] is None


def test_partial_prediction_keeps_numeric_diagnostic_but_no_full_score(bank):
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.hierarchical_perfume import evaluate_reference
    from fragrance_ai.recommender.profile_match import assess_recipe_profiles, profile_search_assessment
    items = IngredientCatalog.load_builtin().ingredients[:2]
    session = SimpleNamespace(prefetch=lambda _: None, shape=lambda _: bank.profiles['lemon'])
    provider = SimpleNamespace(complete_reference_bank=bank, core=SimpleNamespace(sha256='fixture'),
        begin_reference_shapes=lambda: session)
    b = fixtures.brief({'lemon': 1, 'newflower': 1})
    lines = [SimpleNamespace(ingredient_id=i.ingredient_id, concentrate_percent=50.) for i in items]
    prediction = evaluate_reference(provider, b, lines, {i.ingredient_id: i for i in items}, {}, 64)
    assert prediction['score'] is None and prediction['partial_profile_score'] is not None
    assert not prediction['target_met']
    points = [{'minutes': t, 'phase': p, 'scent_profile': {'citrus': 1.},
        'full_reference_prediction': prediction if j == 0 else None}
        for j, (t,p) in enumerate(zip((0,15,60,240,480), ('opening','opening','heart','drydown','drydown')))]
    assessment = assess_recipe_profiles(b, {'citrus': 1.}, points, [.1,.1,.3,.25,.25])
    assert assessment['score'] is None and not assessment['target_met']
    assert assessment['partial_profile_score'] == prediction['partial_profile_score']
    assert assessment['nominal'] is None and assessment['temporal'] == []
    assert assessment['partial_profile_diagnostics']['nominal']['score'] is not None
    private = profile_search_assessment(b, {'citrus': 1.}, points, [.1,.1,.3,.25,.25])
    assert private['score'] == assessment['partial_profile_score'] and private['optimization_only']
    assert not private['target_met'] and private['temporal']
    assert assessment['score'] is None


def test_lotion_partial_search_retains_candidate_but_never_approves(monkeypatch):
    from tests.test_lotion_v21 import fixture
    from fragrance_ai.recommender.lotion_reference_objective import ObservedReferenceBank
    from fragrance_ai.recommender.lotion_optimizer import _optimize_lotion_transport
    from fragrance_ai.platform.lotion_inputs import LotionOptimizationRequest
    value, catalog = fixture()
    bank = ObservedReferenceBank.__new__(ObservedReferenceBank)
    bank.metadata, bank.annotation_extension, bank.resolution = {}, None, None
    bank.odor_space = SimpleNamespace(contract=lambda: {})
    bank.sha256, bank.parent_sha256 = 'b'*64, 'a'*64
    bank.endpoints = ('e1', 'e2', 'e3')
    target = np.array([.8,.15,.05])
    bank.profiles = {'citrus': np.stack([target]*2)}
    bank.background = np.stack([np.array([.1,.1,.8])]*2)
    bank.assert_current = lambda: None
    bank.targets = lambda brief, rows: ([{'profiles': bank.profiles['citrus'],
        'concepts': {'citrus': 1}, 'avoided': []} for _ in rows], ['reference_missing:example'])
    predictor = SimpleNamespace(provider=SimpleNamespace(endpoints=bank.endpoints,
        component_model_sha256='a'*64), prefetch=lambda _: None, shape=lambda _: bank.profiles['citrus'])
    value.update(brief='citrus scent', evaluation_mode='observed_reference')
    for material in value['simulation']['materials']:
        material['odor_threshold_mg_m3'] = 1e-8
    monkeypatch.setattr('fragrance_ai.recommender.lotion_reference_objective.configured_reference', lambda r,p: bank)
    monkeypatch.setattr('fragrance_ai.recommender.lotion_surrogate.attach_release_prediction', lambda actual,*a,**kw: actual)
    result = _optimize_lotion_transport(LotionOptimizationRequest.model_validate(value), catalog, _shape_predictor=predictor)
    assert result['score'] is None and result['partial_profile_score'] is not None
    assert not result['recipe'] and not result['profile_target_met']
    assert result['closest_candidate'] and result['timepoint_assessments']
    assert result['perceptual_evaluation']['physical_presence_passed']
