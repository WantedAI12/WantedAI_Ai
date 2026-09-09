from dataclasses import asdict, replace

import pytest

from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog
from fragrance_ai.recommender.models import RecipeConstraints
from fragrance_ai.recommender.odor_descriptors import load_builtin_odor_descriptor_lexicon, exclusive_odor_spans
from fragrance_ai.recommender.odor_integrity import (
    CONCEPT_ODOR_PROJECTION as NEW, EXPANDED_ODOR_PROJECTION as OLD,
    assess_odor_assertions, odor_assertion_representation, registry_odor_rejection,
)


@pytest.mark.parametrize('row', [r for r in load_builtin_odor_descriptor_lexicon().descriptors if r.formula_supported],
                         ids=lambda r: r.descriptor)
def test_all_declared_aliases_are_invariant_to_duplicate_evidence(row):
    base = ('goodscents:' + row.descriptor, 'goodscents:woody')
    repeated = base + tuple('leffingwell:' + a for a in row.aliases)
    assert assess_odor_assertions(base, NEW) == assess_odor_assertions(repeated, NEW)


def test_old_projection_is_frozen_and_new_one_fixes_duplicate_bias():
    base = ('goodscents:minty', 'goodscents:woody')
    repeated = base + ('goodscents:mint',)
    assert assess_odor_assertions(base, OLD) != assess_odor_assertions(repeated, OLD)
    assert assess_odor_assertions(base, NEW) == assess_odor_assertions(repeated, NEW)


@pytest.mark.parametrize('bad,status', [('odorless', 'conflicting_odor_reports'),
    ('noaroma', 'conflicting_odor_reports'), ('sulfurous', 'unmodeled_odor_descriptors')])
def test_no_relaxation_of_conflicting_or_unsupported_evidence(bad, status):
    assert assess_odor_assertions(('goodscents:minty', 'goodscents:' + bad), NEW) == (status, ())


def test_evidence_retains_unknown_words_and_sources_without_inventing_intensity():
    r = odor_assertion_representation(('goodscents:mint', 'leffingwell:minty',
                                      'goodscents:mystery', 'names:rose', 'goodscents:waxy'))
    assert r['concepts']['minty']['sources'] == ['goodscents', 'leffingwell']
    assert set(r['concepts']) == {'minty', 'waxy'}
    assert r['unprojected_assertions'] == ['goodscents:mystery']
    assert r['ignored_nonodor_assertions'] == ['names:rose']
    assert not r['intensity_measured']


def parse(text):
    return NaturalLanguageBriefParser(IngredientCatalog([])).parse(
        text, RecipeConstraints(enable_semantic_ontology=False))


@pytest.mark.parametrize('texts', [('green fruity scent', 'green-fruity scent', '풋과일 향'),
                                 ('tropical scent', '열대과일 향', '열대 과일 향')])
def test_compound_aliases_have_one_consistent_projection(texts):
    profiles = [parse(t).target_profile for t in texts]
    assert profiles[0] == profiles[1] == profiles[2]


def test_compounds_do_not_swallow_independent_mentions():
    matches = {'green': [(0, 5), (17, 22)], 'green fruity': [(0, 12)], 'fruity': [(6, 12)]}
    chosen = exclusive_odor_spans(matches, {'green fruity'})
    assert chosen == {'green': [(17, 22)], 'green fruity': [(0, 12)], 'fruity': []}
    assert parse('green fruity and woody scent').target_profile['woody'] > 0


@pytest.mark.parametrize('text', ['rose without floral', 'jasmine without floral'])
def test_exclusion_cannot_be_reintroduced_by_parent_category(text):
    brief = parse(text)
    assert brief.target_profile['floral'] == 0
    assert 'floral' in brief.avoided_dimensions


def test_negative_compound_does_not_forbid_an_independent_generic_family():
    brief = parse('fruity scent without green fruity')
    assert brief.target_profile['fruity'] == 1
    assert brief.avoided_descriptors == ['green_fruity']
    assert 'fruity' not in brief.avoided_dimensions


def test_catalog_reprojection_preserves_every_nonprofile_field_and_observation():
    from tests.test_odor_integrity import _positive
    from scripts.build_concept_catalog_v39 import reproject
    old = _positive()
    observed = replace(old, ingredient_id='observed_test', name='Observed fixture', data_source='odor-observed:test')
    blocked = replace(old, ingredient_id='registry_blocked', name='Blocked fixture', blocked=True, formulation_ready=False)
    new, details = reproject(IngredientCatalog([old, observed, blocked]))
    projected = new.ingredients[0]
    assert registry_odor_rejection(projected) is None
    assert projected.odor_projection_version == NEW
    assert new.ingredients[1:] == [observed, blocked]
    assert len(details) == 1
    assert {k: v for k, v in asdict(projected).items() if k not in ('profile', 'odor_projection_version')} == {
        k: v for k, v in asdict(old).items() if k not in ('profile', 'odor_projection_version')}
