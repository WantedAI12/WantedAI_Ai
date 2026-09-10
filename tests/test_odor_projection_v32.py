from dataclasses import replace, asdict
from copy import deepcopy
import json

import pytest

from fragrance_ai.recommender.odor_integrity import (
    assess_odor_assertions, registry_odor_rejection, legacy_registry_line_ids,
    LEGACY_ODOR_PROJECTION as OLD, EXPANDED_ODOR_PROJECTION as NEW,
    COARSE_ODOR_ALIASES, projection_maps,
)
from tests.test_odor_integrity import _positive


@pytest.mark.parametrize('term,axis', [('apple','fruity'), ('lemon','citrus'),
    ('soapy','clean'), ('smoke','smoky'), ('gardenia','white_floral')])
def test_exact_source_words_expand_only_the_versioned_model(term, axis):
    source = ('goodscents:' + term,)
    assert assess_odor_assertions(source)[0] == 'no_positive_odor_evidence'
    status, profile = assess_odor_assertions(source, NEW)
    assert status == 'supported_public_odor_descriptors'
    assert dict(profile)[axis] == 1.
    assert assess_odor_assertions(source, OLD) == assess_odor_assertions(source)


def test_synonym_repetition_cannot_reweight_the_profile():
    a = assess_odor_assertions(('goodscents:citrus','goodscents:woody'), NEW)
    b = assess_odor_assertions(('goodscents:citrus','goodscents:lemon','goodscents:orange',
        'goodscents:woody','leffingwell:lemon'), NEW)
    assert a == b


def test_detailed_mappings_and_unsupported_terms_take_precedence():
    for term in ('waxy','jasmine','herbal','vanilla','leather'):
        assert assess_odor_assertions(('goodscents:'+term,), NEW) == assess_odor_assertions(('goodscents:'+term,), OLD)
    for term in ('sulfurous','hydrocarbon','creosote'):
        assert assess_odor_assertions(('goodscents:apple','goodscents:'+term), NEW)[0] == 'unmodeled_odor_descriptors'
    for term in ('odorless','noaroma'):
        assert assess_odor_assertions(('goodscents:apple','goodscents:'+term), NEW)[0] == 'conflicting_odor_reports'


def test_neither_names_taste_nor_partial_words_are_evidence():
    for source in [('flavordb:apple',), ('names:lemon',), ('goodscents:appleunknown',),
                   ('goodscents:warm',), ('goodscents:light',)]:
        assert assess_odor_assertions(source, NEW)[0] == 'no_positive_odor_evidence'


def test_all_alias_destinations_exist_and_no_word_is_ambiguous():
    aliases = [a for values in COARSE_ODOR_ALIASES.values() for a in values]
    assert len(aliases) == len(set(aliases))
    assert set(COARSE_ODOR_ALIASES) <= set(projection_maps()[0])


def test_profile_binding_requires_correct_projection_and_unchanged_raw_lineage():
    material = _positive()
    ref = json.loads(material.odor_evidence_refs[0])
    ref['descriptor'] = 'apple'
    assertions = ('goodscents:apple',)
    status, profile = assess_odor_assertions(assertions, NEW)
    changed = replace(material, odor_projection_version=NEW, odor_assertions=assertions,
        odor_evidence_refs=(json.dumps(ref),), profile=dict(profile), odor_evidence_status=status)
    assert registry_odor_rejection(changed) is None
    assert registry_odor_rejection(replace(changed, odor_projection_version='')) is not None
    assert registry_odor_rejection(replace(changed, odor_projection_version='future-unknown')) is not None
    assert registry_odor_rejection(replace(changed, odor_evidence_refs=())) is not None
    assert registry_odor_rejection(material) is None


def test_legacy_history_stays_legacy_and_unknown_version_is_quarantined():
    material = _positive()
    line = {'ingredient_id': material.ingredient_id, 'data_source': material.data_source,
            'odor_integrity_version': material.odor_integrity_version}
    assert not legacy_registry_line_ids({'recipe':[line]})
    assert not legacy_registry_line_ids({'recipe':[{**line,'odor_projection_version':NEW}]})
    assert legacy_registry_line_ids({'recipe':[{**line,'odor_projection_version':'unknown'}]})
    with pytest.raises(ValueError):
        assess_odor_assertions(('goodscents:apple',), 'unknown')


@pytest.mark.parametrize('field,value', [('price_per_kg', 0.), ('availability', 1.),
    ('risk_tier', 0), ('odor_assertions', []), ('cas_number', None)])
def test_projection_audit_rejects_unrelated_changes(field, value):
    from scripts.audit_odor_projection_catalog import audit
    original = asdict(_positive())
    if field == 'availability':
        original[field] = .75
    before = {'ingredients':[original], 'registry_sha256':'a'*64}
    after = deepcopy(before)
    after['ingredients'][0]['odor_projection_version'] = NEW
    assert audit(before, after)['profile_changed'] == 0
    after['ingredients'][0][field] = value
    with pytest.raises(ValueError):
        audit(before, after)


def test_extension_policy_preserves_existing_positive_profiles_not_new_negative_findings():
    from scripts.build_odor_integrity_catalog import preserve_existing_positive_profiles
    from fragrance_ai.recommender.catalog import IngredientCatalog
    old = _positive()
    new = replace(old, profile={'floral':1.}, odor_projection_version=NEW, pyramid='base')
    result, count = preserve_existing_positive_profiles(IngredientCatalog([new]), IngredientCatalog([old]))
    assert count == 1 and result.ingredients[0].profile == old.profile
    assert result.ingredients[0].odor_projection_version == old.odor_projection_version
    blocked = replace(new, blocked=True, formulation_ready=False, odor_evidence_status='conflicting_odor_reports')
    result, count = preserve_existing_positive_profiles(IngredientCatalog([blocked]), IngredientCatalog([old]))
    assert count == 0 and result.ingredients[0] == blocked
    pending = replace(old, blocked=True, formulation_ready=False)
    result, count = preserve_existing_positive_profiles(IngredientCatalog([new]), IngredientCatalog([pending]))
    assert count == 0 and result.ingredients[0] == new
    with pytest.raises(ValueError):
        preserve_existing_positive_profiles(IngredientCatalog([replace(new,odor_assertions=('goodscents:apple',))]),
                                            IngredientCatalog([old]))
