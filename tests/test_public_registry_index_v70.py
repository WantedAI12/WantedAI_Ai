from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from fragrance_ai.platform.public_registry_index import PublicRegistryIndex, valid_cas
from fragrance_ai.platform.public_evidence import PublicEvidenceStore
from tests.test_public_evidence_v70 import public as public  # noqa: F401


@pytest.mark.parametrize('cas,valid', [('60-12-8', True), ('97-18-7', True), ('60-12-9', False),
    ('', False), (None, False), ('60-12-8; 50-00-0', False), ('60-12-8 extra', False)])
def test_exact_cas_with_check_digit(cas, valid):
    assert valid_cas(cas) is valid


def index_fixture():
    identity = {'ingredient_id': 'a', 'catalog_cas_number': '60-12-8', 'catalog_structure_smiles': 'CCO',
        'structure_sha256': hashlib.sha256(b'CCO').hexdigest(), 'canonical_cas_number_changed': False,
        'status': 'structure_corroborated_aliases_not_grade_certification',
        'matches': [{'source_ids': ['STRUCTURE'], 'cas_aliases': ['60-12-8']}]}
    index = {'framework': 'K_REACH', 'source_ids': ['K_REACH_LIST'], 'source_row_count': 40000,
        'source_scope': 'test_fixture_not_real_regulatory_evidence', 'active_material_count': 2,
        'matched_material_count': 1, 'compliance_verified': False, 'manufacturing_approval': False,
        'coverage': [{'ingredient_id': 'a', 'status': 'inventory_match'},
                     {'ingredient_id': 'b', 'status': 'no_inventory_match'}],
        'observations': [{'ingredient_id': 'a', 'cas_number': '60-12-8', 'source_id': 'K_REACH_LIST',
                          'status': 'inventory_identity_only_not_safety_or_registration_approval'}]}
    documents = {key: {'url': 'https://example.invalid/' + key, 'sha256': '0' * 64}
                 for key in ('STRUCTURE', 'K_REACH_LIST')}
    return {'material_identities': [identity], 'regulatory_indexes': [index]}, documents


def test_entire_population_preserves_unmatched_and_never_clears():
    value, documents = index_fixture()
    index = PublicRegistryIndex(value, documents)
    result = index.findings('K_REACH', [{'ingredient_id': key, 'concentrate_percent': 50.} for key in ('a', 'b')], documents)
    assert len(result['material_coverage']) == 2 and len(result['findings']) == 1
    assert not result['compliance_verified'] and not result['business_registration_verified']
    none = index.findings('K_REACH', [{'ingredient_id': 'b', 'concentrate_percent': 100.}], documents)
    assert none['status'] == 'source_list_screened_not_cleared'
    assert index.findings('EU_REACH', [{'ingredient_id': 'a', 'concentrate_percent': 100.}], documents) is None


@pytest.mark.parametrize('mode', ['duplicate', 'population', 'unknown_source', 'approval', 'bad_alias', 'unbound_structure'])
def test_index_rejects_population_provenance_and_approval_mutation(mode):
    value, documents = index_fixture()
    index = value['regulatory_indexes'][0]
    if mode == 'duplicate':
        index['coverage'][1] = deepcopy(index['coverage'][0])
    elif mode == 'population':
        index['active_material_count'] = 3
    elif mode == 'unknown_source':
        index['observations'][0]['source_id'] = 'UNKNOWN'
    elif mode == 'approval':
        index['compliance_verified'] = True
    elif mode == 'bad_alias':
        value['material_identities'][0]['matches'][0]['cas_aliases'] = ['60-12-9']
    else:
        value['material_identities'][0]['catalog_structure_smiles'] = 'CCC'
    with pytest.raises(ValueError):
        PublicRegistryIndex(value, documents)


def test_reused_material_id_cannot_inherit_different_structure_or_cas():
    value, documents = index_fixture()
    index = PublicRegistryIndex(value, documents)
    material = SimpleNamespace(ingredient_id='a', cas_number='60-12-8', structure_smiles='CCO')
    assert index.coverage_for(material)['cas_aliases'] == ['60-12-8']
    material.structure_smiles = 'CCC'
    with pytest.raises(ValueError, match='different catalog'):
        index.coverage_for(material)
    with pytest.raises(ValueError, match='recipe identity'):
        index.findings('K_REACH', [{'ingredient_id': 'a', 'cas_number': '64-17-5', 'concentrate_percent': 100.}], documents)


def test_historical_data_does_not_inherit_current_registry(public):
    _, path, _ = public
    value = json.loads(path.read_text())
    fixture, _ = index_fixture()
    index = fixture['regulatory_indexes'][0]
    index.update(source_ids=['K_REACH'], active_material_count=1, matched_material_count=1,
        coverage=[{'ingredient_id': 'phenethyl_alcohol', 'status': 'inventory_match'}],
        observations=[{'ingredient_id': 'phenethyl_alcohol', 'source_id': 'K_REACH', 'status': 'needs_review'}])
    value['regulatory_indexes'] = [index]
    path.write_text(json.dumps(value), encoding='utf-8')
    store = PublicEvidenceStore(path, hashlib.sha256(path.read_bytes()).hexdigest())
    current = store.screen([{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}],
                           category='body_lotion', concentration=1.)
    historical = store.screen([{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}],
                              category='body_lotion', concentration=1., version='old')
    assert current['frameworks'][2]['public_registry_check']['findings']
    assert 'public_registry_check' not in historical['frameworks'][2]
    # Shared raw publications across histories are hash-checked once, not N+1.
    assert len(store.members) == len({str(path) for path, _ in store.members})


def test_fda_screen_preserves_product_use_exceptions():
    from scripts.merge_official_evidence_v70 import fda_matches
    assert fda_matches({'name': 'Chloroform', 'aliases': [], 'cas_number': '67-66-3', 'structure_smiles': 'ClC(Cl)Cl'}) == [
        ('Chloroform', 'named_chemical_match_requires_rule_scope_review')]
    assert fda_matches({'name': 'not mercury', 'aliases': [], 'cas_number': '60-12-8', 'structure_smiles': 'OCCc1ccccc1'}) == []
    assert fda_matches({'name': 'mercury fixture', 'aliases': [], 'cas_number': None, 'structure_smiles': '[Hg]'}) == [
        ('Mercury compounds', 'element_class_match_requires_metal_basis_and_impurity_review')]


def test_ifra_sums_direct_stock_and_natural_contributions_by_standard():
    base = {'rule_id': 'IFRA_TEST', 'rule_name': 'fixture', 'category_limits': {
        'eau_de_parfum': {'kind': 'numeric_finished_product_limit', 'maximum_finished_product_percent': 1.3}},
        'source_url': 'https://example.invalid/fixture', 'source_sha256': '0' * 64, 'pdf_pages': [1]}
    findings = [{**base, 'ingredient_id': 'direct', 'material_active_strength_percent': 50.,
                 'contribution_coefficient_percent': 100., 'contribution_kind': 'direct_added_material'},
                {**base, 'ingredient_id': 'natural', 'material_active_strength_percent': 100.,
                 'contribution_coefficient_percent': 5., 'contribution_kind': 'natural_annex_estimate_not_batch_GCMS'}]
    lines = [{'ingredient_id': 'direct', 'concentrate_percent': 20.}, {'ingredient_id': 'natural', 'concentrate_percent': 80.}]
    result = PublicRegistryIndex._ifra_totals(findings, lines, 'eau_de_parfum', 10.)
    check, = result['checks']
    assert check['finished_product_percent'] == pytest.approx(1.4)
    assert check['source_limit_exceeded'] and check['contains_annex_estimates']
    assert not check['batch_composition_verified'] and not result['full_IFRA_conformity_verified']
    unknown = PublicRegistryIndex._ifra_totals(findings, lines, 'unknown_product', 10.)
    assert unknown['checks'][0]['limit']['maximum_finished_product_percent'] is None
    assert not unknown['source_limit_exceeded']  # unknown is not cleared or zero
    assert unknown['checks'][0]['status'] == 'notebox_or_scope_review_required'


@pytest.mark.parametrize('concentration', [0., -1., float('nan'), float('inf'), 101.])
def test_ifra_rejects_invalid_concentration(concentration):
    with pytest.raises(ValueError, match='concentration'):
        PublicRegistryIndex._ifra_totals([], [{'ingredient_id': 'a', 'concentrate_percent': 100.}], 'body_lotion', concentration)


def test_ifra_rejects_repeated_formula_lines():
    with pytest.raises(ValueError, match='unique'):
        PublicRegistryIndex._ifra_totals([], [{'ingredient_id': 'a', 'concentrate_percent': 50.}] * 2, 'body_lotion', 10.)
