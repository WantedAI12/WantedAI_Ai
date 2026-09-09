from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from fragrance_ai.recommender.science import MolecularProperties
from fragrance_ai.recommender.physical_evidence import enrich_properties, evidence_contract


def prop(**changes):
    base = MolecularProperties('extended', '64-17-5', 46.07, .1, 20., 1, 1, 0, 10.,
                               None, None, None, 'calculated-structure:test', '2026-09-09')
    return replace(base, **changes)


def ingredient(**changes):
    return SimpleNamespace(**{'ingredient_id':'extended','cas_number':'64-17-5',
        'structure_smiles':'CCO','active_strength_percent':100., **changes})


@pytest.fixture
def index(tmp_path):
    path = tmp_path/'evidence.json'
    value = {'schema':'physical-evidence-index/v1','human_recipe_validation':False,
        'by_structure': {'CCO': {'molecular_weight':46.07,'odor_threshold_ppm':.02,
            'unit':'ppmv','identity_basis':'exact_isomeric_graph','source_ref':'published-test-threshold'}},
        'by_cas': {'64-17-5': {'molecular_weight':46.07,'vapor_pressure_pa_25c':100.,
            'boiling_point_c':78.,'identity_basis':'exact_CAS_and_molecular_weight','source_ref':'supplied-test-property'}}}
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


def test_exact_graph_and_cas_join_fill_only_missing_fields(index):
    before = prop()
    result = enrich_properties([ingredient()], {'extended':before},index_path=index)['extended']
    assert result.odor_threshold_ppm == .02 and result.vapor_pressure_pa_25c == 100.
    assert result.boiling_point_c == 78. and ';identity-joined-evidence:' in result.source_ref
    assert before.odor_threshold_ppm is None
    present = prop(odor_threshold_ppm=.5,vapor_pressure_pa_25c=5.,boiling_point_c=50.)
    assert enrich_properties([ingredient()],{'extended':present},index_path=index)['extended'] is present


@pytest.mark.parametrize('item,properties', [
    (ingredient(active_strength_percent=10.),prop()),
    (ingredient(),prop(molecular_weight=150.)),
    (ingredient(structure_smiles='CC',cas_number='74-84-0'),prop()),
    (ingredient(structure_smiles=None,cas_number=None),prop())])
def test_diluted_mismatched_or_unmatched_material_does_not_gain_evidence(index,item,properties):
    assert enrich_properties([item],{'extended':properties},index_path=index)['extended'] is properties


@pytest.mark.parametrize('field,value', [('unit','mg/m3'),('odor_threshold_ppm',-1.),
    ('molecular_weight',None),('odor_threshold_ppm',True),('identity_basis','fuzzy_name')])
def test_bad_units_values_or_identity_fail_closed(index,field,value):
    data = json.loads(index.read_text())
    data['by_structure']['CCO'][field] = value
    index.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError):
        enrich_properties([ingredient()],{'extended':prop()},index_path=index)


def test_evidence_hash_changes_with_snapshot_and_absence_is_explicit(index,tmp_path):
    original = evidence_contract(index)
    data = json.loads(index.read_text())
    data['by_structure']['CCO']['odor_threshold_ppm'] = .003
    index.write_text(json.dumps(data,indent=2),encoding='utf-8')
    assert evidence_contract(index)['index_sha256'] != original['index_sha256']
    assert enrich_properties([ingredient()],{'extended':prop()},index_path=index)['extended'].odor_threshold_ppm == .003
    assert evidence_contract(tmp_path/'absent.json')['available'] is False


def test_real_bundled_evidence_inventory():
    contract = evidence_contract()
    assert contract['exact_structure_threshold_records'] == 268
    assert contract['exact_cas_property_records'] == 29
    assert contract['joined_values_are_not_formula_sensory_validation']
