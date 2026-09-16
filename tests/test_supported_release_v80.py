"""Missing evidence is visible even in a user-authorized supported-scope release."""
from copy import deepcopy
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from fragrance_ai.recommender.odor_release_scope import make_scope, validate_scope
from fragrance_ai.recommender.odor_space import compile_targets, target_coverage
from scripts.prepare_supported_release_v80 import conditional_observed, leading_number, published_reference


def fixture():
    space = {'concepts': [{'id': str(i), 'kind': 'odor', 'resolution': {}} for i in range(134)],
        'reference_bindings': {str(i): 'ref' for i in range(132)},
        'resolution_extension': {'required_reference_ids_before_deployment': [str(i) for i in range(134)]}}
    space['concepts'][-1]['resolution']['semantic_role'] = 'absence_condition'
    space['release_scope'] = make_scope(space, {'ref': []})
    return space


def test_supported_scope_preserves_unavailable_audit_but_can_be_released():
    space = fixture()
    value = validate_scope(space, {'ref': []})
    assert value['ready'] and value['generation_targets'] == 132
    assert value['not_standalone_generation_targets'] == 2
    assert not space['release_scope']['concept_status']['133']['enabled']


@pytest.mark.parametrize('change', ['invent_support', 'drop_record', 'drop_requirement', 'authorization', 'missing_profile'])
def test_deployment_scope_cannot_hide_bad_or_missing_data(change):
    space = fixture()
    profiles = {'ref': []}
    if change == 'invent_support':
        space['release_scope']['concept_status']['133']['enabled'] = True
    elif change == 'drop_record':
        space['concepts'].pop()
    elif change == 'drop_requirement':
        space['resolution_extension']['required_reference_ids_before_deployment'].pop()
    elif change == 'authorization':
        space['release_scope']['authorization'] = 'silently_remove_gate'
    else:
        profiles = {}
    with pytest.raises(ValueError):
        validate_scope(space, profiles)


def test_concentrated_observed_salience_is_not_zero_support():
    rows = [{'id': str(i), 'graph': str(i), 'applicability': p, 'use': p} for i,p in
        enumerate(([.99,.01], [.001,.999], [.002,.998]))]
    profile, meta = conditional_observed(rows, ['target','other'], 'target')
    assert profile is not None and meta['effective_source_observations'] < 3
    assert meta['distinct_identity_groups'] == 3 and meta['limited_support']
    assert meta['measured_profile'] and not meta['population_validated']
    np.testing.assert_allclose(profile.sum(-1), 1)


def test_no_observation_is_not_an_invented_reference():
    rows = [{'id':'0','graph':'x','applicability':[0.,1.], 'use':[0.,1.]}]
    assert conditional_observed(rows, ['target','other'], 'target') == (None,None)
    assert leading_number('12.2 ab') == 12.2
    assert leading_number('n.d.') == 0
    with pytest.raises(ValueError):
        leading_number('not measured')


def test_partial_publication_reference_never_passes_complete_target():
    from tests.test_odor_space_v77 import brief, rows
    b = brief({'citrus':1.})
    node = {'id':'citrus','coarse_projection':{'citrus':1.}}
    bank = SimpleNamespace(odor_space=SimpleNamespace(rows={'citrus':node},sha256='x',
        value={'reference_bindings':{'citrus':'citrus'},'endpoint_routes':{}}),
        profiles={'citrus':np.array([[.5,.5],[.5,.5]])},endpoints=['a','b'],
        metadata={'citrus':{'complete_source_attribute_coverage':False,'unmapped_source_attributes':['X']}})
    targets, missing = compile_targets(bank,b,rows(b))
    coverage = target_coverage(targets,missing)
    assert coverage['searchable'] and not coverage['complete']
    assert 'source_reference_incomplete:citrus' in missing


def test_turnip_taste_scores_cannot_be_relabelled_odor_intensity():
    labels = ['Apple','Cooked Swede','Green vegetable','Sweetcorn','Savoury','Sweet','Earthy','Starchy','Tannin','Wet']
    root = ET.fromstring('<article><table-wrap id="foods-09-01719-t004"><table><tr><td>Aroma</td></tr>'+''.join(
        '<tr><td>'+label+'</td>'+''.join('<td>1</td>' for _ in range(8))+'</tr>' for label in labels)+
        '<tr><td>Taste</td></tr><tr><td>Bitter</td>'+''.join('<td>100</td>' for _ in range(8))+'</tr></table></table-wrap></article>')
    names = ['apple','spinach','corn','savoury','sweet','earthy','potato','tea','musty']
    space = {'aliases':{n:n for n in names}, 'reference_bindings':{n:n for n in names}}
    reference = {'profiles':{n:[[.2,.8],[.3,.7]] for n in names}}
    original = deepcopy(reference)
    _, meta = published_reference(root, {'sha256':'fixture'}, 'turnip', space, reference)
    assert len(meta['native_measurement_rows']) == 10
    assert meta['unmapped_source_attributes'] == ['Cooked Swede']
    assert meta['source_attribute_weight_coverage_by_condition'] == [.9]*7
    assert not meta['complete_source_attribute_coverage'] and not meta['measured_profile']
    assert reference == original
