"""Backend source-derived contracts; synthetic fixtures are not model benchmarks."""
from copy import deepcopy
import pytest

from fragrance_ai.platform.backend_wire_contract import normalize_formula_response
from fragrance_ai.platform import response_transport as transport
from fragrance_ai.platform.rd_evidence import EvidenceStore
from tests.test_rd_contract import data as data, client_for, reviewed_request, POLICY
from tests.test_public_evidence_v70 import public as public


@pytest.mark.parametrize('value', ['unvalidated_text_target_diagnostic', 'heuristic_only', None, .87])
def test_simulation_confidence_is_safe_for_java_numeric_projection(value):
    original = {'confidence': 'heuristic_only', 'simulation_confidence': value}
    result = normalize_formula_response(original)
    assert result['simulation_confidence'] == (value if type(value) is float else None)
    if isinstance(value, str):
        assert result['simulation_confidence_kind'] == value
    assert original['simulation_confidence'] == value
    assert normalize_formula_response(result) == result


def test_large_diagnostics_are_lossless_without_mutating_cached_results(monkeypatch):
    monkeypatch.setattr(transport, 'MAX_BYTES', 100_000)
    original = {'recipe': [{'ingredient_id': 'fixture', 'concentrate_percent': 100.}],
        'profile_target_met': False, 'score': 85.56,
        'simulation': {'status': 'diagnostic', 'materials': ['fixture_' + str(i) for i in range(20000)]},
        'regulatory': {'status': 'blocked', 'tabs': [{'status': 'blocked'}]}}
    before = deepcopy(original)
    result = transport.pack_lotion_response(original)
    assert len(transport.encoded(result)) < transport.MAX_BYTES
    assert result['recipe'] == original['recipe'] and result['score'] == 85.56
    assert result['regulatory'] == original['regulatory'] and not result['profile_target_met']
    assert result['simulation']['status'] == 'diagnostic'
    assert transport.unpack_lotion_response(result) == original == before
    result['recipe'][0]['concentrate_percent'] = 99.
    assert original == before
    with pytest.raises(ValueError):
        transport.unpack_lotion_response(result)


@pytest.mark.parametrize('corruption', ['checksum', 'size', 'data', 'marker', 'bomb'])
def test_corrupt_archives_are_rejected(monkeypatch, corruption):
    monkeypatch.setattr(transport, 'MAX_BYTES', 100_000)
    value = transport.pack_lotion_response({'simulation': ['x'] * 50000})
    archive = value[transport.FIELD]
    if corruption == 'checksum':
        archive['decoded_sha256'] = '0' * 64
    elif corruption == 'size':
        archive['decoded_bytes'] -= 1
    elif corruption == 'data':
        archive['data'] = 'invalid!!!'
    elif corruption == 'marker':
        value['simulation']['archive_field_index'] = 5
    else:
        archive['decoded_bytes'] = 128 * 1024 * 1024 + 1
    with pytest.raises(ValueError):
        transport.unpack_lotion_response(value)


def test_small_responses_remain_plain_and_uncompressible_required_fields_fail_explicitly(monkeypatch):
    value = {'score': 90., 'recipe': [], 'simulation': {'rows': []}}
    assert transport.pack_lotion_response(value) == value
    monkeypatch.setattr(transport, 'MAX_BYTES', 100)
    with pytest.raises(ValueError, match='budget'):
        transport.pack_lotion_response({'recipe': ['required'] * 100})


def test_backend_empty_diagnostic_policy_bridge_and_revision(data, public):
    catalog, _, _, assessment, _ = data
    client, calls = client_for(catalog, EvidenceStore(public_store=public[0]))
    body = {**reviewed_request(), 'evidence_policy': {}, 'diagnostic_only': True,
            'lines': assessment['lines']}
    with client:
        prepared = client.post('/v2/briefs/prepare', json=body)
        assert prepared.status_code == 200, prepared.text
        review = prepared.json()
        assert review['status'] == 'ready' and review['review_id']
        assert review['evidence_policy_context']['operational_recommendation_allowed'] is False
        result = client.post('/v2/formulas/reassess', json={**body, 'confirmed_review_id': review['review_id']})
        assert result.status_code == 200, result.text
        evaluation = result.json()
        assert evaluation['diagnostic_only'] and not evaluation['candidates']
        assert evaluation['input_snapshot']['evidence_policy_context'] == review['evidence_policy_context']
        assert calls[0]['fixed_formula_weights']
        candidate = evaluation['diagnostic_candidates'][0]
        assert not candidate['recommendation_allowed']
        revised = client.post('/v2/briefs/revise', json={'source': {'evaluation': evaluation,
            'candidate_id': candidate['candidate_id'], 'backend_version_id': 'fixture-version'},
            'instruction': 'more woody'})
        assert revised.status_code == 200, revised.text
        assert revised.json()['request']['diagnostic_only'] is True
        assert revised.json()['prepared']['evidence_policy_context'] == review['evidence_policy_context']
        assert not revised.json()['state_changed']


def test_operational_empty_policy_requires_answers_and_never_uses_diagnostic_caps(data):
    catalog, factory, *_ = data
    client, calls = client_for(catalog, factory())
    body = {**reviewed_request(), 'evidence_policy': {}}
    with client:
        review = client.post('/v2/briefs/prepare', json=body).json()
        assert review['status'] == 'needs_input' and review['review_id'] is None
        assert set(review['missing_fields']) == {'evidence_policy.' + key for key in POLICY}
        assert 'evidence_policy_context' not in review
        response = client.post('/v2/briefs/clarify', json={**body, 'prepared_result_id': review['result_id'],
            'answers': {'evidence_policy.' + key: value for key, value in POLICY.items()}})
        assert response.status_code == 200, response.text
        assert response.json()['prepared']['status'] == 'ready'
        assert response.json()['prepared']['evidence_policy'] == POLICY
        assert not calls


@pytest.mark.parametrize('value', [True, -1., '1000', None])
def test_invalid_policy_answers_cannot_be_confirmed(data, value):
    catalog, factory, *_ = data
    client, _ = client_for(catalog, factory())
    body = {**reviewed_request(), 'evidence_policy': {}}
    with client:
        review = client.post('/v2/briefs/prepare', json=body).json()
        result = client.post('/v2/briefs/clarify', json={**body, 'prepared_result_id': review['result_id'],
            'answers': {'evidence_policy.finished_batch_mass_g': value}})
        assert result.status_code == 422
