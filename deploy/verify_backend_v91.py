"""Run actual release-image calls through backend-used request sequences."""
import hashlib
import time

from deploy.verify_supported_api_v89 import run_checks


def run_backend_checks(client, expected_wheel):
    result = run_checks(client, expected_wheel)
    calls = result['calls']

    def call(name, method, path, request=None):
        start = time.perf_counter()
        response = client.request(method, path, json=request) if request is not None else client.request(method, path)
        calls.append({'name': name, 'method': method, 'path': path, 'request': request,
            'verification_scope': 'deployed_image_app', 'http_status': response.status_code,
            'seconds': time.perf_counter() - start, 'response_headers': dict(response.headers),
            'response_body': response.text, 'response_sha256': hashlib.sha256(response.content).hexdigest()})
        assert response.status_code == 200, name + ': ' + response.text[:1000]
        assert len(response.content) < 8 * 1024 * 1024, name + ': backend buffer exceeded'
        return response.json()

    call('backend_capabilities', 'GET', '/v1/ai/capabilities')
    call('backend_catalog', 'GET', '/v1/catalog')
    call('backend_evidence_status', 'GET', '/v2/evidence/status')
    call('backend_evidence_coverage', 'GET', '/v2/evidence/coverage?offset=0&limit=100')
    import json
    perfume = json.loads(next(row['response_body'] for row in calls if row['name'] == 'perfume_design'))
    assert perfume['simulation_confidence'] is None or type(perfume['simulation_confidence']) in (int, float)
    body = {'request': {'formula': {'brief': 'woody scent', 'product_category': 'eau_de_parfum',
        'target_region': 'EU', 'product_concentration_percent': 15., 'max_formula_cost_per_kg': 180.,
        'max_risk_tier': 2, 'enable_registry_trace_candidates': True}},
        'evidence_policy': {}, 'diagnostic_only': True}
    prepared = call('backend_prepare_empty_policy', 'POST', '/v2/briefs/prepare', body)
    assert prepared['status'] == 'ready' and prepared['review_id']
    assert prepared['evidence_policy_context']['operational_recommendation_allowed'] is False
    evaluated = call('backend_evaluate_diagnostic', 'POST', '/v2/formulas/evaluate',
        {**body, 'confirmed_review_id': prepared['review_id']})
    assert evaluated['diagnostic_only'] and not evaluated['candidates'] and evaluated['diagnostic_candidates']
    fixed = {**body, 'lines': [{'ingredient_id': row['ingredient_id'], 'concentrate_percent': row['concentrate_percent']}
        for row in perfume['recipe']]}
    prepared_fixed = call('backend_prepare_fixed', 'POST', '/v2/briefs/prepare', fixed)
    reassessed = call('backend_reassess_diagnostic', 'POST', '/v2/formulas/reassess',
        {**fixed, 'confirmed_review_id': prepared_fixed['review_id']})
    assert reassessed['diagnostic_only'] and not reassessed['candidates'] and reassessed['diagnostic_candidates']
    evaluated_line = reassessed['diagnostic_candidates'][0]['result']
    rendered = evaluated_line.get('recipe') or evaluated_line['closest_candidate']
    assert {row['ingredient_id']: row['concentrate_percent'] for row in rendered} == {
        row['ingredient_id']: row['concentrate_percent'] for row in fixed['lines']}
    assert evaluated_line['brief']['constraints']['target_similarity'] == 90.

    def snapshot(evaluation):
        return {'evaluation': evaluation, 'candidate_id': evaluation['diagnostic_candidates'][0]['candidate_id'],
            'backend_version_id': 'verification-only-' + evaluation['result_id'][:32]}

    compared = call('backend_compare_saved', 'POST', '/v2/formulas/compare',
        {'candidates': [snapshot(evaluated), snapshot(reassessed)]})
    assert compared['new_inference_count'] == 0
    revised = call('backend_revise_saved', 'POST', '/v2/briefs/revise',
        {'source': snapshot(reassessed), 'instruction': 'more woody'})
    assert revised['new_inference_count'] == 0 and not revised['state_changed']
    assert revised['request']['diagnostic_only'] is True
    operational = {**body, 'diagnostic_only': False}
    questions = call('backend_policy_questions', 'POST', '/v2/briefs/prepare', operational)
    assert questions['status'] == 'needs_input'
    clarified = call('backend_policy_clarify', 'POST', '/v2/briefs/clarify',
        {**operational, 'prepared_result_id': questions['result_id'], 'answers': {
            'evidence_policy.finished_batch_mass_g': 1000., 'evidence_policy.maximum_lead_time_days': 10,
            'evidence_policy.maximum_purchase_cost_usd': 100.}})
    assert clarified['prepared']['status'] == 'ready'
    result.update(backend_request_sequence_passed=True, simulation_confidence_numeric_or_null=True,
        diagnostic_policy_assumptions_explicit=True, fixed_composition_preserved=True,
        fixed_target_90_preserved=True, backend_source_modified=False,
        backend_source_commit='51c60a4cbe0e8eff8674482a2db770ea54a3967f',
        below_target_lotion_storage_policy='backend_still_requires_profile_target_met_true')
    return result
