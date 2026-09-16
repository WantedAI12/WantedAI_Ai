"""Read-only deployed-image QA; test histories are explicitly synthetic."""
import hashlib
import json
import math
import time


class Recorder:
    def __init__(self, client):
        self.client, self.calls = client, []

    def call(self, name, method, path, request=None, *, expected=200, stream=False):
        started = time.perf_counter()
        response = self.client.request(method, path, **({'json': request} if request is not None else {}))
        self.calls.append({'name': name, 'method': method, 'path': path, 'request': request,
            'verification_scope': 'deployed_image_app', 'http_status': response.status_code,
            'expected_http_status': expected, 'seconds': time.perf_counter()-started,
            'response_headers': dict(response.headers), 'response_body': response.text,
            'response_sha256': hashlib.sha256(response.content).hexdigest()})
        if response.status_code != expected:
            raise AssertionError(f'{name}: expected {expected}, received {response.status_code}: {response.text[:400]}')
        return response.text if stream else response.json()

    def post(self, name, path, request, **kwargs):
        return self.call(name, 'POST', path, request, **kwargs)

    def get(self, name, path):
        return self.call(name, 'GET', path)


def frames(text):
    result = []
    for frame in text.replace('\r\n', '\n').split('\n\n'):
        event = next((row[6:].strip() for row in frame.splitlines() if row.startswith('event:')), None)
        data = '\n'.join(row[5:].lstrip() for row in frame.splitlines() if row.startswith('data:'))
        if data:
            result.append({'event': event, 'data': json.loads(data)})
    assert result and not any(row['event'] == 'error' for row in result)
    return result


def handoff(qa):
    cap = qa.get('capabilities', '/v1/ai/capabilities')
    assert cap
    qa.get('catalog', '/v1/catalog')
    qa.get('evidence_status', '/v2/evidence/status')
    qa.get('lotion_references', '/v1/applications/body-lotion/references')
    vocabulary = qa.get('odor_expressions', '/v1/odor-expressions?limit=10')
    assert vocabulary['items']
    qa.post('korean_intent', '/v1/odor-expressions/interpret', {'text': '아쿠아틱 우디 향'})
    policy = {'finished_batch_mass_g': 1000., 'maximum_lead_time_days': 10, 'maximum_purchase_cost_usd': 100.}
    partial = {'request': {'formula': {'brief': 'woody scent'}}, 'evidence_policy': policy}
    prepared = qa.post('clarification_prepare', '/v2/briefs/prepare', partial)
    assert prepared['status'] == 'needs_input' and prepared['review_id'] is None
    clarified = qa.post('clarification_answer', '/v2/briefs/clarify', {**partial,
        'prepared_result_id': prepared['result_id'], 'answers': {
            'request.formula.product_category': 'eau_de_parfum', 'request.formula.target_region': 'EU',
            'request.formula.product_concentration_percent': 15., 'request.formula.max_formula_cost_per_kg': 180.}})
    assert clarified['prepared']['status'] == 'ready' and not clarified['state_changed']
    actor = {'actor_id': 'qa-system', 'display_name': 'QA fixture', 'kind': 'system'}
    history = {'formula_id': 'qa-fixture-not-a-production-record', 'formula_name': 'Synthetic QA history',
        'events': [{'event_id': 'qa-event', 'occurred_at': '2026-09-16T00:00:00Z',
            'category': 'formula', 'event_type': 'qa.recorded', 'title': 'Synthetic test event',
            'actor': actor, 'version_id': 'qa-v1', 'changes': []}],
        'versions': [{'version_id': 'qa-v1', 'version_label': 'QA V1',
            'created_at': '2026-09-16T00:00:00Z', 'actor': actor}]}
    report = qa.post('audit_report', '/v1/reports/audit', history)
    assert not report['provenance']['approval_inferred']
    assert report['provenance']['llm_calls'] == 0
    audit = frames(qa.post('audit_stream', '/v1/audit-logs/stream', history, stream=True))
    assert sum(r['event'] == 'audit.entry' for r in audit) == 1
    assert sum(r['event'] == 'version.entry' for r in audit) == 1
    assert audit[-1]['event'] == 'done' and audit[-1]['data']['data']['status'] == 'completed'
    assert [r['data']['sequence'] for r in audit] == list(range(len(audit)))
    filtered = frames(qa.post('version_filtered_stream', '/v1/audit-logs/stream', {
        **history, 'selected_version_ids': ['qa-v1'], 'period_start': '2026-09-15T00:00:00Z',
        'period_end': '2026-09-17T00:00:00Z'}, stream=True))
    assert [r['data']['data']['version_id'] for r in filtered if r['event'] == 'version.entry'] == ['qa-v1']
    streamed = frames(qa.post('report_stream', '/v1/reports/audit/stream', history, stream=True))
    rebuilt = dict(streamed[0]['data']['data'])
    for row in streamed[1:-1]:
        item = row['data']['data']
        rebuilt[item['section']] = item['value']
    rebuilt.pop('generated_at')
    report.pop('generated_at')
    assert rebuilt == report
    qa.post('invalid_formula_input', '/v1/formulas', {'brief': ''}, expected=422)
    qa.post('unknown_formula_field', '/v1/formulas', {'brief': 'woody scent', 'approve': True}, expected=422)
    return {'audit_history_kind': 'synthetic_fixture_not_persisted', 'audit_and_version_sse': True,
            'report_sse_matches_json': True, 'clarification_round_trip': True, 'invalid_inputs_rejected': True}


def diagnostic(qa):
    formula = {'brief': 'woody scent', 'max_risk_tier': 2, 'enable_registry_trace_candidates': True,
        'target_region': 'EU', 'product_category': 'eau_de_parfum', 'product_concentration_percent': 15.,
        'max_formula_cost_per_kg': 180., 'target_similarity': 90.}
    request = {'request': {'formula': formula}, 'evidence_policy': {'finished_batch_mass_g': 1000.,
        'maximum_lead_time_days': 10, 'maximum_purchase_cost_usd': 100.}, 'diagnostic_only': True}
    prepared = qa.post('diagnostic_prepare', '/v2/briefs/prepare', request)
    assert prepared['status'] == 'ready'
    evaluated = qa.post('diagnostic_evaluate', '/v2/formulas/evaluate', {
        **request, 'confirmed_review_id': prepared['review_id']})
    assert evaluated['diagnostic_only'] and not evaluated['candidates']
    assert not evaluated['manufacturing_approval'] and evaluated['diagnostic_candidates']
    candidate = evaluated['diagnostic_candidates'][0]
    payload = candidate['result']
    assert payload['confidence'] is None or type(payload['confidence']) in (float, int)
    lines = payload.get('recipe') or payload.get('closest_candidate')
    assert lines
    fixed = {**request, 'lines': [{k: row[k] for k in ('ingredient_id', 'concentrate_percent')} for row in lines]}
    fixed_review = qa.post('diagnostic_fixed_prepare', '/v2/briefs/prepare', fixed)
    reassessed = qa.post('diagnostic_reassess', '/v2/formulas/reassess', {
        **fixed, 'confirmed_review_id': fixed_review['review_id']})
    assert reassessed['diagnostic_only'] and reassessed['diagnostic_candidates']
    assert not reassessed['candidates'] and not reassessed['manufacturing_approval']
    second = reassessed['diagnostic_candidates'][0]
    def stored(value, selected, version):
        return {'evaluation': value, 'candidate_id': selected['candidate_id'], 'backend_version_id': version}
    source = stored(evaluated, candidate, 'qa-generated')
    comparison = qa.post('saved_candidate_compare', '/v2/formulas/compare', {
        'candidates': [source, stored(reassessed, second, 'qa-fixed')]})
    assert comparison['new_inference_count'] == 0 and not comparison['state_changed']
    revision = qa.post('saved_candidate_revise', '/v2/briefs/revise', {
        'source': source, 'instruction': '우디함을 높여줘'})
    assert revision['new_inference_count'] == 0 and not revision['state_changed']
    operational = {**request, 'diagnostic_only': False}
    op_review = qa.post('operational_prepare', '/v2/briefs/prepare', operational)
    blocked = qa.post('operational_evidence_gate', '/v2/formulas/evaluate', {
        **operational, 'confirmed_review_id': op_review['review_id']}, expected=422)
    assert blocked['detail']['status'] == 'abstained'
    qa.post('stale_review_rejected', '/v2/formulas/evaluate', {**request, 'confirmed_review_id': '0'*64}, expected=409)
    return {'diagnostic_evaluate_http': 200, 'diagnostic_reassess_http': 200,
        'saved_compare_http': 200, 'saved_revision_http': 200,
        'operational_missing_evidence_http': 422, 'evidence_registered_or_modified': False}


def retired(qa, index):
    from fragrance_ai.recommender.retired_blends import SOURCE_RECORDS, RetiredBlendFilter, POLICY_VERSION
    if type(index) is not int or not 0 <= index < len(SOURCE_RECORDS):
        raise ValueError('retired case index must identify one of the ten recorded results')
    record = SOURCE_RECORDS[index]
    lotion = record['product'] == 'body_lotion'
    request = {'brief': record['brief'], 'max_risk_tier': 2, 'target_similarity': 90.}
    if lotion:
        request['registry_pool'] = 'conditional_research'
    else:
        request.update(enable_registry_trace_candidates=True, require_full_profile_match=True)
    path = '/v1/applications/body-lotion/design' if lotion else '/v1/formulas'
    value = qa.post('retired_' + record['product'] + '_' + record['case_id'].replace('-', '_'), path, request)
    lines = value.get('recipe') or value.get('closest_candidate') or []
    assert lines, 'no replacement composition was returned'
    assert not RetiredBlendFilter(record['product'], record['concentration_percent']).reject_lines(lines), 'retired blend was reselected'
    assert abs(math.fsum(row['concentrate_percent'] for row in lines)-100.) < .002
    assert all(row['concentrate_percent'] > 0 for row in lines)
    score = value.get('score') if lotion else value.get('calculated_profile_similarity')
    target_met = value.get('profile_target_met') if lotion else value.get('full_profile_target_met')
    assert type(score) in (float, int) and math.isfinite(score)
    if target_met:
        assert score >= 90.-1e-8 and value['recipe']
    if not lotion:
        assert value['confidence'] is None or type(value['confidence']) in (float, int)
    duration = qa.calls[-1]['seconds']
    return {'policy': POLICY_VERSION, 'product': record['product'], 'case_id': record['case_id'],
        'retired_composition_returned': False, 'replacement_composition_returned': True,
        'score': score, 'target_met': target_met, 'recipe_returned': bool(value['recipe']),
        'replacement_material_count': len(lines), 'seconds': duration,
        'within_web_function_300_seconds': duration < 300., 'score_kind': 'model_points_not_human_accuracy'}


def rate_limit(qa):
    body = {'formula': {'brief': 'woody scent'}}
    for i in range(30):
        qa.post('prepare_' + str(i), '/v1/briefs/prepare', body)
    qa.post('rate_limit_rejected', '/v1/briefs/prepare', body, expected=429)
    assert qa.calls[-1]['response_headers']['retry-after'] == '60'
    return {'requests_accepted': 30, 'next_request_http': 429, 'recipe_inference_calls': 0}


def run_suite(client, expected_wheel, suite, case_index=None):
    qa = Recorder(client)
    result = {'passed': False, 'suite': suite, 'case_index': case_index}
    try:
        assert qa.get('health', '/health')['wheel_sha256'] == expected_wheel
        functions = {'handoff': handoff, 'diagnostic': diagnostic, 'rate_limit': rate_limit}
        result.update(retired(qa, case_index) if suite == 'retired' else functions[suite](qa))
        if result.get('within_web_function_300_seconds') is False:
            raise TimeoutError('replacement generation exceeded the deployed 300-second web budget')
        result['passed'] = True
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error)[:1000])
    result['calls'] = qa.calls
    return result
