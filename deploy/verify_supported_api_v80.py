"""The same bounded request assertions for an installed image and HTTPS API."""
import hashlib
import json
import time


def verify_stream(text, expected):
    progress, results = [], []
    for frame in text.replace('\r\n','\n').split('\n\n'):
        event = next((line[6:].strip() for line in frame.splitlines() if line.startswith('event:')), None)
        data = '\n'.join(line[5:].lstrip() for line in frame.splitlines() if line.startswith('data:'))
        if not data:
            continue
        value = json.loads(data)
        if event == 'error':
            raise ValueError('SSE explicit error')
        if event == 'progress':
            percent = value['percent']
            if isinstance(percent,bool) or not isinstance(percent,(int,float)) or not 0<=percent<=100:
                raise ValueError('invalid SSE percentage')
            progress.append(percent)
        if event == 'result':
            results.append(value)
    if not progress or progress != sorted(progress) or progress[-1]!=100 or results != [expected]:
        raise ValueError('SSE contract mismatch')
    return {'progress':progress,'exact_result_match':True}


def run_checks(client, expected_wheel, save=lambda *_: None):
    checks = []
    def call(name, method, path, body=None):
        start = time.perf_counter()
        response = client.request(method, path, json=body) if body is not None else client.request(method, path)
        if response.status_code != 200:
            raise ValueError(name+': HTTP '+str(response.status_code))
        value = response.json()
        save(name, value)
        row = {'name': name, 'http': response.status_code, 'seconds': time.perf_counter()-start,
            'sha256': hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        checks.append(row)
        print(json.dumps(row), flush=True)
        return value
    health = call('health','GET','/health')
    assert health['wheel_sha256'] == expected_wheel
    cap = call('capabilities','GET','/v1/ai/capabilities')
    space = cap['odor_expression']['hierarchical_space']
    assert space['data_version'] == 'v80'
    count = space['generation_scope']['generation_targets']
    names = call('active_vocabulary','GET','/v1/odor-expressions?limit=100')
    assert names['total'] == count and all(r['generation_support']['enabled'] for r in names['items'])
    excluded = call('inactive_hidden','GET','/v1/odor-expressions?q=industrial')
    assert all(r['id'] != 'industrial' for r in excluded['items'])
    retained = call('inactive_audit','GET','/v1/odor-expressions?q=industrial&include_unavailable=true')
    assert any(r['id']=='industrial' and not r['generation_support']['enabled'] for r in retained['items'])
    for term in ('burnt candle','chalky','seminal, sperm-like','pine - in (pine oil)','butyric','chlorine','sausage','turnip'):
        value = call('interpret_'+term,'POST','/v1/odor-expressions/interpret',{'text':term+' scent'})
        target = value['representation']['hierarchical_target']
        assert target['searchable'], term
        if term == 'turnip':
            assert not target['coverage']['complete']
        else:
            assert target['coverage']['complete'], term
    common = {'max_risk_tier':2,'enable_registry_trace_candidates':True,'target_similarity':95,'max_ingredients':12}
    examples = []
    for term in ('burnt candle', 'sausage'):
        body = {**common, 'brief': term+' scent'}
        value = call('perfume_'+term,'POST','/v1/formulas',body)
        lines = value.get('recipe') or value.get('closest_candidate')
        assert lines and abs(sum(x['concentrate_percent'] for x in lines)-100)<.002
        assert value['full_profile_assessment']['reference_assessment']['intent']['searchable']
        examples.append({'brief':term,'full_score':value.get('calculated_profile_similarity'),
            'passed_95':value['full_profile_target_met'],'ingredient_count':len(lines)})
    partial = call('lotion_partial_turnip','POST','/v1/applications/body-lotion/design',
        {'brief':'turnip scent','max_risk_tier':2,'target_similarity':95})
    assert partial['closest_candidate'] and not partial['recipe'] and partial['score'] is None
    assert not partial['profile_target_met'] and partial['partial_profile_score'] is not None
    cached = call('cache_repeated_request','POST','/v1/formulas',{**common,'brief':'sausage scent'})
    assert cached['formula_id'] == value['formula_id']
    response = client.post('/v1/formulas/stream',json={**common,'brief':'sausage scent'})
    assert response.status_code == 200
    stream = verify_stream(response.text, cached)
    checks.append({'name':'formula_SSE','http':200,'exact_result_match':stream['exact_result_match']})
    save('formula_SSE', stream)
    blocked = call('unsupported_positive_preserved','POST','/v1/formulas',{**common,'brief':'industrial scent'})
    assert not blocked['recipe'] and not blocked['closest_candidate']
    assert not blocked['full_profile_target_met']
    peak = None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    except ImportError:
        import psutil
        peak = getattr(psutil.Process().memory_info(), 'peak_wset', None)
    return {'passed':True,'checks':checks,'generation_targets':count,'examples':examples,
        'memory_peak_bytes':peak,'scope':'software_contract_and_source_support_not_human_accuracy'}
