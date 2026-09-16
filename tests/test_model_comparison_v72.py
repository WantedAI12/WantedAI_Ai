import gzip
import hashlib
import json

import pytest

from scripts.summarize_model_comparison_v72 import load_run, summarize


def save_run(tmp_path, name, versions=('separated','shared'), missing=False):
    root = tmp_path/name
    (root/'responses').mkdir(parents=True)
    cases = [{'id':str(i),'group':'fixed','brief':f'scent {i}'} for i in range(2)]
    protocol = {'cases':cases,'products':['perfume','body_lotion'],'target':95.,
        'versions':{v:{} for v in versions}}
    (root/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    rows = []
    for version in versions:
        for product in protocol['products']:
            for case in cases:
                if missing and version == versions[-1] and product == 'body_lotion' and case['id'] == '1':
                    continue
                score = 96. if version in ('shared','candidate') else 93.
                value = {'score':score,'profile_target_met':score >= 95.}
                request = {'brief':case['brief']}
                filename = f'{version}-{product}-{case["id"]}.json.gz'
                raw = gzip.compress(json.dumps({'request':request,'response':value}).encode(),mtime=0)
                (root/'responses'/filename).write_bytes(raw)
                rows.append({'version':version,'product':product,'case_id':case['id'],
                    'brief':case['brief'],'group':case['group'],'target':95.,'http':200,
                    'score':score,'target_met':score >= 95.,'response_file':filename,
                    'response_sha256':hashlib.sha256(raw).hexdigest()})
    (root/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    return root, rows


def test_completed_comparison_has_whole_denominators_and_no_automatic_promotion(tmp_path):
    a,_ = save_run(tmp_path,'ablation')
    b,_ = save_run(tmp_path,'improved',('candidate',))
    report = summarize([a,b])
    assert report['complete'] and report['selection_allowed']
    assert report['expected'] == report['evaluated'] == 12
    assert report['leaders_by_pass_count']['perfume'] == ['shared','candidate']
    assert report['pairs']['separated->shared:perfume']['pass_gains'] == 2
    assert report['pairs']['shared->candidate:body_lotion']['pass_losses'] == 0
    assert not report['human_accuracy_measured'] and not report['automatic_promotion_performed']


def test_incomplete_never_selects_winner_or_shrinks_denominator(tmp_path):
    a,_ = save_run(tmp_path,'ablation',missing=True)
    with pytest.raises(ValueError,match='incomplete'):
        summarize([a])
    report = summarize([a],allow_partial=True)
    assert not report['selection_allowed'] and report['leaders_by_pass_count'] == {}
    group = report['summaries']['shared:body_lotion']
    assert group['pass_rate_of_full_suite_percent'] == 50.
    assert group['expected'] == 2 and group['missing'] == 1


@pytest.mark.parametrize('mutation',['duplicate','false_pass','case','nan','threshold'])
def test_corrupt_rows_are_rejected(tmp_path,mutation):
    a,rows = save_run(tmp_path,'ablation')
    if mutation == 'duplicate':
        rows.append(rows[0])
    elif mutation == 'false_pass':
        rows[0]['target_met'] = True
    elif mutation == 'case':
        rows[0]['brief'] = 'different scent'
    elif mutation == 'nan':
        rows[0]['score'] = float('nan')
    else:
        rows[0]['target'] = 90.
    (a/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    with pytest.raises(ValueError):
        load_run(a)


def test_modified_raw_response_is_rejected(tmp_path):
    a,rows = save_run(tmp_path,'ablation')
    (a/'responses'/rows[0]['response_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        load_run(a)


def test_different_request_suite_cannot_be_merged(tmp_path):
    a,_ = save_run(tmp_path,'ablation')
    b,_ = save_run(tmp_path,'improved',('candidate',))
    protocol = json.loads((b/'protocol.json').read_text())
    protocol['products'].reverse()
    (b/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='do not match'):
        summarize([a,b])


def test_read_only_summary_never_treats_unverified_rows_as_promotable(tmp_path):
    a,_ = save_run(tmp_path,'ablation')
    report = summarize([a],verify_responses=False)
    assert report['complete'] and not report['selection_allowed']
