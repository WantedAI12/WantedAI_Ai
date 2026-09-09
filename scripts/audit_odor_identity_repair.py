"""Verify source-identity repair changes only evidence-derived material fields."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def audit(old, new):
    before = {i['ingredient_id']:i for i in old['ingredients']}
    after = {i['ingredient_id']:i for i in new['ingredients']}
    assert before.keys() == after.keys(), 'identity inventory changed'
    assert old['registry_sha256'] == new['registry_sha256'] and old['registry_stats'] == new['registry_stats']
    allowed = {'odor_assertions','odor_evidence_refs','odor_evidence_status','profile','pyramid','blocked',
        'blocked_reason','formulation_ready','structure_properties','structure_properties_version'}
    rows = []
    for key, a in before.items():
        b = after[key]
        if a == b: continue
        fields = {f for f in a if a[f] != b[f]}
        assert fields <= allowed, (key, fields-allowed)
        assert set(a['odor_assertions']) <= set(b['odor_assertions']), 'existing assertions discarded'
        assert set(a['odor_evidence_refs']) <= set(b['odor_evidence_refs']), 'existing references discarded'
        links = [json.loads(r)['identity_resolution'] for r in b['odor_evidence_refs'] if 'identity_resolution' in json.loads(r)]
        assert links and all(link['method']=='fingerprinted_CAS_CID_unique_cross_source_registry_identity'
            and link['registry_sha256']==new['registry_sha256'] and link['identity_source_sha256'] for link in links)
        assert all(key == 'registry_'+link['registry_id'].split(':',1)[-1] for link in links)
        rows.append({'ingredient_id':key,'changed_fields':sorted(fields),
            'before_status':a['odor_evidence_status'],'after_status':b['odor_evidence_status'],
            'before_active':a['formulation_ready'],'after_active':b['formulation_ready'],
            'identity_links':links[:1]})
    return {'changed_materials':len(rows),
        'new_active':sum(not r['before_active'] and r['after_active'] for r in rows),
        'newly_blocked':sum(r['before_active'] and not r['after_active'] for r in rows),
        'unchanged_materials':len(before)-len(rows), 'prices_availability_risk_caps_unchanged':True,
        'existing_assertions_and_references_preserved':True, 'rows':rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True,type=Path)
    p.add_argument('--new',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    if args.output.exists(): p.error('new output required')
    report = audit(json.loads(gzip.decompress(args.old.read_bytes())),json.loads(gzip.decompress(args.new.read_bytes())))
    report['old_sha256'] = hashlib.sha256(args.old.read_bytes()).hexdigest()
    report['new_sha256'] = hashlib.sha256(args.new.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__ == '__main__': main()
