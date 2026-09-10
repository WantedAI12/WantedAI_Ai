"""Audit a versioned semantic projection change without treating it as new measurements."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def audit(old, new):
    before = {r['ingredient_id']: r for r in old['ingredients']}
    after = {r['ingredient_id']: r for r in new['ingredients']}
    if (before.keys() != after.keys() or old['registry_sha256'] != new['registry_sha256']
            or old.get('registry_stats') != new.get('registry_stats')):
        raise ValueError('registry identity inventory changed')
    allowed = {'odor_projection_version', 'profile', 'pyramid', 'odor_evidence_status',
        'formulation_ready', 'blocked', 'blocked_reason', 'structure_properties',
        'structure_properties_version', 'registry_structural_alerts'}
    changes = []
    for key, original in before.items():
        a = {**original, 'odor_projection_version': original.get('odor_projection_version', '')}
        b = after[key]
        if a.keys() != b.keys():
            raise ValueError('material schema changed outside the projection field')
        fields = {f for f in a.keys() | b.keys() if a.get(f) != b.get(f)}
        if fields - allowed:
            raise ValueError(f'non-projection fields changed: {key}: {sorted(fields-allowed)}')
        if not key.startswith('registry_') and fields:
            raise ValueError('curated core profile changed')
        old_active = bool(a['formulation_ready'] and not a['blocked'])
        new_active = bool(b['formulation_ready'] and not b['blocked'])
        if key.startswith('registry_') and b['odor_projection_version'] != 'explicit-odor-projection-2':
            if not (old_active and new_active and b['odor_projection_version'] == a['odor_projection_version']
                    and b['profile'] == a['profile'] and b['pyramid'] == a['pyramid']):
                raise ValueError('expanded profiles are not version bound or validly preserved')
        if fields - {'odor_projection_version'}:
            changes.append({'ingredient_id':key, 'changed_fields': sorted(fields),
                'profile_changed': a['profile'] != b['profile'],
                'old_active': old_active, 'new_active': new_active,
                'single_axis_profile': sum(v > 0 for v in b['profile'].values()) == 1,
                'old_status':a['odor_evidence_status'], 'new_status':b['odor_evidence_status']})
    return {'scope':'same_source_assertions_versioned_semantic_projection_not_new_odor_measurements',
        'rows':changes, 'catalog_rows':len(before),
        'new_active':sum(not r['old_active'] and r['new_active'] for r in changes),
        'newly_blocked':sum(r['old_active'] and not r['new_active'] for r in changes),
        'profile_changed':sum(r['profile_changed'] for r in changes),
        'new_single_axis_profiles':sum(not r['old_active'] and r['new_active'] and r['single_axis_profile'] for r in changes),
        'assertions_refs_identity_price_availability_risk_caps_preserved':True,
        'single_axis_means_measured_purity':False, 'human_similarity_measured':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output path')
    raw_old, raw_new = args.old.read_bytes(), args.new.read_bytes()
    report = audit(json.loads(gzip.decompress(raw_old)), json.loads(gzip.decompress(raw_new)))
    report.update(old_sha256=hashlib.sha256(raw_old).hexdigest(), new_sha256=hashlib.sha256(raw_new).hexdigest())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
