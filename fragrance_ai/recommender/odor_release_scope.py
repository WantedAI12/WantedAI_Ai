"""Explicit supported-scope release policy; never erase unavailable source terms."""
from collections import Counter

VERSION = 'supported-odor-release-scope/v80'
DECISION = 'user_authorized_deploy_supported_scope_disable_unavailable_generation'


def make_scope(space, profiles):
    states = {}
    bindings = space['reference_bindings']
    for node in space['concepts']:
        key = node['id']
        ref = bindings.get(key)
        if ref is not None and ref not in profiles:
            raise ValueError('bound reference is missing; do not hide corrupt data')
        components = space.get('compositional_bindings', {}).get(key, [])
        if ref or components and all(bindings.get(c) in profiles for c in components):
            status, enabled = 'reference_connected', True
        elif node.get('resolution', {}).get('semantic_role') == 'absence_condition':
            status, enabled = 'separate_absence_control_not_positive_odor', False
        elif node.get('kind') == 'quality':
            status, enabled = 'modifier_requires_supported_odor_anchor', False
        else:
            status, enabled = 'inactive_no_quantitative_generation_reference', False
        states[key] = {'enabled': enabled, 'status': status, 'reference': ref,
            'source_record_retained': True,
            'impossible_in_principle_claimed': False}
    return {'schema': VERSION, 'authorization': DECISION, 'concept_status': states,
        'summary': {'total_inventory': len(states), 'generation_targets': sum(v['enabled'] for v in states.values()),
            'not_standalone_generation_targets': sum(not v['enabled'] for v in states.values()),
            'status_counts': dict(Counter(v['status'] for v in states.values()))},
        'missing_positive_policy': 'partial_candidate_when_other_targets_are_resolved',
        'missing_exclusion_policy': 'block_not_silently_ignore',
        'actual_human_similarity_claimed': False}


def validate_scope(space, profiles):
    value = space.get('release_scope')
    required = space.get('resolution_extension', {}).get('required_reference_ids_before_deployment')
    ids = {n['id'] for n in space['concepts']}
    if (not isinstance(required, list) or len(required)!=134 or len(set(required))!=134
            or not set(required) <= ids):
        raise ValueError('original reference audit IDs must remain present')
    expected = make_scope(space, profiles)
    if value != expected:
        raise ValueError('inconsistent or unauthorized supported release scope')
    if not expected['summary']['generation_targets']:
        raise ValueError('no supported generation target')
    return {'ready': True, **expected['summary'], 'original_134_ids_preserved': True,
        'inactive_records_not_advertised_as_supported': True,
        'authorization': DECISION}
