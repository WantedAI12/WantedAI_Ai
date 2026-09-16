"""Recover usable source support without inventing measured intensities.

Additive to a frozen V78 artifact. Sparse descriptors yield labelled exemplar
estimates; stereochemical alternatives stay visible. No recipe is used to set
the target. The deployment hold and every previous reference remain intact.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY = 'uncertainty_aware_source_support/v79'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_connectivity(graphs):
    from rdkit import Chem
    identities = set()
    for graph in graphs:
        if '.' in graph:
            return False
        mol = Chem.MolFromSmiles(graph)
        if mol is None:
            return False
        Chem.RemoveStereochemistry(mol)
        identities.add(Chem.MolToSmiles(mol, isomericSmiles=False))
    return len(identities) == 1


def source_route(key, record):
    """Descriptor support != chemical-name identity != no-odor condition."""
    if key == 'odorless':
        return None, {}, 'absence_condition_not_positive_odor'
    named = record.get('named_identities', {})
    annotated = record.get('annotated_identities', {})
    # The label benzoin is also used for a resin. Its descriptor records are
    # usable exemplars, but the homonymous chemical is not that resin.
    if key != 'benzoin' and len(named) > 1 and same_connectivity(named):
        return 'named_structure_alternatives_model_reference', named, None
    if annotated:
        return 'limited_source_model_reference', annotated, None
    return None, {}, 'no_exact_source_exemplar'


def lexical_parts(node, aliases, bindings):
    """Exact explicit compound words only, never a nearest-family fallback."""
    words = re.split(r'[\s_-]+', node['label_en'].casefold().strip())
    parts = [aliases.get(w) for w in words]
    if len(parts) >= 2 and all(p != node['id'] and bindings.get(p) for p in parts):
        return list(dict.fromkeys(parts))
    return []


def lexical_interpretations(nodes, aliases, bindings):
    """Retain idiomatic aliases but expose a conflicting literal scent name."""
    routes = {}
    for key, node in nodes.items():
        if node.get('contextual_only') or node.get('kind') != 'odor' or not bindings.get(key):
            continue
        label = node['label_en'].casefold().strip()
        if aliases.get(label) != key:
            continue
        for suffix in (' scent', ' smell', ' odor', ' odour'):
            phrase = label+suffix
            selected = aliases.get(phrase)
            if selected is not None and selected != key and bindings.get(selected):
                routes[phrase] = {'selected_concept': selected, 'literal_concept': key,
                    'selection_basis': 'existing_idiomatic_alias_pending_interpretation',
                    'literal_reference_available': True, 'requires_interpretation': True}
    return routes


def recover(space, reference, evidence, predict, model_sha, evidence_sha):
    nodes = {r['id']: r for r in space['concepts']}
    bindings = space['reference_bindings']
    extension = reference['resolution_extension']
    additions = extension['new_reference_metadata']
    before = deepcopy(reference['profiles'])
    before_metadata = deepcopy(reference['concept_metadata'])
    frozen = list(space['resolution_extension']['required_reference_ids_before_deployment'])
    if len(frozen) != 134 or len(set(frozen)) != 134:
        raise ValueError('frozen deployment requirement changed')
    decisions = []
    for key, record in sorted(evidence.items()):
        if key not in nodes:
            raise ValueError('evidence has unknown concept identity')
        if bindings.get(key):
            continue
        kind, group, reason = source_route(key, record)
        if kind is None:
            decisions.append({'concept_id': key, 'status': reason,
                'annotated_identity_count': len(record.get('annotated_identities', {}))})
            if key == 'odorless':
                nodes[key]['resolution'].update(semantic_role='absence_condition',
                    status='absence_condition_requires_product_background',
                    quantitative_reference_available=False, positive_odor_profile_applicable=False)
            continue
        graphs = sorted(group)
        outputs = predict(graphs)
        x = np.stack([outputs[h] for h in ('applicability', 'use')], axis=1).astype(float)
        if (x.shape != (len(graphs), 2, len(reference['endpoints'])) or
                not np.isfinite(x).all() or np.any(x < 0) or np.any(x.sum(-1) <= 0)):
            raise ValueError('invalid source-anchored molecular prediction')
        x /= x.sum(-1, keepdims=True)
        mean = x.mean(0)
        mean /= mean.sum(-1, keepdims=True)
        metadata = {'reference_kind': kind, 'support_policy': POLICY,
            'reference_is_model_estimate': True, 'measured_profile': False,
            'source_model_sha256': model_sha, 'source_graphs_used': graphs,
            'distinct_identity_groups': len(graphs), 'external_source_records': group,
            'external_evidence_sha256': evidence_sha, 'source_annotation_is_not_intensity': True,
            'candidate_catalog_used': False, 'recipe_outcomes_used': False,
            'population_validated': False, 'human_error_interval': None,
            'between_identity_dispersion': float(np.abs(x-mean).mean()) if len(graphs)>1 else None,
            'dispersion_is_not_human_error_interval': True,
            'scope': 'source_label_exemplars_not_all_possible_meanings_or_natural_material_composition'}
        if kind == 'named_structure_alternatives_model_reference':
            metadata.update(same_connectivity=True,
                identity_assumption='equal_source_alternatives_not_physical_mixture',
                identity_alternatives=[{'smiles': g, 'profiles': p.tolist(), 'source': group[g]}
                    for g, p in zip(graphs, x)],
                stereochemistry_specific_accuracy_validated=False)
        if key == 'benzoin':
            metadata['homonymous_chemical_identity_not_used_as_resin'] = True
        refkey = 'exemplar_v79_'+key
        reference['profiles'][refkey] = mean.tolist()
        reference['concept_metadata'][refkey] = metadata
        additions[refkey] = metadata
        bindings[key] = refkey
        nodes[key].update(reference_key=refkey, quantitative_status='source_exemplar_estimate_connected')
        nodes[key]['resolution'].update(status='source_exemplar_estimate_connected',
            semantic_role='odor_target', quantitative_reference_available=True,
            reference=refkey, reference_evidence_kind=kind)
        decisions.append({'concept_id': key, 'status': 'source_exemplar_estimate_connected',
            'source_identity_count': len(graphs), 'reference': refkey, 'kind': kind})

    # An exact plural alias is not a distinct chemical odor. Limit this
    # English inflection to the observed source label, not arbitrary stemming.
    for key, node in nodes.items():
        if bindings.get(key) or node.get('contextual_only'):
            continue
        label = node['label_en'].casefold()
        if label == 'raisins' and bindings.get(space['aliases'].get('raisin')):
            owner = space['aliases']['raisin']
            bindings[key] = bindings[owner]
            node.update(reference_key=bindings[key], quantitative_status='exact_inflection_alias_connected')
            node['resolution'].update(status='exact_inflection_alias_connected',
                reference=bindings[key], quantitative_reference_available=True,
                alias_evidence={'method': 'English_raisin_plural', 'canonical_concept': owner})
            decisions.append({'concept_id': key, 'status': 'exact_inflection_alias_connected',
                'reference': bindings[key]})
            continue
        parts = lexical_parts(node, space['aliases'], bindings)
        if not parts:
            continue
        source_refs = [bindings[p] for p in parts]
        mean = np.mean([reference['profiles'][r] for r in source_refs], axis=0)
        mean /= mean.sum(-1, keepdims=True)
        refkey = 'composition_v79_'+key
        metadata = {'reference_kind': 'explicit_lexical_reference_composition',
            'source_child_references': source_refs, 'explicit_components': parts,
            'weight_semantics': 'equal_explicit_lexical_parts_not_recipe_mass',
            'reference_is_model_estimate': any(reference['concept_metadata'].get(r, {}).get('reference_is_model_estimate', False) for r in source_refs),
            'measured_profile': False, 'candidate_catalog_used': False, 'recipe_outcomes_used': False}
        reference['profiles'][refkey] = mean.tolist()
        reference['concept_metadata'][refkey] = metadata
        additions[refkey] = metadata
        bindings[key] = refkey
        node.update(reference_key=refkey, quantitative_status='explicit_composition_connected')
        node['resolution'].update(status='explicit_composition_connected', reference=refkey,
            quantitative_reference_available=True, reference_evidence_kind=metadata['reference_kind'])
        decisions.append({'concept_id': key, 'status': 'explicit_composition_connected',
            'parts': parts, 'reference': refkey})

    # Language warm starts do not define the final full-reference objective.
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    for key, node in nodes.items():
        refkey = bindings.get(key)
        if refkey and not node['coarse_projection']:
            q = np.asarray(reference['profiles'][refkey]).mean(0)
            coarse = {axis: sum(float(q[reference['endpoints'].index(name)])
                for name in ATLAS_PROJECTION.get(axis, ()) if name in reference['endpoints'])
                for axis in SCENT_DIMENSIONS}
            total = sum(coarse.values())
            if total:
                node['coarse_projection'] = {k: v/total for k, v in coarse.items() if v>0}
                node['projection_basis'] = 'fixed_reference_warm_start_only_final_full_146_axis_evaluation'
    if any(reference['profiles'][k] != v for k, v in before.items()) or any(
            reference['concept_metadata'][k] != v for k, v in before_metadata.items()):
        raise ValueError('previous quantitative references must remain unchanged')
    ledger = [{'concept_id': k, 'label': n['label_en'], **n['resolution']}
        for k, n in nodes.items() if n['resolution']['originally_missing_v77']]
    if len(ledger) != 3595:
        raise ValueError('original audited scope changed')
    extension.update(support_policy=POLICY, counts=dict(Counter(r['status'] for r in ledger)),
        source_support_recovery={'previous_reference_count': len(before),
            'added_profiles': len(reference['profiles'])-len(before),
            'previous_references_unchanged': True, 'measured_profiles_added': 0,
            'external_evidence_sha256': evidence_sha},
        required_reference_ids_before_deployment=frozen)
    space.update(version='v79', resolution_extension={k: v for k, v in extension.items() if k != 'new_reference_metadata'})
    space['lexical_interpretations'] = lexical_interpretations(nodes, space['aliases'], bindings)
    reference['hierarchical_extension']['reference_bindings'] = bindings
    from fragrance_ai.recommender.odor_resolution import validate_resolution, deployment_readiness
    validate_resolution(reference)
    report = {'support_policy': POLICY, 'decisions': decisions,
        'previous_reference_count': len(before), 'reference_count': len(reference['profiles']),
        'measured_profiles_added': 0, 'deployment_readiness': deployment_readiness(space, reference['profiles']),
        'ledger_count': len(ledger), 'code_sha256': sha(__file__)}
    return report, ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-profile', type=Path, required=True)
    p.add_argument('--external-evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile-output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.profile_output.exists():
        raise ValueError('new output paths required; previous evidence stays frozen')
    os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'
    parent = json.loads(a.parent_profile.read_text(encoding='utf8'))
    def read(role):
        spec = parent[role]
        path = ROOT/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('parent binding mismatch: '+role)
        return json.loads(path.read_text(encoding='utf8'))
    space, reference = read('odor_space'), read('lotion_target_reference')
    from fragrance_ai.recommender.formulation_core import FormulationCore
    core = FormulationCore(ROOT/parent['formulation_core']['path'], parent['formulation_core']['sha256'])
    evidence = json.loads(a.external_evidence.read_text(encoding='utf8'))
    report, ledger = recover(space, reference, evidence, core.molecular, core.sha256, sha(a.external_evidence))
    a.output.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    write(a.output/'space.json', space)
    reference['hierarchical_extension'].update(space_path='space.json', space_sha256=sha(a.output/'space.json'))
    write(a.output/'reference.json', reference)
    for role, name in [('odor_space', 'space.json'), ('lotion_target_reference', 'reference.json')]:
        path = a.output/name
        parent[role] = {'path': path.resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    write(a.profile_output, parent)
    write(a.output/'resolution-ledger.json', ledger)
    report.update(space_sha256=sha(a.output/'space.json'), reference_sha256=sha(a.output/'reference.json'),
        model_sha256=core.sha256, default_profile_changed=False, deployed=False)
    write(a.output/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'decisions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
