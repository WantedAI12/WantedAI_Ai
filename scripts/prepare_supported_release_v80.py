"""Recover remaining source support, then publish an explicit supported scope."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def paper(pid, directory):
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'+pid+'/fullTextXML'
    path = directory/(pid+'.xml')
    if not path.exists():
        with urlopen(Request(url, headers={'User-Agent': 'PerfumeryResearch/1.0'}), timeout=30) as r:
            raw = r.read(3_000_001)
        if len(raw) > 3_000_000:
            raise ValueError('oversized publication')
        root = ET.fromstring(raw)
        if root.tag != 'article' or not root.findall('.//table-wrap'):
            raise ValueError('not a full-text scientific article')
        path.write_bytes(raw)
    root = ET.fromstring(path.read_bytes())
    return root, {'article': 'https://pmc.ncbi.nlm.nih.gov/articles/'+pid+'/',
        'retrieval_url': url, 'sha256': sha(path),
        'license': ' '.join(''.join(x.itertext()) for x in root.findall('.//license')).strip()}


def table_rows(root, key):
    table = root.find(".//table-wrap[@id='"+key+"']")
    if table is None:
        raise ValueError('required source table missing')
    return [[' '.join(''.join(c.itertext()).split()) for c in row] for row in table.findall('.//tr')]


def leading_number(value):
    if value.strip().casefold() == 'n.d.':
        return 0.
    match = re.match(r'^\s*(\d+(?:\.\d+)?)\b', value)
    if not match:
        raise ValueError('unrecognized source numeric cell: '+value)
    return float(match.group(1))


def conditional_observed(rows, endpoints, endpoint):
    """Preserve all positive observations; effective n is uncertainty, not a ban."""
    raw = np.asarray([[r[h] for r in rows] for h in ('applicability', 'use')], float)
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw.sum(-1) <= 0):
        raise ValueError('invalid source observation')
    shapes = raw/raw.sum(-1, keepdims=True)
    weights = shapes[0, :, endpoints.index(endpoint)]**2
    positive = weights > 0
    if not np.any(positive):
        return None, None
    weights /= weights.sum()
    profile = np.einsum('n,hnd->hd', weights, shapes)
    profile /= profile.sum(-1, keepdims=True)
    return profile, {'positive_source_observations': int(positive.sum()),
        'distinct_identity_groups': len({r['graph'] or r['id'] for r,p in zip(rows, positive) if p}),
        'effective_source_observations': float(1/(weights@weights)),
        'source_observation_ids': [r['id'] for r,p in zip(rows, positive) if p],
        'positive_observation_weights': weights[positive].tolist(),
        'source_endpoint_names': [endpoint], 'reference_kind': 'limited_observed_endpoint_reference',
        'measured_profile': True, 'reference_is_model_estimate': False,
        'population_validated': False, 'human_error_interval': None,
        'limited_support': True, 'candidate_catalog_used': False, 'recipe_outcomes_used': False}


def published_reference(root, source, kind, space, reference):
    """An explicit, auditable bridge; native data are not re-labelled 146-axis measurements."""
    aliases, bindings = space['aliases'], space['reference_bindings']
    if kind == 'sausage':
        raw = [r for r in table_rows(root, 't0010') if len(r)==8 and r[0].isdigit()]
        if len(raw) != 24:
            raise ValueError('sausage table identity/count changed')
        items = [{'native_attribute': r[1], 'values': [leading_number(r[3]), leading_number(r[4])],
            'descriptor_terms': [v.strip().replace('cucumber like', 'cucumber') for v in r[7].split(',')],
            'source_cells': r} for r in raw]
        context = 'cooked_Sichuan_and_Cantonese_sausages_not_all_sausage_styles'
        native_measurement = 'reported_OAV_not_recipe_mass_or_measured_mixture_intensity'
    else:
        raw = table_rows(root, 'foods-09-01719-t004')
        section, items = None, []
        definitions = {'Apple': ['apple'], 'Cooked Swede': [], 'Green vegetable': ['spinach'],
            'Sweetcorn': ['corn'], 'Savoury': ['savoury'], 'Sweet': ['sweet'], 'Earthy': ['earthy'],
            'Starchy': ['potato'], 'Tannin': ['tea'], 'Wet': ['musty']}
        for r in raw:
            if len(r)==1:
                section = r[0]
            elif section == 'Aroma' and len(r)==9:
                if r[0] not in definitions:
                    raise ValueError('unknown turnip aroma label')
                items.append({'native_attribute': r[0], 'values': [leading_number(v) for v in r[1:8]],
                    'descriptor_terms': definitions[r[0]], 'source_cells': r})
        if len(items) != 10:
            raise ValueError('turnip aroma section count changed; taste/flavour must stay excluded')
        context = 'seven_batches_steamed_pureed_turnip_not_fresh_turnip'
        native_measurement = 'panel_aroma_scores_0_to_100_taste_and_retronasal_flavour_excluded'
    vectors, weights, unresolved, mapping = [], [], [], []
    for row in items:
        refs = [bindings.get(aliases.get(t)) for t in row['descriptor_terms']]
        if not refs or any(r not in reference['profiles'] for r in refs):
            unresolved.append(row['native_attribute'])
            continue
        vectors.append(np.mean([reference['profiles'][r] for r in refs], axis=0))
        weights.append(row['values'])
        mapping.append({'native_attribute': row['native_attribute'], 'reference_keys': refs,
            'mapping_basis': 'published_odor_attribute_or_table2_reference_definition',
            'within_attribute_weights': 'equal_explicit_descriptors'})
    if not vectors:
        raise ValueError('no usable publication attributes')
    w = np.asarray(weights).T
    totals = np.asarray([r['values'] for r in items]).sum(0)
    if np.any(w.sum(-1) <= 0) or np.any(totals <= 0):
        raise ValueError('empty publication condition')
    coverage = w.sum(-1)/totals
    scenarios = np.einsum('sn,nhd->shd', w/w.sum(-1, keepdims=True), np.asarray(vectors))
    profile = scenarios.mean(0)
    profile /= profile.sum(-1, keepdims=True)
    metadata = {'reference_kind': 'published_native_attribute_projection', 'measured_profile': False,
        'reference_is_model_estimate': True, 'source': source, 'native_measurement': native_measurement,
        'reference_context': context, 'native_measurement_rows': items, 'explicit_attribute_mapping': mapping,
        'unmapped_source_attributes': unresolved, 'complete_source_attribute_coverage': not unresolved,
        'source_attribute_weight_coverage_by_condition': coverage.tolist(),
        'project_authored_projection_not_measured_full_profile': True,
        'condition_profiles': scenarios.tolist(), 'default_condition_weights': 'equal_published_conditions',
        'raw_food_composition_or_recipe_not_recreated': True, 'population_validated': False,
        'human_error_interval': None, 'candidate_catalog_used': False, 'recipe_outcomes_used': False}
    return profile, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists() or a.profile_output.exists():
        raise ValueError('new output required')
    os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'
    parent = json.loads(a.parent_profile.read_text(encoding='utf8'))
    def read(role):
        p = ROOT/parent[role]['path']
        if sha(p) != parent[role]['sha256']:
            raise ValueError('parent drift: '+role)
        return json.loads(p.read_text(encoding='utf8'))
    space, reference = read('odor_space'), read('lotion_target_reference')
    original = deepcopy(reference['profiles'])
    nodes = {n['id']: n for n in space['concepts']}
    additions = reference['resolution_extension']['new_reference_metadata']
    changed = []
    def connect(key, profile, metadata):
        refkey = 'supported_v80_'+key
        reference['profiles'][refkey] = profile.tolist()
        reference['concept_metadata'][refkey] = additions[refkey] = metadata
        space['reference_bindings'][key] = refkey
        nodes[key].update(reference_key=refkey, quantitative_status='supported_reference_connected')
        nodes[key]['resolution'].update(status='supported_reference_connected', reference=refkey,
            quantitative_reference_available=True, reference_evidence_kind=metadata['reference_kind'])
        changed.append({'concept_id': key, 'reference': refkey, 'kind': metadata['reference_kind']})
    from fragrance_ai.research.atlas_profiles import load_atlas
    rows, endpoints, atlas_source = load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    rows = [r for r in rows if r['level']=='high']
    if endpoints != reference['endpoints']:
        raise ValueError('observed endpoint identity mismatch')
    for key, endpoint in {'burntcandle': 'BURNT CANDLE', 'chalky': 'CHALKY',
        'seminalspermlike': 'SEMINAL, SPERM-LIKE', 'pineinpineoil': 'TURPENTINE, PINE OIL'}.items():
        profile, metadata = conditional_observed(rows, endpoints, endpoint)
        if profile is not None:
            metadata.update(atlas_source=atlas_source, descriptor_scope=endpoint)
            connect(key, profile, metadata)
            space['endpoint_routes'][key] = [endpoint]
    from fragrance_ai.recommender.formulation_core import FormulationCore
    core = FormulationCore(ROOT/parent['formulation_core']['path'], parent['formulation_core']['sha256'])
    for key in ('butyric', 'chlorine'):
        index = core.fine_endpoints.index(key)
        graphs = sorted(g for g, indices in core.manifest['source_annotations'].items() if index in indices and '.' not in g)
        if not graphs:
            continue
        raw = core.molecular(graphs)
        x = np.stack([raw[h] for h in ('applicability', 'use')], axis=1)
        x /= x.sum(-1, keepdims=True)
        mean = x.mean(0)
        connect(key, mean, {'reference_kind': 'limited_source_model_reference', 'support_policy': 'uncertainty_aware_source_support/v79',
            'source_model_sha256': core.sha256, 'source_graphs_used': graphs, 'distinct_identity_groups': len(graphs),
            'annotation_indices': [index], 'measured_profile': False, 'reference_is_model_estimate': True,
            'population_validated': False, 'human_error_interval': None,
            'between_identity_dispersion': float(np.abs(x-mean).mean()) if len(graphs)>1 else None,
            'candidate_catalog_used': False, 'recipe_outcomes_used': False,
            'chemical_hazard_permission_not_granted_by_odor_name': True})
    a.output.mkdir(parents=True)
    sources = a.output/'sources'
    sources.mkdir()
    for key, pid in [('sausage','PMC11582465'), ('turnip','PMC7700530')]:
        root, source = paper(pid, sources)
        profile, metadata = published_reference(root, source, key, space, reference)
        connect(key, profile, metadata)
    # A cooked-sausage study does not silently account for an explicit smoke request.
    space['aliases'].pop('smoked sausage', None)
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    for row in changed:
        node = nodes[row['concept_id']]
        if not node['coarse_projection']:
            q = np.asarray(reference['profiles'][row['reference']]).mean(0)
            coarse = {k: sum(float(q[endpoints.index(n)]) for n in ATLAS_PROJECTION.get(k, ())) for k in SCENT_DIMENSIONS}
            total = sum(coarse.values())
            if total:
                node['coarse_projection'] = {k:v/total for k,v in coarse.items() if v>0}
                node['projection_basis'] = 'fixed_reference_warm_start_only_final_full_146_axis_evaluation'
    from fragrance_ai.recommender.odor_release_scope import make_scope, validate_scope
    space['release_scope'] = make_scope(space, reference['profiles'])
    for key, node in nodes.items():
        node['generation_support'] = space['release_scope']['concept_status'][key]
    extension = reference['resolution_extension']
    extension.update(counts=dict(Counter(n['resolution']['status'] for n in nodes.values() if n['resolution']['originally_missing_v77'])),
        publication_and_observation_recovery=changed)
    space.update(version='v80', resolution_extension={k:v for k,v in extension.items() if k!='new_reference_metadata'})
    assert all(reference['profiles'][k] == v for k,v in original.items())
    from fragrance_ai.recommender.odor_resolution import validate_resolution
    validate_resolution(reference)
    validate_scope(space, reference['profiles'])
    write(a.output/'space.json', space)
    reference['hierarchical_extension'].update(space_path='space.json', space_sha256=sha(a.output/'space.json'),
        reference_bindings=space['reference_bindings'])
    write(a.output/'reference.json', reference)
    for role, name in [('odor_space','space.json'), ('lotion_target_reference','reference.json')]:
        p = a.output/name
        parent[role] = {'path': p.resolve().relative_to(ROOT).as_posix(), 'sha256': sha(p)}
    write(a.profile_output, parent)
    report = {'version':'v80', 'connections':changed, 'previous_profiles_unchanged':True,
        'reference_profiles':len(reference['profiles']), 'release_scope':space['release_scope']['summary'],
        'original_134_disposition': {k:space['release_scope']['concept_status'][k]
            for k in extension['required_reference_ids_before_deployment']},
        'source_sha256':sha(__file__), 'space_sha256':sha(a.output/'space.json'),
        'reference_sha256':sha(a.output/'reference.json'), 'deployed':False}
    write(a.output/'report.json', report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
