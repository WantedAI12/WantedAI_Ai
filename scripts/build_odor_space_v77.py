"""Build a source-pinned, open odor vocabulary and independent reference bank.

All existing concepts survive. Extra historical vocabulary is vocabulary, not
measured intensity. Reference profiles use observed Atlas identities only.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def fetch(url):
    with urlopen(Request(url, headers={'User-Agent': 'PerfumeryAI-source-vocabulary-audit'}), timeout=30) as response:
        raw = response.read(15_000_001)
    if len(raw) > 15_000_000 or raw.lstrip().startswith((b'<!DOCTYPE', b'<html')):
        raise ValueError('unexpected source content')
    return raw


def key(text):
    return re.sub(r'[^a-z0-9가-힣]', '', text.casefold())


def build(base, reference, core, observed_rows):
    import numpy as np
    from fragrance_ai.recommender.lotion_reference_objective import ROUTES, normalize
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    from scripts.extend_reference_annotations_v71 import annotation_routes, fit_annotation_profiles

    rows = {r['id']: deepcopy(r) for r in base['concepts']}
    aliases = {}
    for concept, row in rows.items():
        for name in [concept, *row['aliases'], *row['source_terms']]:
            aliases.setdefault(key(name), set()).add(concept)
    # Exact measurement labels stay distinguishable, including multiword labels.
    routes = {k: tuple(v) for k, v in ROUTES.items()}
    for endpoint in reference['endpoints']:
        matches = aliases.get(key(endpoint), set())
        concept = next(iter(matches)) if len(matches) == 1 else 'atlas_' + key(endpoint)
        if concept not in rows:
            projection = {family: 1. for family in SCENT_DIMENSIONS if endpoint in ROUTES.get(family, ())}
            total = sum(projection.values()) or 1.
            projection = {k: v/total for k, v in projection.items()}
            rows[concept] = {'id': concept, 'label_en': endpoint.lower(), 'aliases': [endpoint.lower()],
                'kind': 'odor', 'hierarchy_paths': [[name] for name in projection],
                'source_terms': [], 'source_uris': ['https://doi.org/10.1520/DS61-EB'],
                'coarse_projection': projection, 'projection_basis': 'named_endpoint_language_route_not_intensity',
                'structure_annotation_support': 0}
        routes.setdefault(concept, (endpoint,))
    # Preserve every family and every established route, not only the 7 old
    # hand-listed citrus/flower aliases. Coarse aliases are lexical identities.
    for family in SCENT_DIMENSIONS:
        if family not in rows:
            rows[family] = {'id': family, 'label_en': family.replace('_', ' '),
                'aliases': [family.replace('_', ' ')], 'kind': 'family', 'hierarchy_paths': [],
                'source_terms': [], 'source_uris': [], 'coarse_projection': {family: 1.},
                'projection_basis': 'public_family_contract', 'structure_annotation_support': 0}
    from fragrance_ai.recommender.brief_parser import KEYWORDS
    for family, names in KEYWORDS.items():
        rows[family]['aliases'] = sorted(set(rows[family]['aliases']) | set(names))
    from fragrance_ai.recommender.odor_descriptors import load_builtin_odor_descriptor_lexicon
    from scripts.build_odor_expression_v61 import canonical
    for descriptor in load_builtin_odor_descriptor_lexicon().descriptors:
        concept = canonical(descriptor.descriptor)
        if concept not in rows:
            owners = {c for c,r in rows.items() if set(descriptor.aliases) & set(r['aliases'])}
            if len(owners) == 1:
                concept = next(iter(owners))
        if concept not in rows:
            rows[concept] = {'id':concept,'label_en':descriptor.descriptor,
                'aliases':list(descriptor.aliases),'kind':'odor','hierarchy_paths':[],
                'source_terms':[],'source_uris':[],'coarse_projection':dict(descriptor.profile),
                'projection_basis':'existing_descriptor_language_contract_not_intensity','structure_annotation_support':0}
        else:
            rows[concept]['aliases'] = sorted(set(rows[concept]['aliases']) | set(descriptor.aliases))

    profiles = deepcopy(reference['profiles'])
    metadata = deepcopy(reference['concept_metadata'])
    all_routes = annotation_routes(core['fine_endpoints'], {'rows': rows}, SCENT_DIMENSIONS)
    fitted, annotation_metadata = fit_annotation_profiles(observed_rows, core['source_annotations'], all_routes)
    for concept, profile in fitted.items():
        if concept not in profiles:
            profiles[concept], metadata[concept] = profile, annotation_metadata[concept]
    # Fixed salience-conditioned observed means; no candidate or model output.
    observed = np.asarray([[r[h] for r in observed_rows] for h in ('applicability', 'use')], float)
    shapes = normalize(observed)
    for concept, names in routes.items():
        if concept in profiles:
            continue
        indices = [reference['endpoints'].index(n) for n in names]
        weights = np.square(shapes[0][:, indices].sum(-1))
        positive = weights > 0
        groups = {r['graph'] or r['id'] for r, ok in zip(observed_rows, positive) if ok}
        effective = float(weights.sum()**2/(weights@weights)) if weights.sum() else 0.
        if len(groups) < 3 or effective < 3:
            continue
        weights /= weights.sum()
        profiles[concept] = normalize(np.einsum('n,hnd->hd', weights, shapes)).tolist()
        metadata[concept] = {'source_endpoint_names': list(names), 'distinct_identity_groups': len(groups),
            'effective_stimuli': effective, 'reference_kind': 'observed_endpoint_conditioned_full_profile'}

    # Exact lexical aliases may select an existing reference, but never inherit
    # a parent's measured status merely because they share a family.
    alias_index = {}
    for concept, row in rows.items():
        for name in [concept, row['label_en'], *row['aliases'], *row['source_terms']]:
            alias_index.setdefault(key(name), set()).add(concept)
    bindings = {}
    for concept, row in rows.items():
        candidates = {concept} & set(profiles)
        if not candidates:
            for ref in profiles:
                if alias_index.get(key(ref)) == {concept}:
                    candidates.add(ref)
        bindings[concept] = next(iter(candidates)) if len(candidates) == 1 else None
        row['reference_key'] = bindings[concept]
        row['quantitative_status'] = 'source_reference_connected' if bindings[concept] else 'reference_missing'
        row['source_annotation_model_connected'] = concept in core['fine_endpoints']
        row['kind'] = row['kind']
    return rows, {**reference, 'profiles': profiles, 'concept_metadata': metadata}, bindings, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-output', type=Path, required=True)
    args = parser.parse_args()
    if args.profile_output.exists():
        raise ValueError('new profile output required')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output/'sources'
    sources.mkdir()
    parent = json.loads(args.parent_profile.read_text(encoding='utf8'))
    def pinned(role):
        spec = parent[role]
        path = ROOT/spec['path']
        if digest(path) != spec['sha256']:
            raise ValueError('parent hash mismatch: ' + role)
        return json.loads(path.read_text(encoding='utf8'))
    core, reference = pinned('formulation_core'), pinned('lotion_target_reference')
    old_reference_count = len(reference['profiles'])
    base_path = ROOT/'fragrance_ai/data/odor_expression_v61.json'
    base = json.loads(base_path.read_text(encoding='utf8'))
    from fragrance_ai.research.atlas_profiles import load_atlas
    observed, endpoints, atlas_source = load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    observed = [r for r in observed if r['level'] == 'high']
    if endpoints != reference['endpoints']:
        raise ValueError('Atlas identity mismatch')
    rows, reference, bindings, routes = build(base, reference, core, observed)

    repository = 'https://github.com/Odeuropa/multilingualTaxonomies'
    commit = json.loads(fetch('https://api.github.com/repos/Odeuropa/multilingualTaxonomies/commits/main'))['sha']
    source_manifest = {'repository': repository, 'commit': commit, 'license': 'CC-BY-4.0',
        'attribution': 'Menini, Paccosi, Tekiroglu and Tonelli (2022), Building a Multilingual Taxonomy of Olfactory Terms with Timestamps',
        'citation': 'https://aclanthology.org/2022.lrec-1.429/', 'files': {}}
    for name in ['LICENSE.md', 'README.md', *['taxonomies-v2/'+language+'_taxonomy.tsv' for language in ('EN','FR','DE','IT')]]:
        raw = fetch(f'https://raw.githubusercontent.com/Odeuropa/multilingualTaxonomies/{commit}/{name}')
        target = sources/Path(name).name
        target.write_bytes(raw)
        source_manifest['files'][name] = {'sha256': digest(target), 'bytes': len(raw)}
    write(sources/'manifest.json', source_manifest)
    # Retain this complete external resource as separately searchable language
    # records. Historical co-occurrences are not automatic scent equivalents.
    external = []
    for language in ('EN','FR','DE','IT'):
        path = sources/(language+'_taxonomy.tsv')
        records = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig')), delimiter='\t'))
        for record in records:
            external.append({'language': language.lower(), 'record': record})
            word = re.sub(r'_(?:adj|noun|verb|adv)$', '', record.get('word', '')).replace('_', ' ').strip()
            if not word:
                continue
            matches = [c for c,r in rows.items() if word.casefold() in [a.casefold() for a in r['aliases']]]
            if len(matches) == 1:
                rows[matches[0]].setdefault('external_source_records', []).append({'language': language.lower(), 'word': record['word']})
                continue
            concept = 'lex_' + language.lower() + '_' + hashlib.sha256(word.encode()).hexdigest()[:16]
            if concept not in rows:
                rows[concept] = {'id': concept, 'label_en': word, 'aliases': [word.casefold()],
                    'kind': 'quality' if record.get('word', '').endswith('_adj') else 'odor',
                    'language': language.lower(), 'hierarchy_paths': [], 'source_terms': [],
                    'source_uris': [repository+'/tree/'+commit+'/taxonomies-v2'],
                    'coarse_projection': {}, 'projection_basis': 'historical_lexical_context_not_measured_odor',
                    'structure_annotation_support': 0, 'contextual_only': True,
                    'reference_key': None, 'quantitative_status': 'reference_missing',
                    'source_annotation_model_connected': False}
                bindings[concept] = None
    aliases = {}
    ambiguous = {}
    for concept, row in rows.items():
        for alias in row['aliases']:
            norm = re.sub(r'\s+', ' ', alias.casefold()).strip()
            if norm in aliases and aliases[norm] != concept:
                ambiguous.setdefault(norm, set()).update((aliases[norm], concept))
            else:
                aliases[norm] = concept
    for alias in ambiguous:
        aliases.pop(alias, None)
    # Existing aliases have a stable owner. Added family/taxonomy records must
    # not make an already supported request such as "musky" ambiguous.
    for row in base['concepts']:
        for alias in row['aliases']:
            aliases[alias.casefold()] = row['id']
    components = {}
    for concept, row in rows.items():
        if row['reference_key'] is not None or row.get('contextual_only'):
            continue
        words = row['label_en'].casefold().split()
        parts = [aliases.get(w) for w in words]
        if len(parts) >= 2 and all(p and p != concept and bindings.get(p) for p in parts):
            components[concept] = list(dict.fromkeys(parts))
            row['quantitative_status'] = 'compositional_reference_connected'
            row['reference_components'] = components[concept]
    space = {'schema': 'hierarchical-odor-space/v77', 'version': 'v77',
        'concepts': list(rows.values()), 'aliases': aliases,
        'ambiguous_aliases': {k: sorted(v) for k,v in ambiguous.items()},
        'external_vocabulary': external, 'source': source_manifest, 'base_registry_sha256': digest(base_path),
        'base_source':base['source'], 'base_attribution':base['attribution'],
        'reference_bindings': bindings, 'endpoint_routes': {k:list(v) for k,v in routes.items()},
        'compositional_bindings': components,
        'coverage_scope': 'complete_union_of_declared_sources_not_every_possible_smell',
        'unknown_terms_must_not_disappear': True, 'new_measured_intensity_axes': False,
        'recipe_outcomes_used': False, 'atlas_source': atlas_source}
    write(args.output/'space.json', space)
    reference['hierarchical_extension'] = {'schema': 'hierarchical-target-compiler/v77',
        'space_path': 'space.json', 'space_sha256': digest(args.output/'space.json'),
        'all_concepts_considered': len(rows), 'reference_bindings': bindings,
        'endpoint_routes': space['endpoint_routes'], 'recipe_outcomes_used': False,
        'candidate_catalog_used': False, 'predicted_odor_profiles_used': False,
        'existing_reference_vectors_unchanged': True}
    write(args.output/'reference.json', reference)
    parent['lotion_target_reference'] = {'path': (args.output/'reference.json').resolve().relative_to(ROOT).as_posix(),
        'sha256': digest(args.output/'reference.json')}
    parent['odor_space'] = {'path': (args.output/'space.json').resolve().relative_to(ROOT).as_posix(),
        'sha256': digest(args.output/'space.json')}
    write(args.profile_output, parent)
    report = {'concepts_before': len(base['concepts']), 'concepts_after': len(rows),
        'references_before': old_reference_count, 'references_after': len(reference['profiles']),
        'reference_connected_concepts': sum(v is not None for v in bindings.values()),
        'reference_missing_concepts': [k for k,v in bindings.items() if v is None],
        'external_language_records': len(external), 'code_sha256': digest(__file__),
        'profile_path': str(args.profile_output.resolve()), 'space_sha256': parent['odor_space']['sha256'],
        'all_existing_concepts_retained': set(r['id'] for r in base['concepts']) <= set(rows),
        'parent_checkpoint_unchanged': True, 'deployed': False}
    write(args.output/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'reference_missing_concepts'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
