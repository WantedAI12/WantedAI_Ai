"""Join existing source odor annotations to measured Atlas rows, not recipes.

All supported ontology terms are considered with the same minimum of three
independent measured identities. Existing reference vectors are preserved.
This adds annotation-conditioned full profiles, NOT new measured endpoints.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotation_routes(endpoints, registry, families):
    def key(text):
        return re.sub(r'[^a-z0-9]', '', text.casefold())
    aliases = defaultdict(set)
    for name, row in registry['rows'].items():
        if row['kind'] != 'odor' and name not in families:
            continue
        for term in [name, *row['aliases'], *row['source_terms']]:
            if term.isascii():
                aliases[key(term)].add(name)
    routes = defaultdict(set)
    for index, term in enumerate(endpoints):
        matches = aliases.get(key(term), set())
        if len(matches) != 1:
            continue
        concept = next(iter(matches))
        routes[concept].add(index)
        projection = registry['rows'][concept]['coarse_projection']
        strongest = max(projection.values(), default=0.)
        for family, value in projection.items():
            # Only a dominant existing language route may label a broad class.
            # Tiny secondary coefficients cannot turn everything into amber.
            if family in families and value >= .5 and value == strongest:
                routes[family].add(index)
    return {k: sorted(v) for k, v in routes.items()}


def fit_annotation_profiles(rows, annotations, routes):
    by_graph = defaultdict(list)
    for row in rows:
        if row['graph']:
            by_graph[row['graph']].append(row)
    # Repeated entries of the same chemical get one identity-level vote.
    normalized = {}
    for graph, records in by_graph.items():
        heads = np.asarray([[r[head] for r in records] for head in ('applicability', 'use')], float)
        totals = heads.sum(-1, keepdims=True)
        if not np.isfinite(heads).all() or np.any(heads < 0) or np.any(totals <= 0):
            raise ValueError('invalid observed profile')
        normalized[graph] = (heads/totals).mean(1)
    profiles, metadata = {}, {}
    for concept, indices in sorted(routes.items()):
        identities = sorted(graph for graph in normalized if set(annotations.get(graph, ())) & set(indices))
        if len(identities) < 3:
            metadata[concept] = {'available': False, 'distinct_identity_groups': len(identities),
                                 'reason': 'fewer_than_three_independent_measured_identities'}
            continue
        observed = np.stack([normalized[graph] for graph in identities])
        mean = observed.mean(0)
        mean /= mean.sum(-1, keepdims=True)
        # Leave-one-identity-out dispersion is retained as a data diagnostic,
        # not reported as human recipe prediction accuracy.
        loo = (observed.sum(0)[None]-observed)/(len(observed)-1)
        overlaps = np.minimum(loo, observed).sum(-1)
        profiles[concept] = mean.tolist()
        metadata[concept] = {'available': True, 'distinct_identity_groups': len(identities),
            'effective_stimuli': len(identities), 'positive_stimuli': sum(len(by_graph[g]) for g in identities),
            'source_identity_groups': identities, 'source_annotation_indices': indices,
            'reference_kind': 'source_annotation_conditioned_observed_full_profiles',
            'annotation_semantic_route_is_project_authored': True,
            'new_measured_descriptor_endpoint': False,
            'leave_identity_out_overlap_mean': float(overlaps.mean()),
            'leave_identity_out_overlap_minimum': float(overlaps.min())}
    return profiles, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', type=Path, required=True)
    parser.add_argument('--source-model-sha256', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.source_model) != args.source_model_sha256 or sha(args.reference) != args.reference_sha256:
        raise ValueError('pinned source mismatch')
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS
    from fragrance_ai.recommender.odor_expression import registry
    from fragrance_ai.research.atlas_profiles import load_atlas
    model = json.loads(args.source_model.read_text(encoding='utf-8'))
    old = json.loads(args.reference.read_text(encoding='utf-8'))
    rows, endpoints, source = load_atlas(ROOT/'.benchmarks/atlas_profiles_v50/source')
    rows = [r for r in rows if r['level'] == 'high']
    if endpoints != old['endpoints'] or endpoints != model['quantitative_endpoints']:
        raise ValueError('observed endpoint identity mismatch')
    language = registry()
    routes = annotation_routes(model['fine_endpoints'], language, SCENT_DIMENSIONS)
    fitted, metadata = fit_annotation_profiles(rows, model['source_annotations'], routes)
    additions = {k: v for k, v in fitted.items() if k not in old['profiles']}
    value = {**old, 'profiles': {**old['profiles'], **additions},
        'concept_metadata': {**old['concept_metadata'], **{k: metadata[k] for k in additions}},
        'annotation_extension': {'version': 'source-annotation-reference-extension/v71',
            'source_model_sha256': args.source_model_sha256, 'prior_reference_sha256': args.reference_sha256,
            'ontology_sha256': language['sha256'], 'atlas_source': source,
            'recipe_outcomes_used': False, 'candidate_catalog_used': False, 'predicted_odor_profiles_used': False,
            'existing_reference_vectors_unchanged': True, 'all_annotation_routes_considered': len(routes),
            'minimum_independent_identities': 3, 'metadata': metadata}}
    assert all(value['profiles'][k] == v for k, v in old['profiles'].items())
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'model.json'
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report = {'reference_sha256': sha(target), 'old_concepts': len(old['profiles']),
        'new_concepts': len(value['profiles']), 'added_concepts': sorted(additions),
        'supported_primary_axes': [name for name in SCENT_DIMENSIONS if name in value['profiles']],
        'unsupported_primary_axes': [name for name in SCENT_DIMENSIONS if name not in value['profiles']],
        'existing_vectors_unchanged': True, 'recipe_outcomes_used': False,
        'code_sha256': sha(__file__)}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
