"""Source-bound categorical odor support, never invented sensory intensities."""
from collections import Counter
import hashlib
import json
import numpy as np

SCHEMA = 'source-bound-fine-odor-features/v1'


def validate_features(features):
    if not isinstance(features, dict) or features.get('schema') != SCHEMA:
        raise ValueError('fine odor provenance missing')
    payload = {k:v for k,v in features.items() if k != 'content_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    if digest != features.get('content_sha256'):
        raise ValueError('fine odor feature content mismatch')
    if features.get('not_observed_does_not_mean_sensory_absence') is not True:
        raise ValueError('fine odor missingness contract invalid')
    vocabulary = features['vocabulary']
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError('fine vocabulary invalid')
    for values in features['by_structure'].values():
        if len(set(values)) != len(values) or any(type(i) is not int or not 0 <= i < len(vocabulary) for i in values):
            raise ValueError('fine support indices invalid')


def catalog_features(catalog, minimum_support=3):
    from ..recommender.odor_integrity import (
        ODOR_SOURCES, AMBIGUOUS_TERMS, NEGATIVE_TERMS, _valid_lineage, normalize_term)
    by_structure = {}
    rejected = 0
    for item in catalog.ingredients:
        if not item.structure_smiles or not item.odor_assertions:
            continue
        if not _valid_lineage(tuple(item.odor_assertions), tuple(item.odor_evidence_refs), item.odor_registry_sha256):
            rejected += 1
            continue
        # Structure identity, not names/CAS substrings. Each concept is binary:
        # repeated sources or aliases do not become extra intensity.
        terms = {normalize_term(a.split(':', 1)[1]) for a in item.odor_assertions
                 if a.split(':', 1)[0] in ODOR_SOURCES}
        terms -= AMBIGUOUS_TERMS | NEGATIVE_TERMS | {'na','noaroma','none','unknown','nan'}
        if terms:
            by_structure.setdefault(item.structure_smiles, set()).update(terms)
    counts = Counter(term for values in by_structure.values() for term in values)
    vocabulary = sorted(term for term, count in counts.items() if count >= minimum_support)
    index = {term: i for i, term in enumerate(vocabulary)}
    bank = {smiles: sorted(index[term] for term in values if term in index)
            for smiles, values in sorted(by_structure.items())}
    bank = {smiles: values for smiles, values in bank.items() if values}
    payload = {'schema': SCHEMA, 'vocabulary': vocabulary, 'by_structure': bank,
        'minimum_structure_support': minimum_support,
        'not_observed_does_not_mean_sensory_absence': True,
        'values_are': 'binary_public_descriptor_support_not_measured_intensity',
        'unverified_lineage_rows_omitted': rejected}
    payload['content_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    return payload


def append_features(x, smiles, features):
    x = np.asarray(x, dtype=float)
    if features.get('schema') != SCHEMA or x.ndim != 2 or x.shape[1] != 1105 or len(x) != len(smiles):
        raise ValueError('fine odor feature contract mismatch')
    vocabulary = features['vocabulary']
    if len(set(vocabulary)) != len(vocabulary) or not vocabulary:
        raise ValueError('unique nonempty fine vocabulary required')
    # Last column marks available support, distinguishing an unknown graph
    # from a literal odorless observation (which is never encoded as zero truth).
    extra = np.zeros((len(x), len(vocabulary)+1))
    for row, graph in enumerate(smiles):
        indices = features['by_structure'].get(graph, [])
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(vocabulary) for i in indices):
            raise ValueError('invalid fine odor feature index')
        extra[row, indices] = 1.
        extra[row, -1] = bool(indices)
    return np.concatenate([x, extra], axis=1)
