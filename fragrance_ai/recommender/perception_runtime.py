"""Independent, hash-bound research model configurations for each product.

Only operators configure local paths. HTTP requests never select models or
solvent scenarios. This module does not promote a research artifact to release.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import threading

PATH_ENV = 'PERFUMERY_AI_PERCEPTION_MANIFEST'
HASH_ENV = 'PERFUMERY_AI_PERCEPTION_MANIFEST_SHA256'
LOTION_PATH_ENV = 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST'
LOTION_HASH_ENV = 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST_SHA256'
_LOAD_LOCK = threading.RLock()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def environment_snapshot(product='perfume'):
    if product not in ('perfume', 'body_lotion'):
        raise ValueError('unknown perception product')
    from .formulation_core import configured_formulation_core
    core = configured_formulation_core()
    if core is not None:
        return ('shared_formulation_core', core.sha256, core._stat(), product)
    path_env, hash_env = (PATH_ENV, HASH_ENV) if product == 'perfume' else (LOTION_PATH_ENV, LOTION_HASH_ENV)
    from .local_runtime import configured_pair
    path, digest = configured_pair(path_env, hash_env, product)
    if not path and not digest:
        return None
    if not path or not re.fullmatch('[0-9a-fA-F]{64}', digest):
        raise ValueError('perception runtime requires a manifest and trusted SHA256 together')
    mode = os.environ.get('PERFUMERY_AI_ENV', 'development').strip().lower()
    if mode == 'production':
        raise ValueError('unpromoted research perception model cannot enter production')
    meta = _metadata(path)
    if meta[1] > 32768:
        raise ValueError('perception manifest too large')
    # Tiny manifest only; large model/registry hashes are checked once per load.
    raw = Path(meta[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest.lower():
        raise ValueError('perception manifest hash mismatch')
    document = json.loads(raw)
    # Old manifests remain perfume-only. A lotion lane must explicitly opt in;
    # it must never inherit the perfume model just because that model is enabled.
    if document.get('product', 'perfume') != product:
        raise ValueError('perception manifest product mismatch')
    if (document.get('schema') != 'perfumery-perception-runtime/v1'
            or document.get('scope') != 'prototype_only'
            or document.get('component_model_version') not in ('v3', 'v4', 'v5')):
        raise ValueError('invalid perception runtime scope')
    artifacts = []
    for name in ('base_model', 'component_model', 'registry'):
        item = document[name]
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('invalid perception artifact digest')
        resolved = (Path(meta[0]).parent / item['path']).resolve(strict=True)
        artifacts.append((name, *_metadata(resolved), item['sha256']))
    return (meta, digest.lower(), mode, tuple(artifacts), product)


@lru_cache(maxsize=4)
def _load(snapshot):
    from .perception_guidance import PerceptionGuidance
    meta, digest, _, artifacts, product = snapshot
    document = json.loads(Path(meta[0]).read_text(encoding='utf-8'))
    if _sha(meta[0]) != digest:
        raise ValueError('perception manifest changed during load')
    for _, path, _, _, expected in artifacts:
        if _sha(path) != expected:
            raise ValueError('perception artifact hash mismatch')
    paths = {name: path for name, path, *_ in artifacts}
    provider = PerceptionGuidance(paths['base_model'], paths['registry'],
        solvent=document['solvent_scenario'], experimental=True, weight=document['guidance_weight'],
        component_model_path=paths['component_model'], component_model_sha256=document['component_model']['sha256'])
    if provider.component_model_version != document['component_model_version']:
        raise ValueError('perception manifest and checkpoint versions differ')
    provider.runtime_manifest_sha256 = digest
    provider.runtime_snapshot = snapshot
    provider.runtime_product = product
    if snapshot != environment_snapshot(product):
        raise ValueError('perception artifacts changed during load')
    return provider


def configured_perception(product='perfume'):
    # functools' cache alone may compute a first miss in multiple threads.
    with _LOAD_LOCK:
        from .formulation_core import configured_formulation_core
        from .formulation_guidance import product_guidance
        core = configured_formulation_core()
        if core is not None:
            return product_guidance(core, product)
        snapshot = environment_snapshot(product)
        return None if snapshot is None else _load(snapshot)


def assert_provider_current(provider):
    check = getattr(provider, 'assert_current', None)
    if check is not None:
        check()
    snapshot = getattr(provider, 'runtime_snapshot', None)
    if snapshot is not None and snapshot != environment_snapshot(getattr(provider, 'runtime_product', 'perfume')):
        raise ValueError('perception runtime changed; reload all API and worker lanes')


def assert_provider_product(provider, product):
    """Unbound explicit component priors are allowed; cross-bound products are not."""
    if product not in ('perfume', 'body_lotion'):
        raise ValueError('unknown perception product')
    bound = getattr(provider, 'runtime_product', None)
    if bound is not None and bound != product:
        raise ValueError('perception provider belongs to a different product')
    assert_provider_current(provider)


def model_contract(provider):
    if provider is None:
        return {'configured': False, 'status': 'legacy_model_configuration', 'full_model_application': False}
    from .perception_guidance import PROJECTION
    from .models import SCENT_DIMENSIONS
    projection = getattr(provider, 'projection', PROJECTION)
    endpoints = set(getattr(provider, 'endpoints', ()))
    supported = sorted(axis for axis,names in projection.items() if set(names) <= endpoints)
    coarse_supported=list(supported)
    if getattr(provider,'complete_reference_dimensions',None) is not None:
        supported=sorted(provider.complete_reference_dimensions)
    return {'configured': True, 'component_model_sha256': getattr(provider, 'component_model_sha256', None),
            'supported_target_dimensions': supported,
            'unmodeled_target_dimensions': sorted(set(SCENT_DIMENSIONS)-set(supported)),
            'coarse_projection_dimensions':coarse_supported,
            'complete_reference_sha256':getattr(provider,'complete_reference_sha256',None),
            'primary_recipe_score_directly_uses_learned_outputs': False,
            'score_role': 'candidate_search_and_auxiliary_prediction_not_primary_strict_profile_score',
            'component_model_version': getattr(provider, 'component_model_version', None),
            'product': getattr(provider, 'runtime_product', 'explicit_shared_component_prior'),
            'manifest_sha256': getattr(provider, 'runtime_manifest_sha256', None),
            'solvent_scenario': provider.solvent, 'search_weight': provider.weight,
            'fine_descriptor_feature_count': len((getattr(provider, 'fine_odor_features', None) or {}).get('vocabulary', [])),
            'fine_descriptor_features_sha256': (getattr(provider, 'fine_odor_features', None) or {}).get('content_sha256'),
            'scope': getattr(provider, 'model_scope', 'prototype_component_prediction_and_additive_search_prior'),
            'manufacturing_approved': False, 'actual_human_accuracy_authorized': False}


def unsupported_lotion_contract(provider):
    return {**model_contract(provider), 'operation': 'lotion', 'applied': False,
            'status': 'unsupported_lotion_matrix' if provider else 'legacy_model_configuration',
            'reason': 'stock-solvent component model is not a lotion partition or mixture-perception model'}
