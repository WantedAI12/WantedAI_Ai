"""Single explicit local model/catalog selection, shared by launchers and SDK.

This file never deploys, downloads a model or changes the process environment.
The workspace profile is outside the wheel to avoid a circular source hash.
"""
from functools import lru_cache
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import threading

PROFILE_ENV = 'PERFUMERY_AI_LOCAL_PROFILE'
DEFAULT_PROFILE = Path(__file__).resolve().parents[2]/'perfumery.local.json'
REFERENCE_ENV = 'PERFUMERY_AI_LOTION_REFERENCE'
_LOCK = threading.RLock()


def local_profile():
    choice = os.environ.get(PROFILE_ENV, '').strip()
    if choice.lower() == 'disabled':
        return None
    production = os.environ.get('PERFUMERY_AI_ENV', '').strip().lower() == 'production'
    if production:
        if choice:
            raise ValueError('local research runtime cannot be used in production')
        return None
    path = (Path(choice) if choice else DEFAULT_PROFILE).resolve()
    if not path.is_file():
        if choice:
            raise ValueError('configured local runtime profile is unavailable')
        return None
    if path.stat().st_size > 32768:
        raise ValueError('local runtime profile too large')
    raw = path.read_bytes()
    if len(raw) > 32768:
        raise ValueError('local runtime profile too large')
    return deepcopy(_parse_profile(str(path), raw))


@lru_cache(maxsize=4)
def _parse_profile(path, raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != 'perfumery-local-runtime/v1' or value.get('scope') != 'local_research':
        raise ValueError('invalid local runtime profile')
    root = Path(path).parent
    result = {'profile_path': path, 'profile_sha256': hashlib.sha256(raw).hexdigest()}
    names = ('catalog', 'perfume', 'body_lotion', 'atlas', 'stock_mixture')
    if 'odor_backbone' in value:
        names += ('odor_backbone',)
    if 'lotion_release' in value:
        names += ('lotion_release',)
    if 'lotion_target_reference' in value:
        names += ('lotion_target_reference',)
    if 'unified_product' in value:
        names += ('unified_product',)
    if 'odor_expression' in value:
        names += ('odor_expression',)
    if 'odor_calibration' in value:
        if 'odor_expression' not in value:
            raise ValueError('odor calibration requires a pinned odor expression model')
        names += ('odor_calibration',)
    for name in names:
        item = value.get(name)
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']
                or not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256'])):
            raise ValueError('local runtime requires all pinned artifacts: ' + name)
        resolved = (root/item['path']).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('local runtime artifact escapes the workspace')
        result[name] = (str(resolved), item['sha256'])
    mode = value.get('lotion_reference', 'atlas')
    if mode not in ('component', 'atlas'):
        raise ValueError('invalid local lotion reference')
    result['lotion_reference'] = mode
    result['language'] = value.get('language')
    return result


def local_snapshot():
    profile = local_profile()
    return (profile['profile_sha256'] if profile else None, os.environ.get(REFERENCE_ENV, '').strip())


def configured_pair(path_env, hash_env, role):
    """Explicit configuration wins as a pair; never fill a partial override."""
    path, digest = os.environ.get(path_env, '').strip(), os.environ.get(hash_env, '').strip()
    if path or digest:
        return path, digest
    profile = local_profile()
    return profile[role] if profile is not None else ('', '')


@lru_cache(maxsize=4)
def _atlas(path, digest, size, mtime):
    from fragrance_ai.research.atlas_profiles import AtlasProfilePredictor
    return AtlasProfilePredictor(path, sha256=digest, experimental=True)


def local_atlas_provider():
    """Frozen molecular parent used to train the explicit stock-assay model."""
    profile = local_profile()
    if profile is None:
        raise ValueError('a pinned local Atlas profile is required')
    path, digest = profile['atlas']
    stat = Path(path).stat()
    with _LOCK:
        return _atlas(path, digest, stat.st_size, stat.st_mtime_ns)


def local_odor_backbone_provider():
    """Select a new odor-only backbone without rebinding old assay training.

    Transport inputs contain physical rates/capacities, not Atlas outputs.
    Their original training parent pins remain mandatory and unchanged.
    """
    profile = local_profile()
    parent = local_atlas_provider()
    if 'odor_backbone' not in profile:
        return parent
    path, digest = profile['odor_backbone']
    stat = Path(path).stat()
    with _LOCK:
        candidate = _atlas(path,digest,stat.st_size,stat.st_mtime_ns)
    if (candidate.parent_checkpoint_sha256 != parent.sha256
            or candidate.artifact_version != 'structured-multioutput-atlas/v66'
            or not candidate.development_nonregression_passed
            or candidate.source_digest != parent.source_digest
            or candidate.endpoints != parent.endpoints
            or any(candidate.models[k]['features'] != parent.models[k]['features'] for k in parent.models)):
        raise ValueError('odor backbone must preserve its verified molecular source and direct parent')
    return candidate


@lru_cache(maxsize=4)
def _bridge(component, path, digest, size, mtime, profile_sha):
    from .lotion_atlas import AtlasLotionGuidance
    provider = AtlasLotionGuidance(_atlas(path, digest, size, mtime), component.structures, experimental=True)
    provider.runtime_component_provider = component
    provider.runtime_local_profile_sha256 = profile_sha
    return provider


def local_lotion_provider(component, *, reference=None):
    profile = local_profile()
    # Explicit legacy component configuration still means component unless the
    # operator also selects an Atlas bridge. Never silently ignore an override.
    from .perception_runtime import LOTION_PATH_ENV, LOTION_HASH_ENV
    explicit_component = bool(os.environ.get(LOTION_PATH_ENV) or os.environ.get(LOTION_HASH_ENV))
    mode = reference or os.environ.get(REFERENCE_ENV, '').strip() or (
        profile['lotion_reference'] if profile and not explicit_component else 'component')
    if mode == 'component':
        return component
    if mode != 'atlas' or profile is None or component is None:
        raise ValueError('Atlas selection requires a complete local runtime profile')
    selected = local_odor_backbone_provider()
    path, digest = str(selected.path), selected.sha256
    stat = Path(path).stat()
    with _LOCK:
        return _bridge(component, path, digest, stat.st_size, stat.st_mtime_ns, profile['profile_sha256'])
