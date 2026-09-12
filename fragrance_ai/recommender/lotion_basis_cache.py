"""Bounded process-local reuse of physics matrices, never recipes or scores."""
import hashlib
import json

from .runtime_cache import InferenceBusy, InferenceCache


_CACHE = InferenceCache(ttl_seconds=300., max_entries=8, max_bytes=8*1024*1024)


def reuse_basis(requests, pool, compute, *, exposure_windows=()):
    payload = {'version':'lotion-linear-basis-2-analytic-exposure',
        'exposure_windows': exposure_windows,
        'requests':[r.model_dump(mode='json') for r in requests],
        'pool':[(i.ingredient_id, i.blocked, i.active_strength_percent, i.vector().tolist()) for i in pool]}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    try:
        return _CACHE.run(key, compute)
    except InferenceBusy:
        # Cache saturation must not introduce a new inference rejection.
        return compute(), 'bypass'
