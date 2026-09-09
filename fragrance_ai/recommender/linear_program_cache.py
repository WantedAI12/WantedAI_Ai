"""Reuse identical successful LP solves within one recipe request only.

The complete numeric problem is the key, not a brief or candidate shortlist.
Timeouts/infeasibility/errors are retried normally. No search budget or final
assessment is skipped, and no result survives the request's model snapshot.
"""
from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import json

import numpy as np
from scipy import sparse

_REQUEST = ContextVar('perfumery_linear_program_request', default=None)
_MAX_ENTRIES = 64
_MAX_BYTES = 8 * 1024 * 1024


def linear_program_request(function):
    @wraps(function)
    def run(*args, **kwargs):
        token = _REQUEST.set({'entries': OrderedDict(), 'bytes': 0})
        try:
            return function(*args, **kwargs)
        finally:
            _REQUEST.reset(token)
    return run


def _problem_key(objective, parameters):
    # Unknown features (callbacks, custom integrality, etc.) bypass reuse.
    if set(parameters) - {'A_ub', 'b_ub', 'A_eq', 'b_eq', 'bounds', 'method', 'options'}:
        return None
    digest = hashlib.sha256()

    def array(value):
        if value is None:
            digest.update(b'none;')
        elif sparse.issparse(value):
            matrix = value.tocsr()
            digest.update(b'csr;')
            digest.update(str(matrix.shape).encode())
            for data in (matrix.indptr, matrix.indices, matrix.data):
                array(data)
        else:
            data = np.asarray(value)
            if data.dtype.kind not in 'biuf':
                raise TypeError('non-numeric solver array')
            digest.update(str((data.shape, data.dtype.str)).encode())
            digest.update(data.tobytes(order='C'))

    try:
        array(objective)
        for name in ('A_ub', 'b_ub', 'A_eq', 'b_eq'):
            digest.update(name.encode())
            array(parameters.get(name))
        # Bounds may contain None; encode the semantic values, not object-array
        # pointers. A change in method, tolerances or time limit changes the key.
        digest.update(json.dumps({name: parameters.get(name) for name in ('bounds', 'method', 'options')},
            sort_keys=True, allow_nan=False, separators=(',', ':')).encode())
    except (TypeError, ValueError):
        return None
    return digest.digest()


def _size(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_size(k) + _size(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(map(_size, value))
    return len(value.encode()) if isinstance(value, str) else 32


def cached_linprog(solver, objective, **parameters):
    request = _REQUEST.get()
    if request is None:
        return solver(objective, **parameters)
    problem = _problem_key(objective, parameters)
    if problem is None:
        return solver(objective, **parameters)
    key = (solver, problem)
    entries = request['entries']
    if key in entries:
        entries.move_to_end(key)
        return deepcopy(entries[key][0])
    result = solver(objective, **parameters)
    if (getattr(result, 'success', False) and getattr(result, 'status', None) == 0
            and getattr(result, 'x', None) is not None and np.isfinite(result.x).all()):
        size = _size(result)
        if size <= _MAX_BYTES:
            while entries and (len(entries) >= _MAX_ENTRIES or request['bytes'] + size > _MAX_BYTES):
                _, (_, removed) = entries.popitem(last=False)
                request['bytes'] -= removed
            entries[key] = (deepcopy(result), size)
            request['bytes'] += size
    return result
