"""Compatibility import for Starlette's transitional httpx2 migration."""

from __future__ import annotations

import warnings

try:
    from starlette.exceptions import StarletteDeprecationWarning
except ImportError:  # Starlette 0.38.x lock predates the warning subclass.
    class StarletteDeprecationWarning(DeprecationWarning):
        pass


# The release lock uses Starlette 0.38.6. Newer Starlette versions emit this
# import-only warning while retaining the httpx fallback. Keep all other
# warnings fatal; remove this shim when the locked test client moves to httpx2.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from fastapi.testclient import TestClient


__all__ = ["TestClient"]
