"""Shared deterministic lifecycle support for SQLite-backed repositories."""

from __future__ import annotations

from types import TracebackType
from typing import Self


class SQLiteConnectionOwner:
    """Make repository connections usable with ``with`` and safe at GC time.

    Concrete repositories keep their existing ``close`` implementations. The
    finalizer is a last-resort guard; application code should still close each
    repository at the end of its request or process lifetime.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors can run after module teardown or a partial __init__.
            pass

    def close(self) -> None:
        raise NotImplementedError
