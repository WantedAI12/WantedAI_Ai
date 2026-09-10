"""Bounded request coalescing and immutable response caching for CPU inference."""

from collections import OrderedDict
from concurrent.futures import Future
import json
import threading
import time
from typing import Callable


class InferenceBusy(RuntimeError):
    pass


class InferenceCache:
    def __init__(
        self,
        *,
        ttl_seconds=120.0,
        max_entries=64,
        max_bytes=16 * 1024 * 1024,
        max_pending=8,
        max_followers=32,
        wait_seconds=110.0,
        clock=time.monotonic,
    ):
        if (
            min(
                ttl_seconds,
                max_entries,
                max_bytes,
                max_pending,
                max_followers,
                wait_seconds,
            )
            <= 0
        ):
            raise ValueError("cache and queue limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_pending = max_pending
        self.max_followers = max_followers
        self.wait_seconds = wait_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._cpu = threading.BoundedSemaphore(1)
        self._cache = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._followers = 0
        self._active = 0
        self._computations = 0
        self._hits = 0
        self._shared = 0

    def stats(self):
        with self._lock:
            return {
                "entries": len(self._cache),
                "bytes": self._bytes,
                "pending": len(self._pending),
                "followers": self._followers,
                "active": self._active,
                "computations": self._computations,
                "hits": self._hits,
                "shared": self._shared,
            }

    def run(
        self, key: str, compute: Callable[[], dict], *, cacheable=True
    ) -> tuple[dict, str]:
        if not cacheable:
            key = object()
        with self._lock:
            now = self.clock()
            for expired in [
                item for item, (deadline, _) in self._cache.items() if deadline <= now
            ]:
                _, data = self._cache.pop(expired)
                self._bytes -= len(data)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                data = cached[1]
            else:
                data = None
                future = self._pending.get(key)
                leader = future is None
                if leader:
                    if len(self._pending) >= self.max_pending:
                        raise InferenceBusy("inference queue is full")
                    future = Future()
                    self._pending[key] = future
                else:
                    if self._followers >= self.max_followers:
                        raise InferenceBusy("shared request limit exceeded")
                    self._followers += 1
                    self._shared += 1
        if data is not None:
            return json.loads(data), "hit"
        if not leader:
            try:
                return json.loads(future.result(timeout=self.wait_seconds)), "shared"
            finally:
                with self._lock:
                    self._followers -= 1
        acquired = False
        try:
            acquired = self._cpu.acquire(timeout=self.wait_seconds)
            if not acquired:
                raise InferenceBusy("inference queue wait expired")
            with self._lock:
                self._active += 1
                self._computations += 1
            payload = compute()
            data = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            with self._lock:
                if cacheable and len(data) <= self.max_bytes:
                    while self._cache and (
                        len(self._cache) >= self.max_entries
                        or self._bytes + len(data) > self.max_bytes
                    ):
                        _, (_, removed) = self._cache.popitem(last=False)
                        self._bytes -= len(removed)
                    self._cache[key] = (self.clock() + self.ttl_seconds, data)
                    self._bytes += len(data)
                self._pending.pop(key)
                future.set_result(data)
            return json.loads(data), "miss" if cacheable else "bypass"
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
                if not future.done():
                    future.set_exception(error)
            raise
        finally:
            if acquired:
                with self._lock:
                    self._active -= 1
                self._cpu.release()
