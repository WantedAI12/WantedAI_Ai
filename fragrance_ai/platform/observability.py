"""Dependency-light Prometheus metrics and structured service logging."""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _label(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class _HistogramState:
    """Fixed-memory cumulative Prometheus histogram state."""

    bucket_counts: list[int] = field(
        default_factory=lambda: [0 for _ in _LATENCY_BUCKETS]
    )
    sample_sum: float = 0.0
    sample_count: int = 0

    def observe(self, value: float) -> None:
        for index, boundary in enumerate(_LATENCY_BUCKETS):
            if value <= boundary:
                self.bucket_counts[index] += 1
        self.sample_sum += value
        self.sample_count += 1

    def snapshot(self) -> tuple[tuple[int, ...], float, int]:
        return tuple(self.bucket_counts), self.sample_sum, self.sample_count


class ServiceMetrics:
    """Thread-safe bounded-cardinality metrics for API and queue operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._request_durations: dict[tuple[str, str], _HistogramState] = {}
        self._in_flight = 0
        self._inference: dict[str, _HistogramState] = {}
        self._jobs: dict[tuple[str, str], int] = defaultdict(int)

    def request_started(self) -> None:
        with self._lock:
            self._in_flight += 1

    def request_finished(
        self, *, method: str, route: str, status_code: int, duration_seconds: float
    ) -> None:
        safe_route = route if route.startswith("/") and len(route) <= 200 else "unknown"
        duration = max(0.0, float(duration_seconds))
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._requests[(method.upper(), safe_route, str(int(status_code)))] += 1
            self._request_durations.setdefault(
                (method.upper(), safe_route), _HistogramState()
            ).observe(duration)

    def observe_inference(self, *, outcome: str, duration_seconds: float) -> None:
        with self._lock:
            self._inference.setdefault(outcome, _HistogramState()).observe(
                max(0.0, float(duration_seconds))
            )

    def observe_job(self, *, kind: str, outcome: str) -> None:
        with self._lock:
            self._jobs[(kind, outcome)] += 1

    @staticmethod
    def _histogram(
        name: str,
        help_text: str,
        samples: dict[tuple[str, str], tuple[tuple[int, ...], float, int]],
        label_names: tuple[str, str],
    ) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for labels, (bucket_counts, sample_sum, sample_count) in sorted(samples.items()):
            base = ",".join(
                f'{key}="{_label(value)}"' for key, value in zip(label_names, labels)
            )
            for bucket, count in zip(_LATENCY_BUCKETS, bucket_counts):
                lines.append(f'{name}_bucket{{{base},le="{bucket:g}"}} {count}')
            lines.append(f'{name}_bucket{{{base},le="+Inf"}} {sample_count}')
            lines.append(f"{name}_sum{{{base}}} {sample_sum:.9f}")
            lines.append(f"{name}_count{{{base}}} {sample_count}")
        return lines

    def render_prometheus(self) -> str:
        with self._lock:
            requests = dict(self._requests)
            durations = {
                key: value.snapshot() for key, value in self._request_durations.items()
            }
            in_flight = self._in_flight
            inference = {key: value.snapshot() for key, value in self._inference.items()}
            jobs = dict(self._jobs)
        lines = [
            "# HELP perfumery_api_requests_total HTTP requests handled.",
            "# TYPE perfumery_api_requests_total counter",
        ]
        for (method, route, status), count in sorted(requests.items()):
            lines.append(
                "perfumery_api_requests_total"
                f'{{method="{_label(method)}",route="{_label(route)}",status="{status}"}} {count}'
            )
        lines.extend(
            [
                "# HELP perfumery_api_in_flight_requests Current HTTP requests in flight.",
                "# TYPE perfumery_api_in_flight_requests gauge",
                f"perfumery_api_in_flight_requests {in_flight}",
            ]
        )
        lines.extend(
            self._histogram(
                "perfumery_api_request_duration_seconds",
                "HTTP request duration.",
                durations,
                ("method", "route"),
            )
        )
        inference_pairs = {(outcome, "core"): values for outcome, values in inference.items()}
        lines.extend(
            self._histogram(
                "perfumery_inference_duration_seconds",
                "Recipe inference duration.",
                inference_pairs,
                ("outcome", "engine"),
            )
        )
        lines.extend(
            [
                "# HELP perfumery_jobs_total Durable jobs by outcome.",
                "# TYPE perfumery_jobs_total counter",
            ]
        )
        for (kind, outcome), count in sorted(jobs.items()):
            lines.append(
                f'perfumery_jobs_total{{kind="{_label(kind)}",outcome="{_label(outcome)}"}} {count}'
            )
        return "\n".join(lines) + "\n"


class JsonLogFormatter(logging.Formatter):
    """One-line JSON logs without serializing request bodies or credentials."""

    _reserved = set(logging.makeLogRecord({}).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._reserved or key in {"message", "asctime"} or key.startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                payload[key] = value
        if record.exc_info:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_json_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
