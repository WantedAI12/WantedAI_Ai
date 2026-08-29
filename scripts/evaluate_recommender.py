"""Run the bundled natural-language perfumery benchmark."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.evaluation import evaluate_benchmark  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=str(ROOT / "benchmarks" / "brief_benchmark.json"),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_benchmark(args.benchmark)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
