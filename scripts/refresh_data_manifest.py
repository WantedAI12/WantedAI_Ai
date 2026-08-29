#!/usr/bin/env python
"""Refresh SHA-256 and byte sizes for declared packaged data assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "fragrance_ai" / "data" / "data_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_dir = args.manifest.parent
    for name, record in manifest["assets"].items():
        path = data_dir / name
        if not path.is_file():
            raise SystemExit(f"declared data asset is missing: {path}")
        record["sha256"] = digest(path)
        record["bytes"] = path.stat().st_size
    manifest["reviewed_on"] = date.today().isoformat()
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
