"""Prepare and check a source-bound target-90 release; never deploy on import."""

import argparse
import json
from pathlib import Path

from deploy import runtime_release_v78 as base


ROOT = base.ROOT
RELEASE_ID = "v87-failed-only-target90-20260916"
registry_path = base.registry_path


def verify(root, expected):
    manifest = base.verify(root, expected, release_id=RELEASE_ID)
    root = Path(root)
    profile = json.loads((root / "perfumery.local.json").read_text(encoding="utf8"))
    space = json.loads((root / profile["odor_space"]["path"]).read_text(encoding="utf8"))
    reference = json.loads((root / profile["lotion_target_reference"]["path"]).read_text(encoding="utf8"))
    if space.get("version") != "v80":
        raise ValueError("source-bound V80 reference scope required")
    from fragrance_ai.recommender.odor_release_scope import validate_scope
    validate_scope(space, reference["profiles"])
    return manifest


def prepare(preparation, output):
    result = base.prepare(preparation, output, release_id=RELEASE_ID)
    verify(output, result["bundle_sha256"])
    return result


def check_installed(root, expected):
    manifest = verify(root, expected)
    result = base.check_installed(root, expected, release_id=RELEASE_ID, data_version="v80")
    from deploy.target_runtime_v87 import create_release_app, verify_target_contract
    app = create_release_app(registry_path=registry_path(root, manifest))
    result["target_contract"] = verify_target_contract(app)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-installed", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.preparation and args.output:
        prepare(args.preparation, args.output)
    elif args.check_installed:
        print(json.dumps(check_installed(args.check_installed, args.sha256)))
    else:
        parser.error("select preparation/output or check-installed/sha256")
