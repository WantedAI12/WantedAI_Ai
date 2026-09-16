"""Check the frozen release's target contract without running recipe inference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads(args.preparation.read_text(encoding="utf8"))
    installed = Path(meta["installed"]).resolve()
    if sha(meta["wheel"]) != meta["wheel_sha256"]:
        raise ValueError("frozen wheel changed")
    for name, digest in meta["deploy_sources"].items():
        if sha(installed / name) != digest:
            raise ValueError("frozen API factory changed")
    sys.path.insert(0, str(installed))
    os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = str(args.bundle.resolve() / "perfumery.local.json")
    os.environ["PERFUMERY_AI_ENV"] = "research"

    import fragrance_ai
    from fastapi.testclient import TestClient
    from deploy.runtime_release_v87 import check_installed, registry_path, verify
    from deploy.target_runtime_v87 import create_release_app, verify_target_contract

    if not Path(fragrance_ai.__file__).resolve().is_relative_to(installed):
        raise ValueError("not the selected installed wheel")
    args.output.mkdir(parents=True, exist_ok=False)
    result = check_installed(args.bundle, args.sha256)
    manifest = verify(args.bundle, args.sha256)
    app = create_release_app(registry_path=registry_path(args.bundle, manifest))
    assert verify_target_contract(app) == result["target_contract"]
    examples = []
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["wheel_sha256"] == meta["wheel_sha256"]
        for target in (None, 95.):
            formula = {"brief": "citrus woody scent"}
            if target is not None:
                formula["target_similarity"] = target
            request = {"formula": formula}
            response = client.post("/v1/briefs/prepare", json=request)
            if response.status_code != 200 or response.json()["effective_target"] != (target or 90.):
                raise ValueError("served target configuration mismatch")
            name = "default90" if target is None else "explicit95"
            raw = args.output / (name + ".response.json")
            raw.write_bytes(response.content)
            request_path = args.output / (name + ".request.json")
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf8")
            examples.append({"name": name, "http": response.status_code,
                             "request_sha256": sha(request_path), "response_sha256": sha(raw),
                             "effective_target": response.json()["effective_target"]})
    result.update(bundle_sha256=args.sha256, examples=examples, deployment_changed=False,
                  transport="isolated_local_ASGI", recipe_inference_calls=0)
    (args.output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
