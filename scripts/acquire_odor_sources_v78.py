"""Pin additional public odor annotations; keep raw archives out of releases."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def get(url, limit=50_000_000):
    with urlopen(
        Request(url, headers={"User-Agent": "PerfumeryAI-source-audit-v78"}), timeout=45
    ) as r:
        raw = r.read(limit + 1)
    if len(raw) > limit or raw.lstrip().startswith((b"<html", b"<!DOCTYPE")):
        raise ValueError("invalid data response")
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    repository = "https://github.com/pyrfume/pyrfume-data"
    commit = json.loads(
        get("https://api.github.com/repos/pyrfume/pyrfume-data/commits/main")
    )["sha"]
    report = {
        "repository": repository,
        "commit": commit,
        "files": {},
        "errors": [],
        "raw_archives_for_local_analysis_only": True,
        "intensity_measurements_must_not_be_inferred_from_binary_annotations": True,
    }
    for name in (
        "LICENSE",
        "README.md",
        "arctander_1960/LICENSE",
        "arctander_1960/behavior_1_sparse.csv",
        "leffingwell/LICENSE",
        *[
            f"{dataset}/{filename}"
            for dataset in (
                "flavornet",
                "goodscents",
                "arctander_1960",
                "leffingwell",
                "sigma_2014",
                "ifra_2019",
                "aromadb",
            )
            for filename in (
                "manifest.toml",
                "molecules.csv",
                "stimuli.csv",
                "behavior.csv",
            )
            if not (dataset == "arctander_1960" and filename == "behavior.csv")
        ],
    ):
        url = f"https://raw.githubusercontent.com/pyrfume/pyrfume-data/{commit}/{name}"
        try:
            raw = get(url)
            lfs = False
            if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
                lines = raw.decode().splitlines()
                expected = next(
                    x.split("sha256:")[1] for x in lines if x.startswith("oid ")
                )
                length = int(next(x.split()[1] for x in lines if x.startswith("size ")))
                raw = get(
                    f"https://media.githubusercontent.com/media/pyrfume/pyrfume-data/{commit}/{name}"
                )
                if len(raw) != length or hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError("LFS content mismatch")
                lfs = True
            path = a.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            report["files"][name] = {
                "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "lfs_verified": lfs,
            }
        except (HTTPError, ValueError) as e:
            report["errors"].append({"file": name, "error": str(e)})
        (a.output / "manifest.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        print(
            json.dumps(
                {
                    "files": len(report["files"]),
                    "errors": len(report["errors"]),
                    "last": name,
                }
            ),
            flush=True,
        )
    print(json.dumps(report["errors"]))


if __name__ == "__main__":
    main()
