"""Append a checksum-verified measured auxiliary dataset to joint training.

All raw source bytes and attribution are preserved. Composition groups, not
rows at neighbouring agitation speeds, are separated before normalization.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402 - bootstrap repository imports for direct execution
from fragrance_ai.recommender.emulsion_science import (  # noqa: E402
    REFERENCE,
    FIELDS,
    features,
    validate_raw,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    destination = args.data / "measured-emulsion-source"
    destination.mkdir(exist_ok=False)
    metadata_raw = urllib.request.urlopen(
        "https://api.figshare.com/v2/articles/30178552", timeout=30
    ).read()
    meta = json.loads(metadata_raw)
    if meta["license"]["name"] != "CC BY 4.0":
        raise ValueError("source license changed; inspect before using")
    descriptor = next(v for v in meta["files"] if v["id"] == 58131787)
    raw = urllib.request.urlopen(descriptor["download_url"], timeout=30).read()
    if (
        len(raw) != 111880
        or hashlib.md5(raw).hexdigest() != "f84715505fb476c6e61d4ca5b420d331"
    ):
        raise ValueError("public experimental file differs from inspected version")
    (destination / "metadata.json").write_bytes(metadata_raw)
    (destination / "Data_Summary.csv").write_bytes(raw)
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    x = validate_raw(np.asarray([v[1:] for v in rows[1:8]], float).T)
    bins = np.asarray([v[0] for v in rows[8:]], float)
    counts = np.asarray([v[1:] for v in rows[8:]], float).T
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.all(np.diff(bins) > 0)
        or not np.allclose(counts.sum(1), 100.0, atol=0.1)
    ):
        raise ValueError("invalid source bins or volume-percent totals")
    y = counts / counts.sum(1, keepdims=True)
    # Composition includes both phase properties, tension and dispersed volume;
    # all rpm values of the same composition stay in one split.
    groups = [",".join(format(v, ".12g") for v in row[1:]) for row in x]
    unique = sorted(
        set(groups),
        key=lambda s: hashlib.sha256(("emulsion-v69:" + s).encode()).hexdigest(),
    )
    mapping = {
        g: (
            0 if i < int(0.7 * len(unique)) else 1 if i < int(0.85 * len(unique)) else 2
        )
        for i, g in enumerate(unique)
    }
    split = np.asarray([mapping[g] for g in groups], np.int8)
    f = features(x)
    train = split == 0
    mean, scale = f[train].mean(0), np.maximum(f[train].std(0), 1e-5)
    z = (f - mean) / scale
    logd = np.log(bins)
    location = y @ logd
    sigma = np.sqrt((y * (logd[None] - location[:, None]) ** 2).sum(1))
    design = np.column_stack((np.ones(len(z)), z))
    regularizer = np.eye(design.shape[1]) * 1.0
    regularizer[0, 0] = 0
    coefficients = np.linalg.solve(
        design[train].T @ design[train] + regularizer,
        design[train].T
        @ np.column_stack((location, np.log(np.maximum(sigma, 0.08))))[train],
    )
    specification = {
        "reference": REFERENCE,
        "raw_fields": FIELDS,
        "diameter_bins_um": bins.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "lognormal_coefficients": coefficients.tolist(),
        "raw_min": x[train].min(0).tolist(),
        "raw_max": x[train].max(0).tolist(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "groups": len(unique),
        "row_ids": rows[0][1:],
        "composition_groups": groups,
        "split_policy": "composition_disjoint_including_all_rpm_for_each_composition",
        "units_note": "source diameter bins micrometres, volume percentages normalized by row sum",
        "target_kind": "measured_steady_state_drop_size_distribution_not_lotion_sensory_similarity",
    }
    np.savez_compressed(
        args.data / "emulsion.npz", raw=x, target=y.astype(np.float32), split=split
    )
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    manifest["emulsion"] = specification
    for p in (
        args.data / "emulsion.npz",
        destination / "metadata.json",
        destination / "Data_Summary.csv",
    ):
        manifest["files"][p.relative_to(args.data).as_posix()] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    manifest["training_sources"]["measured_emulsion_rows"] = len(x)
    manifest["training_sources"]["measured_emulsion_scope"] = specification[
        "target_kind"
    ]
    (args.data / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(x),
                "composition_groups": len(unique),
                "split_rows": np.bincount(split).tolist(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
