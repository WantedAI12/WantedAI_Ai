"""Compose a verified molecular refit and decoder; recheck all held-out tasks."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    import numpy as np
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.models import Ingredient
    from fragrance_ai.recommender.aligned_autoregressive import objective, rollout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    operator = FormulationCore(args.operator, sha(args.operator))
    profile = FormulationCore(args.profile, sha(args.profile))
    comparison = json.loads(args.comparison.read_text())
    if (
        not comparison["eligible_for_operator_revalidation"]
        or comparison["candidate_sha256"] != profile.sha256
    ):
        raise ValueError("isolated source-profile comparison required")
    permitted = ("molecule_in.", "molecule_out.", "quantitative_head.")
    unchanged = all(
        np.array_equal(v, operator.arrays[k])
        for k, v in profile.arrays.items()
        if not k.startswith(permitted)
    )
    if not unchanged:
        raise ValueError("nonmolecular parameter paths changed")
    arrays = {
        **profile.arrays,
        **{k: v for k, v in operator.arrays.items() if k.startswith("autoregressive.")},
    }
    m = copy.deepcopy(operator.manifest)
    m["accepted_for_local_inference"] = False
    m["profile_refit"] = {
        "schema": "isolated-molecular-profile-refit/v75",
        "nonmolecular_parameter_arrays_exact": True,
        "source_disjoint_profile_improvement": True,
        "profile_checkpoint_sha256": profile.sha256,
        "operator_checkpoint_sha256": operator.sha256,
        "comparison_sha256": sha(args.comparison),
        "profile_training_protocol_sha256": sha(args.profile.parent / "protocol.json"),
    }
    m["autoregressive"]["base_heads_frozen_and_byte_identical"] = False
    np.savez_compressed(args.output / "weights.npz", **arrays)
    m["weights"] = {"path": "weights.npz", "sha256": sha(args.output / "weights.npz")}
    m["evaluation"]["gates"].update(
        profile_refit_verified=True,
        autoregressive_refit_revalidated=False,
        frozen_backbone_identity=False,
    )
    write(args.output / "candidate.json", m)
    core = FormulationCore(
        args.output / "candidate.json",
        sha(args.output / "candidate.json"),
        allow_candidate=True,
    )
    items = [
        Ingredient(**row)
        for row in json.loads(
            (args.bank / "materials.json").read_text(encoding="utf-8")
        )
    ]
    latent, missing = core.autoregressive_latents(items)
    lookup = {item.ingredient_id: value for item, value in zip(items, latent)}
    groups = json.loads((args.operator.parent / "episodes.json").read_text())
    banks = {
        name: dict(np.load(args.bank / (name + ".npz"), allow_pickle=False))
        for name in ("perfume", "body_lotion")
    }
    reports = []
    for g in groups:
        if g["split"] != 2:
            continue
        path = args.operator.parent / g["file"]
        if sha(path) != g["sha256"]:
            raise ValueError("frozen operator test episodes changed")
        data = dict(np.load(path, allow_pickle=False))
        bank = banks[g["name"]]
        new_latents = np.stack([lookup[key] for key in bank["ids"]])
        losses = []
        for i in range(g["count"]):
            ids = data["members"][i]
            p = bank["profiles"][:, ids][None]
            values = (
                new_latents[ids][None],
                p,
                data["targets"][i : i + 1],
                data["responses"][i : i + 1],
                data["initial"][i : i + 1],
                np.array([g["product"]]),
                data["lower"][i : i + 1],
                data["upper"][i : i + 1],
                data["prices"][i : i + 1],
                data["budget"][i : i + 1],
            )
            options = {
                "time_weights": data["time_weights"][i : i + 1],
                "avoided": data["avoided"][i : i + 1],
            }
            result, _ = rollout(arrays, *values, **options)
            loss, _, _ = objective(
                p, values[2], values[3], result, values[5], **options
            )
            if (
                np.any(result < values[6] - 1e-8)
                or np.any(result > values[7] + 1e-8)
                or abs(float(result.sum()) - 1) > 1e-6
                or float((result * values[8]).sum()) > float(values[9][0]) + 1e-4
            ):
                raise ValueError("refit composition lost mass/price feasibility")
            losses.append(float(loss[0]))
        reports.append({"product": g["name"], "candidates": g["n"], "losses": losses})
        print(g["name"], g["n"], float(np.mean(losses)), flush=True)
    results = {}
    for name in banks:
        values = [
            value for r in reports if r["product"] == name for value in r["losses"]
        ]
        results[name] = {
            "count": len(values),
            "mean_worst_loss": float(np.mean(values)),
            "passed95": sum(value <= 0.05 + 1e-8 for value in values),
            "mean_score": 100 * (1 - float(np.mean(values))),
        }
    fixed = operator.manifest["autoregressive"]["evaluation"]["results"]["fixed"][
        "products"
    ]
    gate = all(
        results[k]["mean_worst_loss"] <= fixed[k]["mean_worst_loss"] for k in results
    )
    m["evaluation"]["gates"]["autoregressive_refit_revalidated"] = gate
    m["accepted_for_local_inference"] = gate
    m["evaluation"]["profile_refit"] = {
        "comparison": comparison,
        "operator_test": results,
        "fixed_optimizer": fixed,
        "missing_chemical_identity": missing,
        "scope": "profile_and_operator_tests_not_whole_service_performance",
    }
    m["training_sources"]["molecular_profile_refit"] = profile.manifest[
        "training_sources"
    ]
    write(args.output / "model.json", m)
    report = {
        "accepted_for_local_inference": gate,
        "model_sha256": sha(args.output / "model.json"),
        "weights_sha256": sha(args.output / "weights.npz"),
        "operator_test": results,
        "fixed": fixed,
        "single_weight_archive": True,
        "nonmolecular_parameter_arrays_exact": unchanged,
        "deployed": False,
    }
    write(args.output / "completion.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
