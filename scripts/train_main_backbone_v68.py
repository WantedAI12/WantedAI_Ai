"""Nested development evaluation of our train-only molecular kernel learner.

All rows/heads/endpoints and V66 molecule/scaffold folds are retained. No recipe
score, test outcome, physical rate, concentration, or missing observation is
manufactured to train this model. No automatic deployment or runtime rebinding.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fragrance_ai.research.atlas_profiles import (
    atlas_features,
    fit_atlas_scientific,
    inverse_atlas_target,
    load_atlas,
)
from fragrance_ai.research.scientific_kernel import (
    MolecularGeometry,
    ScientificKernel,
    fit_geometry,
    fit_alignment,
    kernel_prior,
)
from fragrance_ai.research.structured_kernel import fit_structured_kernel
from fragrance_ai.research.perception_validation import (
    group_folds,
    profile_errors,
    paired_profile_comparison,
    summarize_profiles,
)

ROOT = Path(__file__).resolve().parents[1]

OPTIONS = [
    (a, r, t, c)
    for a in (0.03, 0.1, 0.3, 1.0, 3.0)
    for r in (0.03, 0.3, 3.0)
    for t in ("sqrt", "log1p")
    for c in (0.5, 0.9)
]
BASELINE_SHA = "1b5af57f7902b4efeee2ab991156623adb8f966c9340d7f456ef3c3d790e1cc9"


def choose(x, y, groups):
    folds = group_folds(groups, 3)
    predictions = {option: np.full(y.shape, np.nan) for option in OPTIONS}
    for fold in range(3):
        train, test = folds != fold, folds == fold
        geometry = fit_geometry(x[train])
        compiled = MolecularGeometry.from_dict(geometry)
        blocks = compiled.blocks(x[train], x[train])
        cross = compiled.blocks(x[test], x[train])
        for regularization in (0.03, 0.3, 3.0):
            for transform in ("sqrt", "log1p"):
                target = (
                    np.sqrt(y[train]) if transform == "sqrt" else np.log1p(y[train])
                )
                alignment = fit_alignment(
                    blocks, target, regularization=regularization, prior=kernel_prior()
                )
                kernel = ScientificKernel(
                    {"geometry": geometry, "alignment": alignment}
                )
                gram = kernel.combine(blocks, x[train], x[train])
                query = kernel.combine(cross, x[test], x[train])
                for alpha in (0.03, 0.1, 0.3, 1.0, 3.0):
                    for coupling in (0.5, 0.9):
                        solution = fit_structured_kernel(
                            gram, target, alpha=alpha, output_coupling=coupling
                        )
                        latent = (
                            query @ solution["coefficients"] + solution["intercept"]
                        )
                        predictions[(alpha, regularization, transform, coupling)][
                            test
                        ] = inverse_atlas_target(latent, transform)
    records = []
    for option, output in predictions.items():
        errors = profile_errors(output, y)
        records.append(
            {
                "alpha": option[0],
                "alignment_regularization": option[1],
                "target_transform": option[2],
                "output_coupling": option[3],
                "loss": float(np.nan_to_num(errors["cosine_distance"], nan=1.0).mean()),
                "mae": float(np.nanmean(errors["mae"])),
            }
        )
    selected = min(records, key=lambda row: (row["loss"], row["mae"]))
    model = fit_atlas_scientific(
        x,
        y,
        selected["alpha"],
        selected["alignment_regularization"],
        target_transform=selected["target_transform"],
        output_coupling=selected["output_coupling"],
    )
    return model, {"selected": selected, "options": records}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    from fragrance_ai.research.atlas_profiles import predict_atlas
    from rdkit.Chem.Scaffolds import MurckoScaffold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = ROOT / ".benchmarks/main_backbone_v66/train-02"
    raw = (source_dir / "model.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError("frozen V66 checkpoint drift")
    baseline = json.loads(raw)
    previous_protocol = json.loads((source_dir / "protocol.json").read_bytes())
    old_predictions = json.loads((source_dir / "predictions.json").read_bytes())
    rows, endpoints, source = load_atlas(ROOT / ".benchmarks/atlas_profiles_v50/source")
    if endpoints != baseline["endpoints"] or source != baseline["source"]:
        raise ValueError("source/endpoint identity mismatch")
    ids = [row["id"] for row in rows]
    if ids != previous_protocol["stimulus_ids"] or ids != old_predictions["ids"]:
        raise ValueError("baseline evaluation population drift")
    x = atlas_features(
        [r["graph"] for r in rows],
        [r["level"] for r in rows],
        baseline["native_profiles"],
        baseline["fine_features"],
    )
    present = np.isfinite(x).all(axis=1)
    groups = [r["graph"] or "unresolved:" + r["id"] for r in rows]
    scaffold = [
        (
            "scaffold:"
            + MurckoScaffold.MurckoScaffoldSmiles(
                smiles=r["graph"], includeChirality=False
            )
        )
        if r["graph"]
        else "unresolved:" + r["id"]
        for r in rows
    ]
    splits = {"molecule": groups, "scaffold": scaffold}
    for name, group in splits.items():
        if (
            group != previous_protocol["outer_splits"][name]["groups"]
            or group_folds(group, 5).tolist()
            != previous_protocol["outer_splits"][name]["folds"]
        ):
            raise ValueError("outer evaluation split changed")
    files = [
        Path(__file__),
        ROOT / "fragrance_ai/research/scientific_kernel.py",
        ROOT / "fragrance_ai/research/structured_kernel.py",
        ROOT / "fragrance_ai/research/atlas_profiles.py",
        source_dir / "predictions.json",
        source_dir / "protocol.json",
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    protocol = {
        "schema": "scientific-backbone-v68-development/v1",
        "source": source,
        "baseline_sha256": BASELINE_SHA,
        "stimulus_ids": ids,
        "endpoints": endpoints,
        "source_rows": len(rows),
        "predictable_rows": int(present.sum()),
        "options": OPTIONS,
        "outer_splits": previous_protocol["outer_splits"],
        "inner_group_folds": 3,
        "selection_metric": "complete_profile_cosine_distance",
        "source_sha256": hashes,
        "acceptance_rule": "both_heads_MAE_and_cosine_nonregression_on_both_splits_without_coverage_loss",
        "new_blind_evaluation": False,
        "recipe_results_used": False,
        "runtime_promoted": False,
    }
    write(args.output / "protocol.json", protocol)
    start, reports, predictions, selections = time.perf_counter(), {}, {}, {}
    for name, group in splits.items():
        folds = group_folds(group, 5)
        reports[name], predictions[name], selections[name] = {}, {}, {}
        for head in ("applicability", "use"):
            y = np.array([row[head] for row in rows])
            output = np.full(y.shape, np.nan)
            selections[name][head] = []
            for fold in range(5):
                train, test = present & (folds != fold), present & (folds == fold)
                train_groups = np.asarray(group)[train].tolist()
                assert not set(train_groups) & set(np.asarray(group)[test])
                model, selected = choose(x[train], y[train], train_groups)
                output[test] = predict_atlas(model, x[test])
                selections[name][head].append({"fold": fold, **selected["selected"]})
                print(
                    json.dumps(
                        {
                            "split": name,
                            "head": head,
                            "fold": fold + 1,
                            "seconds": round(time.perf_counter() - start, 2),
                        }
                    ),
                    flush=True,
                )
            old = np.array(old_predictions[name][head]["candidate"], float)
            old[old < 0] = np.nan
            reports[name][head] = {
                "comparison": paired_profile_comparison(old, output, y, group),
                "baseline": summarize_profiles(old, y, group),
                "candidate": summarize_profiles(output, y, group),
            }
            predictions[name][head] = np.where(np.isfinite(output), output, -1).tolist()
    models, choices = {}, {}
    for head in ("applicability", "use"):
        y = np.array([row[head] for row in rows])
        models[head], choices[head] = choose(
            x[present], y[present], np.asarray(groups)[present].tolist()
        )
    accepted = all(
        value["comparison"]["baseline_mae_minus_candidate_mae"] >= 0
        and value["comparison"]["baseline_cosine_distance_minus_candidate"] >= 0
        and all(
            value["candidate"][metric]["defined_profiles"]
            >= value["baseline"][metric]["defined_profiles"]
            for metric in ("mae", "cosine_distance")
        )
        for split in reports.values()
        for value in split.values()
    )
    artifact = {
        **baseline,
        "models": models,
        "quantitative_target_model_version": "scientific-molecular-atlas/v68",
        "previous_backbone_sha256": BASELINE_SHA,
        "training_executed": True,
        "development_nonregression_passed": accepted,
        "training_protocol_sha256": hashlib.sha256(
            (args.output / "protocol.json").read_bytes()
        ).hexdigest(),
    }
    write(args.output / "model.json", artifact)
    report = {
        "scope": "existing_source_development_not_user_accuracy",
        "measurements": reports,
        "outer_selections": selections,
        "full_training_selection": choices,
        "development_nonregression_passed": accepted,
        "runtime_promoted": False,
        "seconds": time.perf_counter() - start,
        "model_sha256": hashlib.sha256(
            (args.output / "model.json").read_bytes()
        ).hexdigest(),
    }
    for path, digest in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError("source changed during training")
    write(args.output / "predictions.json", {"ids": ids, **predictions})
    write(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "accepted": accepted,
                "seconds": report["seconds"],
                "comparison": {
                    name: {head: value["comparison"] for head, value in split.items()}
                    for name, split in reports.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
