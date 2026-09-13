"""Prepare a complete local-source, multi-task teacher-fidelity curriculum.

Old frozen predictors are teachers OFFLINE ONLY. They have seen old development
data; student holdouts here measure compression fidelity, not fresh human odor
generalization. Source process summaries generate procedural, not sensory labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402 - numerical thread settings must be applied first


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from fragrance_ai.recommender.local_runtime import (
        local_odor_backbone_provider,
        local_profile,
    )
    from fragrance_ai.recommender.fine_odor_model import configured_fine_odor
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.formulation_process import (
        ACTIONS,
        TEMPLATES,
        SOURCES,
        process_context,
        process_history,
        procedure_target,
    )
    from scripts.train_odor_expression_v61 import group_key

    profile, atlas, fine = (
        local_profile(),
        local_odor_backbone_provider(),
        configured_fine_odor(),
    )
    identities = configured_perception().structures
    candidates = (
        set(atlas.fine["by_structure"])
        | set(atlas.native)
        | {row[0] for row in identities.values() if row[0]}
    )
    graphs, exclusions = set(), []
    for graph in sorted(candidates):
        molecule = Chem.MolFromSmiles(graph)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            exclusions.append(
                {
                    "graph": graph,
                    "reason": "invalid_or_disconnected_not_single_molecule",
                }
            )
        else:
            graphs.add(Chem.MolToSmiles(molecule, isomericSmiles=True))
    graphs = sorted(graphs)
    group_keys = [group_key(g) for g in graphs]
    codes = [
        int(hashlib.sha256(("610909" + k).encode()).hexdigest()[:8], 16) % 100
        for k in group_keys
    ]
    split = np.array([0 if k < 70 else 1 if k < 85 else 2 for k in codes], np.int8)
    x, qhigh, qlow, f = [], [], [], []
    for offset in range(0, len(graphs), 256):
        batch = graphs[offset : offset + 256]
        x.append(atlas._query_features(batch, reference_level="high"))
        hi, lo = (
            atlas.predict(batch, reference_level="high"),
            atlas.predict(batch, reference_level="low"),
        )
        qhigh.append(np.concatenate([hi["applicability"], hi["use"]], -1))
        qlow.append(np.concatenate([lo["applicability"], lo["use"]], -1))
        f.append(fine.predict(batch))
        if offset % 2048 == 0:
            print(
                json.dumps(
                    {
                        "phase": "offline_teacher",
                        "processed": offset,
                        "total": len(graphs),
                    }
                ),
                flush=True,
            )
    x, qhigh, qlow, f = map(
        lambda rows: np.concatenate(rows).astype(np.float32), (x, qhigh, qlow, f)
    )
    np.savez_compressed(
        args.output / "molecules.npz", x=x, high=qhigh, low=qlow, fine=f, split=split
    )
    write(
        args.output / "molecule-identities.json",
        {"graphs": graphs, "groups": group_keys, "exclusions": exclusions},
    )
    # Original values/IDs/splits are retained for all synthetic physical rows.
    transport_path = ROOT / ".benchmarks/unified_product_v60/run-02/dataset.npz"
    with np.load(transport_path, allow_pickle=False) as archive:
        transport = {k: archive[k].copy() for k in archive.files}
    np.savez_compressed(args.output / "transport.npz", **transport)
    rng = np.random.default_rng(690913)
    contexts, ids, numbers, targets, checks, splits, provenance = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    max_steps = max(map(len, TEMPLATES.values()))
    for template, sequence in TEMPLATES.items():
        for case in range(320):
            values = {
                "peak_temperature_c": float(rng.uniform(65, 80)) if case % 3 else None,
                "mixing_minutes": float(rng.uniform(1, 20)),
                "measured_ph": float(rng.uniform(1, 13)),
                "fragrance_addition_temperature_c": None
                if case % 2
                else float(rng.uniform(25, 60)),
                "batch_mass_g": float(rng.choice([400, 500, 1000, 2000])),
                "method_code": int(
                    rng.choice(
                        [
                            0,
                            0,
                            1
                            if template == "cold_lotion"
                            else 2
                            if template == "hot_lotion"
                            else 0,
                        ]
                    )
                ),
                "mixer_kind": str(rng.choice(["high_shear", "overhead", "manual"])),
                "pe9010_present": bool(case % 2),
            }
            # Include clean process contexts as well as conflict/unknown cases.
            if case % 4 == 0:
                values.update(
                    peak_temperature_c=72.5 if template == "hot_lotion" else None,
                    measured_ph=5.5,
                    mixer_kind="high_shear",
                    method_code=0,
                )
            case_split = int(rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))
            for length in range(len(sequence) + 1):
                completed = list(sequence[:length])
                for corrupted in (False, True) if length >= 2 else (False,):
                    history = completed.copy()
                    if corrupted:
                        history[-2], history[-1] = history[-1], history[-2]
                    index, flag = procedure_target(template, history, values)
                    step_ids, step_values = process_history(history)
                    padded_ids, padded_values = (
                        np.zeros(max_steps, np.int64),
                        np.zeros((max_steps, 12), np.float32),
                    )
                    padded_ids[: len(step_ids)], padded_values[: len(step_ids)] = (
                        step_ids,
                        step_values,
                    )
                    contexts.append(process_context(template, values))
                    ids.append(padded_ids)
                    numbers.append(padded_values)
                    targets.append(index)
                    checks.append(flag)
                    splits.append(case_split)
                    provenance.append(
                        {
                            "template": template,
                            "context_group": f"{template}:{case}",
                            "history_length": length,
                            "corrupted_order": corrupted,
                        }
                    )
    np.savez_compressed(
        args.output / "process.npz",
        context=np.asarray(contexts),
        ids=np.asarray(ids),
        numbers=np.asarray(numbers),
        action=np.asarray(targets),
        checks=np.asarray(checks),
        split=np.asarray(splits, np.int8),
    )
    write(
        args.output / "process-provenance.json",
        {
            "sources": SOURCES,
            "rows": provenance,
            "label_kind": "project_authored_source_procedure_teacher_not_manufacturing_measurements",
        },
    )
    metadata = {
        "schema": "shared-formulation-data/v69",
        "created_from_profile_sha256": profile["profile_sha256"],
        "fine_endpoints": list(fine.endpoints),
        "quantitative_endpoints": list(atlas.endpoints),
        "actions": list(ACTIONS),
        "fine_features": atlas.fine,
        "native_profiles": atlas.native,
        "source_annotations": fine.annotations,
        "structures": dict(identities),
        "atlas_source": json.loads(Path(profile["odor_backbone"][0]).read_text())[
            "source"
        ],
        "teacher_bindings": {
            k: {"path": v[0], "sha256": v[1]}
            for k, v in profile.items()
            if k
            in (
                "odor_backbone",
                "odor_expression",
                "odor_calibration",
                "unified_product",
            )
        },
        "training_sources": {
            "molecular_rows": len(x),
            "source_candidate_graphs": len(candidates),
            "excluded_graphs": len(exclusions),
            "synthetic_transport_rows": len(transport["raw"]),
            "source_procedure_rows": len(targets),
            "process_templates": list(TEMPLATES),
            "molecular_target_kind": "frozen_v68_and_v61_v62_teacher_predictions",
            "transport_target_kind": "synthetic_physical_transition_operators",
            "process_target_kind": "documented_procedure_selection_and_conflicts",
            "new_human_measurements": 0,
            "measured_manufacturing_outcomes": 0,
            "evaluation_scope": "student_scaffold_holdout_teacher_fidelity_not_new_human_blind_validation",
        },
        "files": {
            name: sha(args.output / name)
            for name in (
                "molecules.npz",
                "molecule-identities.json",
                "transport.npz",
                "process.npz",
                "process-provenance.json",
            )
        },
        "source_transport_sha256": sha(transport_path),
        "seconds": time.perf_counter() - start,
    }
    write(args.output / "manifest.json", metadata)
    print(
        json.dumps(
            {
                "status": "prepared",
                **metadata["training_sources"],
                "seconds": metadata["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
