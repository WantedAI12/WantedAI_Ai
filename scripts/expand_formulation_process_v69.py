"""Increase independent numeric process contexts, retaining old raw datasets.

More synthetic contexts do not mean more supplier sources or observed products.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402 - bootstrap repository imports for direct execution
from fragrance_ai.recommender.formulation_process import (  # noqa: E402
    TEMPLATES,
    SOURCES,
    process_context,
    process_history,
    procedure_target,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=3200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in meta["files"].items():
        if hashlib.sha256((args.base / name).read_bytes()).hexdigest() != digest:
            raise ValueError("base dataset has drifted")
    for name in ("molecules.npz", "molecule-identities.json", "transport.npz"):
        shutil.copy2(args.base / name, args.output / name)
    rng = np.random.default_rng(690914)
    result = {k: [] for k in ("context", "ids", "numbers", "action", "checks", "split")}
    records = []
    maximum = max(map(len, TEMPLATES.values()))
    for template, sequence in TEMPLATES.items():
        for case in range(args.contexts):
            # Equal representation around decision surfaces, not only random
            # interior examples which encourage a broad "hold" classifier.
            boundary = case % 3 == 0
            values = {
                "peak_temperature_c": float(
                    rng.choice([70.0, 75.0]) + rng.uniform(-2, 2)
                )
                if boundary
                else float(rng.uniform(60, 85)),
                "measured_ph": float(rng.choice([3.0, 12.0]) + rng.uniform(-0.8, 0.8))
                if boundary
                else float(rng.uniform(1, 13.5)),
                "mixing_minutes": float(rng.uniform(1, 30)),
                "fragrance_addition_temperature_c": None
                if case % 2
                else float(rng.uniform(20, 60)),
                "batch_mass_g": float(rng.choice([400, 500, 1000, 2000])),
                "method_code": int(rng.choice([0, 0, 0, 1, 2])),
                "mixer_kind": str(
                    rng.choice(["high_shear", "high_shear", "manual", "overhead"])
                ),
                "pe9010_present": bool(case % 2),
            }
            if case % 4 == 0:
                values.update(
                    peak_temperature_c=72.5 if template == "hot_lotion" else None,
                    measured_ph=5.5,
                    mixer_kind="high_shear",
                    method_code=0,
                )
            if case % 7 == 0:
                values.update(peak_temperature_c=None, measured_ph=None)
            split = int(rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))
            context = process_context(template, values)
            for length in range(len(sequence) + 1):
                for corrupt in (False, True) if length >= 2 else (False,):
                    completed = list(sequence[:length])
                    if corrupt:
                        completed[-2], completed[-1] = completed[-1], completed[-2]
                    target, checks = procedure_target(template, completed, values)
                    ids, numbers = process_history(completed)
                    pi, pv = (
                        np.zeros(maximum, np.int64),
                        np.zeros((maximum, 12), np.float32),
                    )
                    pi[: len(ids)], pv[: len(ids)] = ids, numbers
                    for key, value in zip(
                        result, (context, pi, pv, target, checks, split)
                    ):
                        result[key].append(value)
                    records.append(
                        {
                            "template": template,
                            "context_group": f"expanded:{template}:{case}",
                            "history_length": length,
                            "corrupted_order": corrupt,
                        }
                    )
    result = {k: np.asarray(v) for k, v in result.items()}
    np.savez_compressed(args.output / "process.npz", **result)
    (args.output / "process-provenance.json").write_text(
        json.dumps(
            {
                "sources": SOURCES,
                "rows": records,
                "label_kind": "expanded_numeric_source_procedure_contexts_not_new_sources_or_measurements",
            }
        ),
        encoding="utf-8",
    )
    meta["training_sources"]["source_procedure_rows"] = len(result["action"])
    meta["training_sources"]["independent_process_contexts"] = 3 * args.contexts
    meta["previous_data_sha256"] = hashlib.sha256(
        (args.base / "manifest.json").read_bytes()
    ).hexdigest()
    meta["process_context_schema"] = "source_relative_context/v2"
    meta["files"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in meta["files"]
    }
    (args.output / "manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(result["action"]),
                "contexts": 3 * args.contexts,
                "splits": np.bincount(result["split"]).tolist(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
