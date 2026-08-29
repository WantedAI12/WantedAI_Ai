"""Export frozen R2 ensemble tensors to a safe, portable NumPy artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fragrance_ai" / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = DATA / "physsim_r2_runtime_weights.npz"
DEFAULT_MANIFEST = DATA / "physsim_r2_runtime_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(member_index: int, state_key: str) -> str:
    return f"member_{member_index}::{state_key}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    import torch

    from fragrance_ai.recommender.numpy_r2 import NumpyR2Model
    from fragrance_ai.research.r2_physsim import R2PhysSimCore

    ensemble_path = DATA / "physsim_r2_ensemble_manifest.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    state_keys: list[str] | None = None
    normalizer_mean: np.ndarray | None = None
    normalizer_std: np.ndarray | None = None
    source_members = []
    equivalence_errors: list[float] = []
    for index, member in enumerate(ensemble["members"]):
        checkpoint_path = DATA / member["file"]
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != member["sha256"]:
            raise RuntimeError("source checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        current_keys = list(payload["model_state_dict"])
        if state_keys is None:
            state_keys = current_keys
        elif current_keys != state_keys:
            raise RuntimeError("ensemble state dictionaries differ")
        current_mean = np.asarray(payload["normalizer"]["mean"], dtype=np.float32)
        current_std = np.asarray(payload["normalizer"]["std"], dtype=np.float32)
        if normalizer_mean is None:
            normalizer_mean = current_mean
            normalizer_std = current_std
        elif not np.array_equal(normalizer_mean, current_mean) or not np.array_equal(
            normalizer_std, current_std
        ):
            raise RuntimeError("ensemble normalizers differ")
        for state_key, tensor in payload["model_state_dict"].items():
            arrays[key(index, state_key)] = (
                tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            )
        torch_model = R2PhysSimCore()
        torch_model.load_state_dict(payload["model_state_dict"])
        torch_model.eval()
        numpy_model = NumpyR2Model(
            {
                state_key: tensor.detach().cpu().numpy()
                for state_key, tensor in payload["model_state_dict"].items()
            }
        )
        rng = np.random.default_rng(20260818 + index)
        for left_size, right_size in ((1, 1), (5, 7), (10, 10)):
            width = max(left_size, right_size)
            left = rng.normal(0.0, 1.0, (left_size, 217)).astype(np.float32)
            right = rng.normal(0.0, 1.0, (right_size, 217)).astype(np.float32)
            left_padded = np.zeros((1, width, 217), dtype=np.float32)
            right_padded = np.zeros((1, width, 217), dtype=np.float32)
            left_mask = np.zeros((1, width), dtype=np.float32)
            right_mask = np.zeros((1, width), dtype=np.float32)
            left_padded[0, :left_size] = left
            right_padded[0, :right_size] = right
            left_mask[0, :left_size] = 1.0
            right_mask[0, :right_size] = 1.0
            with torch.inference_mode():
                torch_prediction = float(
                    torch_model(
                        torch.from_numpy(left_padded),
                        torch.from_numpy(left_mask),
                        torch.from_numpy(right_padded),
                        torch.from_numpy(right_mask),
                    )[0]
                )
            numpy_prediction = numpy_model.predict(left, right)
            equivalence_errors.append(abs(torch_prediction - numpy_prediction))
        source_members.append(
            {
                "file": member["file"],
                "sha256": checkpoint_sha,
                "model_seed": int(member["model_seed"]),
                "weight": float(member["weight"]),
            }
        )
    assert (
        state_keys is not None
        and normalizer_mean is not None
        and normalizer_std is not None
    )
    maximum_equivalence_error = max(equivalence_errors, default=float("inf"))
    equivalence_tolerance = 1e-5
    if maximum_equivalence_error > equivalence_tolerance:
        raise RuntimeError(
            "portable R2 inference differs from Torch reference: "
            f"{maximum_equivalence_error} > {equivalence_tolerance}"
        )
    arrays["normalizer_mean"] = normalizer_mean
    arrays["normalizer_std"] = normalizer_std
    np.savez_compressed(args.output.resolve(), **arrays)
    manifest = {
        "schema_version": "1.0",
        "runtime": "numpy_only_r2_inference_v1",
        "distribution_contract": {
            "source_serialized_checkpoints_packaged": False,
            "source_serialized_checkpoints_required_at_runtime": False,
            "portable_weights_allow_pickle": False,
        },
        "artifact_file": args.output.name,
        "artifact_sha256": sha256_file(args.output.resolve()),
        "ensemble_manifest_sha256": sha256_file(ensemble_path),
        "descriptor_contract_sha256": ensemble["descriptor_contract_sha256"],
        "state_keys": state_keys,
        "members": source_members,
        "numeric_contract": {
            "dtype": "float32_weights_float64_dynamics",
            "gelu": "exact_erf",
            "layer_norm_epsilon": 1e-5,
            "dropout": "disabled_inference",
            "torch_reference_required_at_build_time_only": True,
        },
        "numeric_equivalence": {
            "passed": True,
            "cases": len(equivalence_errors),
            "maximum_absolute_error": maximum_equivalence_error,
            "tolerance": equivalence_tolerance,
            "input_contract": "deterministic_standardized_descriptor_mixtures",
        },
    }
    args.manifest.resolve().write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact_sha256": manifest["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
