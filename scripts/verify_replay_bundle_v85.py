"""All-pair graph-to-score parity plus fresh-process CPU bundle verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = "disabled"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def main():
    import numpy as np
    from fragrance_ai.recommender.replay_mixture import ReplayMixtureModel, digest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", choices=("fixed", "replay"))
    args = p.parse_args()
    if args.child:
        import psutil
        mode = args.child
        source = args.bundle / mode / "model.json"
        result = read(args.bundle / "result.json")["bundles"][mode]
        started = time.perf_counter()
        runtime = ReplayMixtureModel(source, result["bundle_sha256"])
        loading = time.perf_counter() - started
        fixture = read(args.output / "fixtures.json")
        expected = np.load(args.output / "expected.npz", allow_pickle=False)[mode]
        actual = []
        for row in fixture["pairs"]:
            actual.append(runtime.predict(row["a"], [1.] * len(row["a"]), row["b"], [1.] * len(row["b"]))["similarity"])
        difference = float(np.max(np.abs(np.asarray(actual) - expected)))
        if difference > 2e-5 or "torch" in sys.modules:
            raise ValueError("full graph CPU parity or Torch-free runtime failed")
        rng, timing = np.random.default_rng(851516), []
        for n in (12, 40, 80):
            a = rng.choice(fixture["graphs"], n, replace=False).tolist()
            b = rng.choice(fixture["graphs"], n, replace=False).tolist()
            wa, wb = rng.dirichlet(np.ones(n)).tolist(), rng.dirichlet(np.ones(n)).tolist()
            for _ in range(3):
                runtime.predict(a, wa, b, wb)
            values = []
            for _ in range(40):
                start = time.perf_counter()
                runtime.predict(a, wa, b, wb)
                values.append((time.perf_counter() - start) * 1000)
            timing.append({"components": n, "p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95))})
        memory = psutil.Process().memory_info()
        report = {"bundle": mode, "verified_pairs": len(actual), "max_prediction_difference": difference,
                  "torch_imported": False, "load_seconds": loading, "timings": timing,
                  "process_peak_bytes": getattr(memory, "peak_wset", memory.rss),
                  "scope": "SMILES_validation_cached_features_and_single_MLP_not_full_recipe_API"}
        (args.output / (mode + "-cpu.json")).write_text(json.dumps(report, indent=2), encoding="utf8")
        return
    import torch
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.research.mixture_mlp import MixtureMLP
    from fragrance_ai.research.mixture_replay_training import PairBank
    from fragrance_ai.research.r2_physsim import load_snitz_pairs, load_ravia_pairs
    from scripts.compare_physmix_v83 import write
    torch.set_num_threads(1)
    protocol = read(args.comparison / "protocol.json")
    source = next(Path(path).parent.parent for path in protocol["data"] if path.replace("\\", "/").endswith("snitz_2013/molecules.csv"))
    snitz, ravia = load_snitz_pairs(source), load_ravia_pairs(source)
    core = FormulationCore(Path(protocol["core"]), protocol["core_sha256"])
    bank = PairBank(core, {"snitz": snitz, "ravia": ravia}, "cpu")
    expected = {}
    for name in ("fixed", "replay"):
        manifest = read(args.bundle / name / "model.json")
        model = MixtureMLP(core.arrays, manifest["mode"]).eval()
        with np.load(args.bundle / name / "weights.npz", allow_pickle=False) as arrays:
            model.load_state_dict({k: torch.as_tensor(arrays[k]) for k in arrays.files})
        values = np.r_[bank.predict(model, "snitz", list(range(len(snitz)))), bank.predict(model, "ravia", list(range(len(ravia))))]
        expected[name] = values
        saved = np.load(args.bundle / name / "transfer_predictions.npz", allow_pickle=False)["prediction"]
        np.testing.assert_allclose(saved, values[len(snitz):], atol=2e-6, rtol=2e-6)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "expected.npz", **expected)
    write(args.output / "fixtures.json", {"pairs": [{"a": list(p.mixture_a), "b": list(p.mixture_b)} for p in [*snitz, *ravia]],
          "graphs": sorted({g for p in [*snitz, *ravia] for g in p.molecules})})
    reports = []
    for name in ("fixed", "replay"):
        subprocess.run([sys.executable, __file__, "--comparison", str(args.comparison.resolve()), "--bundle", str(args.bundle.resolve()),
                        "--output", str(args.output.resolve()), "--child", name], check=True)
        reports.append(read(args.output / (name + "-cpu.json")))
    # Integrity rejection is tested with a deliberately incorrect expected hash,
    # without modifying the real candidate bundle.
    rejected = False
    try:
        ReplayMixtureModel(args.bundle / "replay/model.json", "0" * 64)
    except ValueError:
        rejected = True
    if not rejected:
        raise ValueError("untrusted bundle checksum accepted")
    write(args.output / "verification.json", {"complete": True, "models": reports, "invalid_checksum_rejected": True,
          "runtime_source_sha256": digest(ROOT / "fragrance_ai/recommender/replay_mixture.py"),
          "bundle_result_sha256": digest(args.bundle / "result.json")})
    print(json.dumps({"verified": True, "reports": reports}), flush=True)


if __name__ == "__main__":
    main()
