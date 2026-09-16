"""Train a true recurrent dose decoder inside a copy of the shared checkpoint.

All prepared molecule rows participate in synthetic feasible inversion episodes.
Each rollout consumes its OWN preceding predicted masses and hidden state.
These labels are controlled-model tasks, not measured perfume/lotion outcomes.
The original molecular encoder and all seven existing heads remain bit-exact.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main():
    import numpy as np
    import torch
    from fragrance_ai.research.autoregressive_network import (
        AutoregressiveDoseCell,
        residual_features,
        project_simplex,
    )
    from fragrance_ai.recommender.autoregressive_neural import SCHEMA, refine_arrays
    from fragrance_ai.recommender.formulation_core import FormulationCore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.steps <= 16 or args.batch_size < 1:
        raise ValueError("positive bounded training configuration required")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    parent_protocol = json.loads(
        (args.parent.parent / "protocol.json").read_text(encoding="utf-8")
    )
    if sha(args.data / "manifest.json") != parent_protocol["data_sha256"]:
        raise ValueError("data does not match the original source-bound checkpoint")
    for name, digest in meta["files"].items():
        if sha(args.data / name) != digest:
            raise ValueError("prepared source data drift: " + name)
    core = FormulationCore(args.parent, args.parent_sha256)
    data = dict(np.load(args.data / "molecules.npz", allow_pickle=False))
    protocol = {
        "schema": "autoregressive-formulation-training/v73",
        "parent_sha256": core.sha256,
        "data_sha256": sha(args.data / "manifest.json"),
        "seed": 730914,
        "epochs": args.epochs,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "all_molecular_rows_used_as_episode_anchors_for_each_product": True,
        "split": "inherited_source_molecule_scaffold_groups_no_cross_split_components",
        "validation_used_for_epoch_selection": True,
        "test_used_for_epoch_selection": False,
        "teacher_forcing": False,
        "backbone_frozen": True,
        "single_output_weight_archive": True,
        "response_basis": "synthetic_positive_release_and_retention_scenarios",
        "targets": "feasible_compositions_under_source_bound_component_profile_proxy",
        "scope": "controlled_inverse_formulation_not_new_human_sensory_accuracy",
        "script_sha256": sha(__file__),
        "network_source_sha256": sha(
            ROOT / "fragrance_ai/research/autoregressive_network.py"
        ),
    }
    write(args.output / "protocol.json", protocol)
    torch.manual_seed(730914)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(730914)
    device = torch.device(args.device)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    x = np.clip(
        (data["x"] - core.arrays["feature_mean"]) / core.arrays["feature_scale"],
        -12.0,
        12.0,
    )

    def linear(value, name):
        return value @ core.arrays[name + ".weight"].T + core.arrays[name + ".bias"]

    latents = np.maximum(
        0.0, linear(np.maximum(0.0, linear(x, "molecule_in")), "molecule_out")
    ).astype(np.float32)
    shapes = data["high"].reshape(-1, 2, 146)
    shapes = shapes / np.maximum(shapes.sum(-1, keepdims=True), 1e-12)
    if not np.isfinite(shapes).all() or np.any(shapes.sum(-1) <= 0):
        raise ValueError("all source rows require finite complete profiles")
    ids, initial, goal, response, products, splits = [], [], [], [], [], []
    maximum = 12
    for split in (0, 1, 2):
        pool = np.flatnonzero(data["split"] == split)
        for anchor in pool:
            for product in (0, 1):
                count = int(rng.integers(2, maximum + 1))
                others = pool[pool != anchor]
                members = np.r_[anchor, rng.choice(others, count - 1, replace=False)]
                order = np.zeros(maximum, np.int64)
                order[:count] = members
                w0, w1 = np.zeros(maximum, np.float32), np.zeros(maximum, np.float32)
                w0[:count], w1[:count] = (
                    rng.dirichlet(np.ones(count)),
                    rng.dirichlet(np.full(count, 0.7)),
                )
                # Randomized environments are declared synthetic. They are not
                # fake measured lotion release coefficients or supplier values.
                rate = np.exp(rng.uniform(-2.0, 1.0, count))
                retention = np.exp(rng.uniform(-2.5, 1.5, count))
                r = np.zeros((3, maximum), np.float32)
                times = (
                    np.array([0.0, 0.3, 1.2])
                    if product == 0
                    else np.array([0.1, 0.8, 2.0])
                )
                r[:, :count] = retention[None] * np.exp(
                    -times[:, None] * rate[None] * (1.0 if product == 0 else 0.25)
                )
                r /= max(float(r.max()), 1e-12)
                ids.append(order)
                initial.append(w0)
                goal.append(w1)
                response.append(r)
                products.append(product)
                splits.append(split)
    episodes = {
        "ids": np.asarray(ids),
        "initial": np.asarray(initial),
        "goal": np.asarray(goal),
        "responses": np.asarray(response),
        "product": np.asarray(products, np.int8),
        "split": np.asarray(splits, np.int8),
    }
    np.savez_compressed(args.output / "episodes.npz", **episodes)
    for split in (0, 1, 2):
        selected = np.flatnonzero(episodes["split"] == split)
        for index in selected:
            actual = episodes["ids"][index, episodes["initial"][index] > 0]
            if np.any(data["split"][actual] != split):
                raise ValueError("episode crosses a frozen source split")
    bank = {
        "latents": tensor(latents),
        "profiles": tensor(shapes),
        "ids": tensor(episodes["ids"], torch.int64),
        "initial": tensor(episodes["initial"]),
        "goal": tensor(episodes["goal"]),
        "responses": tensor(episodes["responses"]),
        "product": tensor(episodes["product"], torch.int64),
    }
    indices = [np.flatnonzero(episodes["split"] == s) for s in (0, 1, 2)]
    model = AutoregressiveDoseCell().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.00003
    )
    write(
        args.output / "started.json",
        {
            "training_executed": False,
            "rows_by_split": [len(i) for i in indices],
            "source_rows": len(data["x"]),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "frozen_parent_parameters": core.manifest["parameter_count"],
            "device": str(device),
        },
    )

    def batch(index):
        members = bank["ids"][index]
        p = bank["profiles"][members].permute(0, 2, 1, 3)
        r = bank["responses"][index]
        w0, oracle = bank["initial"][index], bank["goal"][index]
        total = torch.einsum("btn,bn->bt", r, oracle).clamp_min(1e-12)
        target = (
            torch.einsum("btn,bn,bhnd->bhtd", r, oracle, p) / total[:, None, :, None]
        )
        return (
            bank["latents"][members],
            p,
            target,
            r,
            w0,
            bank["product"][index],
            w0 > 0,
        )

    def evaluate(selected, *, fixed=False):
        model.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(selected), args.batch_size):
                values = batch(selected[start : start + args.batch_size])
                latent, p, q, r, w0, product, mask = values
                if fixed:
                    w, previous = w0, torch.zeros_like(w0)
                    for step in range(args.steps):
                        f, _ = residual_features(
                            p, q, r, w, previous, product, step, mask
                        )
                        next_w = project_simplex(
                            w - 0.5 * f[:, :, 0] * f[:, :, 6] + 0.2 * previous, mask
                        )
                        previous, w = next_w - w, next_w
                else:
                    w, _ = model(*values, steps=args.steps)
                _, predicted = residual_features(
                    p, q, r, w, torch.zeros_like(w), product, 0, mask
                )
                _, first = residual_features(
                    p, q, r, w0, torch.zeros_like(w0), product, 0, mask
                )
                tv = 0.5 * (predicted - q).abs().sum(-1).amax(dim=(1, 2))
                initial_tv = 0.5 * (first - q).abs().sum(-1).amax(dim=(1, 2))
                cosine = torch.einsum("bhtd,bhtd->bht", predicted, q) / (
                    predicted.norm(dim=-1) * q.norm(dim=-1)
                ).clamp_min(1e-12)
                score = torch.minimum(1 - tv, cosine.amin(dim=(1, 2))) * 100.0
                if (
                    not torch.isfinite(w).all()
                    or (w < 0).any()
                    or (w.sum(-1) - 1).abs().max() > 1e-5
                ):
                    raise ValueError("invalid autoregressive mass distribution")
                rows.extend(
                    np.column_stack(
                        (
                            product.cpu().numpy(),
                            tv.cpu().numpy(),
                            initial_tv.cpu().numpy(),
                            score.cpu().numpy(),
                        )
                    ).tolist()
                )
        rows = np.asarray(rows)
        return {
            str(product): {
                "count": int((rows[:, 0] == product).sum()),
                "mean_worst_tv": float(rows[rows[:, 0] == product, 1].mean()),
                "initial_mean_worst_tv": float(rows[rows[:, 0] == product, 2].mean()),
                "passed95": int((rows[rows[:, 0] == product, 3] >= 95.0).sum()),
                "mean_score": float(rows[rows[:, 0] == product, 3].mean()),
            }
            for product in (0, 1)
        }

    best_loss = float("inf")
    best = None
    curve = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for start in range(0, len(indices[0]), args.batch_size):
            if start == 0:
                order = rng.permutation(indices[0])
            values = batch(order[start : start + args.batch_size])
            _, trace = model(*values, steps=args.steps)
            importance = torch.arange(1, args.steps + 1, device=device)
            loss = (trace * importance).sum(-1).mean() / importance.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        schedule.step()
        validation = evaluate(indices[1])
        objective = sum(v["mean_worst_tv"] for v in validation.values())
        if objective < best_loss:
            best_loss = objective
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            np.savez_compressed(
                args.output / "best_decoder.npz",
                **{k: v.numpy() for k, v in best.items()},
            )
        item = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
            "validation_objective": objective,
            "best_validation_objective": best_loss,
            "seconds": time.perf_counter() - started,
        }
        curve.append(item)
        write(args.output / "learning_curve.json", curve)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            args.output / "training_checkpoint.pt",
        )
        print(json.dumps(item), flush=True)
    model.load_state_dict(best)
    test = evaluate(indices[2])
    baseline = evaluate(indices[2], fixed=True)
    arrays = {
        **{k: v.copy() for k, v in core.arrays.items()},
        **{"autoregressive." + k: v.numpy().copy() for k, v in best.items()},
    }
    parity_values = batch(indices[2][:8])
    with torch.no_grad():
        expected, _ = model(*parity_values, steps=args.steps)
    latent, p, q, r, w, product, mask = [
        v.detach().cpu().numpy() for v in parity_values
    ]
    actual, _ = refine_arrays(
        arrays, latent, p, q, r, w, product, steps=args.steps, mask=mask
    )
    parity = float(np.max(np.abs(actual - expected.cpu().numpy())))
    gates = {
        "autoregressive_vs_initial": all(
            v["mean_worst_tv"] < v["initial_mean_worst_tv"] for v in test.values()
        ),
        "autoregressive_vs_fixed_optimizer": all(
            test[k]["mean_worst_tv"] <= baseline[k]["mean_worst_tv"]
            and test[k]["passed95"] >= baseline[k]["passed95"]
            for k in test
        ),
        "autoregressive_cpu_export": parity < 2e-5,
        "frozen_backbone_identity": all(
            np.array_equal(v, arrays[k]) for k, v in core.arrays.items()
        ),
    }
    np.savez_compressed(args.output / "weights.npz", **arrays)
    evaluation = {
        "test": test,
        "fixed_optimizer": baseline,
        "gates": gates,
        "cpu_export_max_error": parity,
        "heldout_recipe_examples": len(indices[2]),
        "test_used_for_selection": False,
        "scope": protocol["scope"],
        "full_400_service_pass_rate_measured": False,
    }
    write(args.output / "evaluation.json", evaluation)
    manifest = copy.deepcopy(core.manifest)
    manifest.update(
        schema=SCHEMA,
        training_executed=True,
        accepted_for_local_inference=all(gates.values()),
        weights={"path": "weights.npz", "sha256": sha(args.output / "weights.npz")},
        parameter_count=core.manifest["parameter_count"]
        + sum(p.numel() for p in model.parameters()),
    )
    manifest["architecture"]["autoregressive_decoder"] = {
        "kind": "shared_molecular_encoder_conditioned_GRU",
        "hidden_width": 48,
        "physical_feedback_features": 16,
        "chemical_embedding": 64,
        "rollout_steps": args.steps,
        "previous_predicted_mass_and_state_reused": True,
        "candidate_count_not_fixed": True,
    }
    manifest["autoregressive"] = {
        "parent_sha256": core.sha256,
        "training_protocol_sha256": sha(args.output / "protocol.json"),
        "episodes_sha256": sha(args.output / "episodes.npz"),
        "evaluation": evaluation,
        "base_heads_frozen_and_byte_identical": True,
        "default_steps": args.steps,
    }
    manifest["evaluation"]["gates"].update(gates)
    manifest["evaluation"]["autoregressive"] = evaluation
    write(args.output / "model.json", manifest)
    write(
        args.output / "completion.json",
        {
            "training_completed": True,
            "epochs": args.epochs,
            "accepted_for_local_inference": all(gates.values()),
            "model_sha256": sha(args.output / "model.json"),
            "weight_sha256": sha(args.output / "weights.npz"),
            "seconds": time.perf_counter() - started,
            "evaluation": evaluation,
        },
    )
    print((args.output / "completion.json").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
