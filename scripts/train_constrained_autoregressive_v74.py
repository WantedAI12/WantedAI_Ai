"""Retrain the recurrent decoder on runtime-scale, constraint-aware inverse tasks.

Held-out recipe seeds are never used for selection. Shared material profiles and
computed transport are NOT independent human sensory outcome measurements.
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
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.recommender.constrained_autoregressive import (
        SCHEMA,
        capped_simplex,
        project_constraints as cpu_project,
        rollout,
    )
    from fragrance_ai.research.constrained_autoregressive_network import (
        ConstrainedAutoregressiveCell,
        features,
        project_constraints,
        cheapest_feasible,
    )
    from fragrance_ai.research.autoregressive_network import AutoregressiveDoseCell

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-per-group", type=int, default=64)
    parser.add_argument("--validation-per-group", type=int, default=16)
    parser.add_argument("--test-per-group", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--operator", choices=["v74", "v75"], default="v74")
    args = parser.parse_args()
    if args.operator == "v75":
        from fragrance_ai.recommender.aligned_autoregressive import SCHEMA, rollout
        from fragrance_ai.research.aligned_autoregressive_network import (
            AlignedAutoregressiveCell,
            features,
        )

        ConstrainedAutoregressiveCell = AlignedAutoregressiveCell
    if (
        min(
            args.epochs,
            args.train_per_group,
            args.validation_per_group,
            args.test_per_group,
        )
        < 1
    ):
        raise ValueError("positive complete curriculum required")
    args.output.mkdir(parents=True, exist_ok=False)
    bank_meta = json.loads((args.bank / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in bank_meta["files"].items():
        if sha(args.bank / name) != digest:
            raise ValueError("runtime bank changed: " + name)
    core = FormulationCore(args.parent, args.parent_sha256)
    if (
        core.version != "shared-formulation-core/v73"
        or bank_meta["model_sha256"] != core.sha256
    ):
        raise ValueError("V73 source-bound runtime bank required")
    torch.manual_seed(740914)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    banks = {
        k: dict(np.load(args.bank / (k + ".npz"), allow_pickle=False))
        for k in ("perfume", "body_lotion")
    }
    sizes = (8, 32, 128, 512, 2048, 0)
    protocol = {
        "schema": "constrained-autoregressive-training/v74",
        "parent_sha256": core.sha256,
        "runtime_bank_sha256": sha(args.bank / "manifest.json"),
        "epochs": args.epochs,
        "rows_per_group": [
            args.train_per_group,
            args.validation_per_group,
            args.test_per_group,
        ],
        "candidate_cardinalities": [8, 32, 128, 512, 2048, "all_eligible"],
        "seed_by_split": [750914, 751914, 752914]
        if args.operator == "v75"
        else [740914, 741914, 742914],
        "steps": 8,
        "split": "independent_recipe_seeds_shared_material_bank_not_molecule_disjoint",
        "validation_used_for_selection": True,
        "test_used_for_selection": False,
        "targets": "planted_feasible_compositions_under_runtime_component_transport_proxy",
        "transport_augmentation": "declared_synthetic_positive_perturbations_of_computed_runtime_coefficients",
        "runtime_material_limits_and_prices_used": True,
        "teacher_forcing": False,
        "base_seven_heads_frozen": True,
        "all_eligible_catalog_members_present_in_training_candidates": True,
        "scope": "controlled_inverse_formulation_not_human_accuracy_or_400_request_service_test",
        "script_sha256": sha(__file__),
        "network_source_sha256": sha(
            ROOT / "fragrance_ai/research/constrained_autoregressive_network.py"
        ),
        "cpu_source_sha256": sha(
            ROOT / "fragrance_ai/recommender/constrained_autoregressive.py"
        ),
    }
    protocol["operator"] = args.operator
    if args.operator == "v75":
        protocol.update(
            schema="aligned-autoregressive-training/v75",
            targets="mixture_targets_and_independent_sparse_intents_no_400_benchmark_labels",
            optimized_warm_start_curriculum=True,
            loss="actually_retained_product_aligned_state_with_avoidance",
            network_source_sha256=sha(
                ROOT / "fragrance_ai/research/aligned_autoregressive_network.py"
            ),
            cpu_source_sha256=sha(
                ROOT / "fragrance_ai/recommender/aligned_autoregressive.py"
            ),
        )
    write(args.output / "protocol.json", protocol)

    groups = []
    for product, (name, bank) in enumerate(banks.items()):
        eligible = np.flatnonzero(
            (bank["caps"] > 0) & np.all(bank["profiles"].sum(-1) > 0, axis=0)
        )
        for size in sizes:
            n = min(size or len(eligible), len(eligible))
            for split, count in enumerate(protocol["rows_per_group"]):
                rng = np.random.default_rng(
                    protocol["seed_by_split"][split] + product * 100 + size
                )
                episodes = []
                for _ in range(count):
                    members = rng.choice(eligible, n, replace=False)
                    upper = bank["caps"][members].copy()
                    lower = np.zeros(n)
                    if rng.random() < 0.35:
                        required = int(rng.integers(n))
                        lower[required] = min(upper[required] * 0.1, 0.01)

                    def sparse_recipe():
                        raw = np.zeros(n)
                        support = rng.choice(
                            n, int(rng.integers(2, min(n, 12) + 1)), replace=False
                        )
                        raw[support] = rng.dirichlet(np.full(len(support), 0.5))
                        return capped_simplex(raw[None], lower[None], upper[None])[0]

                    oracle = sparse_recipe()
                    initial = sparse_recipe()
                    if rng.random() < 0.5:
                        initial = 0.7 * oracle + 0.3 * initial
                    prices = bank["prices"][members]
                    budget = max(
                        float(oracle @ prices) * float(rng.uniform(1.001, 1.3)), 1.0
                    )
                    initial = cpu_project(
                        initial[None],
                        lower[None],
                        upper[None],
                        prices[None],
                        np.array([budget]),
                    )[0]
                    # No external target or benchmark recipe enters this generator.
                    response = bank["responses"][:, members] * np.exp(
                        rng.normal(0, 0.2, bank["responses"][:, members].shape)
                    )
                    response /= np.maximum(response.max(-1, keepdims=True), 1e-30)
                    episodes.append(
                        (
                            members,
                            oracle,
                            initial,
                            lower,
                            upper,
                            prices,
                            budget,
                            response,
                        )
                    )
                payload = {
                    key: np.asarray(
                        [row[i] for row in episodes],
                        dtype=np.int64 if key == "members" else np.float32,
                    )
                    for i, key in enumerate(
                        (
                            "members",
                            "oracle",
                            "initial",
                            "lower",
                            "upper",
                            "prices",
                            "budget",
                            "responses",
                        )
                    )
                }
                if args.operator == "v75":
                    from fragrance_ai.recommender.exposure_normalization import (
                        normalized_exposure,
                    )

                    targets, avoidance, times = [], [], []
                    for i in range(count):
                        members = payload["members"][i]
                        shapes = bank["profiles"][:, members][None]
                        q, _ = normalized_exposure(
                            shapes,
                            payload["responses"][i : i + 1],
                            payload["oracle"][i : i + 1],
                        )
                        avoid = np.zeros_like(q[0])
                        if rng.random() < 0.25:
                            axes = rng.choice(
                                q.shape[-1], min(3, q.shape[-1]), replace=False
                            )
                            q[:] = 0
                            q[:, :, :, axes] = rng.dirichlet(np.ones(len(axes)))
                        if rng.random() < 0.25:
                            axis = int(np.argmin(q[0].sum(axis=(0, 1))))
                            avoid[:, :, axis] = 1
                        times.append(np.r_[0.0, rng.dirichlet(np.ones(q.shape[2] - 1))])
                        targets.append(q[0])
                        avoidance.append(avoid)
                    payload.update(
                        targets=np.asarray(targets, np.float32),
                        avoided=np.asarray(avoidance, np.float32),
                        time_weights=np.asarray(times, np.float32),
                    )
                filename = f"{name}-{n}-split{split}.npz"
                np.savez_compressed(args.output / filename, **payload)
                groups.append(
                    {
                        "product": product,
                        "name": name,
                        "n": n,
                        "split": split,
                        "count": count,
                        "file": filename,
                        "sha256": sha(args.output / filename),
                        "values": payload,
                    }
                )
    write(
        args.output / "episodes.json",
        [{k: v for k, v in g.items() if k != "values"} for g in groups],
    )
    # Fixed material banks; only episode batches occupy the autograd graph.
    tbanks = {
        name: {
            k: torch.as_tensor(bank[k], dtype=torch.float32, device=device)
            for k in ("profiles", "latents")
        }
        for name, bank in banks.items()
    }

    def batch(group, indices):
        v = {
            k: torch.as_tensor(value[indices], device=device)
            for k, value in group["values"].items()
        }
        material = v["members"]
        p = tbanks[group["name"]]["profiles"][:, material].permute(1, 0, 2, 3)
        r = v["responses"]
        from fragrance_ai.recommender.exposure_normalization import (
            torch_normalized_exposure,
        )

        target, _ = torch_normalized_exposure(p, r, v["oracle"])
        target = v.get("targets", target)
        product = torch.full(
            (len(indices),), group["product"], dtype=torch.int64, device=device
        )
        values = (
            tbanks[group["name"]]["latents"][material],
            p,
            target,
            r,
            v["initial"],
            product,
            v["lower"],
            v["upper"],
            v["prices"],
            v["budget"],
        )
        return (
            values + (v["time_weights"], v["avoided"])
            if args.operator == "v75"
            else values
        )

    def options(values):
        return (
            {"time_weights": values[10], "avoided": values[11]}
            if len(values) > 10
            else {}
        )

    def model_call(values):
        return model(*values[:10], **options(values))

    model = ConstrainedAutoregressiveCell().to(device)
    previous = AutoregressiveDoseCell().to(device).eval()
    previous.load_state_dict(
        {
            k.removeprefix("autoregressive."): torch.as_tensor(v.copy(), device=device)
            for k, v in core.arrays.items()
            if k.startswith("autoregressive.")
        }
    )
    # New feedback inputs and trust control start separately; old recurrent and
    # chemistry features are warm-started, not discarded or silently swapped.
    with torch.no_grad():
        model.chemistry.load_state_dict(previous.chemistry.state_dict())
        model.cell.weight_ih[:, :80].copy_(previous.cell.weight_ih)
        model.cell.weight_ih[:, 80:].zero_()
        model.cell.weight_hh.copy_(previous.cell.weight_hh)
        model.cell.bias_ih.copy_(previous.cell.bias_ih)
        model.cell.bias_hh.copy_(previous.cell.bias_hh)
        model.controls.weight[:2].copy_(previous.controls.weight)
        model.controls.bias[:2].copy_(previous.controls.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.00003
    )

    def metrics(values, mode):
        latent, p, q, r, w0, product, lower, upper, prices, budget = values[:10]
        cheap = cheapest_feasible(lower, upper, prices)
        w0 = project_constraints(w0, lower, upper, prices, budget, cheap)
        if mode == "learned":
            w, _ = model_call(values)
        elif mode == "v73":
            raw, _ = previous(latent, p, q, r, w0, product, upper > lower)
            w = project_constraints(raw, lower, upper, prices, budget, cheap)
        elif mode == "fixed":
            w = w0
            previous_mass = torch.zeros_like(w)
            for step in range(8):
                f, before, _ = features(
                    p,
                    q,
                    r,
                    w,
                    previous_mass,
                    product,
                    step,
                    upper > lower,
                    lower,
                    upper,
                    prices,
                    budget,
                    **options(values),
                )
                proposed = project_constraints(
                    w - 0.3 * f[:, :, 16] * f[:, :, 17] + 0.2 * previous_mass,
                    lower,
                    upper,
                    prices,
                    budget,
                    cheap,
                )
                delta = proposed - w
                proposed = (
                    w
                    + (0.2 / delta.abs().sum(-1).clamp_min(1e-12)).clamp_max(1)[:, None]
                    * delta
                )
                _, after, _ = features(
                    p,
                    q,
                    r,
                    proposed,
                    delta,
                    product,
                    step,
                    upper > lower,
                    lower,
                    upper,
                    prices,
                    budget,
                    **options(values),
                )
                chosen = torch.where((after <= before + 1e-8)[:, None], proposed, w)
                previous_mass, w = chosen - w, chosen
        else:
            w = w0
        _, loss, _ = features(
            p,
            q,
            r,
            w,
            w * 0,
            product,
            0,
            upper > lower,
            lower,
            upper,
            prices,
            budget,
            **options(values),
        )
        feasible = (
            torch.isfinite(w).all(-1)
            & (w >= lower - 1e-6).all(-1)
            & (w <= upper + 1e-6).all(-1)
            & ((w.sum(-1) - 1).abs() < 1e-5)
            & ((w * prices).sum(-1) <= budget + 1e-3)
        )
        if not feasible.all():
            raise ValueError("constraint-violating evaluation")
        return loss, w

    def batch_size(n):
        return max(1, min(16, 2048 // n))

    def evaluate(split, mode):
        model.eval()
        rows = []
        with torch.no_grad():
            for g in groups:
                if g["split"] != split:
                    continue
                for start in range(0, g["count"], batch_size(g["n"])):
                    index = np.arange(
                        start, min(g["count"], start + batch_size(g["n"]))
                    )
                    loss, _ = metrics(batch(g, index), mode)
                    rows.extend(
                        [[g["product"], g["n"], float(v)] for v in loss.cpu().numpy()]
                    )
        a = np.asarray(rows)

        def summarize(selected):
            return {
                "count": int(len(selected)),
                "mean_worst_loss": float(selected[:, 2].mean()),
                "passed95": int((selected[:, 2] <= 0.05 + 1e-8).sum()),
                "mean_score": float(100 * (1 - selected[:, 2].mean())),
            }

        return {
            "products": {
                name: summarize(a[a[:, 0] == i]) for i, name in enumerate(banks)
            },
            "cardinalities": {
                str(int(n)): summarize(a[a[:, 1] == n]) for n in np.unique(a[:, 1])
            },
            "rows": rows,
        }

    if args.operator == "v75":
        # Half the initial states have already gone through a complete fixed
        # optimizer. These are hard warm states, not fabricated real outcomes.
        with torch.no_grad():
            for g in groups:
                for start in range(0, g["count"] // 2, batch_size(g["n"])):
                    index = np.arange(
                        start, min(g["count"] // 2, start + batch_size(g["n"]))
                    )
                    _, warm = metrics(batch(g, index), "fixed")
                    g["values"]["initial"][index] = warm.cpu().numpy()
                np.savez_compressed(args.output / g["file"], **g["values"])
                g["sha256"] = sha(args.output / g["file"])
        write(
            args.output / "episodes.json",
            [{k: v for k, v in g.items() if k != "values"} for g in groups],
        )
    started = time.perf_counter()
    curve = []
    best_loss = float("inf")
    best = None
    rng = np.random.default_rng(741400)
    write(
        args.output / "started.json",
        {
            "training_executed": False,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "groups": len(groups),
            "training_rows": sum(g["count"] for g in groups if g["split"] == 0),
        },
    )
    for epoch in range(args.epochs):
        model.train()
        work = []
        for g in groups:
            if g["split"] == 0:
                order = rng.permutation(g["count"])
                work.extend(
                    (g, order[start : start + batch_size(g["n"])])
                    for start in range(0, g["count"], batch_size(g["n"]))
                )
        losses = []
        for i in rng.permutation(len(work)):
            g, index = work[i]
            _, trace = model_call(batch(g, index))
            importance = torch.arange(1, 9, device=device)
            loss = (trace * importance).sum(-1).mean() / importance.sum()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite recurrent training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        validation = evaluate(1, "learned")
        objective = sum(v["mean_worst_loss"] for v in validation["products"].values())
        if objective < best_loss:
            best_loss = objective
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            np.savez_compressed(
                args.output / "best_decoder.npz",
                **{k: v.numpy() for k, v in best.items()},
            )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation["products"],
            "objective": objective,
            "best": best_loss,
            "seconds": time.perf_counter() - started,
        }
        curve.append(row)
        write(args.output / "learning_curve.json", curve)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            args.output / "training_checkpoint.pt",
        )
        print(json.dumps(row), flush=True)
    model.load_state_dict(best)
    results = {
        mode: evaluate(2, mode) for mode in ("initial", "v73", "fixed", "learned")
    }
    arrays = {
        **{
            k: v.copy()
            for k, v in core.arrays.items()
            if not k.startswith("autoregressive.")
        },
        **{"autoregressive." + k: v.numpy().copy() for k, v in best.items()},
    }
    parity = []
    for name in banks:
        for size in (8, 2048):
            g = next(
                g
                for g in groups
                if g["name"] == name and g["n"] == size and g["split"] == 2
            )
            values = batch(g, np.array([0]))
            with torch.no_grad():
                expected, _ = model_call(values)
            actual, _ = rollout(
                arrays,
                *[v.cpu().numpy() for v in values[:10]],
                **{k: v.cpu().numpy() for k, v in options(values).items()},
            )
            parity.append(float(np.abs(actual - expected.cpu().numpy()).max()))
    test = results["learned"]["products"]
    gates = {
        "autoregressive_vs_initial": all(
            test[k]["mean_worst_loss"]
            < results["initial"]["products"][k]["mean_worst_loss"]
            for k in test
        ),
        "autoregressive_vs_fixed_optimizer": all(
            test[k]["mean_worst_loss"]
            <= results["fixed"]["products"][k]["mean_worst_loss"]
            for k in test
        ),
        "autoregressive_vs_v73": all(
            test[k]["mean_worst_loss"]
            <= results["v73"]["products"][k]["mean_worst_loss"]
            for k in test
        ),
        "autoregressive_cpu_export": max(parity) < 2e-5,
        "frozen_backbone_identity": all(
            np.array_equal(v, arrays[k])
            for k, v in core.arrays.items()
            if not k.startswith("autoregressive.")
        ),
        "autoregressive_constraints": True,
    }
    evaluation = {
        "results": results,
        "gates": gates,
        "cpu_export_max_error": max(parity),
        "test_used_for_selection": False,
        "scope": protocol["scope"],
        "service_full400_measured": False,
    }
    write(args.output / "evaluation.json", evaluation)
    np.savez_compressed(args.output / "weights.npz", **arrays)
    manifest = copy.deepcopy(core.manifest)
    manifest.update(
        schema=SCHEMA,
        training_executed=True,
        accepted_for_local_inference=all(gates.values()),
        weights={"path": "weights.npz", "sha256": sha(args.output / "weights.npz")},
        parameter_count=1868933 + sum(p.numel() for p in model.parameters()),
    )
    manifest["architecture"]["autoregressive_decoder"].update(
        physical_feedback_features=24,
        cap_and_price_projection_each_step=True,
        learned_trust_region=True,
        complete_profile_worst_case_gradient=True,
    )
    manifest["autoregressive"] = {
        "parent_sha256": core.sha256,
        "training_protocol_sha256": sha(args.output / "protocol.json"),
        "episodes_sha256": sha(args.output / "episodes.json"),
        "evaluation": evaluation,
        "base_heads_frozen_and_byte_identical": True,
        "default_steps": 8,
    }
    manifest["evaluation"]["gates"].update(gates)
    manifest["evaluation"]["autoregressive"] = evaluation
    write(args.output / "model.json", manifest)
    completion = {
        "training_completed": True,
        "epochs": args.epochs,
        "accepted_for_local_inference": all(gates.values()),
        "model_sha256": sha(args.output / "model.json"),
        "weights_sha256": sha(args.output / "weights.npz"),
        "seconds": time.perf_counter() - started,
        "gates": gates,
        "results": {k: v["products"] for k, v in results.items()},
    }
    write(args.output / "completion.json", completion)
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
