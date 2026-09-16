"""Learn one recurrent controller on true headspace and lotion exposure objectives.

All six support sizes and both product tasks are retained. Fresh recipe seeds,
unchanged constraints and independent held-out tasks. Per-step line-search work
is matched; differences in total rollout budget are reported explicitly.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf8",
    )


def main():
    import numpy as np
    import torch
    from fragrance_ai.recommender.formulation_core import (
        FormulationCore,
        SYSTEM_VERSION,
    )
    from fragrance_ai.recommender.models import Ingredient
    from fragrance_ai.recommender.science import ScientificPropertyStore
    from fragrance_ai.recommender.physical_evidence_v76 import enrich
    from fragrance_ai.recommender.nonlinear_inverse import NonlinearDoseObjective
    from fragrance_ai.recommender.constrained_autoregressive import (
        capped_simplex,
        project_constraints as cpu_project,
    )
    from fragrance_ai.recommender.aligned_autoregressive import rollout as cpu_rollout
    from fragrance_ai.recommender.exposure_normalization import normalized_exposure
    from fragrance_ai.research.aligned_autoregressive_network import (
        AlignedAutoregressiveCell,
        features,
        objective,
    )
    from fragrance_ai.research.constrained_autoregressive_network import (
        project_constraints,
        cheapest_feasible,
    )
    from fragrance_ai.recommender.formulation_views import shared_views
    from fragrance_ai.recommender.lotion_atlas import (
        AtlasLotionGuidance,
        AtlasLotionShapes,
    )
    from fragrance_ai.recommender.reference_observations import (
        ComponentReferenceObservations,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("core", "bank", "evidence", "observations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, choices=(8, 16), default=8)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--warm-start-controller", type=Path)
    parser.add_argument("--warm-start-state", type=Path)
    parser.add_argument("--test-seed", type=int, default=761019)
    parser.add_argument('--full-reference-perfume',action='store_true',
        help='Train perfume inverse control against both 146-axis source-profile heads used by the current runtime')
    parser.add_argument('--hidden-size',type=int,choices=(48,96,128),default=48)
    parser.add_argument('--early-stopping-patience',type=int,default=0)
    parser.add_argument(
        "--gradient-geometry", choices=("simplex_tangent", "hill_simplex_tangent")
    )
    args = parser.parse_args()
    if args.warm_start_controller and args.warm_start_state:
        raise ValueError("choose one explicit controller initialization")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = "disabled"
    if args.epochs < 1 or args.draws < 16 or args.early_stopping_patience<0:
        raise ValueError(
            "complete positive training and at least 16 physical prior scenarios required"
        )
    core = FormulationCore(args.core, sha(args.core), allow_candidate=True)
    if core.version != SYSTEM_VERSION or any(
        v is not True
        for k, v in core.manifest["evaluation"]["gates"].items()
        if k != "nonlinear_inverse_fidelity"
    ):
        raise ValueError(
            "joint observed/physical source gates must pass before inverse learning"
        )
    manifest = json.loads((args.bank / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["files"].items():
        if sha(args.bank / name) != digest:
            raise ValueError("runtime bank changed")
    all_items = [
        Ingredient(**r)
        for r in json.loads((args.bank / "materials.json").read_text(encoding="utf8"))
    ]
    lookup = {r.ingredient_id: i for i, r in enumerate(all_items)}
    latents, missing = core.autoregressive_latents(all_items)
    store = ScientificPropertyStore.load_builtin()
    try:
        properties = store.with_catalog_structures(
            all_items, store.get_many(list(lookup))
        )
    finally:
        store.close()
    properties = enrich(all_items, properties, (str(args.evidence), sha(args.evidence)))
    full = NonlinearDoseObjective(all_items, properties, 15.0, draws=args.draws)

    def physical(members, concentration):
        selected = np.asarray(members, int)
        base = full.model
        obj = object.__new__(NonlinearDoseObjective)
        obj.draws = args.draws
        obj.model = SimpleNamespace(
            coefficients=base.coefficients[:, :, selected] * (concentration / 15.0),
            transport=base.transport[selected],
            suppression=base.suppression,
            interaction=base.interaction
            if np.array_equal(selected, np.arange(len(all_items)))
            else base.interaction[np.ix_(selected, selected)],
            total_moles=base.total_moles[selected] * (concentration / 15.0),
            gain=base.gain[selected],
            base_moles=(100 - concentration) / 46.06844,
        )
        return obj

    banks = {
        name: dict(np.load(args.bank / (name + ".npz"), allow_pickle=False))
        for name in ("perfume", "body_lotion")
    }
    provider = AtlasLotionGuidance(
        shared_views(core)[0], core.manifest["structures"], experimental=True
    )
    shapes = AtlasLotionShapes(provider)
    shapes.observations = ComponentReferenceObservations(
        args.observations, sha(args.observations)
    )
    for name, bank in banks.items():
        indices = np.array([lookup[key] for key in bank["ids"]])
        bank["all_indices"] = indices
        bank["latents"] = latents[indices]
        if name == "body_lotion" or args.full_reference_perfume:
            chosen = [all_items[i] for i in indices]
            shapes.prefetch(chosen)
            bank["profiles"] = np.stack(
                [
                    shapes.shape(i)
                    if shapes.shape(i) is not None
                    else np.zeros((2, 146))
                    for i in chosen
                ],
                axis=1,
            )
        np.savez_compressed(args.output / (name + "-bank.npz"), **bank)
    protocol = {
        "schema": "nonlinear_inverse_training/v82" if args.full_reference_perfume else "nonlinear_inverse_training/v76",
        "epochs": args.epochs,
        "physical_draws": args.draws,
        'full_reference_perfume':args.full_reference_perfume,
        'controller_hidden_size':args.hidden_size,
        'early_stopping_patience':args.early_stopping_patience,
        'width_expansion_preserves_initial_controller':True,
        'profile_coordinates':{name:{'heads':int(bank['profiles'].shape[0]),
            'descriptors':int(bank['profiles'].shape[2])} for name,bank in banks.items()},
        "steps": args.steps,
        "parent_steps": int(core.manifest["autoregressive"]["default_steps"]),
        "equal_parent_and_candidate_rollout_budget": args.steps
        == int(core.manifest["autoregressive"]["default_steps"]),
        "train_per_group": 64,
        "validation_per_group": 16,
        "test_per_group": 32,
        "support_sizes": [8, 32, 128, 512, 2048, "all_eligible"],
        "products": list(banks),
        "prior_recipe_seeds_reused": False,
        "test_used_for_selection": False,
        "full400_requests_used_in_training": False,
        "split": "independent_recipe_seeds_shared_known_material_bank_not_new_human_or_molecule_holdout",
        "matched_line_search_trials": 10,
        "physical_boundary_direction": "one_sided_1e-8_probe_not_fictitious_signal",
        "training_gradient": "first_order_environment_feedback_with_recurrent_BPTT",
        "parent_sha256": core.sha256,
        "evidence_sha256": sha(args.evidence),
        "bank_sha256": sha(args.bank / "manifest.json"),
        "observations_sha256": sha(args.observations),
        "source_script_sha256": sha(__file__),
        "source_physics_sha256": sha(
            ROOT / "fragrance_ai/recommender/nonlinear_inverse.py"
        ),
        "scope": "computed_inverse_tasks_not_measured_recipe_similarity",
        "selection": "validation_productwise_mean_and_pass_count_nonregression_then_mean",
        "loss": "terminal_loss_plus_0.1_path_loss_plus_5_train_parent_regret_plus_pass_retention_margin",
        "parent_training_teacher_steps": args.steps,
        "pass_retention_training_margin_loss": 0.049,
        "learning_rate": args.learning_rate,
        "test_seed": args.test_seed,
        "gradient_geometry": args.gradient_geometry,
        "parent_gradient_geometry": core.manifest["autoregressive"].get(
            "gradient_geometry"
        ),
        "mass_projection": "support_preserving_roundoff_repair_and_strict_interior_backward/v76",
        "recipe_teacher": "sparse_selected_support_with_required_minima_and_sufficient_capacity",
        "zero_mass_candidates_preserved_exactly": True,
        "warm_start_sha256": sha(args.warm_start_controller)
        if args.warm_start_controller
        else None,
        "warm_start_state_sha256": sha(args.warm_start_state)
        if args.warm_start_state
        else None,
    }
    source_paths = [
        Path(__file__),
        *[
            ROOT / "fragrance_ai" / folder / name
            for folder, name in (
                ("recommender", "nonlinear_inverse.py"),
                ("recommender", "mass_projection.py"),
                ("recommender", "simplex_geometry.py"),
                ("recommender", "aligned_autoregressive.py"),
                ("recommender", "constrained_autoregressive.py"),
                ("recommender", "dose_refinement.py"),
                ("recommender", "science.py"),
                ("recommender", "mixture_physics.py"),
                ("research", "aligned_autoregressive_network.py"),
                ("research", "constrained_autoregressive_network.py"),
                ('research','autoregressive_capacity.py'),
            )
        ],
    ]
    protocol["source_files_sha256"] = {
        str(path.relative_to(ROOT)): sha(path) for path in source_paths
    }
    for path in source_paths:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(args.output / "protocol.json", protocol)
    groups = []
    for product, (name, bank) in enumerate(banks.items()):
        eligible = np.flatnonzero(
            (bank["caps"] > 0) & np.all(bank["profiles"].sum(-1) > 0, axis=0)
        )
        for size in (8, 32, 128, 512, 2048, 0):
            n = min(size or len(eligible), len(eligible))
            for split, count in enumerate((64, 16, 32)):
                rng = np.random.default_rng(
                    (args.test_seed if split == 2 else 760916 + 1000 * split)
                    + product * 100
                    + size
                )
                data = {
                    key: []
                    for key in (
                        "members",
                        "oracle",
                        "initial",
                        "lower",
                        "upper",
                        "prices",
                        "budget",
                        "responses",
                        "targets",
                        "avoided",
                        "time_weights",
                        "concentration",
                    )
                }
                for _ in range(count):
                    member = np.sort(rng.choice(eligible, n, replace=False))
                    upper = bank["caps"][member]
                    lower = np.zeros(n)
                    if rng.random() < 0.35:
                        key = int(rng.integers(n))
                        lower[key] = min(0.01, upper[key] * 0.1)

                    def recipe():
                        value = np.zeros(n)
                        chosen = rng.choice(
                            n, int(rng.integers(2, min(n, 12) + 1)), replace=False
                        )
                        selected = np.zeros(n, bool)
                        selected[chosen] = True
                        selected |= lower > 0
                        if upper[selected].sum() < 1:
                            for index in rng.permutation(np.flatnonzero(~selected)):
                                selected[index] = True
                                if upper[selected].sum() >= 1:
                                    break
                        chosen = np.flatnonzero(selected)
                        value[chosen] = rng.dirichlet(np.ones(len(chosen)) * 0.5)
                        restricted_upper = np.where(selected, upper, 0.0)
                        recipe_weights = capped_simplex(
                            value[None], lower[None], restricted_upper[None]
                        )[0]
                        if np.any(recipe_weights[~selected] != 0.0):
                            raise ValueError(
                                "teacher projection activated absent ingredients"
                            )
                        return recipe_weights

                    oracle = recipe()
                    initial = recipe()
                    if rng.random() < 0.5:
                        initial = 0.7 * oracle + 0.3 * initial
                    prices = bank["prices"][member]
                    budget = max(
                        1.0, float(oracle @ prices) * float(rng.uniform(1.001, 1.3))
                    )
                    initial = cpu_project(
                        initial[None],
                        lower[None],
                        upper[None],
                        prices[None],
                        np.array([budget]),
                    )[0]
                    response = bank["responses"][:, member]
                    p = bank["profiles"][:, member]
                    concentration = (
                        float(rng.choice([5.0, 15.0, 30.0])) if product == 0 else 15.0
                    )
                    if product == 0:
                        target, _ = physical(
                            bank["all_indices"][member], concentration
                        ).predict(p, oracle)
                    else:
                        target = normalized_exposure(
                            p[None], response[None], oracle[None]
                        )[0][0]
                    if rng.random() < 0.25:
                        target[:] = 0.0
                        axes = rng.choice(target.shape[-1], 3, replace=False)
                        target[:, :, axes] = rng.dirichlet(np.ones(3))
                    mask = np.zeros_like(target)
                    if rng.random() < 0.25:
                        mask[:, :, int(np.argmin(target.sum((0, 1))))] = 1.0
                    tw = np.r_[0.0, rng.dirichlet(np.ones(target.shape[1] - 1))]
                    for key, value in zip(
                        data,
                        (
                            member,
                            oracle,
                            initial,
                            lower,
                            upper,
                            prices,
                            budget,
                            response,
                            target,
                            mask,
                            tw,
                            concentration,
                        ),
                    ):
                        data[key].append(value)
                data = {
                    key: np.asarray(
                        v, dtype=np.int64 if key == "members" else np.float64
                    )
                    for key, v in data.items()
                }
                filename = f"{name}-{n}-split{split}.npz"
                np.savez_compressed(args.output / filename, **data)
                groups.append(
                    {
                        "name": name,
                        "product": product,
                        "n": n,
                        "split": split,
                        "count": count,
                        "file": filename,
                        "sha256": sha(args.output / filename),
                        "data": data,
                    }
                )
                print(json.dumps({"prepared": filename, "count": count}), flush=True)
    write(
        args.output / "episodes.json",
        [{k: v for k, v in g.items() if k != "data"} for g in groups],
    )
    torch.set_num_threads(1)
    torch.manual_seed(760916)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignedAutoregressiveCell(hidden_size=args.hidden_size).to(device)
    state = {
        key.removeprefix("autoregressive."): torch.tensor(value.copy(), device=device)
        for key, value in core.arrays.items()
        if key.startswith("autoregressive.")
    }
    from fragrance_ai.research.autoregressive_capacity import initialize_from_controller
    initialize_from_controller(model,state)
    previous = AlignedAutoregressiveCell(hidden_size=state['cell.weight_hh'].shape[1]).to(device).eval()
    previous.load_state_dict(state)
    if args.warm_start_controller:
        warm = FormulationCore(
            args.warm_start_controller,
            sha(args.warm_start_controller),
            allow_candidate=True,
        )
        if any(
            not np.array_equal(value, warm.arrays[key])
            for key, value in core.arrays.items()
            if not key.startswith("autoregressive.")
        ):
            raise ValueError(
                "warm start must retain the same non-controller parameters"
            )
        model.load_state_dict(
            {
                key.removeprefix("autoregressive."): torch.tensor(
                    value.copy(), device=device
                )
                for key, value in warm.arrays.items()
                if key.startswith("autoregressive.")
            }
        )

    if args.warm_start_state:
        snapshot = torch.load(
            args.warm_start_state, map_location=device, weights_only=True
        )
        if snapshot.get("parent_sha256") != core.sha256:
            raise ValueError("warm state must share the exact predictive parent")
        model.load_state_dict(snapshot["model"])

    def tensor(value):
        return torch.tensor(value, device=device, dtype=torch.float32)

    def batch(group, i):
        row = {k: v[i] for k, v in group["data"].items()}
        bank = banks[group["name"]]
        ids = row["members"]
        p = bank["profiles"][:, ids]
        values = (
            tensor(bank["latents"][ids][None]),
            tensor(p[None]),
            tensor(row["targets"][None]),
            tensor(row["responses"][None]),
            tensor(row["initial"][None]),
            torch.tensor([group["product"]], device=device),
            tensor(row["lower"][None]),
            tensor(row["upper"][None]),
            tensor(row["prices"][None]),
            tensor(np.array([row["budget"]])),
        )
        options = {
            "time_weights": tensor(row["time_weights"][None]),
            "avoided": tensor(row["avoided"][None]),
            "gradient_geometry": args.gradient_geometry,
        }
        cpu_physics = None
        if group["product"] == 0:
            cpu_physics = physical(
                bank["all_indices"][ids], float(row["concentration"])
            )
            options["physics"] = cpu_physics.torch(device)
        return values, options, cpu_physics

    def fixed(values, options):
        latent, p, q, r, w, product, lower, upper, prices, budget = values
        cheap = cheapest_feasible(lower, upper, prices)
        w = project_constraints(w, lower, upper, prices, budget, cheap)
        last = torch.zeros_like(w)
        for step in range(args.steps):
            f, before, _ = features(
                p,
                q,
                r,
                w,
                last,
                product,
                step,
                upper > lower,
                lower,
                upper,
                prices,
                budget,
                **options,
            )
            proposal = project_constraints(
                w - 0.3 * f[:, :, 16] * f[:, :, 17] + 0.2 * last,
                lower,
                upper,
                prices,
                budget,
                cheap,
            )
            delta = proposal - w
            delta *= torch.minimum(
                torch.ones(1, device=device),
                0.25 / delta.abs().sum(-1).clamp_min(1e-30),
            )[:, None]
            best = w
            best_loss = before
            for fraction in (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.0078125,
                0.00390625,
                0.001953125,
            ):
                candidate = w + fraction * delta
                if options.get("physics") is not None:
                    loss, _, _ = options["physics"](
                        p,
                        q,
                        r,
                        candidate,
                        product,
                        time_weights=options["time_weights"],
                        avoided=options["avoided"],
                        compute_gradient=False,
                    )
                else:
                    loss, _, _ = objective(p, q, r, candidate, product, **options)
                good = loss < best_loss - 1e-10
                best = torch.where(good[:, None], candidate, best)
                best_loss = torch.where(good, loss, best_loss)
            last, w = best - w, best
        return w

    def evaluate(split, modes=("learned",)):
        result = {name: {mode: [] for mode in modes} for name in banks}
        model.eval()
        with torch.no_grad():
            for group in groups:
                if group["split"] != split:
                    continue
                for i in range(group["count"]):
                    values, options, _ = batch(group, i)
                    for mode in modes:
                        mode_options = dict(options)
                        if mode.startswith("parent"):
                            mode_options["gradient_geometry"] = core.manifest[
                                "autoregressive"
                            ].get("gradient_geometry")
                        w = (
                            values[4]
                            if mode == "initial"
                            else fixed(values, options)
                            if mode == "fixed"
                            else (model if mode == "learned" else previous)(
                                *values,
                                steps=int(
                                    core.manifest["autoregressive"]["default_steps"]
                                )
                                if mode == "parent"
                                else args.steps,
                                **mode_options,
                            )[0]
                        )
                        if options.get("physics") is not None:
                            loss = options["physics"](
                                values[1],
                                values[2],
                                values[3],
                                w,
                                values[5],
                                time_weights=options["time_weights"],
                                avoided=options["avoided"],
                                compute_gradient=False,
                            )[0]
                        else:
                            loss = objective(
                                values[1], values[2], values[3], w, values[5], **options
                            )[0]
                        result[group["name"]][mode].append(float(loss.item()))
        return result

    baseline_validation = evaluate(1, ("parent",))
    baseline_stats = {
        name: {
            "mean_loss": float(np.mean(v["parent"])),
            "passed95": int(np.sum(np.asarray(v["parent"]) <= 0.05 + 1e-8)),
        }
        for name, v in baseline_validation.items()
    }
    write(args.output / "baseline-validation.json", baseline_validation)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.1
    )
    rng = np.random.default_rng(760917)
    start = time.perf_counter()
    best = float("inf")
    history = []
    jobs = [(g, i) for g in groups if g["split"] == 0 for i in range(g["count"])]
    parent_losses = {}
    with torch.no_grad():
        for g, i in jobs:
            values, options, _ = batch(g, i)
            options["gradient_geometry"] = core.manifest["autoregressive"].get(
                "gradient_geometry"
            )
            _, trace = previous(*values, steps=args.steps, **options)
            parent_losses[(g["file"], i)] = float(trace[0, -1])
    write(
        args.output / "training-parent-losses.json",
        [
            {"file": key[0], "row": key[1], "loss": value}
            for key, value in parent_losses.items()
        ],
    )
    selected = None
    selected_epoch = None
    stale_epochs = 0
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(jobs)
        losses = []
        for g, i in jobs:
            values, options, _ = batch(g, i)
            optimizer.zero_grad(set_to_none=True)
            _, traces = model(*values, steps=args.steps, **options)
            terminal = traces[:, -1]
            teacher = parent_losses[(g["file"], i)]
            loss = (
                terminal.mean()
                + 0.1 * traces.mean()
                + 5 * torch.relu(terminal - teacher).mean()
            )
            if teacher <= 0.05:
                loss = loss + 20 * torch.relu(terminal - 0.049).mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite recurrent physical training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(1)
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "parent_sha256": core.sha256,
            },
            args.output / f"epoch-{epoch:03d}.pt",
        )
        score = float(np.mean([np.mean(v["learned"]) for v in validation.values()]))
        validation_stats = {
            name: {
                "mean_loss": float(np.mean(v["learned"])),
                "passed95": int(np.sum(np.asarray(v["learned"]) <= 0.05 + 1e-8)),
            }
            for name, v in validation.items()
        }
        eligible = all(
            v["mean_loss"] <= baseline_stats[name]["mean_loss"] + 1e-7
            and v["passed95"] >= baseline_stats[name]["passed95"]
            for name, v in validation_stats.items()
        )
        if eligible and score < best:
            best = score
            selected = deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            selected_epoch = epoch
            stale_epochs = 0
            torch.save(
                {"model": selected, "epoch": epoch, "parent_sha256": core.sha256},
                args.output / "training_checkpoint.pt",
            )
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation": validation_stats,
            "baseline_validation": baseline_stats,
            "productwise_selection_eligible": eligible,
            "selected_epoch": selected_epoch,
            "seconds": time.perf_counter() - start,
            'stale_validation_epochs':stale_epochs,
        }
        history.append(row)
        write(args.output / "learning_curve.json", history)
        print(json.dumps(row), flush=True)
        if args.early_stopping_patience and selected is not None and stale_epochs>=args.early_stopping_patience:
            stopped_early = True
            break
    if selected is not None:
        model.load_state_dict(selected)
    model.eval()
    test = evaluate(
        2,
        ("initial", "fixed", "parent", "learned")
        + (
            ("parent_same_steps",)
            if args.steps != int(core.manifest["autoregressive"]["default_steps"])
            else ()
        ),
    )
    arrays = dict(core.arrays)
    arrays.update(
        {
            "autoregressive." + k: v.detach().cpu().numpy()
            for k, v in model.state_dict().items()
        }
    )
    summary = {
        name: {
            mode: {
                "mean_loss": float(np.mean(v)),
                "passed95": int(np.sum(np.array(v) <= 0.05 + 1e-8)),
                "rows": len(v),
            }
            for mode, v in modes.items()
        }
        for name, modes in test.items()
    }
    parity = []
    for g in groups:
        if g["split"] != 2:
            continue
        values, options, cp = batch(g, 0)
        with torch.no_grad():
            expected = model(*values, steps=args.steps, **options)[0].cpu().numpy()
        cpu_values = [v.detach().cpu().numpy() for v in values]
        cpu_options = {
            key: value.detach().cpu().numpy()
            for key, value in options.items()
            if key not in ("physics", "gradient_geometry")
        }
        cpu_options["physics"] = cp
        cpu_options["gradient_geometry"] = options["gradient_geometry"]
        actual, _ = cpu_rollout(arrays, *cpu_values, steps=args.steps, **cpu_options)
        from fragrance_ai.recommender.aligned_autoregressive import (
            objective as cpu_objective,
        )

        lo = cpu_objective(
            cpu_values[1],
            cpu_values[2],
            cpu_values[3],
            actual,
            cpu_values[5],
            **cpu_options,
        )[0]
        le = cpu_objective(
            cpu_values[1],
            cpu_values[2],
            cpu_values[3],
            expected,
            cpu_values[5],
            **cpu_options,
        )[0]
        parity.append(
            {
                "product": g["name"],
                "n": g["n"],
                "mass_max_abs": float(np.max(np.abs(actual - expected))),
                "objective_abs": float(np.max(np.abs(lo - le))),
                "simplex_error": float(abs(actual.sum() - 1)),
            }
        )
    gates = {
        "autoregressive_vs_initial": all(
            v["learned"]["mean_loss"] <= v["initial"]["mean_loss"] + 1e-7
            for v in summary.values()
        ),
        "autoregressive_vs_fixed_optimizer": all(
            v["learned"]["mean_loss"] <= v["fixed"]["mean_loss"] + 1e-7
            for v in summary.values()
        ),
        "autoregressive_vs_parent": all(
            v["learned"]["mean_loss"] <= v["parent"]["mean_loss"] + 1e-7
            for v in summary.values()
        ),
        "autoregressive_pass_count_retention": all(
            v["learned"]["passed95"] >= v["parent"]["passed95"]
            for v in summary.values()
        ),
        "autoregressive_productwise_validation": selected is not None,
        "autoregressive_cpu_export": all(v["objective_abs"] <= 2e-4 for v in parity),
        "autoregressive_constraints": all(v["simplex_error"] <= 1e-6 for v in parity),
    }
    gates["nonlinear_inverse_fidelity"] = all(gates.values())
    report = {
        "summary": summary,
        "gates": gates,
        "parity": parity,
        "raw_test_losses": test,
        "selected_epoch": selected_epoch,
        "actual_human_similarity_claimed": False,
        "test_used_for_selection": False,
        "physical_draws": args.draws,
        "candidate_steps": args.steps,
        "parent_steps": int(core.manifest["autoregressive"]["default_steps"]),
        "fixed_steps": args.steps,
        'epochs_completed':len(history),'maximum_epochs':args.epochs,
        'early_stopped_on_validation':stopped_early,'controller_hidden_size':args.hidden_size,
        'controller_parameter_count':sum(v.numel() for v in model.parameters()),
    }
    write(args.output / "evaluation.json", report)
    np.savez_compressed(args.output / "weights.npz", **arrays)
    m = deepcopy(core.manifest)
    m["weights"] = {"path": "weights.npz", "sha256": sha(args.output / "weights.npz")}
    m["evaluation"]["gates"].update(gates)
    m["evaluation"]["gates"].pop("autoregressive_vs_v73", None)
    m["autoregressive"].update(
        physical_draws=args.draws,
        nonlinear_inverse_trained=True,
        gradient_geometry=args.gradient_geometry,
        default_steps=args.steps,
        nonlinear_inverse_report_sha256=sha(args.output / "evaluation.json"),
        training_protocol_sha256=sha(args.output / "protocol.json"),
        predictive_parent_sha256=core.sha256,
        full_reference_perfume=args.full_reference_perfume,
        profile_coordinates=protocol['profile_coordinates'],
        hidden_size=args.hidden_size,
        equal_horizon_weight_superiority_demonstrated=bool(args.steps==int(core.manifest['autoregressive']['default_steps'])
            and all(v['learned']['mean_loss']<v['parent']['mean_loss']-1e-7 for v in summary.values())),
    )
    m["architecture"]["autoregressive_decoder"]["rollout_steps"] = args.steps
    m['architecture']['autoregressive_decoder']['hidden_size'] = args.hidden_size
    m['parameter_count'] = sum(v.size for k,v in arrays.items() if '.' in k)
    m["accepted_for_local_inference"] = all(m["evaluation"]["gates"].values())
    write(args.output / "model.json", m)
    print(
        json.dumps(
            {
                "model_sha256": sha(args.output / "model.json"),
                "accepted": m["accepted_for_local_inference"],
                "summary": summary,
                "gates": gates,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
