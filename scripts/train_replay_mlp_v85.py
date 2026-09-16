"""Nested replay-policy learning followed by fresh, budget-matched MLP training.

Each outer held-out mixture is absent from policy-world fitting/validation and
live model selection. Previous public-data exposure is still declared: this is
development evidence, not a new sealed human validation or a service rollout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PERFUMERY_AI_LOCAL_PROFILE"] = "disabled"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_physmix_v83 import metrics, pair_key, sha, source_audit, write  # noqa: E402
from scripts.compare_mixture_mlp_v84 import mixture_partitions  # noqa: E402


def inner_split(pairs, available, seed):
    import numpy as np
    keys = sorted({pair_key(pairs[i]) for i in available})
    order = np.random.default_rng(seed).permutation(len(keys))
    validation = {keys[i] for i in order[:max(1, round(.25 * len(keys)))]}
    return {"train": [i for i in available if pair_key(pairs[i]) not in validation],
            "validation": [i for i in available if pair_key(pairs[i]) in validation]}


def save_branch(branch, folder, binding):
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "history.json", [{**{k: v for k, v in row.items() if k != "observation"},
                                    "observation": asdict(row["observation"])} for row in branch.history])
    if branch.selected is not None:
        torch.save({"state_dict": branch.selected["state"], "mode": branch.mode,
                    "selected_epoch": branch.selected["epoch"], "validation_rmse": branch.selected["rmse"],
                    "binding": binding}, folder / "model.pt")
        return {"mode": branch.mode, "epochs": branch.step, "selected_epoch": branch.selected["epoch"],
                "validation_rmse": branch.selected["rmse"], "sha256": sha(folder / "model.pt")}
    return {"mode": branch.mode, "epochs": branch.step, "status": "no_selected_checkpoint"}


def main():
    import numpy as np
    import torch
    from fragrance_ai.recommender.formulation_core import FormulationCore
    from fragrance_ai.research.mixture_mlp import MixtureMLP
    from fragrance_ai.research.mixture_replay_training import PairBank, TrainingBranch, validation_ensemble
    from fragrance_ai.research.replay_search import Observation, Policy, improve_policy, replay, run_live
    from fragrance_ai.research.r2_physsim import load_snitz_pairs, load_ravia_pairs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--history-source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(.25)
        torch.backends.cuda.matmul.allow_tf32 = False
    core = FormulationCore(args.core, sha(args.core))
    snitz, ravia = load_snitz_pairs(args.data_root), load_ravia_pairs(args.data_root)
    if (len(snitz), len(ravia)) != (360, 182):
        raise ValueError("complete public pair sets required")
    seeds = [850042, 850123, 850456]
    splits = [split for seed in seeds for split in mixture_partitions(snitz, seed)]
    for split in splits:
        inner = [inner_split(snitz, split["train"], split["seed"] + split["fold"] * 1009 + offset)
                 for offset in (500003, 700009)]
        forbidden = set(split["test"]) | set(split["validation"])
        if any(forbidden & set(row["train"] + row["validation"]) for row in inner):
            raise ValueError("outer evaluation or selection labels leaked into replay worlds")
        split["inner_worlds"] = inner
    sources = [Path(__file__), ROOT / "fragrance_ai/research/replay_search.py",
               ROOT / "fragrance_ai/research/mixture_replay_training.py", ROOT / "fragrance_ai/research/mixture_mlp.py",
               ROOT / "fragrance_ai/research/mixture_mlp_numpy.py", ROOT / "fragrance_ai/research/physmix_comparison.py",
               ROOT / "fragrance_ai/research/r2_physsim.py", ROOT / "scripts/compare_mixture_mlp_v84.py",
               ROOT / "scripts/compare_physmix_v83.py", ROOT / "fragrance_ai/recommender/formulation_core.py"]
    files = [args.data_root / p for p in ("snitz_2013/molecules.csv", "snitz_2013/behavior.csv",
             "ravia_2020/molecules.csv", "ravia_2020/stimuli.csv", "ravia_2020/behavior_2.csv")]
    protocol = {"schema": "nested-replay-mlp/v85", "core": str(args.core.resolve()), "core_sha256": core.sha256,
                "sources": {str(path): sha(path) for path in sources}, "data": {str(path): sha(path) for path in files},
                "source_audit": source_audit(args.data_root), "seeds": seeds, "splits": splits,
                "modes": list(MixtureMLP.MODES), "live_policies": ["fixed", "replay"],
                "work_budget_epochs": 120, "branch_epoch_limit": 60, "branch_patience": 8,
                "training": {"lr": .0003, "weight_decay": .0001, "batch_size": 16,
                             "loss": "0.7_MSE_plus_0.3_rank", "dropout_rng": "branch_and_epoch_local"},
                "policy_worlds_per_outer_fold": 2, "numeric_policy_candidates": 109,
                "replay_quality_tolerance": .001, "ensemble_regularization": .001,
                "policy_learning_labels": "outer_training_subset_only",
                "checkpoint_and_ensemble_selection": "outer_inner_validation_only",
                "no_test_or_transfer_labels_in_policy": True, "all_360_pairs_in_each_seed": True,
                "previous_public_data_exposure": True, "molecule_cold_claimed": False,
                "parent_molecular_encoder_frozen": True, "physical_evaluator_unchanged": True,
                "deployed": False, "external_LLM_calls": 0, "device": args.device,
                "promotion_policy": {"no_automatic_deployment": True, "max_mae_regression": .002,
                                     "max_ravia_mae_regression": .003, "minimum_work_reduction": .1,
                                     "at_least_two_improving_seed_maes": True}}
    historical_protocol = None
    if args.history_source is not None:
        historical_protocol = read(args.history_source / "protocol.json")
        for key in ("core_sha256", "data", "splits", "training", "modes", "branch_epoch_limit", "branch_patience"):
            if historical_protocol[key] != protocol[key]:
                raise ValueError("history reuse would change data, split, or training conditions")
        for path in (ROOT / "fragrance_ai/research/mixture_replay_training.py", ROOT / "fragrance_ai/research/mixture_mlp.py",
                     ROOT / "fragrance_ai/research/physmix_comparison.py", ROOT / "fragrance_ai/research/r2_physsim.py",
                     ROOT / "fragrance_ai/recommender/formulation_core.py"):
            if historical_protocol["sources"][str(path)] != sha(path):
                raise ValueError("history belongs to a different training implementation")
    protocol["history_source"] = ({"path": str(args.history_source.resolve()),
                                   "protocol_sha256": sha(args.history_source / "protocol.json")}
                                  if args.history_source else None)
    if args.resume:
        if read(args.output / "protocol.json") != protocol:
            raise ValueError("resume source/data/protocol drift")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "protocol.json", protocol)
        (args.output / "source").mkdir()
        for path in sources:
            (args.output / "source" / path.name).write_bytes(path.read_bytes())
    binding = {"core_sha256": core.sha256, "protocol_sha256": sha(args.output / "protocol.json")}
    bank = PairBank(core, {"snitz": snitz, "ravia": ravia}, args.device)
    np.savez_compressed(args.output / "molecular_embeddings.npz", embeddings=bank.molecular_embeddings)
    records = []
    for ordinal, split in enumerate(splits):
        tag = f'seed-{split["seed"]}-fold-{split["fold"]}'
        folder = args.output / tag
        if (folder / "result.json").exists():
            record = read(folder / "result.json")
            for policy_name in protocol["live_policies"]:
                for row in record["policies"][policy_name]["models"]:
                    if sha(folder / policy_name / row["mode"] / "model.pt") != row["sha256"]:
                        raise ValueError("resume checkpoint integrity failure")
            records.append(record)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        task_start, worlds, history_cost = time.perf_counter(), [], 0.
        seed = split["seed"] + split["fold"] * 1009
        for world_id, inner in enumerate(split["inner_worlds"]):
            world = {}
            for mode in MixtureMLP.MODES:
                if args.history_source is not None:
                    old = args.history_source / tag / f"inner-{world_id}" / mode
                    history = read(old / "history.json")
                    checkpoint = torch.load(old / "model.pt", map_location="cpu", weights_only=True)
                    if checkpoint["binding"]["protocol_sha256"] != protocol["history_source"]["protocol_sha256"]:
                        raise ValueError("historical model bound to a different run")
                    selected = history[checkpoint["selected_epoch"] - 1]["observation"]
                    if abs(-selected["score"] - checkpoint["validation_rmse"]) > 1e-12:
                        raise ValueError("historical selection and checkpoint disagree")
                    world[mode] = [Observation(**row["observation"]) for row in history]
                    history_cost += sum(o.seconds for o in world[mode])
                    destination = folder / f"inner-{world_id}" / mode
                    destination.mkdir(parents=True, exist_ok=True)
                    write(destination / "history.json", history)
                    write(destination / "reused.json", {"history_path": str(old.resolve()),
                          "history_sha256": sha(old / "history.json"), "checkpoint_sha256": sha(old / "model.pt"),
                          "historical_protocol_sha256": protocol["history_source"]["protocol_sha256"]})
                    continue
                trainer = TrainingBranch(bank, mode, inner["train"], inner["validation"], seed + 90001 * (world_id + 1))
                try:
                    while True:
                        observation = trainer.advance()
                        if observation.terminal:
                            break
                finally:
                    save_branch(trainer, folder / f"inner-{world_id}" / mode, binding)
                world[mode] = [row["observation"] for row in trainer.history]
                history_cost += sum(o.seconds for o in world[mode])
            worlds.append(world)
            # The pool expands; policy versions are selected only from nested
            # training history, before live outer validation/test predictions.
            chosen, search = improve_policy(worlds, budget=120, quality_tolerance=.001)
            write(folder / f"policy-cycle-{world_id}.json", {**search, "policy": asdict(chosen), "binding": binding})
            print(json.dumps({"task": tag, "history_worlds": len(worlds), "policy_candidates": search["evaluated_policies"]}), flush=True)
        frozen_policy = chosen
        write(folder / "frozen_policy.json", {"policy": asdict(frozen_policy), "sha256": frozen_policy.digest, "binding": binding})
        policies = {}
        for policy_name, policy in (("fixed", Policy(kind="round_robin")), ("replay", frozen_policy)):
            trainers = {mode: TrainingBranch(bank, mode, split["train"], split["validation"], seed) for mode in MixtureMLP.MODES}
            policy_folder = folder / policy_name
            policy_folder.mkdir(parents=True, exist_ok=True)
            live_started = time.perf_counter()
            try:
                trace = run_live(tuple(trainers), lambda name: trainers[name].advance(), policy, budget=120,
                                 on_reveal=lambda obs, decision: write(policy_folder / "progress.json", decision))
            finally:
                models = [save_branch(trainer, policy_folder / mode, binding) for mode, trainer in trainers.items()]
            # Replaying the just-completed live trace must reproduce every
            # action and selected result without running a single extra epoch.
            current_world = {mode: [row["observation"] for row in trainer.history] for mode, trainer in trainers.items()}
            replayed = replay(current_world, policy, budget=120)
            if replayed != trace:
                raise ValueError("live/replay decision parity failed")
            write(policy_folder / "trace.json", trace)
            active = [mode for mode, trainer in trainers.items() if trainer.selected is not None]
            val = [trainers[mode].selected["validation_prediction"] for mode in active]
            target = [snitz[i].similarity for i in split["validation"]]
            weights, selection = validation_ensemble(val, target)
            single = selection["best_single"]
            test_predictions, transfer_predictions = [], []
            for mode in active:
                model = trainers[mode].load_selected()
                test_predictions.append(bank.predict(model, "snitz", split["test"]))
                transfer_predictions.append(bank.predict(model, "ravia", list(range(len(ravia)))))
            test_values, transfer_values = np.asarray(test_predictions), np.asarray(transfer_predictions)
            policies[policy_name] = {"policy": asdict(policy), "policy_sha256": policy.digest,
                "models": models, "active_modes": active, "ensemble_weights": weights.tolist(), "selection": selection,
                "test_indices": split["test"], "test_predictions": (weights @ test_values).tolist(),
                "transfer_predictions": (weights @ transfer_values).tolist(),
                "single_test_predictions": test_values[single].tolist(),
                "single_transfer_predictions": transfer_values[single].tolist(),
                "selection_rmse": float(np.sqrt(np.mean((weights @ np.asarray(val) - target)**2))),
                "work_epochs": trace["work"], "measured_training_seconds": trace["seconds"],
                "optimizer_updates": sum(row["optimizer_updates"] for trainer in trainers.values() for row in trainer.history),
                "live_wall_seconds": time.perf_counter() - live_started, "live_replay_identical": True,
                "stop_reason": trace["stop_reason"]}
            write(policy_folder / "result.json", policies[policy_name])
        record = {"seed": split["seed"], "fold": split["fold"], "policies": policies,
                  "history_training_seconds": history_cost, "total_task_seconds": time.perf_counter() - task_start,
                  "history_reused": args.history_source is not None,
                  "policy_trained_from_outer_training_only": True, "binding": binding}
        write(folder / "result.json", record)
        records.append(record)
        progress = {"completed_tasks": ordinal + 1, "expected_tasks": 15, "complete": ordinal == 14, "last_task": tag}
        write(args.output / "progress.json", progress)
        print(json.dumps({**progress, "fixed_epochs": policies["fixed"]["work_epochs"],
                          "replay_epochs": policies["replay"]["work_epochs"]}), flush=True)
    aggregate(records, protocol, snitz, ravia, args.output)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def aggregate(records, protocol, snitz, ravia, output):
    import numpy as np
    truth = np.asarray([p.similarity for p in snitz])
    transfer_truth = np.asarray([p.similarity for p in ravia])
    scores = {}
    for policy in protocol["live_policies"]:
        for variant in ("ensemble", "single"):
            field = "" if variant == "ensemble" else "single_"
            summed, counts = np.zeros((3, len(snitz))), np.zeros((3, len(snitz)))
            transfer = []
            for row in records:
                r = row["policies"][policy]
                s, ix = protocol["seeds"].index(row["seed"]), r["test_indices"]
                summed[s, ix] += r[field + "test_predictions"]
                counts[s, ix] += 1
                transfer.append(r[field + "transfer_predictions"])
            if np.any(counts == 0):
                raise ValueError("incomplete test coverage")
            oof, transferred = summed / counts, np.mean(transfer, axis=0)
            key = policy + "_" + variant
            scores[key] = {"snitz": metrics(oof.mean(0), truth), "seed_snitz": [metrics(p, truth) for p in oof],
                           "ravia": metrics(transferred, transfer_truth)}
            np.savez_compressed(output / (key + ".npz"), oof=oof, target=truth, counts=counts,
                                transfer=transferred, transfer_target=transfer_truth)
    costs = {policy: {"work_epochs": sum(r["policies"][policy]["work_epochs"] for r in records),
                       "optimizer_updates": sum(r["policies"][policy]["optimizer_updates"] for r in records),
                       "training_seconds": sum(r["policies"][policy]["measured_training_seconds"] for r in records),
                       "live_wall_seconds": sum(r["policies"][policy]["live_wall_seconds"] for r in records)}
             for policy in protocol["live_policies"]}
    result = {"complete": True, "tasks": len(records), "scores": scores, "costs": costs,
              "upfront_history_training_seconds": sum(r["history_training_seconds"] for r in records),
              "new_history_training_seconds": sum(r["history_training_seconds"] for r in records if not r.get("history_reused", False)),
              "total_experiment_seconds": sum(r["total_task_seconds"] for r in records),
              "protocol_sha256": sha(output / "protocol.json"), "deployed": False,
              "model_profile_threshold_changed": False, "new_real_sensory_accuracy_claimed": False}
    write(output / "comparison.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
