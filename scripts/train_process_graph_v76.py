"""Train source-legal next-action sets and process checks; preserve other paths.

Only ordered-process encoders and action/check readouts may change. Synthetic
source-policy histories are explicitly separate from the measured outcome data.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    from fragrance_ai.recommender.formulation_core import (
        FormulationCore,
        forward_arrays,
    )
    from fragrance_ai.research.formulation_network import (
        FormulationNetwork,
        export_arrays,
    )
    from fragrance_ai.recommender.formulation_process import (
        TEMPLATES,
        ACTIONS,
        process_context,
        process_history,
        SOURCES,
    )
    from fragrance_ai.recommender.formulation_process_graph import (
        process_state,
        VERSION,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--contexts-per-template", type=int, default=1200)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    core = FormulationCore(a.core, sha(a.core), allow_candidate=True)
    rng = np.random.default_rng(760918)
    rows = []
    for template in TEMPLATES:
        for case in range(a.contexts_per_template):
            split = (
                0
                if case < int(0.7 * a.contexts_per_template)
                else 1
                if case < int(0.85 * a.contexts_per_template)
                else 2
            )
            values = {
                "peak_temperature_c": float(rng.choice([68.0, 70.0, 72.5, 75.0, 78.0])),
                "measured_ph": float(rng.choice([2.0, 3.0, 5.5, 7.0, 12.0, 13.0])),
                "mixing_minutes": float(rng.uniform(1, 20)),
                "batch_mass_g": float(rng.choice([400, 500, 1000])),
                "method_code": 0,
                "mixer_kind": "high_shear",
                "pe9010_present": bool(case % 2),
            }
            if case % 3 == 0:
                values.update(
                    peak_temperature_c=72.5,
                    measured_ph=5.5,
                    batch_mass_g=400 if template == "cold_lotion" else 500,
                )
            history = []
            for length in range(36):
                state = process_state(template, history, values)
                ids, numbers = process_history(history)
                rows.append(
                    (
                        process_context(template, values),
                        ids,
                        numbers,
                        state["mask"],
                        state["checks"],
                        split,
                        template,
                        case,
                    )
                )
                if state["allowed"] == ["hold"] or "done" in history:
                    break
                choices = state["allowed"]
                if length > 26 and "batch_record" in choices:
                    action = "batch_record"
                else:
                    action = str(rng.choice(choices))
                if length % 3 == 1:
                    invalid = [x for x in ACTIONS if x not in choices and x != "hold"]
                    wrong = history + [str(rng.choice(invalid))]
                    ws = process_state(template, wrong, values)
                    wi, wv = process_history(wrong)
                    rows.append(
                        (
                            process_context(template, values),
                            wi,
                            wv,
                            ws["mask"],
                            ws["checks"],
                            split,
                            template,
                            case,
                        )
                    )
                history.append(action)
    maximum = max(len(r[1]) for r in rows)
    ids = np.zeros((len(rows), maximum), np.int64)
    numbers = np.zeros((len(rows), maximum, 12), np.float32)
    for j, row in enumerate(rows):
        ids[j, : len(row[1])] = row[1]
        numbers[j, : len(row[1])] = row[2]
    contexts = np.array([r[0] for r in rows])
    allowed = np.array([r[3] for r in rows])
    checks = np.array([r[4] for r in rows])
    splits = np.array([r[5] for r in rows])
    np.savez_compressed(
        a.output / "histories.npz",
        context=contexts,
        ids=ids,
        numbers=numbers,
        allowed=allowed,
        checks=checks,
        split=splits,
    )
    protocol = {
        "schema": VERSION,
        "parent_sha256": core.sha256,
        "epochs": a.epochs,
        "rows": len(rows),
        "split_counts": np.bincount(splits).tolist(),
        "contexts_per_template": a.contexts_per_template,
        "split": "independent_numeric_contexts_all_prefixes_and_corruptions_grouped",
        "label_kind": "project_authored_source_dependency_constraints_not_observed_batches",
        "all_training_rows_used": True,
        "sources": SOURCES,
        "history_sha256": sha(a.output / "histories.npz"),
        "source_graph_sha256": sha(
            ROOT / "fragrance_ai/recommender/formulation_process_graph.py"
        ),
    }
    (a.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(760918)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = FormulationNetwork(
        core.feature_width,
        450,
        292,
        len(ACTIONS),
        blend_outputs=109,
        aqueous_outputs=18,
    ).to(device)
    model.load_state_dict(
        {
            k: torch.tensor(v.copy())
            for k, v in core.arrays.items()
            if "." in k and not k.startswith("autoregressive.")
        }
    )
    permitted = (
        "step_embedding.",
        "process_input.",
        "process_state.",
        "action_head.",
        "check_head.",
    )
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith(permitted))
    tensors = {
        k: torch.tensor(v, device=device)
        for k, v in dict(
            context=contexts, ids=ids, numbers=numbers, allowed=allowed, checks=checks
        ).items()
    }

    def forward(indices):
        c, si, sv = (
            tensors["context"][indices],
            tensors["ids"][indices],
            tensors["numbers"][indices],
        )
        length = int((si != 0).sum(-1).max())
        si, sv = si[:, :length], sv[:, :length]
        state = torch.zeros((len(indices), 96), device=device)
        for j in range(length):
            embedded = torch.cat((model.step_embedding(si[:, j]), sv[:, j]), -1)
            updated = torch.tanh(
                model.process_input(embedded) + model.process_state(state)
            )
            state = torch.where((si[:, j] != 0)[:, None], updated, state)
        fused = torch.cat(
            (
                torch.zeros((len(indices), 768), device=device),
                torch.relu(model.context_in(c)),
                state,
            ),
            -1,
        )
        h = torch.relu(model.shared_in(torch.relu(model.fusion(fused))))
        for block in model.shared:
            h = block(h)
        h = model.observed_dropout(h)
        return model.action_head(h), model.check_head(h)

    sets = [np.flatnonzero(splits == s) for s in range(3)]

    def evaluate(which):
        model.eval()
        valid = []
        correct = []
        with torch.no_grad():
            for start in range(0, len(sets[which]), 512):
                index = sets[which][start : start + 512]
                action, check = forward(index)
                valid.extend(
                    tensors["allowed"][index]
                    .gather(1, action.argmax(-1)[:, None])[:, 0]
                    .cpu()
                    .tolist()
                )
                correct.extend(
                    ((check.sigmoid() >= 0.5) == (tensors["checks"][index] > 0.5))
                    .float()
                    .mean(-1)
                    .cpu()
                    .tolist()
                )
        return {
            "raw_valid_next_action_rate": float(np.mean(valid)),
            "check_accuracy": float(np.mean(correct)),
            "rows": len(valid),
        }

    before = evaluate(1)
    best = float("inf")
    history = []
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        weight_decay=0.00001,
    )
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(sets[0])
        losses = []
        for offset in range(0, len(order), 512):
            index = order[offset : offset + 512]
            optimizer.zero_grad(set_to_none=True)
            action, check = forward(index)
            permitted_logits = action.masked_fill(tensors["allowed"][index] == 0, -1e30)
            loss = (
                torch.logsumexp(action, -1) - torch.logsumexp(permitted_logits, -1)
            ).mean() + 0.3 * torch.nn.functional.binary_cross_entropy_with_logits(
                check, tensors["checks"][index]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(1)
        criterion = (
            2 * (1 - metrics["raw_valid_next_action_rate"])
            + 1
            - metrics["check_accuracy"]
        )
        if criterion < best:
            best = criterion
            selected = deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            selected_epoch = epoch
            torch.save(
                {"model": selected, "epoch": epoch}, a.output / "training_checkpoint.pt"
            )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                **metrics,
                "seconds": time.perf_counter() - start,
            }
        )
        (a.output / "learning_curve.json").write_text(
            json.dumps(history, indent=2), encoding="utf8"
        )
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(selected)
    model.eval()
    test = evaluate(2)
    arrays = dict(core.arrays)
    arrays.update(export_arrays(model))
    unchanged = all(
        np.array_equal(v, arrays[k])
        for k, v in core.arrays.items()
        if not k.startswith(permitted)
    )
    if not unchanged:
        raise ValueError("nonprocess weights changed during isolated process learning")
    index = sets[2][:11]
    with torch.no_grad():
        action, check = forward(index)
    zeros = np.zeros((len(index), 1, core.feature_width), np.float32)
    computed = forward_arrays(
        arrays,
        zeros,
        np.zeros((len(index), 1), np.float32),
        contexts[index],
        ids[index],
        numbers[index],
    )
    parity = max(
        float(np.max(np.abs(computed["action"] - action.cpu().numpy()))),
        float(np.max(np.abs(computed["check"] - check.cpu().numpy()))),
    )
    report = {
        "before_validation": before,
        "test": test,
        "selected_epoch": selected_epoch,
        "nonprocess_weights_exact": unchanged,
        "permitted_parameter_prefixes": list(permitted),
        "cpu_export_max_error": parity,
        "eligible_for_assembly": test["raw_valid_next_action_rate"] >= 0.98
        and test["check_accuracy"] >= 0.98
        and parity <= 0.0002,
        "parent_sha256": core.sha256,
        "source_graph_sha256": protocol["source_graph_sha256"],
    }
    np.savez_compressed(a.output / "weights.npz", **arrays)
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
