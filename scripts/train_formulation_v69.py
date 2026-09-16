"""Jointly train one checkpoint on all prepared v69 local-source task rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402 - thread settings precede numerical imports
import torch  # noqa: E402
from torch import nn  # noqa: E402
from fragrance_ai.research.formulation_network import FormulationNetwork, export_arrays  # noqa: E402
from fragrance_ai.recommender.formulation_core import (  # noqa: E402
    VERSION,
    forward_arrays,
)
from fragrance_ai.recommender.unified_transport import (  # noqa: E402
    features,
    baseline_kernel,
    kernel_mask,
    correction_gate,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(
        (a * b).sum(1), denominator, out=np.zeros(len(a)), where=denominator > 0
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--observed-data", type=Path)
    parser.add_argument("--physical-mixtures", type=Path)
    parser.add_argument("--parent-core", type=Path)
    parser.add_argument(
        "--worst-profile-loss",
        action="store_true",
        help="optimize worst complete profile head as well as source magnitude fidelity",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="train molecular encoder and quantitative readout while preserving context/process-only paths",
    )
    parser.add_argument(
        "--profile-shape-loss",
        action="store_true",
        help="also distill full normalized output shapes, not only per-column log magnitude",
    )
    args = parser.parse_args()
    if args.observed_data and (not args.physical_mixtures or not args.parent_core or args.profile_only):
        raise ValueError('V76 requires physical mixture labels, parent core and whole-network training')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("positive training budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in meta["files"].items():
        if sha(args.data / name) != digest:
            raise ValueError("training dataset drift: " + name)
    protocol = {
        "schema": "shared-formulation-training/v69",
        "data_sha256": sha(args.data / "manifest.json"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": 690913,
        "all_rows_in_each_task_used_each_epoch": True,
        "old_networks_called_at_inference": False,
        "validation_selection": "sum_of_normalized_teacher_errors_plus_procedure_error",
        "test_used_for_epoch_selection": False,
        "initial_checkpoint_sha256": sha(args.initial) if args.initial else None,
        "full_profile_shape_loss": args.profile_shape_loss,
        "worst_profile_shape_loss": args.worst_profile_loss,
        "profile_only_parameter_paths": args.profile_only,
        "local_gate": {
            "quantitative_median_cosine_min": 0.95,
            "fine_probability_mae_max": 0.035,
            "physical_transition_rmse_max": 0.035,
            "procedure_accuracy_min": 0.98,
            "cpu_export_max_error": 0.0002,
        },
        "evaluation_scope": meta["training_sources"]["evaluation_scope"],
    }
    if args.observed_data:
        protocol.update(schema='source_separated_joint_training/v76',
            observed_data_sha256=sha(args.observed_data/'manifest.json'),
            physical_mixture_sha256=sha(args.physical_mixtures),
            parent_core_sha256=sha(args.parent_core),
            evaluation_scope='source_scaffold_and_formulation_combination_holdouts_not_formula_human_similarity')
    write(args.output / "protocol.json", protocol)
    torch.manual_seed(690913)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(690913)
    device = torch.device(args.device)
    m = dict(np.load(args.data / "molecules.npz", allow_pickle=False))
    t = dict(np.load(args.data / "transport.npz", allow_pickle=False))
    p = dict(np.load(args.data / "process.npz", allow_pickle=False))
    from fragrance_ai.recommender.emulsion_science import (
        baseline as emulsion_baseline,
        context as emulsion_context,
    )

    e = dict(np.load(args.data / "emulsion.npz", allow_pickle=False))
    train = m["split"] == 0
    feature_mean, feature_scale = (
        np.zeros(m["x"].shape[1], np.float32),
        np.ones(m["x"].shape[1], np.float32),
    )
    feature_mean[1024:1040] = m["x"][train, 1024:1040].mean(0)
    feature_scale[1024:1040] = np.maximum(m["x"][train, 1024:1040].std(0), 1e-4)
    # Keep binary fingerprint/annotations binary and normalize native amplitude.
    feature_scale[1040:1059] = np.maximum(
        np.sqrt((m["x"][train, 1040:1059] ** 2).mean(0)), 0.1
    )
    qscale = np.maximum(
        np.sqrt(
            np.mean(np.log1p(np.r_[m["high"][train], m["low"][train]]) ** 2, axis=0)
        ),
        0.1,
    )
    tf = features(t["raw"]).astype(np.float32)
    transport_mean = tf[t["split"] == 0].mean(0)
    transport_scale = np.maximum(tf[t["split"] == 0].std(0), 1e-4)
    normalization = {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "quantitative_scale": qscale,
        "transport_mean": transport_mean,
        "transport_scale": transport_scale,
    }
    x = np.clip((m["x"] - feature_mean) / feature_scale, -12, 12)
    zero = np.clip(-feature_mean / feature_scale, -12, 12)
    transport_context = np.zeros((len(tf), 64), np.float32)
    transport_context[:, 4:12] = (tf - transport_mean) / transport_scale
    transport_context[:, 12] = 1.0
    # Train the set aggregation and revision readout on actual multi-component
    # inputs. These labels preserve the old additive product proxy; they are NOT
    # newly measured synergy/suppression or professional formulation decisions.
    from fragrance_ai.recommender.lotion_atlas import ATLAS_PROJECTION
    from fragrance_ai.recommender.models import SCENT_DIMENSIONS

    projection = np.zeros((146, 19), np.float32)
    for j, axis in enumerate(SCENT_DIMENSIONS):
        for name in ATLAS_PROJECTION.get(axis, ()):
            projection[meta["quantitative_endpoints"].index(name), j] = 1.0
    mix_ids, mix_masses, mix_context, mix_q, mix_f, mix_delta, mix_split = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for split_id, count in enumerate((6000, 1000, 1000)):
        pool = np.flatnonzero(m["split"] == split_id)
        for _ in range(count):
            n = int(rng.integers(2, 9))
            members = rng.choice(pool, n, replace=False)
            fractions = rng.dirichlet(np.full(n, 0.8)).astype(np.float32)
            ids = np.zeros(8, np.int64)
            ids[:n] = members
            mass = np.zeros(8, np.float32)
            mass[:n] = fractions
            q, fine_target = (
                fractions @ m["high"][members],
                fractions @ m["fine"][members],
            )
            coarse = q[:146] @ projection
            coarse /= max(float(coarse.sum()), 1e-12)
            desired = rng.dirichlet(0.15 + 3 * coarse).astype(np.float32)
            context = np.zeros(64, np.float32)
            context[int(rng.integers(0, 3))] = 1.0
            context[25:44], context[44:63] = desired, coarse
            context[63] = float(rng.uniform(-4.0, -0.5))
            mix_ids.append(ids)
            mix_masses.append(mass)
            mix_context.append(context)
            mix_q.append(q)
            mix_f.append(fine_target)
            mix_delta.append(desired - coarse)
            mix_split.append(split_id)
    mix = {
        "ids": np.asarray(mix_ids),
        "masses": np.asarray(mix_masses),
        "context": np.asarray(mix_context),
        "quantitative": np.asarray(mix_q),
        "fine": np.asarray(mix_f),
        "revision": np.asarray(mix_delta),
        "split": np.asarray(mix_split, np.int8),
    }
    if args.physical_mixtures:
        physical=dict(np.load(args.physical_mixtures,allow_pickle=False))
        mix={key:physical[{'masses':'masses'}.get(key,key)] for key in mix}
        mix['context'][:,12]=2.
    np.savez_compressed(args.output / "mixture-training.npz", **mix)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    bank = {
        "x": tensor(x),
        "high": tensor(np.log1p(m["high"]) / qscale),
        "low": tensor(np.log1p(m["low"]) / qscale),
        "fine": tensor(m["fine"]),
        "tc": tensor(transport_context),
        "pt": tensor(t["target"]),
        "base": tensor(np.log(np.maximum(baseline_kernel(t["raw"]), 1e-30))),
        "mask": tensor(kernel_mask(t["raw"]), torch.bool),
        "gate": tensor(correction_gate(t["raw"])),
        "pc": tensor(p["context"]),
        "pi": tensor(p["ids"], torch.int64),
        "pv": tensor(p["numbers"]),
        "pa": tensor(p["action"], torch.int64),
        "pf": tensor(p["checks"]),
    }
    bank.update(
        mi=tensor(mix["ids"], torch.int64),
        mm=tensor(mix["masses"]),
        mc=tensor(mix["context"]),
        mq=tensor(np.log1p(mix["quantitative"]) / qscale),
        mf=tensor(mix["fine"]),
        mr=tensor(mix["revision"]),
    )
    eb = emulsion_baseline(e["raw"], meta["emulsion"])
    bank.update(
        ec=tensor(emulsion_context(e["raw"], meta["emulsion"])),
        eb=tensor(eb),
        ey=tensor(e["target"]),
    )
    observed_meta=None
    if args.observed_data:
        observed_meta=json.loads((args.observed_data/'manifest.json').read_text(encoding='utf8'))
        for name,digest in observed_meta['files'].items():
            if sha(args.observed_data/name)!=digest:
                raise ValueError('observed training source drift')
        pair=dict(np.load(args.observed_data/'odor-pairs.npz',allow_pickle=False))
        aqueous=dict(np.load(args.observed_data/'liquid.npz',allow_pickle=False))
        from fragrance_ai.recommender.observed_formulation import aqueous_context
        bank.update(pair_x=tensor(np.clip((pair['x']-feature_mean)/feature_scale,-12,12)),
                    pair_ids=tensor(pair['ids'],torch.int64), pair_y=tensor(pair['target']),
                    aqueous_c=tensor(aqueous_context(aqueous['x'])),
                    aqueous_y=tensor(aqueous['normalized_target']),aqueous_mask=tensor(aqueous['mask']))
    model = FormulationNetwork(x.shape[1], 450, 292, len(meta["actions"]),
                               blend_outputs=109 if observed_meta else 0,aqueous_outputs=18 if observed_meta else 0).to(device)
    prior = np.clip(m["fine"][train].mean(0), 1e-5, 1 - 1e-5)
    with torch.no_grad():
        model.fine_head.bias.copy_(tensor(np.log(prior / (1 - prior))))
    if args.initial:
        missing, unexpected = model.load_state_dict(
            torch.load(args.initial, map_location="cpu", weights_only=True)["model"],
            strict=False,
        )
        if unexpected or set(missing) - {"emulsion_head.weight", "emulsion_head.bias",'blend_head.weight','blend_head.bias','aqueous_head.weight','aqueous_head.bias'}:
            raise ValueError("warm-start architecture mismatch")
    fine_weights = tensor(np.clip(1 / np.sqrt(np.maximum(prior, 0.01)), 1, 8))
    fine_weights /= fine_weights.mean()
    if args.profile_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("molecule_in.", "molecule_out.", "quantitative_head."))
            )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0002 if args.initial else 0.0008,
        weight_decay=0.0001,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0.00002
    )
    indices = {
        task: [np.flatnonzero(data["split"] == i) for i in range(3)]
        for task, data in [
            ("molecule", m),
            ("transport", t),
            ("process", p),
            ("mixture", mix),
            ("emulsion", e),
        ]
    }
    if observed_meta:
        indices.update({name:[np.flatnonzero(data['split']==i) for i in range(3)]
                        for name,data in [('pair',pair),('aqueous',aqueous)]})
        if any(not len(group) for groups in indices.values() for group in groups):
            raise ValueError('every task requires populated train, validation and test groups')
        with torch.no_grad():
            freq=np.clip(pair['target'][indices['pair'][0]].mean(0),1e-4,1-1e-4)
            model.blend_head.bias.copy_(tensor(np.log(freq/(1-freq))))
    parameter_count = sum(v.numel() for v in model.parameters())

    def shape_loss(predicted, target):
        # The lotion optimizer compares normalized FULL 146-D distributions.
        # Per-column log MSE can hide small coordinated errors in those shapes.
        # No recipe request, pass label or reference-target score enters here.
        scale = tensor(qscale)
        p = (
            torch.expm1((predicted * scale).clamp(-20, 20))
            .clamp_min(1e-7)
            .reshape(-1, 2, 146)
        )
        t = torch.expm1(target * scale).clamp_min(0).reshape(-1, 2, 146)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-12)
        if args.worst_profile_loss:
            tv = 0.5 * (p - t).abs().sum(-1)
            cosine = 1 - nn.functional.cosine_similarity(p, t, dim=-1)
            return torch.maximum(tv, cosine).amax(-1).mean()
        return (
            0.3 * torch.abs(p - t).sum(-1).mean()
            + 0.2 * (1 - nn.functional.cosine_similarity(p, t, dim=-1)).mean()
        )

    write(
        args.output / "started.json",
        {
            "parameter_count": parameter_count,
            "device": str(device),
            "rows": {k: [len(ids) for ids in groups] for k, groups in indices.items()},
            "training_executed": False,
        },
    )

    def inputs(task, ids, low=False):
        b = len(ids)
        context = torch.zeros((b, 64), device=device)
        si = torch.zeros((b, 0), dtype=torch.int64, device=device)
        sv = torch.zeros((b, 0, 12), device=device)
        if task == "molecule":
            mx = bank["x"][ids].clone()
            if low:
                mx[:, -2], mx[:, -1] = 0.0, 1.0
            return mx[:, None], torch.ones((b, 1), device=device), context, si, sv
        if task == "mixture":
            return bank["x"][bank["mi"][ids]], bank["mm"][ids], bank["mc"][ids], si, sv
        if task == 'pair':
            context[:,3]=1.
            context[:,12]=-1.
            return bank['pair_x'][bank['pair_ids'][ids]],torch.ones((b,2),device=device),context,si,sv
        mx = tensor(zero).expand(b, 1, -1)
        mw = torch.zeros((b, 1), device=device)
        if task == "transport":
            return mx, mw, bank["tc"][ids], si, sv
        if task == "emulsion":
            return mx, mw, bank["ec"][ids], si, sv
        if task == 'aqueous':
            return mx,mw,bank['aqueous_c'][ids],si,sv
        return mx, mw, bank["pc"][ids], bank["pi"][ids], bank["pv"][ids]

    def transition(out, ids):
        logits = (
            bank["base"][ids] + out.reshape(-1, 2, 5) * bank["gate"][ids, None, None]
        )
        return torch.softmax(logits.masked_fill(~bank["mask"][ids], -1e30), -1)

    stability_calibration={'slope':1.,'intercept':0.,'fit_split':'not_fitted'}

    def evaluate(split):
        model.eval()
        report = {}
        with torch.no_grad():
            for task in indices:
                outputs = []
                for offset in range(0, len(indices[task][split]), args.batch_size):
                    ids = indices[task][split][offset : offset + args.batch_size]
                    result = model(*inputs(task, ids))
                    if task == "molecule":
                        low = model(*inputs(task, ids, True))
                        outputs.append(
                            (
                                result["quantitative"].cpu().numpy(),
                                low["quantitative"].cpu().numpy(),
                                torch.sigmoid(result["fine"]).cpu().numpy(),
                            )
                        )
                    elif task == "mixture":
                        outputs.append(
                            (
                                result["quantitative"].cpu().numpy(),
                                result["revision"].cpu().numpy(),
                            )
                        )
                    elif task == "transport":
                        outputs.append(
                            transition(result["transport"], ids).cpu().numpy()
                        )
                    elif task in ('pair','aqueous'):
                        outputs.append(result['blend' if task=='pair' else 'aqueous'].cpu().numpy())
                    elif task == "emulsion":
                        outputs.append(
                            torch.softmax(result["emulsion"] + bank["eb"][ids], -1)
                            .cpu()
                            .numpy()
                        )
                    else:
                        outputs.append(
                            (
                                result["action"].argmax(-1).cpu().numpy(),
                                torch.sigmoid(result["check"]).cpu().numpy(),
                            )
                        )
                ids = indices[task][split]
                if task == "molecule":
                    hi, lo, fp = [
                        np.concatenate([o[i] for o in outputs]) for i in range(3)
                    ]
                    qp = np.maximum(
                        0.0, np.expm1(np.clip(np.r_[hi, lo] * qscale, -20, 20))
                    )
                    qt = np.r_[m["high"][ids], m["low"][ids]]
                    report[task] = {
                        "quantitative_log_rmse": float(
                            np.sqrt(np.mean((np.log1p(qp) - np.log1p(qt)) ** 2))
                        ),
                        "quantitative_median_cosine": float(np.median(cosine(qp, qt))),
                        "quantitative_mean_cosine": float(np.mean(cosine(qp, qt))),
                        "fine_probability_mae": float(
                            np.mean(np.abs(fp - m["fine"][ids]))
                        ),
                        "rows": len(ids),
                    }
                    head_p, head_t = qp.reshape(-1, 2, 146), qt.reshape(-1, 2, 146)
                    head_p = head_p / np.maximum(head_p.sum(-1, keepdims=True), 1e-30)
                    head_t = head_t / np.maximum(head_t.sum(-1, keepdims=True), 1e-30)
                    report[task]["complete_profile_worst_tv"] = float(
                        (0.5 * np.abs(head_p - head_t).sum(-1)).max(-1).mean()
                    )
                elif task == "mixture":
                    qp, delta = [
                        np.concatenate([o[i] for o in outputs]) for i in range(2)
                    ]
                    qp = np.maximum(0.0, np.expm1(np.clip(qp * qscale, -20, 20)))
                    report[task] = {
                        "additive_proxy_median_cosine": float(
                            np.median(cosine(qp, mix["quantitative"][ids]))
                        ),
                        "revision_deficit_rmse": float(
                            np.sqrt(np.mean((delta - mix["revision"][ids]) ** 2))
                        ),
                        "rows": len(ids),
                    }
                    head_p = qp.reshape(-1, 2, 146)
                    head_t = mix["quantitative"][ids].reshape(-1, 2, 146)
                    head_p = head_p / np.maximum(head_p.sum(-1, keepdims=True), 1e-30)
                    head_t = head_t / np.maximum(head_t.sum(-1, keepdims=True), 1e-30)
                    report[task]["complete_profile_worst_tv"] = float(
                        (0.5 * np.abs(head_p - head_t).sum(-1)).max(-1).mean()
                    )
                elif task == "transport":
                    pred = np.concatenate(outputs)
                    report[task] = {
                        "rmse": float(np.sqrt(np.mean((pred - t["target"][ids]) ** 2))),
                        "maximum_mass_error": float(np.max(np.abs(pred.sum(-1) - 1))),
                        "nonnegative": bool(np.all(pred >= 0)),
                        "rows": len(ids),
                    }
                elif task == 'pair':
                    from sklearn.metrics import average_precision_score
                    logits=np.concatenate(outputs)
                    pred=1/(1+np.exp(-np.clip(logits,-40,40)))
                    target=pair['target'][ids]
                    support=target.sum(0)
                    eligible=(support>=3)&(support<len(target))&(pair['target'][indices['pair'][0]].sum(0)>=3)
                    aps=[average_precision_score(target[:,j],pred[:,j]) for j in np.flatnonzero(eligible)]
                    report[task]={'rows':len(ids),'macro_average_precision':float(np.mean(aps)) if aps else None,
                                  'macro_eligible_labels':int(eligible.sum()),'all_labels':target.shape[1],
                                  'constant_baseline_macro_ap':float(target[:,eligible].mean()) if eligible.any() else None,
                                  'log_loss':float(-np.mean(target*np.log(np.clip(pred,1e-7,1))+(1-target)*np.log(np.clip(1-pred,1e-7,1)))),
                                  'source_scaffold_disjoint':True,'ratios_observed':False,'human_accuracy':None}
                elif task == 'aqueous':
                    from sklearn.metrics import roc_auc_score, brier_score_loss
                    pred=np.concatenate(outputs)
                    target=aqueous['normalized_target'][ids]
                    mask=aqueous['mask'][ids].copy()
                    mask[:,0]=False
                    raw_probability=1/(1+np.exp(-np.clip(pred[:,0],-40,40)))
                    probability=1/(1+np.exp(-np.clip(stability_calibration['slope']*pred[:,0]+stability_calibration['intercept'],-40,40)))
                    stability=aqueous['target'][ids,0]
                    report[task]={'rows':len(ids),'stability_auc':float(roc_auc_score(stability,probability)) if len(set(stability))==2 else None,
                                  'stability_brier':float(brier_score_loss(stability,probability)),
                                  'uncalibrated_stability_brier':float(brier_score_loss(stability,raw_probability)),
                                  'constant_stability_brier':float(np.mean((stability-aqueous['target'][indices['aqueous'][0],0].mean())**2)),
                                  'normalized_outcome_rmse':float(np.sqrt(np.mean((pred[mask]-target[mask])**2))),
                                  'constant_outcome_rmse':float(np.sqrt(np.mean(target[mask]**2))),
                                  'composition_combination_disjoint':True,'missing_outcomes_filled_as_truth':False}
                elif task == "emulsion":
                    pred = np.concatenate(outputs)
                    target = e["target"][ids]
                    reference = np.exp(eb[ids])
                    mean = e["target"][e["split"] == 0].mean(0)
                    logd = np.log(meta["emulsion"]["diameter_bins_um"])

                    def errors(value):
                        return float(np.sqrt(np.mean(((value - target) @ logd) ** 2)))

                    report[task] = {
                        "rows": len(ids),
                        "composition_disjoint": True,
                        "log_geometric_diameter_rmse": errors(pred),
                        "physical_baseline_log_diameter_rmse": errors(reference),
                        "train_mean_log_diameter_rmse": errors(
                            np.tile(mean, (len(ids), 1))
                        ),
                        "cdf_rmse": float(
                            np.sqrt(np.mean((pred.cumsum(-1) - target.cumsum(-1)) ** 2))
                        ),
                        "scope": "measured_emulsion_not_lotion_or_fragrance_validation",
                    }
                else:
                    ap, cp = [np.concatenate([o[i] for o in outputs]) for i in range(2)]
                    report[task] = {
                        "next_action_accuracy": float(np.mean(ap == p["action"][ids])),
                        "check_accuracy": float(
                            np.mean((cp >= 0.5) == p["checks"][ids])
                        ),
                        "rows": len(ids),
                    }
        return report

    def criterion(r):
        return (
            (r['pair']['log_loss']+r['aqueous']['stability_brier']+.1*r['aqueous']['normalized_outcome_rmse'] if observed_meta else 0.)
            +
            r["molecule"]["quantitative_log_rmse"]
            + 3 * r["molecule"]["fine_probability_mae"]
            + 8 * r["transport"]["rmse"]
            + 2 * (1 - r["process"]["next_action_accuracy"])
            + 1
            - r["mixture"]["additive_proxy_median_cosine"]
            + r["mixture"]["revision_deficit_rmse"]
            + 0.05 * r["emulsion"]["log_geometric_diameter_rmse"]
            + (
                2
                * (
                    r["molecule"]["complete_profile_worst_tv"]
                    + r["mixture"]["complete_profile_worst_tv"]
                )
                if args.worst_profile_loss
                else 0
            )
        )

    start, history, best_score, selected, best = (
        time.perf_counter(),
        [],
        float("inf"),
        0,
        None,
    )
    print(
        json.dumps(
            {
                "phase": "joint_training",
                "parameter_count": parameter_count,
                "device": str(device),
                "training_rows": {k: len(v[0]) for k, v in indices.items()},
            }
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        jobs = []
        for task in indices:
            order = rng.permutation(indices[task][0])
            repeats = (
                (False, True)
                if task == "molecule"
                else (False,) * 8
                if task == "emulsion"
                else (False,)
            )
            for low in repeats:
                jobs.extend(
                    (task, order[i : i + args.batch_size], low)
                    for i in range(0, len(order), args.batch_size)
                )
        if observed_meta:
            # Equal task exposure rather than 2 aqueous updates versus hundreds
            # of pair/process updates. Every original training row still appears.
            by_task={task:[job for job in jobs if job[0]==task] for task in indices}
            maximum=max(map(len,by_task.values()))
            jobs=[]
            for task,original in by_task.items():
                jobs.extend(original)
                for j in range(maximum-len(original)):
                    order=rng.permutation(indices[task][0])[:args.batch_size]
                    jobs.append((task,order,bool(j%2) if task=='molecule' else False))
        rng.shuffle(jobs)
        losses = {task: [] for task in indices}
        for task, ids, low in jobs:
            optimizer.zero_grad(set_to_none=True)
            output = model(*inputs(task, ids, low))
            if task == "molecule":
                q = bank["low" if low else "high"][ids]
                qloss = nn.functional.mse_loss(output["quantitative"], q)
                floss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        output["fine"], bank["fine"][ids], reduction="none"
                    )
                    * fine_weights
                ).mean()
                loss = qloss + floss
                if args.profile_shape_loss or args.worst_profile_loss:
                    loss = loss + shape_loss(output["quantitative"], q)
            elif task == "mixture":
                loss = (
                    nn.functional.mse_loss(output["quantitative"], bank["mq"][ids])
                    + (
                        nn.functional.binary_cross_entropy_with_logits(
                            output["fine"], bank["mf"][ids], reduction="none"
                        )
                        * fine_weights
                    ).mean()
                    + 4 * nn.functional.mse_loss(output["revision"], bank["mr"][ids])
                )
                if args.profile_shape_loss or args.worst_profile_loss:
                    loss = loss + shape_loss(output["quantitative"], bank["mq"][ids])
            elif task == "transport":
                predicted, target = (
                    transition(output["transport"], ids),
                    bank["pt"][ids],
                )
                loss = 5 * nn.functional.mse_loss(
                    predicted, target
                ) + 0.002 * nn.functional.mse_loss(
                    torch.log(predicted.clamp_min(1e-10)),
                    torch.log(target.clamp_min(1e-10)),
                )
            elif task == 'pair':
                loss=nn.functional.binary_cross_entropy_with_logits(output['blend'],bank['pair_y'][ids])
            elif task == 'aqueous':
                prediction,target=output['aqueous'],bank['aqueous_y'][ids]
                mask=bank['aqueous_mask'][ids,1:]
                regression=nn.functional.smooth_l1_loss(prediction[:,1:],target[:,1:],reduction='none')
                loss=nn.functional.binary_cross_entropy_with_logits(prediction[:,0],target[:,0])+(regression*mask).sum()/mask.sum().clamp_min(1)
            elif task == "emulsion":
                logits = output["emulsion"] + bank["eb"][ids]
                logp = torch.log_softmax(logits, -1)
                target = bank["ey"][ids]
                loss = 0.2 * (
                    -(target * logp).sum(-1).mean()
                    + 5
                    * nn.functional.mse_loss(logp.exp().cumsum(-1), target.cumsum(-1))
                )
            else:
                loss = nn.functional.cross_entropy(
                    output["action"], bank["pa"][ids]
                ) + 0.3 * nn.functional.binary_cross_entropy_with_logits(
                    output["check"], bank["pf"][ids]
                )
            if not torch.isfinite(loss):
                raise ArithmeticError("nonfinite multi-task loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses[task].append(float(loss.detach()))
        schedule.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation = evaluate(1)
            score = criterion(validation)
            selection_key=(0,score)
            if observed_meta:
                failed=sum((validation['aqueous']['stability_brier']>=validation['aqueous']['constant_stability_brier'],
                            validation['aqueous']['normalized_outcome_rmse']>=validation['aqueous']['constant_outcome_rmse'],
                            validation['process']['next_action_accuracy']<.98,
                            validation['molecule']['quantitative_median_cosine']<.95,
                            validation['mixture']['revision_deficit_rmse']>.03))
                selection_key=(failed,score)
            row = {
                "epoch": epoch,
                "loss": {k: float(np.mean(v)) for k, v in losses.items()},
                "validation": validation,
                "selection_loss": score,
                "validation_failed_gates": selection_key[0],
                "seconds": time.perf_counter() - start,
            }
            history.append(row)
            if selection_key < ((999,float('inf')) if best is None else best_score):
                best_score, selected = selection_key, epoch
                best = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                torch.save(
                    {
                        "model": best,
                        "epoch": epoch,
                        "data_sha256": protocol["data_sha256"],
                    },
                    args.output / "training_checkpoint.pt",
                )
            write(args.output / "learning_curve.json", history)
            print(
                json.dumps({"phase": "training", **row, "selected_epoch": selected}),
                flush=True,
            )
    model.load_state_dict(best)
    model.eval()
    if observed_meta:
        from fragrance_ai.research.probability_calibration import fit_logit_calibration
        with torch.no_grad():
            logits=model(*inputs('aqueous',indices['aqueous'][1]))['aqueous'][:,0].cpu().numpy()
        stability_calibration=fit_logit_calibration(logits,aqueous['target'][indices['aqueous'][1],0])
    validation, test = evaluate(1), evaluate(2)
    arrays = {**export_arrays(model), **normalization}
    np.savez_compressed(args.output / "weights.npz", **arrays)
    # Export is checked independently, including nonempty, ordered history.
    parity = {}
    with torch.no_grad():
        for task in indices:
            ids = indices[task][2][:7]
            data = inputs(task, ids)
            expected = model(*data)
            actual = forward_arrays(arrays, *(v.cpu().numpy() for v in data))
            parity[task] = max(
                float(np.max(np.abs(actual[k] - expected[k].cpu().numpy())))
                for k in expected
            )
    gates = {
        "quantitative_fidelity": test["molecule"]["quantitative_median_cosine"] >= 0.95,
        "fine_fidelity": test["molecule"]["fine_probability_mae"] <= 0.035,
        "transport_fidelity": test["transport"]["rmse"] <= 0.035,
        "procedure_fidelity": test["process"]["next_action_accuracy"] >= 0.98,
        "mixture_proxy_fidelity": test["mixture"]["additive_proxy_median_cosine"]
        >= 0.95,
        "revision_deficit": test["mixture"]["revision_deficit_rmse"] <= 0.03,
        "measured_emulsion_vs_mean": test["emulsion"]["log_geometric_diameter_rmse"]
        < test["emulsion"]["train_mean_log_diameter_rmse"],
        "cpu_export": max(parity.values()) <= 0.0002,
    }
    if observed_meta:
        gates.update(observed_pair_holdout=test['pair']['macro_average_precision']>test['pair']['constant_baseline_macro_ap'],
                     measured_formulation_holdout=test['aqueous']['stability_brier']<test['aqueous']['constant_stability_brier']
                         and test['aqueous']['normalized_outcome_rmse']<test['aqueous']['constant_outcome_rmse'])
    report = {
        "scope": protocol["evaluation_scope"],
        "validation": validation,
        "test": test,
        "selected_epoch": selected,
        "epochs_completed": args.epochs,
        "gates": gates,
        "all_prepared_rows_used": True,
        "export_max_abs_error": parity,
        "seconds": time.perf_counter() - start,
        "human_similarity_percent": None,
        "actual_manufacturing_outcome_accuracy": None,
        "stability_calibration": stability_calibration if observed_meta else None,
    }
    write(args.output / "evaluation.json", report)
    manifest = {
        "schema": VERSION,
        "single_shared_checkpoint": True,
        "training_executed": True,
        "accepted_for_local_inference": all(gates.values()),
        "parameter_count": parameter_count,
        "weights": {"path": "weights.npz", "sha256": sha(args.output / "weights.npz")},
        "architecture": {
            "molecule_features": x.shape[1],
            "molecule_encoder": [x.shape[1], 384, 256],
            "set_aggregation": "mass_weighted_mean_variance_and_context_attention",
            "ordered_process_state": 96,
            "shared_trunk": [992, 384, 256],
            "shared_residual_blocks": 2,
            "heads": [
                "fine450",
                "quantitative292",
                "transport10",
                "process_action",
                "checks8",
                "revision19",
                "emulsion101",
            ],
        },
        "process_context_schema": meta.get("process_context_schema"),
        "emulsion": meta["emulsion"],
        "actions": meta["actions"],
        "fine_endpoints": meta["fine_endpoints"],
        "quantitative_endpoints": meta["quantitative_endpoints"],
        "fine_features": meta["fine_features"],
        "native_profiles": meta["native_profiles"],
        "source_annotations": meta["source_annotations"],
        "structures": meta["structures"],
        "atlas_source": meta["atlas_source"],
        "training_sources": {
            **meta["training_sources"],
            "mixture_proxy_rows": len(mix["ids"]),
            "mixture_target_kind": "additive_existing_prediction_proxy_not_measured_interactions",
            "revision_target_kind": "desired_minus_predicted_profile_not_professional_decision_labels",
        },
        "teacher_bindings": meta["teacher_bindings"],
        "evaluation": report,
        "data_sha256": protocol["data_sha256"],
        "source_sha256": {
            name: sha(ROOT / name)
            for name in (
                "fragrance_ai/research/formulation_network.py",
                "fragrance_ai/recommender/formulation_core.py",
                "fragrance_ai/recommender/formulation_process.py",
                "scripts/train_formulation_v69.py",
            )
        },
    }
    if observed_meta:
        from fragrance_ai.recommender.formulation_core import FormulationCore, SYSTEM_VERSION
        parent=FormulationCore(args.parent_core,sha(args.parent_core))
        # Preserve one archive while marking the recurrent decoder NOT revalidated
        # until the separate nonlinear inverse-learning stage has completed.
        arrays.update({k:v for k,v in parent.arrays.items() if k.startswith('autoregressive.')})
        np.savez_compressed(args.output/'weights.npz',**arrays)
        manifest.update(schema=SYSTEM_VERSION,accepted_for_local_inference=False,
            weights={'path':'weights.npz','sha256':sha(args.output/'weights.npz')},
            parameter_count=sum(v.size for k,v in arrays.items() if '.' in k),
            autoregressive=parent.manifest['autoregressive'], complete_reference_guidance=True,
            blend_labels=observed_meta['odor_pairs']['labels'],aqueous=observed_meta['liquid'],
            joint_observed_refit={'schema':'source_separated_joint_refit/v76','data_sha256':sha(args.observed_data/'manifest.json'),
                                 'physical_teacher_sha256':sha(args.physical_mixtures),'parent_sha256':parent.sha256})
        manifest['joint_observed_refit'].update(goal_independent_predictive_path=True,exact_revision_arithmetic=True,
                                               task_balanced_replay=True,all_original_training_rows_retained=True)
        manifest['architecture']['heads'].extend(['blend_annotations109','observed_aqueous_outcomes18'])
        manifest['aqueous']['stability_calibration']=stability_calibration
        manifest['architecture']['autoregressive_decoder']=parent.manifest['architecture']['autoregressive_decoder']
        manifest['evaluation']['gates']['nonlinear_inverse_fidelity']=False
        manifest['training_sources'].update(observed_pair_rows=observed_meta['odor_pairs']['prepared_rows'],
            observed_liquid_rows=observed_meta['liquid']['source_rows'],
            mixture_target_kind='shared_forward_saturation_suppression_evaporation_prior_not_measured_labels')
    write(args.output / "model.json", manifest)
    print(
        json.dumps(
            {
                "status": "accepted_local_candidate"
                if all(gates.values())
                else "candidate_gate_failed",
                "model_sha256": sha(args.output / "model.json"),
                **report,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
