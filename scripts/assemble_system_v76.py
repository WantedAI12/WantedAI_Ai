"""Assemble trained V76 process parameters without changing other learned paths."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from fragrance_ai.recommender.formulation_core import (
        FormulationCore,
        forward_arrays,
    )
    from fragrance_ai.recommender.formulation_process_graph import VERSION

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inverse", type=Path, required=True)
    p.add_argument("--process", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-graph-source", type=Path)
    p.add_argument("--coverage", type=Path, required=True)
    a = p.parse_args()
    inverse = FormulationCore(a.inverse, sha(a.inverse))
    inverse_protocol_path = a.inverse.parent / "protocol.json"
    if (
        sha(inverse_protocol_path)
        != inverse.manifest["autoregressive"]["training_protocol_sha256"]
    ):
        raise ValueError("inverse training protocol changed")
    inverse_protocol = json.loads(inverse_protocol_path.read_text(encoding="utf8"))
    inverse_report_path = a.inverse.parent / "evaluation.json"
    if (
        sha(inverse_report_path)
        != inverse.manifest["autoregressive"]["nonlinear_inverse_report_sha256"]
    ):
        raise ValueError("inverse evaluation report changed")
    inverse_report = json.loads(inverse_report_path.read_text(encoding="utf8"))
    coverage = json.loads(a.coverage.read_text(encoding="utf8"))
    if coverage["evidence_index_sha256"] != inverse_protocol["evidence_sha256"]:
        raise ValueError("coverage and inverse physical evidence differ")
    report = json.loads((a.process / "report.json").read_text(encoding="utf8"))
    protocol = json.loads((a.process / "protocol.json").read_text(encoding="utf8"))
    graph_path = ROOT / "fragrance_ai/recommender/formulation_process_graph.py"
    graph_sha = sha(graph_path)
    training_graph = a.training_graph_source or graph_path
    semantic_identity = ast.dump(
        ast.parse(training_graph.read_text(encoding="utf8"))
    ) == ast.dump(ast.parse(graph_path.read_text(encoding="utf8")))
    if (
        report["eligible_for_assembly"] is not True
        or report["nonprocess_weights_exact"] is not True
        or sha(training_graph) != report["source_graph_sha256"]
        or sha(training_graph) != protocol["source_graph_sha256"]
        or not semantic_identity
    ):
        raise ValueError("process training, identity, or source provenance failed")
    prefixes = (
        "step_embedding.",
        "process_input.",
        "process_state.",
        "action_head.",
        "check_head.",
    )
    if tuple(report["permitted_parameter_prefixes"]) != prefixes:
        raise ValueError("process training expanded its permitted parameter scope")
    with np.load(a.process / "weights.npz", allow_pickle=False) as source:
        if set(source.files) != set(inverse.arrays):
            raise ValueError("checkpoint parameter schema mismatch")
        if any(
            not np.array_equal(source[k], v)
            for k, v in inverse.arrays.items()
            if not k.startswith(prefixes + ("autoregressive.",))
        ):
            raise ValueError(
                "process and inverse stages do not share an identical predictive parent"
            )
        arrays = {
            k: source[k].copy() if k.startswith(prefixes) else v.copy()
            for k, v in inverse.arrays.items()
        }
    rng = np.random.default_rng(761020)
    molecules = rng.normal(size=(12, 3, inverse.feature_width)).astype(np.float32)
    masses = rng.uniform(size=(12, 3)).astype(np.float32)
    context = rng.uniform(size=(12, 64)).astype(np.float32)
    history = np.zeros((12, 0), np.int64)
    numbers = np.zeros((12, 0, 12), np.float32)
    before = forward_arrays(
        inverse.arrays, molecules, masses, context, history, numbers
    )
    after = forward_arrays(arrays, molecules, masses, context, history, numbers)
    preserved = {
        k: bool(np.array_equal(v, after[k]))
        for k, v in before.items()
        if k not in ("action", "check")
    }
    if not all(preserved.values()):
        raise ValueError("process assembly changed a no-history predictive path")
    if sha(a.process / "histories.npz") != protocol["history_sha256"]:
        raise ValueError("process holdout records changed")
    with np.load(a.process / "histories.npz", allow_pickle=False) as histories:
        maximum_history = int(histories["ids"].shape[1])
        selected = np.flatnonzero(histories["split"] == 2)
        valid, checks = [], []
        for start in range(0, len(selected), 256):
            index = selected[start : start + 256]
            computed = forward_arrays(
                arrays,
                np.zeros((len(index), 1, inverse.feature_width), np.float32),
                np.zeros((len(index), 1), np.float32),
                histories["context"][index],
                histories["ids"][index],
                histories["numbers"][index],
            )
            valid.extend(
                histories["allowed"][index, computed["action"].argmax(-1)].tolist()
            )
            checks.extend(
                ((computed["check"] >= 0) == (histories["checks"][index] > 0.5))
                .mean(-1)
                .tolist()
            )
    repeated = {
        "raw_valid_next_action_rate": float(np.mean(valid)),
        "check_accuracy": float(np.mean(checks)),
        "rows": len(selected),
    }
    if (
        repeated["raw_valid_next_action_rate"] < 0.98
        or repeated["check_accuracy"] < 0.98
    ):
        raise ValueError(
            "assembled process predictions failed the full saved process holdout"
        )
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / "weights.npz", **arrays)
    m = deepcopy(inverse.manifest)
    m['autoregressive'].update(
        objective_family='shared_nonlinear_headspace_and_lotion_exposure/v76',
        predictive_parent_sha256=inverse_protocol['parent_sha256'],
        mass_projection=inverse_protocol.get('mass_projection'),
        non_controller_parameters_preserved_during_inverse_stage=True)
    if all('parent_same_steps' in value for value in inverse_report['summary'].values()):
        m['autoregressive']['equal_horizon_weight_superiority_demonstrated']=all(
            value['learned']['mean_loss']<value['parent_same_steps']['mean_loss']
            for value in inverse_report['summary'].values())
        m['autoregressive']['comparison_scope']='updated_configuration_vs_previous_default_not_claim_of_equal_compute_weight_superiority'
    m['evaluation']['scope_qualification']={
        'pair_split':'component_scaffold_partition_within_pair_annotations',
        'pretrained_component_knowledge_used':True,
        'global_cross_task_unseen_molecule_claimed':False,
        'pair_scores_are_annotation_recovery_not_measured_mixture_accuracy':True}
    m["evaluation"]["nonlinear_inverse"] = {
        key: value
        for key, value in inverse_report.items()
        if key not in ("raw_test_losses", "parity")
    }
    m["scientific_coverage"] = {k: v for k, v in coverage.items() if k != "rows"}
    m["scientific_coverage"]["coverage_report_sha256"] = sha(a.coverage)
    m["scientific_evidence_index_sha256"] = coverage["evidence_index_sha256"]
    m["weights"] = {"path": "weights.npz", "sha256": sha(a.output / "weights.npz")}
    m["process_graph_training"] = {
        "schema": VERSION,
        "source_legal_holdout_passed": True,
        "report_sha256": sha(a.process / "report.json"),
        "weights_sha256": sha(a.process / "weights.npz"),
        "protocol_sha256": sha(a.process / "protocol.json"),
        "source_graph_sha256": graph_sha,
        "training_graph_source_sha256": sha(training_graph),
        "max_training_history_steps": maximum_history,
        "format_only_ast_identity_verified": semantic_identity,
        "label_kind": protocol["label_kind"],
        "test": repeated,
        "no_history_prediction_arrays_exact": preserved,
        "source_policy_not_measured_manufacturing_accuracy": True,
    }
    m["evaluation"]["gates"]["process_dependency_graph_fidelity"] = True
    (a.output / "model.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf8"
    )
    model_sha = sha(a.output / "model.json")
    reference = json.loads(a.reference.read_text(encoding="utf8"))
    # Only rebind the checkpoint parent; measured/reference values stay unchanged.
    reference["parent_atlas_sha256"] = model_sha
    (a.output / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf8"
    )
    final = FormulationCore(a.output / "model.json", model_sha)
    final.assert_current()
    result = {
        "model_sha256": model_sha,
        "inverse_sha256": inverse.sha256,
        "process_report_sha256": sha(a.process / "report.json"),
        "reference_sha256": sha(a.output / "reference.json"),
        "no_history_predictions_exact": preserved,
        "deployed": False,
    }
    (a.output / "assembly.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
