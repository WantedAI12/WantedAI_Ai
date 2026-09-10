from __future__ import annotations

import base64
import csv
import json
import shutil
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    sha256_file,
    signing_payload,
)
from fragrance_ai.recommender.concentration_response import (
    CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
    concentration_response_from_environment,
)
from fragrance_ai.recommender.continual_training import (
    finalize_blind_challenge,
    prepare_blind_challenge,
    process_learning_jobs,
)
from fragrance_ai.recommender.continuous_improvement import (
    ContinuousImprovementController,
    ContinualImprovementError,
    PROSPECTIVE_DATASET_ACQUISITION_ARTIFACT_TYPE,
    bootstrap_seed,
    load_production_concentration_response,
)
from fragrance_ai.recommender.service import NaturalLanguagePerfumeryAI


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _intensity(dilution: float) -> float:
    return max(0.0, min(100.0, 95.0 + 20.0 * __import__("math").log10(dilution)))


def _candidate_bundle(
    tmp_path: Path, candidate_id: str = "candidate-v1"
) -> tuple[Path, Path, dict]:
    training_path = tmp_path / "training.csv"
    challenge_path = tmp_path / "challenge.csv"
    concentrations = (0.0001, 0.001, 0.01, 0.1)
    training_rows = []
    for molecule in range(40):
        for concentration_index, concentration in enumerate(concentrations):
            training_rows.append(
                {
                    "row_id": f"T-{molecule}-{concentration_index}",
                    "source_id": f"train-source-{molecule % 2}",
                    "target_id": f"train-target-{molecule}",
                    "molecule_id": f"train-molecule-{molecule}",
                    "scaffold_id": f"train-scaffold-{molecule}",
                    "dilution_fraction": concentration,
                    "intensity": _intensity(concentration),
                    "label_origin": "external_human_measurement",
                    "evidence_class": "retrospective_external_human",
                }
            )
    challenge_rows = []
    outcome_rows = []
    for molecule in range(40):
        for concentration_index, concentration in enumerate(concentrations):
            row_id = f"E-{molecule}-{concentration_index}"
            challenge_rows.append(
                {
                    "row_id": row_id,
                    "source_id": f"external-source-{molecule % 2}",
                    "target_id": f"external-target-{molecule}",
                    "molecule_id": f"external-molecule-{molecule}",
                    "scaffold_id": f"external-scaffold-{molecule}",
                    "dilution_fraction": concentration,
                }
            )
            outcome_rows.append(
                {"row_id": row_id, "intensity": _intensity(concentration)}
            )
    _write_csv(
        training_path,
        [
            "row_id",
            "source_id",
            "target_id",
            "molecule_id",
            "scaffold_id",
            "dilution_fraction",
            "intensity",
            "label_origin",
            "evidence_class",
        ],
        training_rows,
    )
    _write_csv(
        challenge_path,
        [
            "row_id",
            "source_id",
            "target_id",
            "molecule_id",
            "scaffold_id",
            "dilution_fraction",
        ],
        challenge_rows,
    )
    prepared = tmp_path / "prepared"
    prepare_blind_challenge(
        training_csv=training_path,
        challenge_inputs_csv=challenge_path,
        output_dir=prepared,
        candidate_id=candidate_id,
    )
    outcomes = tmp_path / "outcomes.csv"
    _write_csv(outcomes, ["row_id", "intensity"], outcome_rows)
    timestamp = tmp_path / "timestamp.tsr"
    timestamp.write_bytes(b"independent-rfc3161-fixture")
    receipt = {
        "schema": "perfumery-external-dataset-receipt/v1",
        "candidate_id": candidate_id,
        "dataset_id": f"external-dataset-{candidate_id}",
        "evidence_class": "prospective_external_human",
        "label_origin": "external_human_measurement",
        "source_ids": ["external-source-0", "external-source-1"],
        "target_ids": [f"external-target-{index}" for index in range(40)],
        "row_count": len(outcome_rows),
        "prediction_sha256": sha256_file(prepared / "predictions.json"),
        "prediction_seal_sha256": sha256_file(prepared / "prediction_seal.json"),
        "outcome_sha256": sha256_file(outcomes),
        "timestamp_response_sha256": sha256_file(timestamp),
        "timestamp_authority_verified": True,
        "prediction_sealed_at": "2026-08-01T00:00:00+00:00",
        "outcome_first_read_at": "2026-08-02T00:00:00+00:00",
        "synthetic_rows": 0,
        "model_generated_label_rows": 0,
        "evaluation_labels_available_during_training": False,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    acquisition_private = Ed25519PrivateKey.generate()
    acquisition_public = acquisition_private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    acquisition_signer = f"external-acquirer-{candidate_id}"
    acquisition_artifacts = {
        "dataset_receipt": receipt_path,
        "outcomes": outcomes,
        "predictions": prepared / "predictions.json",
        "prediction_seal": prepared / "prediction_seal.json",
        "timestamp_response": timestamp,
    }
    acquisition_envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": f"{candidate_id}-acquisition",
        "artifact_type": PROSPECTIVE_DATASET_ACQUISITION_ARTIFACT_TYPE,
        "signer_id": acquisition_signer,
        "signer_role": "external_evidence_acquirer",
        "scope": {
            "candidate_id": candidate_id,
            "dataset_id": receipt["dataset_id"],
            "evidence_class": receipt["evidence_class"],
            "label_origin": receipt["label_origin"],
            "source_ids": sorted(receipt["source_ids"]),
            "target_ids": sorted(receipt["target_ids"]),
            "row_count": receipt["row_count"],
            "prediction_sealed_at": receipt["prediction_sealed_at"],
            "outcome_first_read_at": receipt["outcome_first_read_at"],
            "timestamp_authority_verified": True,
        },
        "issued_at": "2026-08-02T00:00:00+00:00",
        "expires_at": "2027-08-02T00:00:00+00:00",
        "artifact_hashes": {
            label: sha256_file(path)
            for label, path in acquisition_artifacts.items()
        },
    }
    acquisition_envelope["signature"] = base64.b64encode(
        acquisition_private.sign(signing_payload(acquisition_envelope))
    ).decode("ascii")
    acquisition_path = tmp_path / "acquisition_authorization.json"
    acquisition_path.write_text(json.dumps(acquisition_envelope), encoding="utf-8")
    trust_root = {
        "signers": {
            acquisition_signer: {
                "public_key": acquisition_public,
                "roles": ["external_evidence_acquirer"],
                "artifact_types": [
                    PROSPECTIVE_DATASET_ACQUISITION_ARTIFACT_TYPE
                ],
            }
        }
    }
    root = tmp_path / "continual"
    result = finalize_blind_challenge(
        prepared_dir=prepared,
        outcomes_csv=outcomes,
        dataset_receipt_json=receipt_path,
        timestamp_response=timestamp,
        acquisition_authorization=acquisition_path,
        inbox_root=root / "inbox",
        bootstrap_draws=2000,
    )
    return root, Path(result["bundle"]), trust_root


def _sign_authorization(bundle: Path, trust_root: dict) -> dict:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    candidate_path = bundle / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    paths = {
        label: bundle / record["path"]
        for label, record in candidate["artifacts"].items()
    }
    model_manifest = json.loads(paths["model_manifest"].read_text(encoding="utf-8"))
    scope = {
        "model_sha256": sha256_file(paths["runtime"]),
        "manifest_sha256": sha256_file(paths["model_manifest"]),
        "algorithm": model_manifest["algorithm"],
        "approved_primary_score_weight": 0.05,
        "candidate_id": candidate["candidate_id"],
        "model_family": "concentration_response",
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "evaluation_report_sha256": sha256_file(paths["evaluation_report"]),
        "dataset_receipt_sha256": sha256_file(paths["dataset_receipt"]),
    }
    artifacts = {
        "manifest": paths["model_manifest"],
        "model": paths["runtime"],
        "candidate_manifest": candidate_path,
        "dataset_receipt": paths["dataset_receipt"],
        "evaluation_report": paths["evaluation_report"],
        "outcomes": paths["outcomes"],
        "predictions": paths["predictions"],
        "prediction_seal": paths["prediction_seal"],
        "timestamp_response": paths["timestamp_response"],
        "training_data": paths["training_data"],
        "challenge_inputs": paths["challenge_inputs"],
        "acquisition_authorization": paths["acquisition_authorization"],
    }
    envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": "continual-production-release-1",
        "artifact_type": CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE,
        "signer_id": "independent-model-release",
        "signer_role": "model_release_approver",
        "scope": scope,
        "issued_at": "2026-08-03T00:00:00+00:00",
        "expires_at": "2027-08-01T00:00:00+00:00",
        "artifact_hashes": {
            label: sha256_file(path) for label, path in artifacts.items()
        },
    }
    envelope["signature"] = base64.b64encode(
        private.sign(signing_payload(envelope))
    ).decode("ascii")
    (bundle / "authorization.json").write_text(json.dumps(envelope), encoding="utf-8")
    combined = json.loads(json.dumps(trust_root))
    combined["signers"]["independent-model-release"] = {
        "public_key": public,
        "roles": ["model_release_approver"],
        "artifact_types": [CONCENTRATION_RESPONSE_AUTHORIZATION_ARTIFACT_TYPE],
    }
    return combined


def test_blind_challenger_promotes_shadow_but_not_unsigned_production(tmp_path: Path) -> None:
    root, _bundle, trust_root = _candidate_bundle(tmp_path)
    controller = ContinuousImprovementController(root, trust_root=trust_root)
    result = controller.run_once()
    decision = result["decisions"][0]
    assert decision["scientific_gate_passed"] is True
    assert decision["shadow_promoted"] is True
    assert decision["production_promoted"] is False
    assert decision["reasons"] == ("signed_production_authorization_missing",)
    assert result["registry"]["champions"]["concentration_response"]["production"][
        "approved_primary_score_weight"
    ] == 0.0
    assert controller.run_once()["processed_now"] == 0
    assert controller.verify_registry()["valid"] is True


def test_signed_candidate_promotes_and_runtime_reverifies_every_artifact(tmp_path: Path) -> None:
    root, bundle, acquisition_trust_root = _candidate_bundle(tmp_path)
    unsigned_controller = ContinuousImprovementController(
        root, trust_root=acquisition_trust_root
    )
    unsigned_controller.run_once()
    request = unsigned_controller.build_authorization_request(
        candidate_id="candidate-v1",
        signer_id="independent-model-release",
        issued_at="2026-08-03T00:00:00+00:00",
        expires_at="2027-08-01T00:00:00+00:00",
    )
    assert request["envelope"]["scope"]["candidate_manifest_sha256"] == sha256_file(
        bundle / "candidate.json"
    )
    assert "signature" not in request["envelope"]
    trust_root = _sign_authorization(bundle, acquisition_trust_root)
    controller = ContinuousImprovementController(root, trust_root=trust_root)
    result = controller.run_once()
    decision = result["decisions"][0]
    assert decision["production_promoted"] is True
    model = load_production_concentration_response(root / "registry.json", trust_root)
    assert model.approved_primary_score_weight == 0.05
    with NaturalLanguagePerfumeryAI(concentration_response=model) as service:
        assert service.physsim_engine.concentration_response.approved_primary_score_weight == 0.05
    value, in_domain = model.intensity(0.01)
    assert in_domain and value > 0.0

    (bundle / "outcomes.csv").write_text("row_id,intensity\ntampered,0\n", encoding="utf-8")
    with pytest.raises(ContinualImprovementError, match="production artifact changed|hash mismatch"):
        load_production_concentration_response(root / "registry.json", trust_root)


def test_synthetic_labels_and_artifact_tampering_fail_closed(tmp_path: Path) -> None:
    root, bundle, trust_root = _candidate_bundle(tmp_path)
    candidate_path = bundle / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["label_origin"] = "synthetic"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decision = ContinuousImprovementController(
        root, trust_root=trust_root
    ).run_once()["decisions"][0]
    assert decision["scientific_gate_passed"] is False
    assert "synthetic_or_model_generated_labels_forbidden" in decision["reasons"]
    assert decision["production_promoted"] is False

    second_root = tmp_path / "tampered-controller"
    second_bundle = second_root / "inbox" / "candidate-v1"
    shutil.copytree(bundle, second_bundle)
    second_candidate_path = second_bundle / "candidate.json"
    second_candidate = json.loads(second_candidate_path.read_text(encoding="utf-8"))
    second_candidate["label_origin"] = "external_human_measurement"
    second_candidate_path.write_text(json.dumps(second_candidate), encoding="utf-8")
    (second_bundle / "runtime.json").write_text("{}\n", encoding="utf-8")
    tampered = ContinuousImprovementController(
        second_root, trust_root=trust_root
    ).run_once()["decisions"][0]
    assert tampered["scientific_gate_passed"] is False
    assert "artifact hash mismatch" in tampered["reasons"][0]


def test_source_overlap_and_human90_claim_are_rejected(tmp_path: Path) -> None:
    root, bundle, trust_root = _candidate_bundle(tmp_path)
    report_path = bundle / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["lineage"]["training_source_ids"].append("external-source-0")
    report["metrics"]["candidate_mae"] += 5.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate_path = bundle / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["artifacts"]["evaluation_report"]["sha256"] = sha256_file(report_path)
    candidate["artifacts"]["evaluation_report"]["bytes"] = report_path.stat().st_size
    candidate["claim_boundary"]["human_olfactory_90_percent_certified"] = True
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decision = ContinuousImprovementController(
        root, trust_root=trust_root
    ).run_once()["decisions"][0]
    assert "training_evaluation_source_overlap" in decision["reasons"]
    assert "human_olfactory_90_claim_forbidden" in decision["reasons"]
    assert (
        "reported_metric_differs_from_sealed_rows:candidate_mae"
        in decision["reasons"]
    )


def test_candidate_cannot_select_or_forge_a_weaker_baseline(tmp_path: Path) -> None:
    root, bundle, trust_root = _candidate_bundle(tmp_path)
    predictions_path = bundle / "predictions.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["predictions"][0]["baseline_prediction"] = 100.0
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    seal_path = bundle / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["prediction_sha256"] = sha256_file(predictions_path)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    receipt_path = bundle / "dataset_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prediction_sha256"] = sha256_file(predictions_path)
    receipt["prediction_seal_sha256"] = sha256_file(seal_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    report_path = bundle / "evaluation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["bootstrap"]["seed"] = bootstrap_seed(sha256_file(predictions_path))
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate_path = bundle / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for label, path in (
        ("predictions", predictions_path),
        ("prediction_seal", seal_path),
            ("dataset_receipt", receipt_path),
            ("evaluation_report", report_path),
    ):
        candidate["artifacts"][label]["sha256"] = sha256_file(path)
        candidate["artifacts"][label]["bytes"] = path.stat().st_size
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decision = ContinuousImprovementController(
        root, trust_root=trust_root
    ).run_once()["decisions"][0]
    assert (
        "sealed_baseline_predictions_do_not_match_current_champion"
        in decision["reasons"]
    )
    assert decision["shadow_promoted"] is False


def test_registry_hash_chain_and_partial_environment_configuration_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _bundle, trust_root = _candidate_bundle(tmp_path)
    controller = ContinuousImprovementController(root, trust_root=trust_root)
    controller.run_once()
    state = json.loads((root / "registry.json").read_text(encoding="utf-8"))
    state["sequence"] += 1
    (root / "registry.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ContinualImprovementError, match="audit commit"):
        controller.verify_registry()

    monkeypatch.setenv("PERFUMERY_AI_CONTINUAL_STATE", str(root / "registry.json"))
    monkeypatch.delenv("PERFUMERY_AI_CONTINUAL_TRUST_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="configured together"):
        concentration_response_from_environment()


def test_watcher_prepares_jobs_only_while_outcomes_are_absent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _candidate_bundle(source, candidate_id="source-candidate")
    root = tmp_path / "automatic"
    good_job = root / "jobs" / "automatic-candidate"
    good_job.mkdir(parents=True)
    shutil.copyfile(source / "training.csv", good_job / "training.csv")
    shutil.copyfile(source / "challenge.csv", good_job / "challenge_inputs.csv")
    controller = ContinuousImprovementController(root)
    events = process_learning_jobs(root, controller, bootstrap_draws=2000)
    assert events == [
        {
            "candidate_id": "automatic-candidate",
            "status": "predictions_prepared",
        }
    ]
    assert (root / "prepared" / "automatic-candidate" / "prediction_seal.json").is_file()

    blocked_job = root / "jobs" / "blocked-candidate"
    blocked_job.mkdir()
    shutil.copyfile(source / "training.csv", blocked_job / "training.csv")
    shutil.copyfile(source / "challenge.csv", blocked_job / "challenge_inputs.csv")
    (blocked_job / "outcomes.csv").write_text("row_id,intensity\n", encoding="utf-8")
    events = process_learning_jobs(root, controller, bootstrap_draws=2000)
    blocked = next(item for item in events if item["candidate_id"] == "blocked-candidate")
    assert blocked["status"] == "failed_closed"
    assert "existed before predictions were prepared" in blocked["error"]
    assert not (root / "prepared" / "blocked-candidate").exists()


def test_one_prospective_evaluation_cannot_select_multiple_challengers(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()
    root, _first_bundle, first_trust = _candidate_bundle(
        first_source, candidate_id="candidate-a"
    )
    _second_root, second_bundle, second_trust = _candidate_bundle(
        second_source, candidate_id="candidate-b"
    )
    destination = root / "inbox" / "candidate-b"
    shutil.copytree(second_bundle, destination)
    combined_trust = {"signers": {**first_trust["signers"], **second_trust["signers"]}}
    result = ContinuousImprovementController(
        root, trust_root=combined_trust
    ).run_once()
    decisions = {item["candidate_id"]: item for item in result["decisions"]}
    assert decisions["candidate-a"]["scientific_gate_passed"] is True
    assert decisions["candidate-b"]["scientific_gate_passed"] is False
    assert (
        "prospective_evaluation_rows_already_consumed"
        in decisions["candidate-b"]["reasons"]
    )
