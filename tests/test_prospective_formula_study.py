from __future__ import annotations

import base64
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from fragrance_ai.recommender.artifact_trust import (
    ARTIFACT_SIGNATURE_SCHEMA,
    sha256_file,
    signing_payload,
)
from fragrance_ai.research import prospective_formula_study as study


@pytest.fixture(scope="module")
def prepared_study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("prospective_formula") / "study"
    result = study.prepare_study(
        root,
        study_id="PFBS-TEST-20260828",
        randomization_seed=123456789,
    )
    assert result["formula_count"] == 24
    assert result["pair_count"] == 120
    assert result["total_assignments"] == 2400
    return root


def _completed_outcomes(
    prepared: Path,
    *,
    prediction_weight: float = 1.0,
    noise_scale: float = 0.8,
) -> Path:
    predictions = json.loads(
        (prepared / "restricted" / "predictions.json").read_text(encoding="utf-8")
    )
    predicted = {
        row["pair_id"]: float(row["predicted_similarity_0_100"])
        for row in predictions["pairs"]
    }
    with (prepared / "public" / "outcomes_template.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    rng = np.random.default_rng(42)
    for row in rows:
        target = prediction_weight * predicted[row["pair_id"]] + (
            1.0 - prediction_weight
        ) * 50.0
        row["similarity_0_100"] = f"{np.clip(target + rng.normal(0, noise_scale), 0, 100):.6f}"
        row["confidence_0_100"] = "90"
        row["intensity_left_0_100"] = "60"
        row["intensity_right_0_100"] = "60"
        row["notes"] = ""
    output = prepared / "external" / "human_outcomes.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(study.OUTCOME_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def _completed_manufacturing(prepared: Path) -> tuple[Path, Path]:
    template = json.loads(
        (
            prepared / "restricted" / "manufacturing_execution_template.json"
        ).read_text(encoding="utf-8")
    )
    evidence_root = prepared / "external" / "evidence"
    documents = {
        "approvals/safety.txt": b"independent safety approval fixture",
        "approvals/ethics.txt": b"independent ethics approval fixture",
        "base/coa.txt": b"product base coa fixture",
        "base/sds.txt": b"product base sds fixture",
    }
    document_hashes = {}
    for relative, content in documents.items():
        path = evidence_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        document_hashes[relative] = sha256_file(path)
    template["execution_authorized"] = True
    template["laboratory_id"] = "independent-lab-01"
    template["safety_approval"] = {
        "approval_id": "SAFETY-001",
        "document": "approvals/safety.txt",
    }
    template["ethics_approval"] = {
        "approval_id": "ETHICS-001",
        "document": "approvals/ethics.txt",
    }
    template["environment"] = {
        "sop_id": "SOP-SENSORY-01",
        "sop_version": "1.0",
        "product_base_id": "ETHANOL-BASE-01",
        "product_base_lot": "BASE-LOT-001",
        "product_base_coa_document": "base/coa.txt",
        "product_base_sds_document": "base/sds.txt",
        "dilution_solvent": "ethanol-water controlled base",
        "maturation_hours": 72.0,
        "test_temperature_c": 22.0,
        "relative_humidity_percent": 50.0,
        "headspace_equilibration_minutes": 15.0,
        "sniff_interval_seconds": 60.0,
        "vial_fill_ml": 5.0,
        "vial_headspace_ml": 15.0,
        "container_lot": "VIAL-LOT-001",
    }
    batch_by_formula = {}
    for batch in template["formula_batches"]:
        batch_id = f"BATCH-{batch['formula_id']}"
        batch["concentrate_batch_id"] = batch_id
        batch_by_formula[batch["formula_id"]] = batch_id
        total = 0.0
        for lot in batch["ingredient_lots"]:
            mass = float(lot["target_concentrate_percent"])
            lot["actual_mass_g"] = mass
            total += mass
            lot["supplier_sku"] = f"SKU-{lot['ingredient_id']}"
            lot["lot_number"] = f"LOT-{lot['ingredient_id']}"
            for kind in ("coa", "sds", "ifra"):
                relative = f"suppliers/{lot['ingredient_id']}/{kind}.txt"
                path = evidence_root / relative
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(
                        f"{kind} fixture for {lot['ingredient_id']}".encode("utf-8")
                    )
                    document_hashes[relative] = sha256_file(path)
                lot[f"{kind}_document"] = relative
        batch["weighed_total_g"] = total
    for index, vial in enumerate(template["sample_vials"], start=1):
        vial["concentrate_batch_id"] = batch_by_formula[vial["formula_id"]]
        finished_mass = 10.0
        concentrate_mass = (
            finished_mass * float(vial["finished_product_concentration_percent"]) / 100.0
        )
        vial["concentrate_mass_g"] = concentrate_mass
        vial["base_mass_g"] = finished_mass - concentrate_mass
        vial["finished_mass_g"] = finished_mass
        vial["vial_id"] = f"VIAL-{index:03d}"
        vial["prepared_at"] = "2026-08-28T01:00:00Z"
    template["document_hashes"] = document_hashes
    output = prepared / "external" / "manufacturing_execution.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template), encoding="utf-8")
    return output, evidence_root


def test_prepare_is_balanced_sealed_and_not_execution_authority(prepared_study: Path):
    verified = study.verify_study_seal(prepared_study)
    predictions = verified["predictions"]
    assert len(predictions["pairs"]) == 120
    assert predictions["human_accuracy_claim"] is False
    assert predictions["outcome_data_accessed"] is False
    pair_types = [row["pair_type"] for row in predictions["pairs"]]
    assert pair_types.count("ordinary_formula_pair") == 100
    assert pair_types.count("identical_formula_control") == 10
    assert pair_types.count("same_formula_concentration_control") == 10

    participant_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    for row in verified["assignments"]:
        participant_counts[row["participant_id"]] = (
            participant_counts.get(row["participant_id"], 0) + 1
        )
        pair_counts[row["pair_id"]] = pair_counts.get(row["pair_id"], 0) + 1
    assert len(participant_counts) == 80
    assert set(participant_counts.values()) == {30}
    assert len(pair_counts) == 120
    assert set(pair_counts.values()) == {20}
    code_a = {row["pair_id"]: row["code_a"] for row in verified["study_key"]}
    left_a_counts: dict[str, int] = {}
    for row in verified["assignments"]:
        if row["code_left"] == code_a[row["pair_id"]]:
            left_a_counts[row["pair_id"]] = left_a_counts.get(row["pair_id"], 0) + 1
    assert len(left_a_counts) == 120
    assert set(left_a_counts.values()) == {10}

    with (prepared_study / "restricted" / "formulas.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        formula_rows = list(csv.DictReader(handle))
    assert formula_rows
    assert all(row["manufacturing_execution_authorized"] == "false" for row in formula_rows)
    assert all(row["supplier_sku"] == "" and row["lot_number"] == "" for row in formula_rows)
    assert not (prepared_study / "external" / "human_outcomes.csv").exists()


def test_seal_detects_tampering(prepared_study: Path, tmp_path: Path):
    copied = tmp_path / "study"
    shutil.copytree(prepared_study, copied)
    predictions = copied / "restricted" / "predictions.json"
    predictions.write_text(predictions.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed file changed"):
        study.verify_study_seal(copied)


def test_outcome_identity_and_ranges_are_fail_closed(prepared_study: Path, tmp_path: Path):
    copied = tmp_path / "study"
    shutil.copytree(prepared_study, copied)
    outcome = _completed_outcomes(copied)
    verified = study.verify_study_seal(copied)
    validated = study.validate_outcomes(verified["assignments"], outcome)
    assert len(validated) == 2400

    with outcome.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["code_left"] = "999"
    with outcome.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(study.OUTCOME_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="identity changed"):
        study.validate_outcomes(verified["assignments"], outcome)


def test_manufacturing_evidence_binds_lots_batches_and_all_vials(
    prepared_study: Path, tmp_path: Path
):
    copied = tmp_path / "study"
    shutil.copytree(prepared_study, copied)
    manufacturing, evidence_root = _completed_manufacturing(copied)
    verified = study.verify_study_seal(copied)
    result = study.validate_manufacturing_execution(
        verified, manufacturing, evidence_root
    )
    assert result["formula_batch_count"] == 24
    assert result["sample_vial_count"] == 240
    assert result["supporting_document_count"] > 20

    payload = json.loads(manufacturing.read_text(encoding="utf-8"))
    payload["sample_vials"][0]["finished_product_concentration_percent"] = 99
    manufacturing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="formula or concentration changed"):
        study.validate_manufacturing_execution(verified, manufacturing, evidence_root)


def test_crossed_bootstrap_can_pass_and_fail_the_preregistered_gate(
    prepared_study: Path, tmp_path: Path
):
    good = tmp_path / "good"
    shutil.copytree(prepared_study, good)
    good_outcome = _completed_outcomes(good, prediction_weight=1.0, noise_scale=0.5)
    good_verified = study.verify_study_seal(good)
    good_rows = study.validate_outcomes(good_verified["assignments"], good_outcome)
    good_analysis = study.evaluate_outcomes(
        good_verified["predictions"],
        good_rows,
        bootstrap_draws=200,
        reliability_repeats=100,
        seed=7,
    )
    assert good_analysis["statistical_ninety_percent_gate_passed"] is True
    assert good_analysis["candidate"]["intervals_95"][
        "absolute_similarity_accuracy_percent"
    ][0] >= 90.0

    poor = tmp_path / "poor"
    shutil.copytree(prepared_study, poor)
    poor_outcome = _completed_outcomes(poor, prediction_weight=0.0, noise_scale=20.0)
    poor_verified = study.verify_study_seal(poor)
    poor_rows = study.validate_outcomes(poor_verified["assignments"], poor_outcome)
    poor_analysis = study.evaluate_outcomes(
        poor_verified["predictions"],
        poor_rows,
        bootstrap_draws=200,
        reliability_repeats=100,
        seed=8,
    )
    assert poor_analysis["statistical_ninety_percent_gate_passed"] is False


def test_signed_external_outcome_is_consumed_once(
    prepared_study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    copied = tmp_path / "study"
    shutil.copytree(prepared_study, copied)
    timestamp_payload = {
        "verified": True,
        "time_stamp": "Aug 27 00:00:00 2026 GMT",
        "timestamp_utc": "2026-08-27T00:00:00+00:00",
        "response_sha256": "a" * 64,
        "ca_sha256": "b" * 64,
        "tsa_sha256": "c" * 64,
    }
    timestamp_directory = copied / "timestamp"
    timestamp_directory.mkdir(parents=True, exist_ok=True)
    timestamp_record = {
        "schema": study.STUDY_SCHEMA,
        "study_id": "PFBS-TEST-20260828",
        "seal_sha256": sha256_file(copied / "seal.json"),
        "verified_at": "2026-08-27T00:01:00Z",
        "timestamp": timestamp_payload,
        "human_outcome_present_at_verification": False,
        "manufacturing_evidence_present_at_verification": False,
    }
    (timestamp_directory / "verification.json").write_text(
        json.dumps(timestamp_record), encoding="utf-8"
    )
    outcome = _completed_outcomes(copied, prediction_weight=1.0, noise_scale=0.5)
    manufacturing, evidence_root = _completed_manufacturing(copied)
    manufacturing_verification = study.validate_manufacturing_execution(
        study.verify_study_seal(copied), manufacturing, evidence_root
    )
    seal_hash = sha256_file(copied / "seal.json")
    protocol_hash = sha256_file(copied / "study_protocol.json")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    envelope = {
        "schema": ARTIFACT_SIGNATURE_SCHEMA,
        "artifact_id": "lab-outcome-PFBS-TEST-20260828",
        "artifact_type": study.LAB_ARTIFACT_TYPE,
        "signer_id": "independent-lab-01",
        "signer_role": "sensory_laboratory",
        "scope": {
            "study_id": "PFBS-TEST-20260828",
            "protocol_sha256": protocol_hash,
            "seal_sha256": seal_hash,
            "manufacturing_sha256": sha256_file(manufacturing),
        },
        "issued_at": "2026-08-28T00:00:00Z",
        "expires_at": "2030-01-01T00:00:00Z",
        "artifact_hashes": {
            "human_outcomes_csv": sha256_file(outcome),
            **{
                label: sha256_file(path)
                for label, path in manufacturing_verification["artifact_paths"].items()
            },
        },
    }
    envelope["signature"] = base64.b64encode(
        private_key.sign(signing_payload(envelope))
    ).decode("ascii")
    envelope_path = tmp_path / "lab-envelope.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    trust_root = {
        "signers": {
            "independent-lab-01": {
                "public_key": public_key.hex(),
                "roles": ["sensory_laboratory"],
                "artifact_types": [study.LAB_ARTIFACT_TYPE],
                "scope_constraints": {"study_id": ["PFBS-TEST-20260828"]},
            }
        }
    }
    trust_path = tmp_path / "trust-root.json"
    trust_path.write_text(json.dumps(trust_root), encoding="utf-8")
    monkeypatch.setattr(
        study,
        "verify_rfc3161_timestamp",
        lambda **_: timestamp_payload,
    )
    report = copied / "final-report.json"
    ledger = copied.parent / "prospective_formula_evidence_ledger.jsonl"
    result = study.finalize_study(
        copied,
        outcomes_path=outcome,
        manufacturing_evidence_path=manufacturing,
        evidence_root=evidence_root,
        signature_envelope_path=envelope_path,
        trust_root_path=trust_path,
        openssl=tmp_path / "unused-openssl",
        timestamp_response_path=tmp_path / "unused.tsr",
        timestamp_ca_path=tmp_path / "unused-ca.pem",
        timestamp_tsa_path=tmp_path / "unused-tsa.crt",
        report_path=report,
        ledger_path=ledger,
        bootstrap_draws=200,
        reliability_repeats=100,
        seed=9,
    )
    assert result["human_olfactory_similarity_90_gate_passed"] is True
    assert report.is_file()
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        study.finalize_study(
            copied,
            outcomes_path=outcome,
            manufacturing_evidence_path=manufacturing,
            evidence_root=evidence_root,
            signature_envelope_path=envelope_path,
            trust_root_path=trust_path,
            openssl=tmp_path / "unused-openssl",
            timestamp_response_path=tmp_path / "unused.tsr",
            timestamp_ca_path=tmp_path / "unused-ca.pem",
            timestamp_tsa_path=tmp_path / "unused-tsa.crt",
            report_path=report,
            ledger_path=ledger,
            bootstrap_draws=100,
            reliability_repeats=100,
            seed=10,
        )
