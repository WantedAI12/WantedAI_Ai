"""Build, sign and verify a cryptographic commercial-release evidence chain.

The script never accepts a claimed report hash in lieu of a file and never
records unsigned evidence.  Sign payloads are emitted for an external
Ed25519/HSM workflow; this program intentionally has no private-key option.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.models import RecipeConstraints, RecipeLine  # noqa: E402
from fragrance_ai.recommender.release import (  # noqa: E402
    CommercialReleaseStore,
    VerifiedRegulatorySignoff,
    signing_payload,
)
from fragrance_ai.recommender.release_spec import ReleaseSpec, sha256_file  # noqa: E402
from fragrance_ai.recommender.supplier import SupplierMaterial, SupplierRegistry  # noqa: E402


def _scope_from_input(path: str | Path) -> ReleaseSpec:
    """Create a verifiable scope from an operator-controlled JSON input.

    Expected shape: ``{lines, constraints, supplier_registry: {records: [...]}}``.
    ``constraints.commercial_supplier_evidence`` holds real document paths.
    No file path or document digest is copied into the signed scope unchecked.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_constraints = {item.name for item in fields(RecipeConstraints)}
    constraints = RecipeConstraints(
        **{
            key: value
            for key, value in dict(payload.get("constraints", {})).items()
            if key in allowed_constraints
        }
    )
    lines = []
    for item in payload.get("lines", []):
        line = dict(item)
        lines.append(
            RecipeLine(
                ingredient_id=str(line["ingredient_id"]),
                name=str(line.get("name", line["ingredient_id"])),
                pyramid=str(line.get("pyramid", "unknown")),
                concentrate_percent=float(line["concentrate_percent"]),
                finished_product_percent=float(line["finished_product_percent"]),
                volume_ml_for_batch=None,
                price_per_kg=0.0,
                availability=1.0,
                risk_tier=0,
                reason="release-scope-input",
                active_strength_percent=float(line.get("active_strength_percent", 100.0)),
                carrier=line.get("carrier"),
            )
        )
    registry_data = dict(payload.get("supplier_registry", {}))
    registry = SupplierRegistry(
        (SupplierMaterial.from_mapping(item) for item in registry_data.get("records", [])),
        registry_data.get("metadata", {}),
    )
    return ReleaseSpec.build(
        lines,
        constraints,
        registry,
        rule_pack_version=constraints.commercial_rule_pack_version,
        data_version=constraints.commercial_data_version,
        model_version=constraints.commercial_model_version,
    )


def _trusted_signers(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("trusted signer file must be a JSON object keyed by signer_id")
    return data


def _signoff(args: argparse.Namespace, scope: ReleaseSpec) -> VerifiedRegulatorySignoff:
    report_path = Path(args.report_file).expanduser().resolve(strict=True)
    return VerifiedRegulatorySignoff(
        release_spec_id=scope.release_spec_id,
        market_region=args.market_region,
        approver_role=args.approver_role,
        organization=args.organization,
        signer_id=args.signer_id,
        approved_on=args.approved_on,
        valid_until=args.valid_until,
        report_ref=args.report_ref,
        report_path=str(report_path),
        report_sha256=sha256_file(report_path),
        signature=(Path(args.signature_file).read_text(encoding="ascii").strip() if getattr(args, "signature_file", None) else ""),
    )


def _add_signoff_arguments(parser: argparse.ArgumentParser, include_signature: bool) -> None:
    parser.add_argument("--scope-input", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--market-region", required=True)
    parser.add_argument("--approver-role", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--signer-id", required=True)
    parser.add_argument("--approved-on", required=True)
    parser.add_argument("--valid-until", required=True)
    parser.add_argument("--report-ref", required=True)
    if include_signature:
        parser.add_argument("--signature-file", required=True, help="base64 Ed25519 signature")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", help="SQLite evidence database; required for record/assess")
    commands = parser.add_subparsers(dest="command", required=True)
    scope_command = commands.add_parser("build-scope")
    scope_command.add_argument("--scope-input", required=True)
    payload_command = commands.add_parser("signing-payload")
    _add_signoff_arguments(payload_command, include_signature=False)
    record_command = commands.add_parser("record")
    _add_signoff_arguments(record_command, include_signature=True)
    record_command.add_argument("--trusted-signers", required=True)
    assess_command = commands.add_parser("assess")
    assess_command.add_argument("--scope-input", required=True)
    assess_command.add_argument("--trusted-signers", required=True)
    assess_command.add_argument("--as-of", default=date.today().isoformat())
    revoke_command = commands.add_parser("revoke")
    revoke_command.add_argument("--release-spec-id", required=True)
    revoke_command.add_argument("--report-sha256", required=True)
    revoke_command.add_argument("--revoked-on", required=True)
    revoke_command.add_argument("--reason", required=True)
    args = parser.parse_args()

    if args.command == "build-scope":
        scope = _scope_from_input(args.scope_input)
        print(json.dumps({"release_spec_id": scope.release_spec_id, "payload": scope.payload}, ensure_ascii=False, indent=2))
        return
    if args.command == "signing-payload":
        scope = _scope_from_input(args.scope_input)
        signoff = _signoff(args, scope)
        payload = signing_payload(signoff)
        print(json.dumps({
            "release_spec_id": scope.release_spec_id,
            "report_sha256": signoff.report_sha256,
            "signing_payload_base64": base64.b64encode(payload).decode("ascii"),
            "signing_payload_sha256": hashlib.sha256(payload).hexdigest(),
        }, ensure_ascii=False, indent=2))
        return
    if args.command in {"record", "assess", "revoke"} and not args.db:
        parser.error("--db is required for record, assess and revoke")

    if args.command == "revoke":
        store = CommercialReleaseStore(args.db)
        store.revoke(args.release_spec_id, args.report_sha256, date.fromisoformat(args.revoked_on), args.reason)
        print(json.dumps({"revoked": True}, ensure_ascii=False, indent=2))
        return

    scope = _scope_from_input(args.scope_input)
    store = CommercialReleaseStore(args.db, _trusted_signers(args.trusted_signers))
    if args.command == "record":
        store.record_verified(scope, _signoff(args, scope))
        output = {"recorded": True, "release_spec_id": scope.release_spec_id}
    else:
        output = store.assess_scope(scope, date.fromisoformat(args.as_of)).__dict__
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
