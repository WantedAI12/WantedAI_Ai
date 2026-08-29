"""Record and report physical stability and pilot-batch evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.artifact_trust import EvidenceTrustRoot  # noqa: E402
from fragrance_ai.recommender.quality import QualityEvidenceStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--trust-root",
        help="independent signer allowlist JSON; required for record-verified",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--formula-id", required=True)
    record.add_argument("--test", required=True)
    record.add_argument("--result", choices=("pass", "fail"), required=True)
    record.add_argument("--date", required=True)
    record.add_argument("--protocol-ref", required=True)
    record.add_argument("--report-ref", required=True)
    record.add_argument("--notes", default="")

    record_verified = commands.add_parser("record-verified")
    record_verified.add_argument("--release-spec-id", required=True)
    record_verified.add_argument("--test", required=True)
    record_verified.add_argument("--result", choices=("pass", "fail"), required=True)
    record_verified.add_argument("--date", required=True)
    record_verified.add_argument("--protocol-path", required=True)
    record_verified.add_argument("--report-path", required=True)
    record_verified.add_argument(
        "--envelope-json", required=True,
        help="signed evidence envelope; it must cover both protocol/report bytes",
    )
    report = commands.add_parser("report")
    report.add_argument("--formula-id", required=True)
    args = parser.parse_args()

    if args.command == "record-verified" and not args.trust_root:
        parser.error("record-verified requires --trust-root")
    trust_root = EvidenceTrustRoot.from_json_file(args.trust_root) if args.trust_root else None
    store = QualityEvidenceStore(args.db, trusted_signers=trust_root)
    if args.command == "record":
        store.record(
            args.formula_id,
            args.test,
            args.result,
            date.fromisoformat(args.date),
            args.protocol_ref,
            args.report_ref,
            args.notes,
        )
        output = {"recorded": True, "status": "legacy_unverified"}
    elif args.command == "record-verified":
        envelope = json.loads(Path(args.envelope_json).read_text(encoding="utf-8"))
        store.record_verified(
            args.release_spec_id,
            args.test,
            args.result,
            date.fromisoformat(args.date),
            args.protocol_path,
            args.report_path,
            envelope,
        )
        output = {"recorded": True, "status": "verified_pending_assessment"}
    else:
        output = store.assess(args.formula_id).__dict__
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
