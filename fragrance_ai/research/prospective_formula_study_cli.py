"""Command line entry point for prospective generated-formula validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prospective_formula_study import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_RELIABILITY_REPEATS,
    create_timestamp_query,
    finalize_study,
    prepare_study,
    record_timestamp_verification,
    verify_study_seal,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, timestamp, verify, or finalize an outcome-blind human "
            "similarity study for generated perfume formulas."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="generate and seal a new study")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--study-id")
    prepare.add_argument("--randomization-seed", type=int)
    prepare.add_argument(
        "--openssl",
        type=Path,
        help="optional OpenSSL executable; creates timestamp/seal.tsq",
    )

    timestamp = subparsers.add_parser(
        "timestamp-query", help="create an RFC3161 request for an existing seal"
    )
    timestamp.add_argument("--study-dir", type=Path, required=True)
    timestamp.add_argument("--openssl", type=Path, required=True)

    verify = subparsers.add_parser("verify-seal", help="verify all sealed bytes")
    verify.add_argument("--study-dir", type=Path, required=True)

    verify_timestamp = subparsers.add_parser(
        "verify-timestamp", help="verify and record an RFC3161 seal timestamp"
    )
    verify_timestamp.add_argument("--study-dir", type=Path, required=True)
    verify_timestamp.add_argument("--openssl", type=Path, required=True)
    verify_timestamp.add_argument("--timestamp-response", type=Path, required=True)
    verify_timestamp.add_argument("--timestamp-ca", type=Path, required=True)
    verify_timestamp.add_argument("--timestamp-tsa", type=Path, required=True)
    verify_timestamp.add_argument("--output", type=Path)

    finalize = subparsers.add_parser(
        "finalize", help="consume independently signed outcomes and score once"
    )
    finalize.add_argument("--study-dir", type=Path, required=True)
    finalize.add_argument("--outcomes", type=Path, required=True)
    finalize.add_argument("--manufacturing-evidence", type=Path, required=True)
    finalize.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="root directory containing every document referenced by manufacturing evidence",
    )
    finalize.add_argument("--signature-envelope", type=Path, required=True)
    finalize.add_argument("--trust-root", type=Path, required=True)
    finalize.add_argument("--openssl", type=Path, required=True)
    finalize.add_argument("--timestamp-response", type=Path, required=True)
    finalize.add_argument("--timestamp-ca", type=Path, required=True)
    finalize.add_argument("--timestamp-tsa", type=Path, required=True)
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--ledger", type=Path, required=True)
    finalize.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    finalize.add_argument(
        "--reliability-repeats", type=int, default=DEFAULT_RELIABILITY_REPEATS
    )
    finalize.add_argument("--seed", type=int, default=20260828)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        result = prepare_study(
            args.output_dir,
            study_id=args.study_id,
            randomization_seed=args.randomization_seed,
            openssl=args.openssl,
        )
    elif args.command == "timestamp-query":
        result = {"timestamp_query": str(create_timestamp_query(args.study_dir, args.openssl))}
    elif args.command == "verify-seal":
        verified = verify_study_seal(args.study_dir)
        result = {
            "study_id": verified["seal"]["study_id"],
            "sealed_files": len(verified["seal"]["files"]),
            "prediction_pairs": len(verified["predictions"]["pairs"]),
            "assignments": len(verified["assignments"]),
            "verified": True,
        }
    elif args.command == "verify-timestamp":
        result = record_timestamp_verification(
            args.study_dir,
            openssl=args.openssl,
            response_path=args.timestamp_response,
            ca_path=args.timestamp_ca,
            tsa_path=args.timestamp_tsa,
            output_path=args.output,
        )
    else:
        result = finalize_study(
            args.study_dir,
            outcomes_path=args.outcomes,
            manufacturing_evidence_path=args.manufacturing_evidence,
            evidence_root=args.evidence_root,
            signature_envelope_path=args.signature_envelope,
            trust_root_path=args.trust_root,
            openssl=args.openssl,
            timestamp_response_path=args.timestamp_response,
            timestamp_ca_path=args.timestamp_ca,
            timestamp_tsa_path=args.timestamp_tsa,
            report_path=args.report,
            ledger_path=args.ledger,
            bootstrap_draws=args.bootstrap_draws,
            reliability_repeats=args.reliability_repeats,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
