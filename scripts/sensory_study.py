"""Create and operate auditable blind sensory studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender.artifact_trust import EvidenceTrustRoot  # noqa: E402
from fragrance_ai.recommender.sensory import SensoryEvaluationStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--trust-root",
        help="independent signer allowlist JSON; required for import-verified",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("result_json")

    study = commands.add_parser("create-study")
    study.add_argument("--brief", required=True)
    study.add_argument("--formula-id", action="append", required=True)
    study.add_argument("--protocol-ref", required=True)

    record = commands.add_parser("record")
    record.add_argument("--study-id", required=True)
    record.add_argument("--blind-code", required=True)
    record.add_argument("--panelist-id", required=True)
    record.add_argument("--similarity", required=True, type=float)
    record.add_argument("--liking", required=True, type=float)
    record.add_argument("--expert", action="store_true")
    record.add_argument("--dimensions", default="{}")
    record.add_argument("--defect", action="append", default=[])

    report = commands.add_parser("report")
    report.add_argument("--formula-id", required=True)
    report.add_argument("--target", type=float, default=90.0)
    report.add_argument("--min-panelists", type=int, default=12)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--output", required=True)

    import_csv = commands.add_parser("import-human-csv")
    import_csv.add_argument("csv")

    import_verified = commands.add_parser("import-verified")
    import_verified.add_argument("--study-id", required=True)
    import_verified.add_argument("csv")
    import_verified.add_argument(
        "--envelope-json", required=True,
        help="signed research-data evidence envelope bound to this study and formula set",
    )

    args = parser.parse_args()
    if args.command == "import-verified" and not args.trust_root:
        parser.error("import-verified requires --trust-root")
    trust_root = EvidenceTrustRoot.from_json_file(args.trust_root) if args.trust_root else None
    store = SensoryEvaluationStore(args.db, trusted_signers=trust_root)
    if args.command == "register":
        payload = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
        lines = payload.get("recipe") or payload.get("closest_candidate") or []
        store.register_formula(
            payload["formula_id"],
            payload["brief"]["original_text"],
            payload.get("raw_similarity_score", payload["similarity_score"]),
            [
                {
                    "ingredient_id": item["ingredient_id"],
                    "concentrate_percent": item["concentrate_percent"],
                }
                for item in lines
            ],
        )
        output = {"registered": payload["formula_id"]}
    elif args.command == "create-study":
        output = store.create_study(
            args.brief, args.formula_id, args.protocol_ref
        )
    elif args.command == "record":
        store.record_evaluation(
            args.study_id,
            args.blind_code,
            args.panelist_id,
            args.similarity,
            args.liking,
            json.loads(args.dimensions),
            args.defect,
            args.expert,
        )
        output = {"recorded": True, "status": "legacy_unverified"}
    elif args.command == "report":
        output = store.formula_evidence(
            args.formula_id, args.target, args.min_panelists
        ).__dict__
    elif args.command == "calibrate":
        artifact = store.fit_calibrator()
        artifact.save(args.output)
        output = artifact.__dict__
    elif args.command == "import-human-csv":
        output = {"imported_rows": store.import_human_csv(args.csv)}
        output["status"] = "legacy_unverified"
    else:
        envelope = json.loads(Path(args.envelope_json).read_text(encoding="utf-8"))
        output = {
            "imported_rows": store.import_verified(args.study_id, args.csv, envelope),
            "status": "verified_external_study",
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
