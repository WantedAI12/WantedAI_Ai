"""Operate the fail-closed continual-improvement champion/challenger loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .recommender.artifact_trust import EvidenceTrustRoot
from .recommender.continual_training import (
    finalize_blind_challenge,
    prepare_blind_challenge,
    process_learning_jobs,
)
from .recommender.continuous_improvement import ContinuousImprovementController


def _root(value: str | None) -> Path:
    configured = value or os.environ.get("PERFUMERY_AI_CONTINUAL_ROOT")
    return Path(configured or "continuous-improvement").expanduser().resolve()


def _trust_root(value: str | None) -> EvidenceTrustRoot:
    configured = value or os.environ.get("PERFUMERY_AI_CONTINUAL_TRUST_ROOT")
    return (
        EvidenceTrustRoot.from_json_file(configured)
        if configured
        else EvidenceTrustRoot()
    )


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _controller(args: argparse.Namespace) -> ContinuousImprovementController:
    return ContinuousImprovementController(
        _root(args.root), trust_root=_trust_root(args.trust_root)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="새 후보를 한 번 평가하고 registry를 갱신")
    run.add_argument("--root")
    run.add_argument("--trust-root")

    status = subparsers.add_parser("status", help="registry와 감사 체인 검증")
    status.add_argument("--root")
    status.add_argument("--trust-root")

    watch = subparsers.add_parser("watch", help="후보 inbox를 지속적으로 감시")
    watch.add_argument("--root")
    watch.add_argument("--trust-root")
    watch.add_argument("--interval-seconds", type=float, default=300.0)

    prepare = subparsers.add_parser(
        "prepare-blind",
        help="결과 열람 없이 challenger 학습·예측·seal 생성",
    )
    prepare.add_argument("--training-csv", required=True)
    prepare.add_argument("--challenge-inputs-csv", required=True)
    prepare.add_argument("--prepared-dir", required=True)
    prepare.add_argument("--candidate-id", required=True)
    prepare.add_argument("--baseline-manifest")
    prepare.add_argument("--baseline-runtime")

    finalize = subparsers.add_parser(
        "finalize-blind",
        help="외부 timestamp/결과 receipt를 검증해 immutable 후보 bundle 생성",
    )
    finalize.add_argument("--prepared-dir", required=True)
    finalize.add_argument("--outcomes-csv", required=True)
    finalize.add_argument("--dataset-receipt", required=True)
    finalize.add_argument("--timestamp-response", required=True)
    finalize.add_argument("--acquisition-authorization", required=True)
    finalize.add_argument("--root")
    finalize.add_argument("--bootstrap-draws", type=int, default=5000)

    authorization = subparsers.add_parser(
        "authorization-request",
        help="현재 shadow의 외부 Ed25519 서명용 정확한 payload 생성",
    )
    authorization.add_argument("--root")
    authorization.add_argument("--trust-root")
    authorization.add_argument("--candidate-id", required=True)
    authorization.add_argument("--signer-id", required=True)
    authorization.add_argument("--issued-at", required=True)
    authorization.add_argument("--expires-at", required=True)
    authorization.add_argument("--artifact-id")
    authorization.add_argument("--output")

    args = parser.parse_args()
    try:
        if args.command == "run":
            controller = _controller(args)
            jobs = process_learning_jobs(_root(args.root), controller)
            result = controller.run_once()
            result["learning_jobs"] = jobs
            _print(result)
        elif args.command == "status":
            _print(_controller(args).status())
        elif args.command == "watch":
            if not 5.0 <= args.interval_seconds <= 86_400.0:
                raise ValueError("--interval-seconds must be between 5 and 86400")
            controller = _controller(args)
            try:
                while True:
                    jobs = process_learning_jobs(_root(args.root), controller)
                    result = controller.run_once()
                    result["learning_jobs"] = jobs
                    if result["processed_now"] or jobs:
                        _print(result)
                    time.sleep(args.interval_seconds)
            except KeyboardInterrupt:
                _print({"status": "stopped"})
        elif args.command == "prepare-blind":
            _print(
                prepare_blind_challenge(
                    training_csv=args.training_csv,
                    challenge_inputs_csv=args.challenge_inputs_csv,
                    output_dir=args.prepared_dir,
                    candidate_id=args.candidate_id,
                    baseline_manifest=args.baseline_manifest,
                    baseline_runtime=args.baseline_runtime,
                )
            )
        elif args.command == "finalize-blind":
            _print(
                finalize_blind_challenge(
                    prepared_dir=args.prepared_dir,
                    outcomes_csv=args.outcomes_csv,
                    dataset_receipt_json=args.dataset_receipt,
                    timestamp_response=args.timestamp_response,
                    acquisition_authorization=args.acquisition_authorization,
                    inbox_root=_root(args.root) / "inbox",
                    bootstrap_draws=args.bootstrap_draws,
                )
            )
        elif args.command == "authorization-request":
            request = _controller(args).build_authorization_request(
                candidate_id=args.candidate_id,
                signer_id=args.signer_id,
                issued_at=args.issued_at,
                expires_at=args.expires_at,
                artifact_id=args.artifact_id,
            )
            if args.output:
                output = Path(args.output).expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
            _print(request)
    except Exception as error:  # noqa: BLE001 - CLI boundary
        _print({"status": "failed_closed", "error": f"{type(error).__name__}:{error}"})
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
