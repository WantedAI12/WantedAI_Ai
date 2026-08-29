"""Run the authenticated multi-tenant perfumery workspace API and UI.

The service exposes formulation projects, immutable formula versions,
distributed inference jobs, metrics, and audit verification. It deliberately
cannot create supplier, quality, sensory, or release-approval evidence.
"""

from __future__ import annotations

import argparse
import os

from .api import authorizer_from_env, create_app
from .platform.audit import audit_log_from_env
from .platform.observability import configure_json_logging
from .platform.store import workspace_store_from_env
from .recommender.service import NaturalLanguagePerfumeryAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--audit-db")
    parser.add_argument("--workspace-db")
    parser.add_argument("--max-concurrent-inference", type=int, default=2)
    parser.add_argument("--max-request-bytes", type=int, default=65_536)
    parser.add_argument("--disable-ui", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.max_concurrent_inference <= 0:
        raise SystemExit("--max-concurrent-inference must be positive")
    if not 4_096 <= args.max_request_bytes <= 10_000_000:
        raise SystemExit("--max-request-bytes must be between 4096 and 10000000")
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit(
            "API dependencies are missing; install perfumery-ai-core[commercial]"
        ) from error

    if args.audit_db:
        os.environ["PERFUMERY_AI_AUDIT_DB"] = args.audit_db
    if args.workspace_db:
        os.environ["PERFUMERY_AI_WORKSPACE_DB"] = args.workspace_db
    configure_json_logging(os.environ.get("PERFUMERY_AI_LOG_LEVEL", "INFO"))
    # Production mode requires OIDC, PostgreSQL, and an audit HMAC key. Partial
    # configuration never downgrades to static tokens or local SQLite.
    authorizer = authorizer_from_env()
    store = workspace_store_from_env()
    audit_log = audit_log_from_env()
    if store.horizontally_scalable:
        from .platform.postgres import PostgresFixedWindowRateLimiter

        rate_limiter = PostgresFixedWindowRateLimiter(
            store,
            requests=int(os.environ.get("PERFUMERY_AI_RATE_LIMIT_REQUESTS", "60")),
            window_seconds=int(os.environ.get("PERFUMERY_AI_RATE_LIMIT_WINDOW", "60")),
        )
    else:
        rate_limiter = None
    app = create_app(
        ai_factory=NaturalLanguagePerfumeryAI,
        authorizer=authorizer,
        audit_log=audit_log,
        workspace_store=store,
        rate_limiter=rate_limiter,
        max_concurrent_inference=args.max_concurrent_inference,
        max_request_bytes=args.max_request_bytes,
        enable_ui=not args.disable_ui,
    )
    # One process owns the local SQLite audit chain.  Horizontal deployments
    # must inject a transactional shared audit backend instead of adding
    # uvicorn workers around this file.
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            access_log=os.environ.get("PERFUMERY_AI_ACCESS_LOG", "1") != "0",
            proxy_headers=os.environ.get("PERFUMERY_AI_TRUST_PROXY_HEADERS", "0") == "1",
        )
    finally:
        audit_log.close()
        store.close()


if __name__ == "__main__":
    main()
