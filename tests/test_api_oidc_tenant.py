from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from fragrance_ai.api import (
    OIDCJWTAuthorizer,
    SlidingWindowRateLimiter,
    TokenAuthorizer,
    authorizer_from_env,
    create_app,
)
from fragrance_ai.recommender.audit_log import AppendOnlyAuditLog


NOW = 1_800_000_000.0


def _claims(*, tenant_id: str = "tenant-a", **overrides):
    claims = {
        "iss": "https://identity.example",
        "aud": ["other-audience", "perfumery-api"],
        "sub": "chemist-7",
        "role": "formulator",
        "tenant_id": tenant_id,
        "iat": NOW - 10,
        "exp": NOW + 600,
    }
    claims.update(overrides)
    return claims


def test_oidc_claim_gate_is_offline_testable_and_fail_closed():
    claims_by_token = {
        "good": _claims(),
        "bad-issuer": _claims(iss="https://attacker.example"),
        "expired": _claims(exp=NOW - 61),
        "unknown-role": _claims(role="release-approver"),
        "missing-tenant": _claims(tenant_id=""),
    }
    authorizer = OIDCJWTAuthorizer(
        issuer="https://identity.example",
        audience="perfumery-api",
        jwks_url="https://identity.example/.well-known/jwks.json",
        claims_decoder=lambda token: claims_by_token[token],
        clock=lambda: NOW,
    )

    principal = authorizer.authenticate("good")
    assert principal is not None
    assert principal.actor_id == "chemist-7"
    assert principal.tenant_id == "tenant-a"
    assert authorizer.permits(principal, "recipe:create")
    assert not authorizer.permits(principal, "audit:verify")
    for invalid_token in ("bad-issuer", "expired", "unknown-role", "missing-tenant"):
        assert authorizer.authenticate(invalid_token) is None


def test_static_tokens_keep_legacy_shape_and_support_explicit_tenant():
    legacy = TokenAuthorizer.from_plaintext({"legacy": ("operator-1", "formulator")})
    tenant_bound = TokenAuthorizer.from_plaintext(
        {"tenant-token": ("operator-2", "formulator", "tenant-b")}
    )
    assert legacy.authenticate("legacy").tenant_id == "default"
    assert tenant_bound.authenticate("tenant-token").tenant_id == "tenant-b"


def test_partial_or_conflicting_oidc_environment_cannot_fall_back_to_tokens(monkeypatch):
    monkeypatch.setenv("PERFUMERY_AI_API_TOKENS", '{"token": {"actor_id": "a", "role": "admin"}}')
    monkeypatch.setenv("PERFUMERY_AI_OIDC_ISSUER", "https://identity.example")
    with pytest.raises(RuntimeError, match="OIDC configuration is incomplete"):
        authorizer_from_env()
    monkeypatch.setenv("PERFUMERY_AI_AUTH_MODE", "static")
    with pytest.raises(RuntimeError, match="cannot be combined"):
        authorizer_from_env()


def test_production_refuses_static_authentication(monkeypatch):
    monkeypatch.setenv("PERFUMERY_AI_ENV", "production")
    monkeypatch.setenv("PERFUMERY_AI_AUTH_MODE", "static")
    monkeypatch.setenv(
        "PERFUMERY_AI_API_TOKENS",
        '{"token": {"actor_id": "admin-1", "role": "admin"}}',
    )
    for variable in (
        "PERFUMERY_AI_OIDC_ISSUER",
        "PERFUMERY_AI_OIDC_AUDIENCE",
        "PERFUMERY_AI_OIDC_JWKS_URL",
    ):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(RuntimeError, match="production requires OIDC"):
        authorizer_from_env()


def test_oidc_token_lifetime_and_permission_narrowing_are_fail_closed():
    claims_by_token = {
        "too-long": _claims(iat=NOW - 10, exp=NOW + 3591),
        "narrow": _claims(
            permissions=["project:read", "formula:edit", "audit:verify"]
        ),
    }
    authorizer = OIDCJWTAuthorizer(
        issuer="https://identity.example",
        audience="perfumery-api",
        jwks_url="https://identity.example/.well-known/jwks.json",
        claims_decoder=lambda token: claims_by_token[token],
        clock=lambda: NOW,
        max_token_lifetime_seconds=3600,
    )

    assert authorizer.authenticate("too-long") is None
    principal = authorizer.authenticate("narrow")
    assert principal is not None
    assert authorizer.permits(principal, "project:read")
    assert authorizer.permits(principal, "formula:edit")
    assert not authorizer.permits(principal, "recipe:create")
    assert not authorizer.permits(principal, "audit:verify")


def test_api_binds_tenant_header_audit_scope_and_rate_limit(tmp_path):
    pytest.importorskip("fastapi")
    from tests._http_client import TestClient

    @dataclass
    class FakeResult:
        def to_dict(self):
            return {"status": "prototype_ready", "formula_id": "sha256:" + "a" * 64}

    class FakeAI:
        def create_recipe(self, brief, constraints):
            return FakeResult()

    claims_by_token = {
        "tenant-a-token": _claims(tenant_id="tenant-a"),
        "tenant-a-rotated-token": _claims(tenant_id="tenant-a"),
        "tenant-b-token": _claims(tenant_id="tenant-b"),
    }
    authorizer = OIDCJWTAuthorizer(
        issuer="https://identity.example",
        audience="perfumery-api",
        jwks_url="https://identity.example/.well-known/jwks.json",
        claims_decoder=lambda token: claims_by_token[token],
        clock=lambda: NOW,
    )
    audit = AppendOnlyAuditLog(tmp_path / "tenant-audit.db")
    app = create_app(
        ai_factory=FakeAI,
        authorizer=authorizer,
        audit_log=audit,
        rate_limiter=SlidingWindowRateLimiter(requests=1, window_seconds=60),
    )
    client = TestClient(app)
    client.__enter__()

    accepted_a = client.post(
        "/v1/recipes",
        headers={
            "Authorization": "Bearer tenant-a-token",
            "X-Tenant-ID": "tenant-a",
        },
        json={"brief": "clean citrus"},
    )
    assert accepted_a.status_code == 200
    assert accepted_a.json()["tenant_id"] == "tenant-a"

    mismatch = client.post(
        "/v1/recipes",
        headers={
            "Authorization": "Bearer tenant-a-token",
            "X-Tenant-ID": "tenant-b",
        },
        json={"brief": "clean citrus"},
    )
    assert mismatch.status_code == 403

    # Tenant A has consumed its one-request principal allowance.  Replacing a
    # bearer credential does not reset it; tenant B remains independent despite
    # sharing the same OIDC subject.
    assert client.post(
        "/v1/recipes",
        headers={"Authorization": "Bearer tenant-a-rotated-token"},
        json={"brief": "clean citrus"},
    ).status_code == 429
    accepted_b = client.post(
        "/v1/recipes",
        headers={
            "Authorization": "Bearer tenant-b-token",
            "X-Tenant-ID": "tenant-b",
        },
        json={"brief": "clean citrus"},
    )
    assert accepted_b.status_code == 200

    rows = audit.connection.execute(
        "SELECT scope_id, payload_json FROM audit_events ORDER BY sequence"
    ).fetchall()
    assert len(rows) == 4
    assert all(row[0].startswith("tenant:") for row in rows)
    assert [json.loads(row[1])["tenant_id"] for row in rows] == [
        "tenant-a",
        "tenant-a",
        "tenant-b",
        "tenant-b",
    ]
    client.__exit__(None, None, None)
    audit.close()
