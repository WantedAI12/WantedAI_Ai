"""Optional authenticated HTTP surface for the perfumery R&D core.

Install ``perfumery-ai-core[commercial]`` to use this module.  Evidence-writing
endpoints are intentionally absent: supplier, sensory, quality, and regulatory
records must enter through their separately authenticated verification flows.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import re
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse
from concurrent.futures import Future, ThreadPoolExecutor

ROLE_PERMISSIONS = {
    "viewer": frozenset({"project:read", "formula:read", "job:read", "catalog:read"}),
    "formulator": frozenset(
        {
            "recipe:create",
            "project:read",
            "project:create",
            "formula:read",
            "formula:create",
            "formula:edit",
            "job:read",
            "job:create",
            "catalog:read",
        }
    ),
    "auditor": frozenset(
        {
            "audit:verify",
            "metrics:read",
            "project:read",
            "formula:read",
            "job:read",
            "catalog:read",
        }
    ),
    "admin": frozenset(
        {
            "recipe:create",
            "audit:verify",
            "metrics:read",
            "project:read",
            "project:create",
            "formula:read",
            "formula:create",
            "formula:edit",
            "job:read",
            "job:create",
            "catalog:read",
        }
    ),
}

LOGGER = logging.getLogger("perfumery_ai.api")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

_DEFAULT_TENANT_ID = "default"
_SAFE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:@"
)
_OIDC_ALLOWED_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512", "EdDSA"}
)


def _identity_value(value: object, field: str) -> str:
    """Validate an identifier before it becomes an audit or limiter key."""

    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(
            f"{field} must be a non-empty identifier of at most 128 characters"
        )
    if any(character not in _SAFE_ID_CHARS for character in value):
        raise ValueError(f"{field} contains unsupported characters")
    return value


def _permission_permitted(principal: "Principal", permission: str) -> bool:
    role_permissions = ROLE_PERMISSIONS.get(principal.role, frozenset())
    if principal.permissions is None:
        return permission in role_permissions
    return permission in role_permissions and permission in principal.permissions


@dataclass(frozen=True)
class Principal:
    actor_id: str
    role: str
    token_fingerprint: str
    tenant_id: str = _DEFAULT_TENANT_ID
    permissions: frozenset[str] | None = None

    @property
    def rate_limit_identity(self) -> str:
        """Principal rate-limit key, cryptographically bound to its tenant.

        The bearer credential is deliberately excluded so refreshing or
        rotating a token cannot reset a subject's per-tenant allowance.
        """

        material = "\x1f".join((self.tenant_id, self.actor_id)).encode("utf-8")
        return "tenant-principal:" + hashlib.sha256(material).hexdigest()


class Authorizer(Protocol):
    def authenticate(self, token: str) -> Principal | None: ...

    @staticmethod
    def permits(principal: Principal, permission: str) -> bool: ...


class TokenAuthorizer:
    """Constant-time bearer-token lookup using only stored SHA-256 digests."""

    def __init__(
        self,
        principals_by_digest: Mapping[str, tuple[str, str] | tuple[str, str, str]],
    ):
        self._principals: dict[str, tuple[str, str, str]] = {}
        for digest, encoded_principal in principals_by_digest.items():
            normalized = digest.lower()
            if len(normalized) != 64 or any(
                ch not in "0123456789abcdef" for ch in normalized
            ):
                raise ValueError("token digest must be 64 hexadecimal characters")
            if len(encoded_principal) == 2:
                actor_id, role = encoded_principal
                tenant_id = _DEFAULT_TENANT_ID
            elif len(encoded_principal) == 3:
                actor_id, role, tenant_id = encoded_principal
            else:
                raise ValueError(
                    "each token principal must contain actor_id, role, and optional tenant_id"
                )
            if not isinstance(role, str) or role not in ROLE_PERMISSIONS:
                raise ValueError(f"unsupported role: {role}")
            self._principals[normalized] = (
                _identity_value(actor_id, "actor_id"),
                role,
                _identity_value(tenant_id, "tenant_id"),
            )

    @classmethod
    def from_plaintext(
        cls,
        values: Mapping[str, tuple[str, str] | tuple[str, str, str]],
    ) -> "TokenAuthorizer":
        for token in values:
            if not isinstance(token, str) or not token or len(token) > 16_384:
                raise ValueError(
                    "token must be a non-empty string of at most 16384 characters"
                )
        return cls(
            {
                hashlib.sha256(token.encode("utf-8")).hexdigest(): principal
                for token, principal in values.items()
            }
        )

    @classmethod
    def from_env(cls, variable: str = "PERFUMERY_AI_API_TOKENS") -> "TokenAuthorizer":
        raw = os.environ.get(variable, "")
        if not raw:
            raise RuntimeError(f"{variable} is required")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{variable} must be a JSON object")
        values: dict[str, tuple[str, str] | tuple[str, str, str]] = {}
        for token, principal in payload.items():
            if not isinstance(token, str) or not isinstance(principal, dict):
                raise RuntimeError("each token principal must be a JSON object")
            actor_id = principal["actor_id"]
            role = principal["role"]
            tenant_id = principal.get("tenant_id", _DEFAULT_TENANT_ID)
            if (
                not isinstance(actor_id, str)
                or not isinstance(role, str)
                or not isinstance(tenant_id, str)
            ):
                raise RuntimeError("token principal fields must be strings")
            values[token] = (actor_id, role, tenant_id)
        return cls.from_plaintext(values)

    def authenticate(self, token: str) -> Principal | None:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for expected, (actor_id, role, tenant_id) in self._principals.items():
            if hmac.compare_digest(digest, expected):
                return Principal(actor_id, role, digest[:16], tenant_id)
        return None

    @staticmethod
    def permits(principal: Principal, permission: str) -> bool:
        return _permission_permitted(principal, permission)


class OIDCJWTAuthorizer:
    """Fail-closed OIDC JWT verifier backed by an issuer's HTTPS JWKS endpoint.

    ``claims_decoder`` is intentionally injectable only for deterministic,
    network-free unit tests and controlled embedding.  A supplied decoder is
    trusted to have performed cryptographic JWT verification; this class still
    enforces issuer, audience, time, role, and tenant claims before issuing a
    principal.  Normal deployments leave it unset and use PyJWT + JWKS.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        role_claim: str = "role",
        tenant_claim: str = "tenant_id",
        permissions_claim: str = "permissions",
        claims_decoder: Callable[[str], Mapping[str, Any]] | None = None,
        clock_skew_seconds: float = 60.0,
        max_token_lifetime_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ):
        parsed_issuer = urlparse(issuer)
        parsed_jwks = urlparse(jwks_url)
        if (
            parsed_issuer.scheme != "https"
            or not parsed_issuer.netloc
            or parsed_issuer.username
            or parsed_issuer.password
        ):
            raise ValueError("OIDC issuer must be an HTTPS URL without credentials")
        if (
            parsed_jwks.scheme != "https"
            or not parsed_jwks.netloc
            or parsed_jwks.username
            or parsed_jwks.password
        ):
            raise ValueError("OIDC JWKS URL must be an HTTPS URL without credentials")
        if not isinstance(audience, str) or not audience.strip() or len(audience) > 512:
            raise ValueError("OIDC audience is required")
        if not isinstance(role_claim, str) or not role_claim.strip():
            raise ValueError("OIDC role claim is required")
        if not isinstance(tenant_claim, str) or not tenant_claim.strip():
            raise ValueError("OIDC tenant claim is required")
        if not isinstance(permissions_claim, str) or not permissions_claim.strip():
            raise ValueError("OIDC permissions claim is required")
        if not 0 <= clock_skew_seconds <= 300:
            raise ValueError("OIDC clock skew must be between 0 and 300 seconds")
        if not 60 <= max_token_lifetime_seconds <= 86_400:
            raise ValueError(
                "OIDC maximum token lifetime must be between 60 and 86400 seconds"
            )
        self.issuer = issuer
        self.audience = audience
        self.jwks_url = jwks_url
        self.role_claim = role_claim
        self.tenant_claim = tenant_claim
        self.permissions_claim = permissions_claim
        self.clock_skew_seconds = float(clock_skew_seconds)
        self.max_token_lifetime_seconds = float(max_token_lifetime_seconds)
        self._clock = clock
        self._claims_decoder = claims_decoder or self._pyjwt_claims_decoder()

    @classmethod
    def from_env(cls) -> "OIDCJWTAuthorizer":
        required = {
            "issuer": "PERFUMERY_AI_OIDC_ISSUER",
            "audience": "PERFUMERY_AI_OIDC_AUDIENCE",
            "jwks_url": "PERFUMERY_AI_OIDC_JWKS_URL",
        }
        missing = [
            variable for variable in required.values() if not os.environ.get(variable)
        ]
        if missing:
            raise RuntimeError(
                "OIDC configuration is incomplete: " + ", ".join(missing)
            )
        try:
            clock_skew = float(
                os.environ.get("PERFUMERY_AI_OIDC_CLOCK_SKEW_SECONDS", "60")
            )
            max_lifetime = float(
                os.environ.get("PERFUMERY_AI_OIDC_MAX_TOKEN_LIFETIME_SECONDS", "3600")
            )
        except ValueError as error:
            raise RuntimeError(
                "OIDC clock skew and maximum lifetime must be numeric"
            ) from error
        return cls(
            issuer=os.environ[required["issuer"]],
            audience=os.environ[required["audience"]],
            jwks_url=os.environ[required["jwks_url"]],
            role_claim=os.environ.get("PERFUMERY_AI_OIDC_ROLE_CLAIM", "role"),
            tenant_claim=os.environ.get("PERFUMERY_AI_OIDC_TENANT_CLAIM", "tenant_id"),
            permissions_claim=os.environ.get(
                "PERFUMERY_AI_OIDC_PERMISSIONS_CLAIM", "permissions"
            ),
            clock_skew_seconds=clock_skew,
            max_token_lifetime_seconds=max_lifetime,
        )

    def _pyjwt_claims_decoder(self) -> Callable[[str], Mapping[str, Any]]:
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "OIDC authentication requires PyJWT; install perfumery-ai-core[commercial]"
            ) from error

        jwks_client = PyJWKClient(
            self.jwks_url,
            cache_keys=True,
            lifespan=300,
            timeout=5,
        )

        def decode(token: str) -> Mapping[str, Any]:
            header = jwt.get_unverified_header(token)
            algorithm = header.get("alg")
            if algorithm not in _OIDC_ALLOWED_ALGORITHMS:
                raise ValueError("JWT algorithm is not allowed")
            token_type = header.get("typ")
            if token_type not in {None, "JWT", "at+jwt"}:
                raise ValueError("JWT type is not an access-token type")
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=[algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
                leeway=self.clock_skew_seconds,
            )

        return decode

    def _principal_from_claims(
        self, claims: Mapping[str, Any], token: str
    ) -> Principal:
        if not isinstance(claims, Mapping):
            raise ValueError("JWT claims must be an object")
        if not hmac.compare_digest(str(claims.get("iss", "")), self.issuer):
            raise ValueError("JWT issuer mismatch")
        audiences = claims.get("aud")
        if isinstance(audiences, str):
            audience_matches = hmac.compare_digest(audiences, self.audience)
        elif isinstance(audiences, (list, tuple)) and all(
            isinstance(item, str) for item in audiences
        ):
            audience_matches = any(
                hmac.compare_digest(item, self.audience) for item in audiences
            )
        else:
            audience_matches = False
        if not audience_matches:
            raise ValueError("JWT audience mismatch")
        now = self._clock()
        for claim_name in ("exp", "iat"):
            value = claims.get(claim_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"JWT {claim_name} claim is required")
        if float(claims["exp"]) <= now - self.clock_skew_seconds:
            raise ValueError("JWT is expired")
        if float(claims["iat"]) > now + self.clock_skew_seconds:
            raise ValueError("JWT was issued in the future")
        if float(claims["exp"]) <= float(claims["iat"]):
            raise ValueError("JWT lifetime is invalid")
        if (
            float(claims["exp"]) - float(claims["iat"])
            > self.max_token_lifetime_seconds
        ):
            raise ValueError("JWT lifetime exceeds service policy")
        not_before = claims.get("nbf")
        if not_before is not None:
            if isinstance(not_before, bool) or not isinstance(not_before, (int, float)):
                raise ValueError("JWT nbf claim is invalid")
            if float(not_before) > now + self.clock_skew_seconds:
                raise ValueError("JWT is not active")
        actor_id = _identity_value(claims.get("sub"), "OIDC subject")
        role = claims.get(self.role_claim)
        if not isinstance(role, str) or role not in ROLE_PERMISSIONS:
            raise ValueError("JWT role is not authorized")
        tenant_id = _identity_value(claims.get(self.tenant_claim), "OIDC tenant")
        permission_value = claims.get(self.permissions_claim)
        permissions: frozenset[str] | None = None
        if permission_value is not None:
            if isinstance(permission_value, str):
                requested_permissions = permission_value.split()
            elif isinstance(permission_value, (list, tuple)) and all(
                isinstance(item, str) for item in permission_value
            ):
                requested_permissions = list(permission_value)
            else:
                raise ValueError("JWT permissions claim is invalid")
            allowed = ROLE_PERMISSIONS[role]
            permissions = frozenset(
                permission
                for permission in requested_permissions
                if permission in allowed
            )
        fingerprint = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        return Principal(actor_id, role, fingerprint, tenant_id, permissions)

    def authenticate(self, token: str) -> Principal | None:
        if not isinstance(token, str) or not token or len(token) > 16_384:
            return None
        try:
            return self._principal_from_claims(self._claims_decoder(token), token)
        except Exception:
            # Authentication failures deliberately do not expose decoder/JWKS
            # detail to a caller and never fall back to static-token auth.
            return None

    @staticmethod
    def permits(principal: Principal, permission: str) -> bool:
        return _permission_permitted(principal, permission)


def authorizer_from_env() -> Authorizer:
    """Choose exactly one authentication mode; partial OIDC config is an error."""

    mode = os.environ.get("PERFUMERY_AI_AUTH_MODE", "").strip().lower()
    if mode not in {"", "static", "oidc"}:
        raise RuntimeError("PERFUMERY_AI_AUTH_MODE must be 'static' or 'oidc'")
    oidc_variables = (
        "PERFUMERY_AI_OIDC_ISSUER",
        "PERFUMERY_AI_OIDC_AUDIENCE",
        "PERFUMERY_AI_OIDC_JWKS_URL",
    )
    oidc_configured = any(os.environ.get(variable) for variable in oidc_variables)
    production = (
        os.environ.get("PERFUMERY_AI_ENV", "development").strip().lower()
        == "production"
    )
    if mode == "static" and oidc_configured:
        raise RuntimeError(
            "static auth mode cannot be combined with OIDC configuration"
        )
    if mode == "oidc" or oidc_configured:
        return OIDCJWTAuthorizer.from_env()
    if production:
        raise RuntimeError("production requires OIDC authentication")
    return TokenAuthorizer.from_env()


class SlidingWindowRateLimiter:
    def __init__(self, requests: int = 30, window_seconds: float = 60.0):
        if requests <= 0 or window_seconds <= 0:
            raise ValueError("rate-limit values must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[identity]
            while events and current - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.requests:
                return False
            events.append(current)
            return True


class _InferenceLane:
    """One AI instance pinned to one owner thread for SQLite thread safety."""

    def __init__(self, ai_factory: Callable[[], Any], index: int):
        self._ai_factory = ai_factory
        self._ai = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"perfumery-inference-{index}",
        )

    def _engine(self):
        if self._ai is None:
            self._ai = self._ai_factory()
        return self._ai

    def warm(self) -> None:
        self._executor.submit(self._engine).result()

    def submit(self, brief: str, constraints: Any) -> Future:
        return self._executor.submit(
            lambda: self._engine().create_recipe(brief, constraints)
        )

    def _close_on_owner_thread(self) -> None:
        if self._ai is None:
            return
        close = getattr(self._ai, "close", None)
        if callable(close):
            close()
        self._ai = None

    def close(self) -> None:
        try:
            self._executor.submit(self._close_on_owner_thread).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


def create_app(
    *,
    ai_factory: Callable[[], Any],
    authorizer: Authorizer,
    audit_log: Any,
    max_concurrent_inference: int = 2,
    rate_limiter: Any | None = None,
    workspace_store: Any | None = None,
    metrics: Any | None = None,
    enable_ui: bool = True,
    max_request_bytes: int = 65_536,
):
    """Create the authenticated synchronous and durable-workspace API.

    Each ``ai_factory`` result is pinned to a dedicated single-thread lane.
    SQLite-backed repositories therefore remain on their creating thread while
    a bounded number of warmed engines are reused across requests.
    """

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel, ConfigDict, Field
        from starlette.concurrency import run_in_threadpool
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "commercial API dependencies are missing; install perfumery-ai-core[commercial]"
        ) from error

    if max_concurrent_inference <= 0:
        raise ValueError("max_concurrent_inference must be positive")
    if not 4_096 <= int(max_request_bytes) <= 10_000_000:
        raise ValueError("max_request_bytes must be between 4096 and 10000000")
    from .platform.observability import ServiceMetrics

    limiter = rate_limiter or SlidingWindowRateLimiter()
    service_metrics = metrics or ServiceMetrics()

    async def append_audit_async(**fields: Any) -> Any:
        """Keep synchronous SQLite/PostgreSQL audit I/O off the event loop."""

        return await run_in_threadpool(audit_log.append, **fields)

    # A process-wide semaphore is deliberately loop-independent.  This keeps
    # the capacity guard correct when ASGI test/deployment workers enter the
    # app through different event loops.
    semaphore = threading.BoundedSemaphore(max_concurrent_inference)
    inference_lanes = [
        _InferenceLane(ai_factory, index) for index in range(max_concurrent_inference)
    ]
    available_lanes: queue.LifoQueue[_InferenceLane] = queue.LifoQueue()
    for lane in inference_lanes:
        available_lanes.put(lane)
    runtime_lock = threading.Lock()
    runtime_closed = False

    def warm_inference_runtime() -> None:
        with runtime_lock:
            if runtime_closed:
                raise RuntimeError("inference runtime is closed")
        for lane in inference_lanes:
            lane.warm()

    def close_inference_runtime() -> None:
        nonlocal runtime_closed
        with runtime_lock:
            if runtime_closed:
                return
            runtime_closed = True
        for lane in inference_lanes:
            lane.close()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await run_in_threadpool(warm_inference_runtime)
            yield
        finally:
            await run_in_threadpool(close_inference_runtime)

    class RecipeRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        brief: str = Field(min_length=1, max_length=4000)
        constraints: dict[str, Any] = Field(default_factory=dict)

    class RequestBodyTooLarge(Exception):
        pass

    class RequestBodyLimitMiddleware:
        """Enforce the limit while receiving, including chunked HTTP bodies."""

        def __init__(self, app, maximum_bytes: int):
            self.app = app
            self.maximum_bytes = maximum_bytes

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            supplied_length = headers.get(b"content-length")
            if supplied_length is not None:
                try:
                    content_length = int(supplied_length.decode("ascii"))
                except (UnicodeDecodeError, ValueError):
                    content_length = self.maximum_bytes + 1
                if content_length < 0 or content_length > self.maximum_bytes:
                    await JSONResponse(
                        status_code=413,
                        content={"detail": "request body exceeds service limit"},
                    )(scope, receive, send)
                    return

            received = 0
            response_started = False
            limit_exceeded = False

            async def limited_receive():
                nonlocal limit_exceeded, received
                message = await receive()
                if message.get("type") == "http.request":
                    received += len(message.get("body", b""))
                    if received > self.maximum_bytes:
                        limit_exceeded = True
                        raise RequestBodyTooLarge
                return message

            async def tracked_send(message):
                nonlocal response_started
                # FastAPI converts receive failures into a generic 400.  Once
                # this middleware has observed the actual size violation, hold
                # that inner response and emit the precise 413 below.
                if limit_exceeded:
                    return
                if message.get("type") == "http.response.start":
                    response_started = True
                await send(message)

            try:
                await self.app(scope, limited_receive, tracked_send)
            except RequestBodyTooLarge:
                limit_exceeded = True
            if limit_exceeded:
                if response_started:
                    raise RuntimeError("request limit exceeded after response start")
                await JSONResponse(
                    status_code=413,
                    content={"detail": "request body exceeds service limit"},
                )(scope, receive, send)

    app = FastAPI(
        title="Perfumery AI Core",
        version="1.4.0",
        lifespan=lifespan,
        description=(
            "Authenticated R&D recipe candidate API. Proxy scores are not "
            "measured human olfactory accuracy or regulatory approval."
        ),
    )
    app.add_middleware(RequestBodyLimitMiddleware, maximum_bytes=max_request_bytes)
    app.state.workspace_store = workspace_store
    app.state.metrics = service_metrics
    app.state.warm_inference_runtime = warm_inference_runtime
    app.state.close_inference_runtime = close_inference_runtime
    app.state.inference_engine_capacity = max_concurrent_inference

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else "req_" + uuid.uuid4().hex
        )
        service_metrics.request_started()
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            route = getattr(request.scope.get("route"), "path", "unknown")
            duration = time.perf_counter() - started
            service_metrics.request_finished(
                method=request.method,
                route=route,
                status_code=500,
                duration_seconds=duration,
            )
            LOGGER.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status_code": 500,
                    "duration_ms": round(duration * 1000.0, 3),
                },
            )
            raise
        route = getattr(request.scope.get("route"), "path", "unknown")
        duration = time.perf_counter() - started
        service_metrics.request_finished(
            method=request.method,
            route=route,
            status_code=status_code,
            duration_seconds=duration,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        if request.url.path.startswith(("/v1/", "/health")):
            response.headers["Cache-Control"] = "no-store"
        LOGGER.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status_code": status_code,
                "duration_ms": round(duration * 1000.0, 3),
            },
        )
        return response

    def principal(
        authorization: str | None = Header(default=None),
        requested_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    ) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="bearer token required")
        authenticated = authorizer.authenticate(authorization[7:])
        if authenticated is None:
            raise HTTPException(status_code=401, detail="invalid bearer token")
        if requested_tenant_id is not None:
            try:
                normalized_requested_tenant = _identity_value(
                    requested_tenant_id, "X-Tenant-ID"
                )
            except ValueError as error:
                raise HTTPException(
                    status_code=403, detail="invalid tenant context"
                ) from error
            if not hmac.compare_digest(
                normalized_requested_tenant, authenticated.tenant_id
            ):
                raise HTTPException(status_code=403, detail="tenant context mismatch")
        if not limiter.allow(authenticated.rate_limit_identity):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return authenticated

    def require(*permissions: str):
        if not permissions or any(not permission for permission in permissions):
            raise ValueError("at least one non-empty permission is required")

        def dependency(current: Principal = Depends(principal)) -> Principal:
            if not all(
                authorizer.permits(current, permission) for permission in permissions
            ):
                raise HTTPException(status_code=403, detail="insufficient role")
            return current

        return dependency

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "release_boundary": "r_and_d_candidate_only"}

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive", "version": "1.4.0"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        storage_ready = workspace_store is None or bool(workspace_store.ping())
        if not storage_ready:
            raise HTTPException(status_code=503, detail="workspace storage unavailable")
        return {
            "status": "ready",
            "workspace_backend": (
                workspace_store.backend if workspace_store is not None else "disabled"
            ),
            "horizontal_scaling": bool(
                workspace_store is not None and workspace_store.horizontally_scalable
            ),
        }

    @app.post("/v1/recipes")
    async def create_recipe(
        request: RecipeRequest,
        current: Principal = Depends(require("recipe:create")),
    ) -> dict[str, Any]:
        from .platform.workspace import constraints_from_payload

        request_id = f"tenant:{current.tenant_id}:request:{uuid.uuid4().hex}"
        brief_sha = hashlib.sha256(request.brief.encode("utf-8")).hexdigest()
        await append_audit_async(
            actor_id=current.actor_id,
            actor_role=current.role,
            event_type="recipe.requested",
            scope_id=request_id,
            payload={"tenant_id": current.tenant_id, "brief_sha256": brief_sha},
        )
        acquired = await run_in_threadpool(semaphore.acquire, True, 0.25)
        if not acquired:
            await append_audit_async(
                actor_id=current.actor_id,
                actor_role=current.role,
                event_type="recipe.rejected_busy",
                scope_id=request_id,
                payload={"tenant_id": current.tenant_id},
            )
            raise HTTPException(status_code=503, detail="inference capacity exhausted")
        lane = None
        try:
            constraints = constraints_from_payload(request.constraints)
            try:
                lane = available_lanes.get_nowait()
            except queue.Empty as error:  # pragma: no cover - semaphore invariant
                raise HTTPException(
                    status_code=503,
                    detail="inference engine unavailable",
                ) from error

            inference_started = time.perf_counter()
            result = await asyncio.wrap_future(lane.submit(request.brief, constraints))
            service_metrics.observe_inference(
                outcome="succeeded",
                duration_seconds=time.perf_counter() - inference_started,
            )
            payload = result.to_dict()
            await append_audit_async(
                actor_id=current.actor_id,
                actor_role=current.role,
                event_type="recipe.completed",
                scope_id=request_id,
                payload={
                    "tenant_id": current.tenant_id,
                    "status": payload.get("status"),
                    "formula_id": payload.get("formula_id", ""),
                    "olfactory_validation_status": payload.get(
                        "olfactory_validation_status"
                    ),
                },
            )
            return {
                "request_id": request_id,
                "tenant_id": current.tenant_id,
                "result": payload,
            }
        except HTTPException:
            raise
        except Exception as error:
            if "inference_started" in locals():
                service_metrics.observe_inference(
                    outcome="failed",
                    duration_seconds=time.perf_counter() - inference_started,
                )
            await append_audit_async(
                actor_id=current.actor_id,
                actor_role=current.role,
                event_type="recipe.failed",
                scope_id=request_id,
                payload={
                    "tenant_id": current.tenant_id,
                    "error_type": type(error).__name__,
                },
            )
            raise HTTPException(
                status_code=422,
                detail="request could not be processed under the configured constraints",
            ) from error
        finally:
            if lane is not None:
                available_lanes.put(lane)
            semaphore.release()

    @app.get("/v1/audit/verify")
    def verify_audit(
        current: Principal = Depends(require("audit:verify")),
    ) -> dict[str, Any]:
        result = audit_log.verify()
        audit_log.append(
            actor_id=current.actor_id,
            actor_role=current.role,
            event_type="audit.verified",
            scope_id=f"tenant:{current.tenant_id}:audit-log",
            payload={
                "tenant_id": current.tenant_id,
                "passed": result["passed"],
                "head_hash": result["head_hash"],
            },
        )
        return result

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics(
        current: Principal = Depends(require("metrics:read")),
    ) -> PlainTextResponse:
        return PlainTextResponse(
            service_metrics.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    if workspace_store is not None:
        from .platform.api_routes import install_workspace_routes

        install_workspace_routes(
            app=app,
            store=workspace_store,
            ai_factory=ai_factory,
            require=require,
            audit_log=audit_log,
            metrics=service_metrics,
        )

    ui_directory = Path(__file__).resolve().parent / "ui"
    if enable_ui and ui_directory.is_dir():
        # Windows/Python MIME registries can report JavaScript as the legacy
        # ``text/javascript`` type. Keep the API contract stable across hosts.
        mimetypes.add_type("application/javascript", ".js", strict=True)
        app.mount("/ui", StaticFiles(directory=ui_directory, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def ui_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/", status_code=307)

    return app
