FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS builder

ENV PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY pyproject.toml README.md LICENSE LICENSE_POLICY.md requirements-ci.lock requirements-runtime.lock /build/
COPY fragrance_ai /build/fragrance_ai
RUN python -m pip install --require-hashes -r requirements-ci.lock && \
    python -m pip wheel --require-hashes --wheel-dir /wheels \
        -r requirements-runtime.lock && \
    python -m build --no-isolation --wheel --outdir /wheels

FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system perfumery && useradd --system --gid perfumery --home /app perfumery
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels \
        "perfumery-ai-core[commercial]==1.4.0" && \
    rm -rf /wheels && \
    mkdir -p /var/lib/perfumery-ai/continuous && \
    chown -R perfumery:perfumery /app /var/lib/perfumery-ai

USER perfumery
EXPOSE 8000
CMD ["perfumery-ai-api", "--host", "0.0.0.0", "--port", "8000"]
