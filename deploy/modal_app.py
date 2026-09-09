"""Pinned legacy V32 deployment; newer releases use deploy/web_app.py."""

import modal

from deploy.web_app import (
    ROOT as ROOT,
    WHEEL as WHEEL,
    WHEEL_SHA256 as WHEEL_SHA256,
    REGISTRY as REGISTRY,
    REGISTRY_SHA256 as REGISTRY_SHA256,
    REMOTE_WHEEL as REMOTE_WHEEL,
    REMOTE_REGISTRY as REMOTE_REGISTRY,
    RUNTIME_CATALOG as RUNTIME_CATALOG,
    RUNTIME_CATALOG_SHA256 as RUNTIME_CATALOG_SHA256,
    REMOTE_CATALOG as REMOTE_CATALOG,
    INDEX_HTML as INDEX_HTML,
    _sha256_file as _sha256_file,
    create_web_app as create_web_app,
)


if modal.is_local():
    if not WHEEL.is_file() or not REGISTRY.is_file():
        raise RuntimeError(
            "Modal deployment requires the sealed wheel and registry artifacts"
        )
    if (
        _sha256_file(WHEEL) != WHEEL_SHA256
        or _sha256_file(REGISTRY) != REGISTRY_SHA256
    ):
        raise RuntimeError("Modal deployment artifact hash mismatch")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "fastapi[standard]==0.116.1",
        "cryptography==46.0.3",
    )
    .add_local_file(WHEEL, REMOTE_WHEEL, copy=True)
    .run_commands(f"python -m pip install --no-cache-dir {REMOTE_WHEEL}")
    .add_local_file(REGISTRY, REMOTE_REGISTRY, copy=True)
)
if RUNTIME_CATALOG_SHA256:
    if modal.is_local() and _sha256_file(RUNTIME_CATALOG) != RUNTIME_CATALOG_SHA256:
        raise RuntimeError("runtime catalog artifact hash mismatch")
    image = image.add_local_file(RUNTIME_CATALOG, REMOTE_CATALOG, copy=True)

app = modal.App("perfumery-ai-core")


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=120,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    return create_web_app()
