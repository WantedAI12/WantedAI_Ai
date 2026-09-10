"""One immutable material and quality-policy snapshot for API and queue workers."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .catalog import IngredientCatalog
from .odor_integrity import ODOR_INTEGRITY_VERSION
from .promotion_activation import PromotionActivationBundle
from .registry_activation import load_runtime_catalog
from .service import NaturalLanguagePerfumeryAI
from .concentration_response import concentration_response_from_environment
from .perception_runtime import configured_perception, model_contract, environment_snapshot, assert_provider_current


MANIFEST_ENV = "PERFUMERY_AI_RUNTIME_CATALOG_MANIFEST"
MANIFEST_HASH_ENV = "PERFUMERY_AI_RUNTIME_CATALOG_MANIFEST_SHA256"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_snapshot():
    package_root = Path(__file__).resolve().parent.parent
    return {"fragrance_ai/" + item.relative_to(package_root).as_posix(): _digest(item)
            for item in package_root.rglob("*.py")}


def _data_snapshot(data_root=None):
    root = Path(data_root) if data_root is not None else Path(__file__).resolve().parent.parent / "data"
    files = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if item.name.endswith((".db-wal", ".db-journal")) and item.stat().st_size:
            raise ValueError("runtime data must be checkpointed immutable snapshots, not active SQLite journals")
        if item.suffix in {".db", ".json", ".npz"}:
            files["fragrance_ai/data/" + item.relative_to(root).as_posix()] = _digest(item)
    return files


def _continual_snapshot():
    values = {name: os.environ.get(name, "").strip() for name in (
        "PERFUMERY_AI_CONTINUAL_STATE", "PERFUMERY_AI_CONTINUAL_TRUST_ROOT")}
    return {name: _digest(path) if path else None for name, path in values.items()}


def _runtime_metadata_snapshot():
    """Cheap change detector after full startup hashes, not a cryptographic proof."""
    root = Path(__file__).resolve().parent.parent
    paths = [path for path in root.rglob("*") if path.is_file() and (
        path.suffix in {".py", ".db", ".json", ".npz"} or path.name.endswith((".db-wal", ".db-journal")))]
    paths.extend(Path(value) for name in ("PERFUMERY_AI_CONTINUAL_STATE", "PERFUMERY_AI_CONTINUAL_TRUST_ROOT")
                 if (value := os.environ.get(name, "").strip()))
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in sorted(paths))


def _material_digest(catalog):
    digest = hashlib.sha256()
    for item in catalog.ingredients:
        digest.update(json.dumps(asdict(item), sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_configured_catalog(manifest_path=None, expected_sha256=None):
    """No silent fallback to 29 materials after a configured snapshot fails."""
    from .local_runtime import configured_pair
    if manifest_path is not None or expected_sha256 is not None:
        path, expected = manifest_path, expected_sha256
    else:
        path, expected = configured_pair(MANIFEST_ENV, MANIFEST_HASH_ENV, 'catalog')
    if not path and not expected:
        return IngredientCatalog.load_builtin(), None
    bundle = load_verified_catalog_bundle(path, expected)
    return bundle['catalog'], bundle['manifest_sha256']


def load_verified_catalog_bundle(path, expected):
    """Load one source-bound snapshot with the metadata needed by API/SDK.

    Unlike the configuration wrapper, this function never falls back to the
    builtin catalog. Preserve the validated activation report, not just rows.
    """
    if not path or not expected:
        raise ValueError("runtime catalog requires both manifest path and trusted manifest SHA-256")
    if not isinstance(expected, str) or len(expected.strip()) != 64:
        raise ValueError("runtime catalog manifest SHA-256 is invalid")
    expected = expected.strip().lower()
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 1_000_000:
        raise ValueError("runtime catalog manifest hash or size mismatch")
    raw = path.read_bytes()
    if len(raw) > 1_000_000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("runtime catalog manifest hash or size mismatch")
    manifest = json.loads(raw)
    if manifest.get("odor_integrity_version") != ODOR_INTEGRITY_VERSION:
        raise ValueError("runtime manifest does not use the corrected odor evidence contract")
    binding = manifest["runtime_catalog"]
    target = (path.parent / binding["path"]).resolve(strict=True)
    if not target.is_relative_to(path.parent):
        raise ValueError("runtime catalog escapes the trusted manifest directory")
    actual = _source_snapshot()
    if actual != manifest.get("runtime_source_sha256"):
        raise ValueError("runtime catalog was built for a different package source")
    assets = manifest.get("runtime_data_sha256")
    current_data = _data_snapshot()
    if not isinstance(assets, dict) or not assets or any(current_data.get(name) != digest for name, digest in assets.items()):
        raise ValueError("runtime catalog was built for different model or physical-property assets")
    catalog, report, stats = load_runtime_catalog(target, expected_sha256=binding["sha256"],
        expected_wheel_sha256=binding["wheel_sha256"], expected_registry_sha256=binding["registry_sha256"])
    return {'catalog': catalog, 'manifest_sha256': expected, 'binding': dict(binding),
            'activation_report': report, 'registry_stats': stats,
            'manifest_path': path, 'catalog_path': target}


class RuntimeAIFactory:
    """Share immutable inputs, not SQLite connections or mutable inference engines."""

    def __init__(self, *, catalog=None, minimum_profile_target=95., require_full_profile_match=True,
                 allow_experimental_safety=False, manifest_sha256=None, promotion_bundle=None, perception_guidance=None):
        if catalog is None:
            catalog, configured_digest = load_configured_catalog()
            manifest_sha256 = manifest_sha256 or configured_digest
        from .local_runtime import local_snapshot
        self._local_snapshot = local_snapshot()
        self.base_catalog = catalog
        self.promotion_bundle = promotion_bundle if promotion_bundle is not None else PromotionActivationBundle.from_environment()
        self.catalog = self.promotion_bundle.merge_catalog(self.base_catalog)
        self.minimum_profile_target = minimum_profile_target
        self.require_full_profile_match = require_full_profile_match or minimum_profile_target is not None
        self.allow_experimental_safety = allow_experimental_safety
        self.perception_guidance = perception_guidance if perception_guidance is not None else configured_perception()
        self._perception_environment = environment_snapshot()
        data_snapshot = _data_snapshot()
        continual_snapshot = _continual_snapshot()
        # Pin the signed champion object once. A newly promoted champion must
        # use a new worker/API snapshot rather than changing queued inference.
        self.concentration_response = concentration_response_from_environment()
        if continual_snapshot != _continual_snapshot():
            raise ValueError("continual model state changed while loading the runtime snapshot")
        self.runtime_contract = {"schema": "perfumery-runtime-policy/v1",
            "material_snapshot_sha256": _material_digest(self.catalog),
            "runtime_source_sha256": hashlib.sha256(json.dumps(_source_snapshot(), sort_keys=True).encode()).hexdigest(),
            "runtime_data_sha256": hashlib.sha256(json.dumps(data_snapshot, sort_keys=True).encode()).hexdigest(),
            "continual_snapshot": continual_snapshot,
            "promotion_snapshot_sha256": hashlib.sha256(json.dumps(asdict(self.promotion_bundle), sort_keys=True, default=str).encode()).hexdigest(),
            "catalog_manifest_sha256": manifest_sha256,
            "local_profile_sha256": self._local_snapshot[0],
            "minimum_profile_target": minimum_profile_target, "strict_full_profile_gate": self.require_full_profile_match,
            "allow_experimental_safety": allow_experimental_safety,
            "score_kind": "full_model_profile_agreement_not_human_similarity"}
        self.runtime_contract['perception_model'] = model_contract(self.perception_guidance)
        self.runtime_contract['product_model'] = {
            'product': 'perfume', 'prediction_model': 'perfume_temporal_mixture',
            'evaluation_version': 'perfume_full_profile/v1',
            'cross_product_scores_comparable': False,
            'body_lotion_matrix_evaluated': False,
        }
        self._snapshot_metadata = _runtime_metadata_snapshot()
        # Validate policy at startup; the temporary instance closes its owned
        # resources here. Runtime engines remain owned by their request lane.
        with self() as probe:
            if probe.catalog.ingredients != self.catalog.ingredients:
                raise ValueError("factory and inference material snapshots differ")
        if data_snapshot != _data_snapshot():
            raise ValueError("runtime data changed while constructing the inference snapshot")

    def __call__(self):
        self.assert_current_snapshot()
        ai = NaturalLanguagePerfumeryAI(catalog=self.base_catalog, promotion_bundle=self.promotion_bundle,
            minimum_profile_target=self.minimum_profile_target, require_full_profile_match=self.require_full_profile_match,
            allow_experimental_safety=self.allow_experimental_safety, concentration_response=self.concentration_response,
            perception_guidance=self.perception_guidance)
        ai.runtime_contract = dict(self.runtime_contract)
        ai._runtime_snapshot_guard = self.assert_current_snapshot
        return ai

    def assert_current_snapshot(self):
        from .local_runtime import local_snapshot
        if self._local_snapshot != local_snapshot():
            raise ValueError('local runtime selection changed; reload API and workers')
        if self._perception_environment != environment_snapshot():
            raise ValueError('perception model policy changed; reload API and workers')
        assert_provider_current(self.perception_guidance)
        if self._snapshot_metadata != _runtime_metadata_snapshot():
            raise ValueError("runtime input snapshot changed; reload API and workers under the same verified policy")

    @classmethod
    def from_environment(cls, *, minimum_profile_target=95., require_full_profile_match=True,
                         manifest_path=None, expected_manifest_sha256=None):
        catalog, digest = load_configured_catalog(manifest_path, expected_manifest_sha256)
        production = os.environ.get("PERFUMERY_AI_ENV", "development").strip().lower() == "production"
        if production and (minimum_profile_target is None or minimum_profile_target < 95):
            raise ValueError("production runtime requires at least the 95-point full-profile gate")
        if production and digest is None:
            raise ValueError("production requires an explicitly pinned runtime catalog; builtin fallback is disabled")
        return cls(catalog=catalog, manifest_sha256=digest, minimum_profile_target=minimum_profile_target,
                   require_full_profile_match=require_full_profile_match, allow_experimental_safety=not production)
