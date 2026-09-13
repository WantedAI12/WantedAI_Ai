"""Lazy compatibility lane for the separate historical stock-aliquot assay.

Never called by the shared perfume/lotion model. Changing its molecular parent
without retraining its assay would invalidate that experiment's lineage.
"""

import hashlib
import json
from pathlib import Path
import threading


class FrozenStockAssay:
    def __init__(self, profile):
        self.profile = profile
        self.sha256 = profile["stock_mixture"][1]
        p = Path(profile["perfume"][0])
        raw = p.read_bytes()
        if len(raw) > 32768 or hashlib.sha256(raw).hexdigest() != profile["perfume"][1]:
            raise ValueError("legacy assay component manifest mismatch")
        self.document = json.loads(raw)
        self.component_sha256 = self.document["component_model"]["sha256"]
        self._model = None
        self._lock = threading.RLock()
        self._paths = [
            Path(profile[k][0]) for k in ("perfume", "stock_mixture", "atlas")
        ]
        self._metadata = self._stat()

    def _stat(self):
        return [(p.stat().st_size, p.stat().st_mtime_ns) for p in self._paths]

    def assert_current(self):
        from .local_runtime import local_profile

        if (
            self._metadata != self._stat()
            or local_profile()["profile_sha256"] != self.profile["profile_sha256"]
        ):
            raise ValueError("frozen assay selection changed")
        if self._model is not None:
            self._model.assert_current()

    def _load(self):
        from .perception_guidance import PerceptionGuidance
        from .stock_mixture import StockMixturePredictor
        from ..research.atlas_profiles import AtlasProfilePredictor

        with self._lock:
            self.assert_current()
            if self._model is None:
                root = Path(self.profile["perfume"][0]).parent
                doc = self.document
                provider = PerceptionGuidance(
                    root / doc["base_model"]["path"],
                    root / doc["registry"]["path"],
                    solvent=doc["solvent_scenario"],
                    weight=doc["guidance_weight"],
                    experimental=True,
                    component_model_path=root / doc["component_model"]["path"],
                    component_model_sha256=self.component_sha256,
                )
                atlas = AtlasProfilePredictor(
                    self.profile["atlas"][0],
                    sha256=self.profile["atlas"][1],
                    experimental=True,
                )
                self._model = StockMixturePredictor(
                    provider,
                    self.profile["stock_mixture"][0],
                    sha256=self.sha256,
                    experimental=True,
                    atlas_predictor=atlas,
                )
            return self._model

    def predict(self, *args, **kwargs):
        return self._load().predict(*args, **kwargs)

    def predict_masses(self, *args, **kwargs):
        return self._load().predict_masses(*args, **kwargs)

    def contract(self):
        self.assert_current()
        return {
            "model_sha256": self.sha256,
            "component_model_sha256": self.component_sha256,
            "application_domain": "stock_aliquot_assay",
            "legacy_compatibility_only": True,
            "loaded": self._model is not None,
            "used_for_perfume_or_lotion_recipes": False,
            "scope": "separate_historical_assay_not_the_shared_formulation_model",
        }
