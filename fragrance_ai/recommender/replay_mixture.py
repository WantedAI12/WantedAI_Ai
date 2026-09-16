"""Checksum-bound CPU mixture predictor plus its learned training policy.

This optional local module does not override the main recipe/physics scorer.
Inputs are relative component composition, not measured gas concentration.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from ..research.mixture_mlp_numpy import NumpyMixtureMLP
from ..research.replay_search import Policy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8192)
def _canonical_graph(graph):
    """Cache immutable identity validation, not concentrations or model scores."""
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(graph) if graph and "." not in graph else None
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("one explicit molecular graph per component required")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


class ReplayMixtureModel:
    def __init__(self, path, expected_sha256):
        path = Path(path).resolve()
        if digest(path) != expected_sha256:
            raise ValueError("mixture bundle manifest checksum mismatch")
        self.manifest = json.loads(path.read_text(encoding="utf8"))
        if self.manifest.get("schema") != "replay-mixture-bundle/v85":
            raise ValueError("unsupported mixture bundle")
        self.sha256 = expected_sha256
        files = {}
        for role in ("weights", "features", "encoder", "policy"):
            record = self.manifest[role]
            source = (path.parent / record["file"]).resolve()
            if source.parent != path.parent or digest(source) != record["sha256"]:
                raise ValueError("mixture bundle file escaped its directory or checksum failed")
            files[role] = source
        with np.load(files["weights"], allow_pickle=False) as archive:
            self.model = NumpyMixtureMLP({key: archive[key] for key in archive.files}, self.manifest["mode"])
        with np.load(files["encoder"], allow_pickle=False) as archive:
            self.encoder = {key: archive[key] for key in archive.files}
        self.features = json.loads(files["features"].read_text(encoding="utf8"))
        record = json.loads(files["policy"].read_text(encoding="utf8"))
        self.policy = Policy(**record["policy"])
        if self.policy.digest != record["sha256"]:
            raise ValueError("policy checksum mismatch")
        if any(not np.isfinite(v).all() for v in self.encoder.values()) or np.any(self.encoder["feature_scale"] <= 0):
            raise ValueError("invalid frozen molecular encoder")
        self._cache, self._lock = OrderedDict(), threading.RLock()

    def _embed(self, graphs):
        from ..research.atlas_profiles import atlas_features
        with self._lock:
            pending = sorted(set(graphs) - self._cache.keys())
            if pending:
                raw = atlas_features(pending, ["high"] * len(pending), self.features["native_profiles"], self.features["fine_features"]).astype(np.float32)
                arrays = self.encoder
                normalized = np.clip((raw - arrays["feature_mean"]) / arrays["feature_scale"], -12, 12)
                value = np.maximum(0, normalized @ arrays["molecule_in.weight"].T + arrays["molecule_in.bias"])
                value = np.maximum(0, value @ arrays["molecule_out.weight"].T + arrays["molecule_out.bias"]).astype(np.float32)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite molecular embedding")
                for graph, embedding in zip(pending, value):
                    self._cache[graph] = embedding
            result = np.stack([self._cache[g] for g in graphs])
            for g in graphs:
                self._cache.move_to_end(g)
            while len(self._cache) > 8192:
                self._cache.popitem(last=False)
            return result

    @staticmethod
    def _composition(graphs, amounts):
        if isinstance(graphs, (str, bytes)) or not graphs or len(graphs) != len(amounts):
            raise ValueError("nonempty molecular graphs and matching relative amounts required")
        combined = {}
        for graph, amount in zip(graphs, amounts):
            if (not isinstance(amount, (int, float, np.integer, np.floating))
                    or isinstance(amount, (bool, np.bool_)) or not np.isfinite(amount) or amount < 0):
                raise ValueError("finite nonnegative relative composition required")
            if not isinstance(graph, str):
                raise ValueError("one explicit molecular graph per component required")
            canonical = _canonical_graph(graph)
            combined[canonical] = combined.get(canonical, 0.) + float(amount)
        total = sum(combined.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("positive finite total composition required")
        keys = sorted(g for g, amount in combined.items() if amount > 0)
        return keys, np.asarray([combined[g] / total for g in keys], np.float32)

    def predict(self, first_graphs, first_amounts, second_graphs, second_amounts):
        a, wa = self._composition(first_graphs, first_amounts)
        b, wb = self._composition(second_graphs, second_amounts)
        value = self._embed(a + b)
        context = np.zeros((1, 64), np.float32)
        context[:, 12] = 2.
        score = self.model(value[:len(a)][None], wa[None], value[len(a):][None], wb[None], context)[0]
        return {"similarity": float(score), "model_sha256": self.sha256,
                "policy_sha256": self.policy.digest, "amount_basis": "relative_component_composition",
                "score_kind": "learned_nominal_mixture_similarity", "product_context": "neutral_lab_mixture",
                "physical_release_prediction": False}
