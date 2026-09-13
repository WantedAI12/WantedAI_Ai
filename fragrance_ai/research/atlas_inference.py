"""Immutable bounded-memory CPU compilation of molecular measurement heads."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .atlas_profiles import (
    _valid_features,
    atlas_kernel,
    inverse_atlas_target,
    predict_atlas,
)
from .scientific_kernel import ScientificKernel


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class _KernelGroup:
    support: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    slices: tuple
    scientific: ScientificKernel | None
    geometry_key: str | None
    scale: np.ndarray | None
    fine_weight: float | None
    normalized: bool

    def kernel(self, query):
        if self.scientific is not None:
            return self.scientific(query, self.support)
        return atlas_kernel(
            query, self.support, self.scale, self.fine_weight, normalize=self.normalized
        )


class CompiledAtlasHeads:
    def __init__(self, models):
        compiled, groups = {}, {}
        width = None
        for name, supplied in models.items():
            model = deepcopy(supplied)
            scientific = model.get("kind") == "atlas-quantitative-profiles/v4"
            keys = ("support", "weights", "intercept") + (
                () if scientific else ("scale",)
            )
            for key in keys:
                array = np.array(model[key], dtype=float, copy=True)
                array.setflags(write=False)
                model[key] = array
            # Complete portable validation happens once, not once per batch.
            predict_atlas(model, model["support"][:1])
            if width is not None and model["support"].shape[1] != width:
                raise ValueError("Atlas heads require the same feature layout")
            width = model["support"].shape[1]
            digest = hashlib.sha256()
            arrays = (
                (model["support"],)
                if scientific
                else (model["support"], model["scale"])
            )
            for array in arrays:
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
            if scientific:
                digest.update(
                    json.dumps(
                        model["kernel_specification"], sort_keys=True, allow_nan=False
                    ).encode()
                )
            else:
                digest.update(
                    repr(
                        (
                            float(model["fine_weight"]),
                            model["kind"] == "atlas-quantitative-profiles/v3",
                        )
                    ).encode()
                )
            groups.setdefault(digest.hexdigest(), []).append((name, model))
            compiled[name] = model
        if not compiled:
            raise ValueError("at least one Atlas measurement head required")
        self.width = width
        result = []
        for heads in groups.values():
            example = heads[0][1]
            coefficients = np.concatenate(
                [model["weights"] for _, model in heads], axis=1
            )
            intercept = np.concatenate([model["intercept"] for _, model in heads])
            coefficients.setflags(write=False)
            intercept.setflags(write=False)
            slices, start = [], 0
            for name, model in heads:
                stop = start + len(model["intercept"])
                slices.append(
                    (name, start, stop, model.get("target_transform", "identity"))
                )
                start = stop
            scientific = (
                ScientificKernel(example["kernel_specification"])
                if example["kind"] == "atlas-quantitative-profiles/v4"
                else None
            )
            geometry_key = None
            if scientific is not None:
                geometry_key = hashlib.sha256(
                    example["support"].tobytes()
                    + json.dumps(
                        example["kernel_specification"]["geometry"],
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
            result.append(
                _KernelGroup(
                    example["support"],
                    coefficients,
                    intercept,
                    tuple(slices),
                    scientific,
                    geometry_key,
                    example.get("scale"),
                    example.get("fine_weight"),
                    example["kind"] == "atlas-quantitative-profiles/v3",
                )
            )
        self.groups = tuple(result)
        self.models = _freeze(compiled)

    def predict(self, features):
        return self._forward(features, diagnostics=False)[0]

    def predict_with_diagnostics(self, features):
        return self._forward(features, diagnostics=True)

    def _forward(self, features, *, diagnostics):
        x = np.asarray(features, float)
        if not _valid_features(x) or x.shape[1] != self.width:
            raise ValueError("invalid compiled Atlas query features")
        output = {
            name: np.empty((len(x), len(model["intercept"])))
            for name, model in self.models.items()
        }
        details = {name: [] for name in self.models} if diagnostics else {}
        for offset in range(0, len(x), 256):
            query = x[offset : offset + 256]
            block_cache = {}
            for group in self.groups:
                if group.scientific is None:
                    kernel = group.kernel(query)
                else:
                    if group.geometry_key not in block_cache:
                        block_cache[group.geometry_key] = (
                            group.scientific.geometry.blocks(query, group.support)
                        )
                    kernel = group.scientific.combine(
                        block_cache[group.geometry_key], query, group.support
                    )
                latent = kernel @ group.coefficients + group.intercept
                diagnostic = (
                    group.scientific.diagnostics(query, kernel)
                    if diagnostics and group.scientific is not None
                    else None
                )
                for name, start, stop, transform in group.slices:
                    output[name][offset : offset + len(query)] = inverse_atlas_target(
                        latent[:, start:stop], transform
                    )
                    if diagnostics:
                        details[name].extend(
                            [
                                {
                                    "basis": "training_geometry_not_calibrated_predictive_uncertainty",
                                    **{
                                        key: value[index].item()
                                        for key, value in diagnostic.items()
                                    },
                                }
                                if diagnostic is not None
                                else {"basis": "unavailable_in_legacy_checkpoint"}
                                for index in range(len(query))
                            ]
                        )
        return output, details
