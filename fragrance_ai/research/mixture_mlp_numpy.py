"""Torch-free CPU execution of V84 mixture-MLP research checkpoints."""
from __future__ import annotations

import numpy as np

from fragrance_ai.research.physmix_numpy import NumpyPairComparison


class NumpyMixtureMLP(NumpyPairComparison):
    MODES = ("pooled_mlp", "normalized_mlp", "set_mlp", "covariance_mlp")

    def __init__(self, arrays, mode):
        if mode not in self.MODES:
            raise ValueError("unknown mixture MLP arm")
        super().__init__(arrays, "capacity_control")
        self.mode = mode

    def distribution(self, embedded, amounts):
        prefix = "interaction.ingredient."
        z = self.linear(self.gelu(self.linear(self.layer(embedded, prefix + "0"), prefix + "1")), prefix + "3")
        weights = amounts / amounts.sum(-1, keepdims=True)
        mean = (weights[..., None] * z).sum(1)
        centered = z - mean[:, None]
        variance = (weights[..., None] * centered**2).sum(1)
        values = [mean, variance]
        if self.mode == "covariance_mlp":
            projected = self.linear(centered, "interaction.projection")
            matrix = projected.transpose(0, 2, 1) @ (weights[..., None] * projected)
            first, second = np.triu_indices(16, k=1)
            values.append(matrix[:, first, second])
        return np.concatenate(values, -1)

    def encode(self, embedded, amounts, context):
        e, w, c = (np.asarray(x, np.float32) for x in (embedded, amounts, context))
        if (
            e.ndim != 3 or e.shape[-1] != 256 or w.shape != e.shape[:2] or c.shape != (len(e), 64)
            or any(not np.isfinite(x).all() for x in (e, w, c)) or np.any(w < 0)
            or np.any(w.sum(-1) <= 0) or not np.isfinite(w.sum(-1)).all()
        ):
            raise ValueError("finite complete molecular mixtures and context required")
        h, mean, variance = self.aggregate(e, w, c)
        values = np.concatenate((mean, variance), -1)
        if self.mode == "pooled_mlp":
            extra = self.linear(self.gelu(self.linear(values, "interaction.0")), "interaction.2")
        elif self.mode == "normalized_mlp":
            extra = self.linear(self.gelu(self.linear(self.layer(values, "interaction.0"), "interaction.1")), "interaction.3")
        else:
            values = np.concatenate((values, self.distribution(e, w)), -1)
            prefix = "interaction.readout."
            extra = self.linear(self.gelu(self.linear(self.layer(values, prefix + "0"), prefix + "1")), prefix + "3")
        return h + .25 * np.tanh(self.arrays["gate"]) * extra, np.log1p(w.sum(-1))
