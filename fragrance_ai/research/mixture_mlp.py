"""Parameter-matched, distribution-aware MLP residuals for mixture research.

These blocks learn dimensionless mixture representations, not measured
concentration laws. The pretrained molecular encoder/backbone remain fixed.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fragrance_ai.research.physmix_comparison import PairComparison


class DistributionResidual(nn.Module):
    """Transform ingredients before pooling; optionally retain cross moments.

    For normalized masses w, z_i = phi(e_i), mu = sum_i w_i z_i,
    C = sum_i w_i (P z_i - P mu)(P z_i - P mu)^T. C is a learned
    representation covariance, not a measured chemical interaction matrix.
    Complexity is linear in ingredient count for fixed projection rank 16.
    """

    def __init__(self, *, covariance: bool):
        super().__init__()
        self.covariance = covariance
        self.ingredient = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 64)
        )
        if covariance:
            self.projection = nn.Linear(64, 16, bias=False)
            self.register_buffer("upper", torch.triu_indices(16, 16, offset=1))
        features, width = (760, 121) if covariance else (640, 139)
        self.readout = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, width), nn.GELU(), nn.Linear(width, 256)
        )

    def moments(self, embedded, amounts):
        weights = amounts / amounts.sum(-1, keepdim=True)
        z = self.ingredient(embedded)
        mean = (weights[..., None] * z).sum(1)
        centered = z - mean[:, None]
        variance = (weights[..., None] * centered.square()).sum(1)
        values = [mean, variance]
        if self.covariance:
            projected = self.projection(centered)
            matrix = projected.transpose(1, 2) @ (weights[..., None] * projected)
            values.append(matrix[:, self.upper[0], self.upper[1]])
        return torch.cat(values, -1)

    def forward(self, embedded, amounts, mean, variance):
        return self.readout(torch.cat((mean, variance, self.moments(embedded, amounts)), -1))


class MixtureMLP(PairComparison):
    MODES = ("pooled_mlp", "normalized_mlp", "set_mlp", "covariance_mlp")

    def __init__(self, arrays, mode):
        if mode not in self.MODES:
            raise ValueError("unknown mixture MLP arm")
        # Reproduce the V83 control and its initialization before any ablation.
        nn.Module.__init__(self)
        baseline = PairComparison(arrays, "capacity_control")
        self.backbone = baseline.backbone
        self.interaction = baseline.interaction
        self.gate = baseline.gate
        self.head = baseline.head
        self.mode = mode
        if mode == "normalized_mlp":
            self.interaction = nn.Sequential(
                nn.LayerNorm(512), nn.Linear(512, 217), nn.GELU(), nn.Linear(217, 256)
            )
        elif mode in ("set_mlp", "covariance_mlp"):
            self.interaction = DistributionResidual(covariance=mode == "covariance_mlp")

    def encode_precomputed(self, embedded, amounts, h, mean, variance):
        """Training-only cache path; cached backbone values are label independent."""
        if self.mode in ("pooled_mlp", "normalized_mlp"):
            extra = self.interaction(torch.cat((mean, variance), -1))
        else:
            extra = self.interaction(embedded, amounts, mean, variance)
        return h + .25 * torch.tanh(self.gate) * extra, amounts.sum(-1).log1p()

    def encode(self, embedded, amounts, context):
        if (
            embedded.ndim != 3 or embedded.shape[-1] != 256
            or amounts.shape != embedded.shape[:2] or context.shape != (len(embedded), 64)
            or any(not torch.isfinite(x).all() for x in (embedded, amounts, context))
            or torch.any(amounts < 0) or torch.any(amounts.sum(-1) <= 0)
            or not torch.isfinite(amounts.sum(-1)).all()
        ):
            raise ValueError("finite complete molecular mixtures and context required")
        return self.encode_precomputed(embedded, amounts, *self.backbone(embedded, amounts, context))

    def pair_head(self, first, ca, second, cb):
        features = torch.cat((
            (first - second).abs(), first * second, F.cosine_similarity(first, second)[:, None],
            (ca - cb).abs()[:, None], (ca * cb)[:, None]
        ), -1)
        return torch.sigmoid(self.head(features).squeeze(-1))

    def forward_precomputed(self, first, second):
        a, ca = self.encode_precomputed(*first)
        b, cb = self.encode_precomputed(*second)
        return self.pair_head(a, ca, b, cb)

    def forward(self, a, wa, b, wb, context):
        first, ca = self.encode(a, wa, context)
        second, cb = self.encode(b, wb, context)
        return self.pair_head(first, ca, second, cb)
