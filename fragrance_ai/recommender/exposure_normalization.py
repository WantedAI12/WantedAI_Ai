"""Scale-invariant positive headspace normalization and its exact derivative."""

import numpy as np


def normalized_exposure(profiles, responses, weights):
    p, r, w = (np.asarray(v, np.float64) for v in (profiles, responses, weights))
    exposure = r * w[:, None, :]
    scale = exposure.max(-1, keepdims=True)
    if np.any(scale <= 0) or not all(np.isfinite(v).all() for v in (p, r, w, exposure)):
        raise ValueError(
            "positive finite headspace required; zero exposure has no odor shape"
        )
    relative = exposure / scale
    total = relative.sum(-1, keepdims=True)
    shares = relative / total
    predicted = np.einsum("btn,bhnd->bhtd", shares, p)
    derivative = (r / scale) / total
    if not np.isfinite(derivative).all():
        raise ValueError("headspace sensitivity outside finite numerical range")
    return predicted, derivative


def torch_normalized_exposure(profiles, responses, weights):
    import torch

    p, r, w = profiles.double(), responses.double(), weights.double()
    exposure = r * w[:, None, :]
    scale = exposure.amax(-1, keepdim=True)
    # The same domain is enforced in research as in NumPy inference.
    if bool((scale <= 0).any()) or not bool(torch.isfinite(exposure).all()):
        raise ValueError(
            "positive finite headspace required; zero exposure has no odor shape"
        )
    relative = exposure / scale
    total = relative.sum(-1, keepdim=True)
    predicted = torch.einsum("btn,bhnd->bhtd", relative / total, p)
    derivative = (r / scale) / total
    return predicted.to(profiles.dtype), derivative
