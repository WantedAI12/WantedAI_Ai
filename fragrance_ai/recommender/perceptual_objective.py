"""One authoritative perfume aggregation for optimization and final scoring."""
import numpy as np

VERSION = 'product-aware-perceptual-objective/v89'

def perfume_loss(predicted, target, time_weights, avoided):
    """Perfume: max(nominal loss, time-weighted loss), then worst head."""
    norm = np.linalg.norm(predicted, axis=-1)
    qnorm = np.linalg.norm(target, axis=-1)
    cosine = (predicted * target).sum(-1) / np.maximum(norm * qnorm, 1e-30)
    losses = np.stack(
        (
            0.5 * np.abs(predicted - target).sum(-1),
            1 - cosine,
            (predicted * avoided).sum(-1),
        ),
        -1,
    )
    kind = losses.argmax(-1)
    points = np.take_along_axis(losses, kind[..., None], -1)[..., 0]
    partial = np.where(
        (kind == 0)[..., None],
        0.5 * np.sign(predicted - target),
        np.where(
            (kind == 1)[..., None],
            cosine[..., None] * predicted / np.maximum(norm[..., None] ** 2, 1e-30)
            - target / np.maximum((norm * qnorm)[..., None], 1e-30),
            avoided,
        ),
    )
    weights = np.array(time_weights, float, copy=True)
    weights[0] = 0.0
    if weights.sum() <= 0:
        raise ValueError("positive time weights required")
    weights /= weights.sum()
    temporal = points @ weights
    total = np.maximum(points[:, 0], temporal)
    head = int(total.argmax())
    coefficients = np.zeros_like(points)
    coefficients[head] = (
        np.eye(len(weights))[0] if points[head, 0] >= temporal[head] else weights
    )
    return float(total[head]), partial * coefficients[..., None]
