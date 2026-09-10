"""Bounded, perfumer-inspired block trials, not measured accord synergy.

Transfer mass between odor-family blocks while preserving each existing
block's internal recipe ratios. The caller must evaluate the actual product
response and all constraints; no bonus points are assigned for using a block.
"""
import numpy as np


def accord_trials(weights, profiles, response, target, lower, caps, *, budget_groups=None, limit=32):
    weights, profiles, response, target, lower, caps = map(
        lambda x: np.asarray(x, dtype=float), (weights, profiles, response, target, lower, caps))
    n = len(weights)
    if (profiles.ndim != 2 or profiles.shape[0] != n or target.shape != (profiles.shape[1],)
            or any(x.shape != (n,) for x in (response, lower, caps))):
        raise ValueError('accord trial shape mismatch')
    if any(not np.isfinite(x).all() for x in (weights, profiles, response, target, lower, caps)):
        raise ValueError('finite accord trial data required')
    if any(np.any(x < 0) for x in (weights, profiles, response, target, lower, caps)) or np.any(lower > caps):
        raise ValueError('nonnegative accord data and ordered bounds required')
    if target.sum() <= 0 or weights.sum() <= 0 or limit <= 0:
        return
    target = target / target.sum()
    raw = (weights * response) @ profiles
    if raw.sum() <= 0:
        return
    deficit = target - raw / raw.sum()
    family = np.argmax(profiles, axis=1)
    budgets = np.zeros(n, dtype=int) if budget_groups is None else np.asarray(budget_groups)
    if budgets.shape != (n,):
        raise ValueError('one budget group per ingredient required')
    moves = []
    for group in np.unique(budgets):
        members = budgets == group
        for donor_axis in np.flatnonzero(deficit < -1e-10):
            donors = np.flatnonzero(members & (family == donor_axis) & (weights > 0))
            if not len(donors):
                continue
            donor_ratio = weights[donors] / weights[donors].sum()
            available = float(np.min((weights[donors] - lower[donors]) / donor_ratio))
            if available <= 1e-12:
                continue
            for receiver_axis in np.flatnonzero(deficit > 1e-10):
                receivers = np.flatnonzero(members & (family == receiver_axis) & (response > 0)
                                          & (profiles[:, receiver_axis] > 0)
                                          & (caps - weights > 1e-12))
                if not len(receivers):
                    continue
                active = receivers[weights[receivers] > 0]
                if len(active):
                    receivers = active
                    receiver_ratio = weights[receivers] / weights[receivers].sum()
                else:
                    # Seed an absent block from at most three currently eligible
                    # materials. This is a trial shortlist, not a catalog cap.
                    rank = np.argsort(-profiles[receivers, receiver_axis], kind='stable')[:3]
                    receivers = receivers[rank]
                    receiver_ratio = profiles[receivers, receiver_axis]
                    receiver_ratio = receiver_ratio / receiver_ratio.sum()
                room = float(np.min((caps[receivers] - weights[receivers]) / receiver_ratio))
                amount = min(available, room)
                if amount > 1e-12:
                    moves.append((deficit[receiver_axis]-deficit[donor_axis], donors, donor_ratio,
                                  receivers, receiver_ratio, amount))
    moves.sort(key=lambda row: -row[0])
    count = 0
    # Coarse-to-fine trials include tiny corrections for high-impact materials.
    for fraction in (.5, .25, .1, .02, 1.):
        for _, donors, donor_ratio, receivers, receiver_ratio, amount in moves:
            proposal = weights.copy()
            proposal[donors] -= fraction * amount * donor_ratio
            proposal[receivers] += fraction * amount * receiver_ratio
            # A complete transfer can round just below the donor floor (for
            # example -2.8e-17 at zero). Restore the original bounds here;
            # downstream mass, cost, profile and product checks stay strict.
            proposal[donors] = np.maximum(lower[donors], proposal[donors])
            proposal[receivers] = np.minimum(caps[receivers], proposal[receivers])
            yield proposal
            count += 1
            if count >= limit:
                return
