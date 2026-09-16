"""Prefix-only replay and live search over immutable evaluation trajectories.

Inspired by historical replay, implemented for this project's numeric search.
No executable policy text, external LLM, evaluator edits, or guessed outcomes.
Scores are maximized; callers must supply only the designated selection score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class Observation:
    branch: str
    step: int
    score: float | None
    seconds: float
    terminal: bool = False
    status: str = "ok"
    artifact_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.branch, str) or not self.branch:
            raise ValueError("nonempty branch identifier required")
        if type(self.step) is not int or self.step < 1:
            raise ValueError("positive integral branch step required")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("finite nonnegative observed cost required")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("nonfinite score must be recorded as a failed attempt")
        if self.status == "ok" and self.score is None:
            raise ValueError("successful attempts require a selection score")
        if self.status != "ok" and self.score is not None:
            raise ValueError("failed attempts cannot earn a quality score")
        if type(self.terminal) is not bool:
            raise ValueError("terminal must be explicit boolean")


@dataclass(frozen=True)
class Policy:
    revision: int = 2
    kind: str = "adaptive"
    warmup: int = 8
    horizon: int = 8
    exploration: float = .25
    patience: int = 12
    cost_power: float = .5
    stop_margin: float = .5
    minimum_improvement: float = 1e-5

    def __post_init__(self):
        if self.revision not in (1, 2):
            raise ValueError("unknown replay policy revision")
        if self.kind not in ("adaptive", "round_robin"):
            raise ValueError("unknown bounded policy family")
        if any(type(x) is not int or x < 1 for x in (self.warmup, self.horizon, self.patience)):
            raise ValueError("positive integral scheduling parameters required")
        if not 0 <= self.exploration <= 2 or not 0 <= self.cost_power <= 1:
            raise ValueError("invalid scheduling coefficients")
        if not math.isfinite(self.stop_margin) or self.stop_margin < 0:
            raise ValueError("finite nonnegative continuation margin required")
        if not math.isfinite(self.minimum_improvement) or self.minimum_improvement <= 0:
            raise ValueError("positive finite improvement resolution required")

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def best_observation(prefix: Mapping[str, Sequence[Observation]]):
    candidates = [o for history in prefix.values() for o in history if o.status == "ok"]
    return max(candidates, key=lambda o: o.score) if candidates else None


def choose(policy: Policy, prefix: Mapping[str, Sequence[Observation]], legal: Sequence[str]):
    """Observe only immutable prefixes. No world, evaluator, or result handle."""
    if not legal:
        return None, "all_branches_terminal"
    ordered = sorted(legal)
    if policy.kind == "round_robin":
        return min(ordered, key=lambda b: (len(prefix[b]), b)), "fixed_equal_allocation"
    warm = [b for b in ordered if len(prefix[b]) < policy.warmup]
    if warm:
        return min(warm, key=lambda b: (len(prefix[b]), b)), "minimum_observed_exploration"
    valid = {b: [o for o in prefix[b] if o.status == "ok"] for b in ordered}
    all_scores = [o.score for row in valid.values() for o in row]
    if not all_scores:
        return None, "no_successful_selection_evidence"
    anchors = {b: max((o.score for o in row), default=-math.inf) for b, row in valid.items()}
    # A completed high-quality branch remains the incumbent. Dropping it when
    # computing legal frontiers would waste work trying to beat a weaker one.
    all_anchors = [max(o.score for o in row if o.status == "ok") for row in prefix.values()
                   if any(o.status == "ok" for o in row)]
    incumbent = max(all_anchors) if policy.revision >= 2 else max(anchors.values())
    finite_anchors = all_anchors if policy.revision >= 2 else [v for v in anchors.values() if math.isfinite(v)]
    scale = max(.001, statistics.pstdev(finite_anchors))
    typical_cost = max(1e-9, statistics.median(o.seconds for row in prefix.values() for o in row))
    total_steps = sum(len(row) for row in prefix.values())
    priorities = []
    for branch in ordered:
        history = valid[branch]
        if not history:
            continue
        # A temporary regression or failure never destroys the best old anchor.
        best = history[0].score
        improved_at = history[0].step
        running = []
        for obs in history:
            if obs.score > best + policy.minimum_improvement:
                best, improved_at = obs.score, obs.step
            running.append((obs.step, best))
        age = prefix[branch][-1].step - improved_at
        if age >= policy.patience:
            continue
        recent = running[-min(6, len(running)):]
        slope = max(0., (recent[-1][1] - recent[0][1]) / max(1, recent[-1][0] - recent[0][0]))
        uncertainty = policy.exploration * scale * math.sqrt(math.log1p(total_steps) / len(history))
        # Relative potential, with softplus so poor branches are not rewarded
        # for being expensive when their potential is negative.
        potential = (anchors[branch] - incumbent + policy.horizon * slope + uncertainty) / scale
        if policy.revision >= 2 and potential < -policy.stop_margin:
            continue
        benefit = scale * (max(potential, 0.) + math.log1p(math.exp(-abs(potential))))
        cost = max(1e-9, statistics.median(o.seconds for o in prefix[branch])) / typical_cost
        priority = benefit / cost**policy.cost_power
        priorities.append((priority, -len(history), branch))
    if not priorities:
        return None, "no_supported_marginal_gain" if policy.revision >= 2 else "all_observed_branches_stalled"
    selected = max(priorities)[2]
    return selected, "observed_gain_uncertainty_per_cost"


def run_live(branches: Sequence[str], advance: Callable[[str], Observation], policy: Policy, *, budget: int,
             on_reveal: Callable[[Observation, dict], None] | None = None):
    """The same controller drives real training and historical replay."""
    if type(budget) is not int or budget < 1 or not branches or len(set(branches)) != len(branches):
        raise ValueError("positive work budget and unique branches required")
    prefix: dict[str, list[Observation]] = {b: [] for b in sorted(branches)}
    decisions, spent = [], 0
    while spent < budget:
        legal = [b for b, row in prefix.items() if not row or not row[-1].terminal]
        # Immutable values and copied tuples prevent accidental future-state access.
        branch, reason = choose(policy, {b: tuple(row) for b, row in prefix.items()}, tuple(legal))
        if branch is None:
            break
        obs = advance(branch)
        if obs.branch != branch or obs.step != len(prefix[branch]) + 1:
            raise ValueError("executor returned a different branch or nonconsecutive step")
        prefix[branch].append(obs)
        spent += 1
        decision = {"work": spent, "branch": branch, "step": obs.step, "reason": reason,
                    "score": obs.score, "seconds": obs.seconds, "status": obs.status}
        decisions.append(decision)
        if on_reveal is not None:
            on_reveal(obs, decision)
    best = best_observation(prefix)
    return {"policy": asdict(policy), "policy_sha256": policy.digest,
            "work": spent, "budget": budget, "seconds": sum(o.seconds for row in prefix.values() for o in row),
            "best": asdict(best) if best else None, "decisions": decisions,
            "traces": {b: [asdict(o) for o in row] for b, row in prefix.items()},
            "stop_reason": "work_budget" if spent == budget else reason}


class UnsupportedReplay(RuntimeError):
    pass


def replay(world: Mapping[str, Sequence[Observation]], policy: Policy, *, budget: int):
    """Never fabricate unrecorded children or treat truncated branches as stalls."""
    cursors = dict.fromkeys(world, 0)
    for branch, row in world.items():
        for i, obs in enumerate(row):
            if obs.branch != branch or obs.step != i + 1 or (i < len(row) - 1 and obs.terminal):
                raise ValueError("invalid immutable branch trajectory")

    def advance(branch):
        offset = cursors[branch]
        if offset >= len(world[branch]):
            raise UnsupportedReplay(f"unrecorded continuation: {branch} step {offset + 1}")
        cursors[branch] += 1
        return world[branch][offset]

    return run_live(tuple(world), advance, policy, budget=budget)


def policy_grid():
    # Numeric candidates, not arbitrary self-modifying code or LLM calls.
    for warmup in (4, 8, 12):
        for horizon in (4, 8, 16):
            for exploration in (0., .25, .75):
                for patience in (8, 16):
                    for cost_power in (0., .5):
                        yield Policy(warmup=warmup, horizon=horizon, exploration=exploration,
                                     patience=patience, cost_power=cost_power)


def improve_policy(worlds, *, budget, quality_tolerance=.001):
    """Quality-first replay selection, with incumbent as a candidate.

    The tolerance is a policy-selection tradeoff, never a recipe/evaluator
    threshold change. It is frozen before final evaluation.
    """
    if not worlds or not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("historical worlds and finite quality tolerance required")
    candidates = [Policy(kind="round_robin"), *policy_grid()]
    rows = []
    for policy in candidates:
        try:
            results = [replay(world, policy, budget=budget) for world in worlds]
        except UnsupportedReplay:
            continue
        if any(r["best"] is None for r in results):
            continue
        rows.append({"policy": asdict(policy), "policy_sha256": policy.digest,
                     "score": statistics.mean(r["best"]["score"] for r in results),
                     "seconds": statistics.mean(r["seconds"] for r in results),
                     "work": statistics.mean(r["work"] for r in results)})
    if not rows:
        raise ValueError("no policy supported by the recorded worlds")
    highest = max(r["score"] for r in rows)
    admissible = [r for r in rows if r["score"] >= highest - quality_tolerance]
    selected = min(admissible, key=lambda r: (r["seconds"], r["work"], -r["score"], r["policy_sha256"]))
    return Policy(**selected["policy"]), {"evaluated_policies": len(rows), "worlds": len(worlds),
                                         "selection": selected, "candidates": rows,
                                         "quality_tolerance": quality_tolerance}
