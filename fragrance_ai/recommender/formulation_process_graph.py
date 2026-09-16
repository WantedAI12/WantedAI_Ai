"""Source-scoped precedence and iterative bench work, not one fixed sequence.

A/B premixes commute. Perfume trials may be revised repeatedly; a new trial
invalidates old downstream filling/quality state. Source condition predicates
remain authoritative, and completing a graph never grants production approval.
"""

import numpy as np
from .formulation_process import ACTIONS, TEMPLATES, procedure_target

VERSION = "source_dependency_graph/v76"


def process_state(template, completed, values):
    _, checks = procedure_target(template, (), values)
    seen = set()
    invalid = False
    prerequisites = {
        "brief": set(),
        "material_review": {"brief"},
        "weigh": {"material_review"},
        "batch_record": {"compare", "revise"}
        if template == "perfume"
        else {"emulsify", "fragrance_add"},
        "quality_review": {"batch_record"},
        "fill_record": {"quality_review"},
        "done": {"fill_record"},
    }
    if template == "perfume":
        prerequisites.update(
            basic_accord={"weigh"},
            modify={"basic_accord"},
            blend={"basic_accord"},
            fix={"blend"},
            dilute={"blend"},
            compare={"dilute"},
            revise={"compare"},
        )
    else:
        prerequisites.update(
            phase_a={"weigh"},
            phase_b={"weigh"},
            heat={"phase_a", "phase_b"},
            emulsify={"heat"}
            if template == "hot_lotion"
            else {"phase_a", "phase_b", "fragrance_add"},
            cool={"emulsify"},
            fragrance_add={"cool"} if template == "hot_lotion" else {"phase_b"},
        )
    relevant = set(TEMPLATES[template])
    for action in completed:
        if action not in relevant or (action == "done" and "done" in seen):
            invalid = True
            break
        if "done" in seen:
            invalid = True
            break
        if (
            template == "perfume"
            and action in {"weigh", "basic_accord", "modify", "blend"}
            and "revise" in seen
        ):
            seen -= {
                "blend",
                "fix",
                "dilute",
                "compare",
                "revise",
                "batch_record",
                "quality_review",
                "fill_record",
                "done",
            }
            if action == "weigh":
                seen -= {"weigh", "basic_accord", "modify"}
            elif action == "basic_accord":
                seen -= {"basic_accord", "modify"}
            elif action == "modify":
                seen.discard("modify")
        if action in seen or not prerequisites[action] <= seen:
            invalid = True
            break
        seen.add(action)
    checks[0] = invalid
    if checks[:5].any():
        allowed = ["hold"]
    elif "done" in seen:
        allowed = ["done"]
    else:
        allowed = [
            action
            for action in TEMPLATES[template]
            if action not in seen and prerequisites[action] <= seen
        ]
        if template == "perfume" and "revise" in seen:
            allowed = [*["weigh", "modify", "blend"], *allowed]
    allowed = list(dict.fromkeys(allowed))
    mask = np.array([action in allowed for action in ACTIONS], np.float32)
    if not mask.any():
        raise ValueError("source process graph has no allowed next action")
    return {
        "allowed": allowed,
        "mask": mask,
        "checks": checks,
        "completed_unique": sorted(seen),
        "version": VERSION,
        "manufacturing_approved": False,
    }
