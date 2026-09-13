"""Use the shared learned revision direction to order, never approve, trials."""

import numpy as np


def rank_formulation_trials(
    trials,
    ingredients,
    incumbent,
    profiles,
    response,
    target,
    *,
    product,
    concentration_percent,
):
    trials = list(trials)
    if len(trials) < 2:
        return trials
    from .formulation_core import configured_formulation_core

    core = configured_formulation_core()
    if core is None:
        return trials
    target = np.asarray(target, float)
    profiles, response, incumbent = map(
        lambda v: np.asarray(v, float), (profiles, response, incumbent)
    )
    # The learned revision task has 19 explicitly named axes. The 146-axis
    # observed-reference solver stays in its own metric, without relabelling.
    if target.shape != (19,) or profiles.shape != (len(ingredients), 19):
        return trials
    graphs = []
    for item in ingredients:
        binding = core.manifest["structures"].get(item.ingredient_id)
        graph = item.structure_smiles or (binding[0] if binding else None)
        if not graph or "." in graph:
            return trials
        if binding and binding[1] is not None and binding[1] != item.cas_number:
            raise ValueError("revision material/CAS mismatch")
        graphs.append(graph)
    current = (incumbent * response) @ profiles
    if current.sum() <= 0 or incumbent.sum() <= 0:
        return trials
    predicted = core.mixture(
        graphs,
        incumbent,
        product=product,
        target=target,
        current=current,
        concentration_percent=concentration_percent,
    )
    direction = predicted["revision_direction"]

    def priority(proposal):
        value = (np.asarray(proposal) * response) @ profiles
        if value.sum() <= 0:
            return -np.inf
        return float(direction @ (value / value.sum() - current / current.sum()))

    # The caller evaluates ALL generated proposals against the unchanged
    # constraints/objective. A network rank or probability is never acceptance.
    return sorted(trials, key=priority, reverse=True)
