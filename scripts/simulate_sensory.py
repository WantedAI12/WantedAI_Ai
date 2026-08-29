"""Run the synthetic sensory proxy without creating human evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fragrance_ai.recommender import NaturalLanguagePerfumeryAI, RecipeConstraints  # noqa: E402
from fragrance_ai.recommender.simulation import SimulatedSensoryEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", required=True)
    parser.add_argument("--target", type=float, default=90.0)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    ai = NaturalLanguagePerfumeryAI()
    result = ai.create_recipe(
        args.brief,
        RecipeConstraints(
            target_similarity=args.target,
            simulation_draws=args.draws,
            require_simulation_pass=False,
        ),
    )
    if not result.closest_candidate:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        raise SystemExit(2)
    ingredient_map = {item.ingredient_id: item for item in ai.catalog.ingredients}
    scientific_twin = ai.temporal_simulator.evaluate(
        result.closest_candidate,
        ingredient_map,
        result.brief,
        ai.scientific_store,
        draws=max(64, args.draws),
        seed=args.seed,
    )
    physsim = ai.physsim_engine.evaluate(
        result.closest_candidate,
        ingredient_map,
        result.brief,
        ai.scientific_store,
    )
    simulation = SimulatedSensoryEngine().evaluate(
        result.closest_candidate,
        ingredient_map,
        result.brief,
        ai.corpus,
        target=args.target,
        draws=args.draws,
        seed=args.seed,
        scientific_twin=scientific_twin,
        physsim=physsim,
    )
    output = {
        "brief": args.brief,
        "formula_id": result.formula_id,
        "scientific_twin": asdict(scientific_twin),
        "physsim": asdict(physsim),
        "simulation": simulation.__dict__,
        "warning": "Physics-informed simulation only; this is not measured human olfactory accuracy.",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
