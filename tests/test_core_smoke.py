import math

from fragrance_ai.ai.unified_ai_system import UnifiedAIConfig, UnifiedFragranceAI
from fragrance_ai.domain.fragrance_chemistry import FragranceChemistry
from fragrance_ai.tools.scientific_validator_tool import NotesComposition, validate_composition


def small_system() -> UnifiedFragranceAI:
    return UnifiedFragranceAI(
        UnifiedAIConfig(
            dl_embedding_dim=32,
            dl_num_layers=1,
            dl_num_heads=4,
            dl_max_length=24,
            moga_population_size=8,
            moga_generations=2,
            device="cpu",
            seed=42,
        )
    )


def test_unified_ai_core_smoke():
    ai = small_system()

    generated = ai.generate_with_dl([1, 2])
    pareto = ai.optimize_with_moga()
    evolved = ai.evolve_with_rl([5.0] * 20, 4.0)

    assert len(generated["notes"][0]) == 20
    assert pareto
    assert math.isclose(sum(evolved["evolved_formula"]), 100.0, abs_tol=1e-6)


def test_chemistry_evaluation_is_bounded():
    result = FragranceChemistry.evaluate_fragrance_complete(
        [("bergamot", 25.0)],
        [("rose", 35.0)],
        [("sandalwood", 40.0)],
    )

    assert 0.0 <= result["harmony"] <= 1.0
    assert 0.0 <= result["balance"] <= 1.0
    assert result["cost"] > 0


def test_scientific_validator_uses_percentage_scale():
    report = validate_composition(
        [
            NotesComposition(1, "citrus", 20.0, "top", 0.9, 0.7, 0.3, True),
            NotesComposition(2, "floral", 40.0, "middle", 0.5, 0.8, 0.6, True),
            NotesComposition(3, "woody", 40.0, "base", 0.1, 0.6, 0.9, False),
        ]
    )

    assert not report.errors
    assert 0.0 <= report.harmony_score <= 1.0
