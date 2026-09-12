"""Phase intent, search diversity, batching, cache and snapshot contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import date
import hashlib
import math
import threading
import time

import numpy as np
import pytest

from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import (
    IngredientCatalog,
    HistoricalReferenceCorpus,
)
from fragrance_ai.recommender.registry_activation import (
    RegistryActivationReport,
    load_runtime_catalog,
    write_runtime_catalog,
)
from fragrance_ai.recommender.runtime_cache import InferenceCache, InferenceBusy
from fragrance_ai.recommender.science import (
    ATMOSPHERIC_PRESSURE_PA,
    ETHANOL_MOLECULAR_WEIGHT,
    TIMEPOINTS_MINUTES,
    TIMEPOINT_WEIGHTS,
    ScientificPropertyStore,
    TemporalMixtureSimulator,
)


def test_reversed_phase_requests_keep_distinct_targets_and_formulas(tmp_path):
    corpus = HistoricalReferenceCorpus(tmp_path / "not-packaged.db")
    with NaturalLanguagePerfumeryAI(corpus=corpus) as ai:
        constraints = RecipeConstraints(
            simulation_draws=64, physics_search_population=3, target_similarity=50
        )
        first = ai.create_recipe(
            "opening citrus, drydown woody musk", constraints, as_of=date(2026, 9, 5)
        )
        reverse = ai.create_recipe(
            "opening woody musk, drydown citrus", constraints, as_of=date(2026, 9, 5)
        )
        korean = ai.parser.parse("오프닝은 시트러스, 잔향은 우디 머스크", constraints)
    assert first.brief.phase_target_profiles == korean.phase_target_profiles
    assert first.brief.phase_target_profiles != reverse.brief.phase_target_profiles
    assert first.formula_id != reverse.formula_id
    assert first.temporal_profile[0]["target_profile"]["citrus"] == 1.0
    assert reverse.temporal_profile[-1]["target_profile"]["citrus"] == 1.0
    assert first.recipe and reverse.recipe


@pytest.mark.parametrize("concentration", [7.5, 15.0, 22.5])
def test_vectorized_responses_match_scalar_model(concentration):
    catalog = IngredientCatalog.load_builtin()
    engine = TemporalMixtureSimulator()
    with ScientificPropertyStore.load_builtin() as store:
        inputs = engine.prepare_response_inputs(catalog.ingredients, store)
        actual = engine.ingredient_response_matrix(inputs, concentration)
        expected = []
        for item in catalog.ingredients:
            prop = store.get(item.ingredient_id)
            mass = concentration / 100 * item.active_strength_percent / 100
            moles = mass / max(1e-9, prop.molecular_weight if prop else 180.0)
            base = max(0, 100 - concentration / 100) / ETHANOL_MOLECULAR_WEIGHT
            pressure = engine._vapor_pressure_prior(item, prop)[0]
            threshold = engine._threshold_prior(item, prop)[0]
            logp = prop.xlogp if prop and prop.xlogp is not None else 2.0
            activity = max(0.5, min(3, math.exp(0.18 * (logp - 2))))
            gas = (
                moles
                / max(1e-12, moles + base)
                * activity
                * pressure
                / ATMOSPHERIC_PRESSURE_PA
                * 1e6
            )
            oav = max(1e-12, gas / threshold)
            half = engine._half_life_minutes(item, prop, pressure)
            expected.append(
                [
                    engine._air_to_receptor_transport(prop)
                    * (oav * 0.5 ** (t / half)) ** 0.55
                    / (1 + (oav * 0.5 ** (t / half)) ** 0.55)
                    for t in TIMEPOINTS_MINUTES
                ]
            )
    assert actual == pytest.approx(np.asarray(expected), abs=1e-12)
    raw = np.maximum(1e-9, np.sum(np.asarray(expected) * TIMEPOINT_WEIGHTS, axis=1))
    expected_factors = np.clip(raw / max(1e-9, float(np.median(raw))), 0.15, 8)
    assert list(
        engine.factors_from_responses(inputs.identifiers, actual).values()
    ) == pytest.approx(expected_factors, abs=1e-12)


def test_swap_search_retains_baseline_and_respects_constraints(tmp_path, monkeypatch):
    with NaturalLanguagePerfumeryAI(
        corpus=HistoricalReferenceCorpus(tmp_path / "not-packaged.db")
    ) as ai:
        constraints = RecipeConstraints(
            simulation_draws=64, physics_search_population=3, target_similarity=50
        )
        real = ai.optimizer.replacement_selections
        monkeypatch.setattr(
            ai.optimizer, "replacement_selections", lambda *args, **kwargs: []
        )
        baseline = ai.create_recipe(
            "clean fresh citrus woody musk", constraints, as_of=date(2026, 9, 5)
        )
        monkeypatch.setattr(ai.optimizer, "replacement_selections", real)
        searched = ai.create_recipe(
            "clean fresh citrus woody musk", constraints, as_of=date(2026, 9, 5)
        )
    assert searched.ingredient_sets_evaluated > 1
    assert searched.ingredient_swaps_evaluated > 0
    assert searched.physics_search_objective >= baseline.physics_search_objective
    assert searched.safety.internal_gate_passed
    assert len(searched.recipe) <= constraints.max_ingredients
    assert sum(line.concentrate_percent for line in searched.recipe) == pytest.approx(
        100, abs=0.001
    )


def test_response_cache_coalesces_concurrent_calls_and_returns_copies():
    cache = InferenceCache()
    started = threading.Event()
    finish = threading.Event()

    def compute():
        started.set()
        assert finish.wait(5)
        return {"values": [1]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        leader = pool.submit(cache.run, "same", compute)
        assert started.wait(5)
        followers = [pool.submit(cache.run, "same", compute) for _ in range(3)]
        deadline = time.monotonic() + 3
        while cache.stats()["followers"] < 3 and time.monotonic() < deadline:
            finish.wait(0.005)
        assert cache.stats()["followers"] == 3
        finish.set()
        results = [leader.result(), *(future.result() for future in followers)]
    assert cache.stats()["computations"] == 1
    assert all(payload == {"values": [1]} for payload, _ in results)
    results[0][0]["values"].append(2)
    assert cache.run("same", compute) == ({"values": [1]}, "hit")
    assert cache.stats()["pending"] == 0


def test_follower_timeout_does_not_cancel_shared_work():
    cache = InferenceCache(wait_seconds=0.03)
    started, finish = threading.Event(), threading.Event()
    def compute():
        started.set()
        assert finish.wait(5)
        return {"ok": True}
    with ThreadPoolExecutor(max_workers=1) as pool:
        leader = pool.submit(cache.run, "same", compute)
        assert started.wait(3)
        with pytest.raises(TimeoutError):
            cache.run("same", compute)
        assert cache.stats()["followers"] == 0
        finish.set()
        assert leader.result()[0] == {"ok": True}
    assert cache.run("same", compute)[1] == "hit"


def test_cache_failure_recovers_and_queue_is_bounded():
    cache = InferenceCache(max_pending=1)
    started = threading.Event()
    finish = threading.Event()

    def broken():
        started.set()
        assert finish.wait(5)
        raise ValueError("test failure")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.run, "first", broken)
        assert started.wait(5)
        with pytest.raises(InferenceBusy):
            cache.run("different", lambda: {})
        finish.set()
        with pytest.raises(ValueError, match="test failure"):
            first.result()
    assert cache.stats()["active"] == 0
    assert cache.stats()["pending"] == 0
    assert cache.run("first", lambda: {"recovered": True})[0] == {"recovered": True}


def test_cache_ttl_lru_bytes_and_bypass():
    clock = [0.0]
    cache = InferenceCache(
        ttl_seconds=10, max_entries=2, max_bytes=30, clock=lambda: clock[0]
    )
    cache.run("a", lambda: {"a": 1})
    cache.run("b", lambda: {"b": 1})
    cache.run("a", lambda: {})
    cache.run("c", lambda: {"c": 1})
    assert cache.run("b", lambda: {"b": 2})[1] == "miss"
    clock[0] = 11
    assert cache.run("b", lambda: {"b": 3})[1] == "miss"
    for _ in range(2):
        assert cache.run("large", lambda: {"text": "x" * 100})[1] == "miss"
        assert (
            cache.run("dynamic", lambda: {"fresh": True}, cacheable=False)[1]
            == "bypass"
        )
    assert cache.stats()["bytes"] <= 30
    assert cache.stats()["entries"] <= 2


def test_runtime_catalog_round_trip_and_hash_binding(tmp_path):
    catalog = IngredientCatalog.load_builtin()
    report = RegistryActivationReport(
        "a" * 64, 29240, 8455, 20785, 0, 0, len(catalog.ingredients), 0, 0, 0
    )
    path = tmp_path / "catalog.json.gz"
    digest = write_runtime_catalog(
        path, catalog, report, {"reference_molecules": 29240}, wheel_sha256="b" * 64
    )
    restored, restored_report, stats = load_runtime_catalog(
        path,
        expected_sha256=digest,
        expected_wheel_sha256="b" * 64,
        expected_registry_sha256="a" * 64,
    )
    assert [asdict(item) for item in restored.ingredients] == [
        asdict(item) for item in catalog.ingredients
    ]
    assert restored.metadata == catalog.metadata
    assert restored_report == replace(report, connected_catalog_rows=47, experimental_formula_candidates=34, active_odorant_candidates=34)
    assert stats["reference_molecules"] == 29240
    assert isinstance(restored.ingredients[0].aliases, tuple)
    with pytest.raises(ValueError, match="binding"):
        load_runtime_catalog(
            path,
            expected_sha256=digest,
            expected_wheel_sha256="c" * 64,
            expected_registry_sha256="a" * 64,
        )
    path.write_bytes(path.read_bytes() + b"corruption")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != digest
    with pytest.raises(ValueError, match="hash mismatch"):
        load_runtime_catalog(
            path,
            expected_sha256=digest,
            expected_wheel_sha256="b" * 64,
            expected_registry_sha256="a" * 64,
        )


def test_no_phase_markers_preserve_empty_phase_contract():
    brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(
        "clean fresh woody musk"
    )
    assert brief.phase_target_profiles == {}


@pytest.mark.parametrize("text", [
    "opening no sweetness, drydown woody musk",
    "rose fragrance, opening no rose",
])
def test_perfume_negative_only_phase_does_not_inherit_a_positive_target(text):
    brief = NaturalLanguageBriefParser(IngredientCatalog.load_builtin()).parse(text)
    opening = dict(brief.phase_target_profiles["opening"])
    targets = TemporalMixtureSimulator.targets_by_time(brief)
    for target, desired, avoided in targets[:2]:
        assert not target.any()
        assert desired == []
        assert avoided == sorted(set(brief.avoided_dimensions) | set(brief.phase_avoided_dimensions["opening"]))
    assert targets[-1][0].sum() > 0
    assert brief.phase_target_profiles["opening"] == opening


def test_local_phase_exclusion_does_not_erase_other_phase_target():
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    brief = parser.parse("opening citrus, drydown woody without citrus")
    assert brief.phase_target_profiles["opening"]["citrus"] == 1
    assert "citrus" in brief.phase_avoided_dimensions["drydown"]
    assert "citrus" in brief.desired_dimensions
    assert "citrus" not in brief.avoided_dimensions
    negative_only = parser.parse("opening no sweetness, drydown woody musk")
    assert negative_only.phase_avoided_dimensions["opening"] == ["gourmand"]
    assert sum(negative_only.phase_target_profiles["opening"].values()) == 0
    with pytest.raises(ValueError, match="충돌"):
        parser.parse("without sweetness; opening citrus, drydown vanilla")


@pytest.mark.parametrize("text", ["citrus opening, woody drydown", "시트러스 첫향, 우디 잔향", "opening citrus, woody drydown", "citrus opening and woody drydown"])
def test_postfix_phase_names_do_not_reverse_targets(text):
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    brief = parser.parse(text)
    assert brief.phase_target_profiles["opening"]["citrus"] == 1
    assert brief.phase_target_profiles["drydown"]["woody"] == 1


@pytest.mark.parametrize("global_text", [
    "no musk; opening citrus, drydown woody",
    "no musk; citrus opening, woody drydown",
    "no musk; citrus opening and woody drydown",
    "머스크 없이; 시트러스 첫향, 우디 잔향",
])
def test_global_exclusions_and_single_phase_local_exclusions_stay_separate(global_text):
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    global_ban = parser.parse(global_text)
    assert all("musky" in values[2] for values in TemporalMixtureSimulator.targets_by_time(global_ban))
    local_ban = parser.parse("rose fragrance, opening no rose")
    targets = TemporalMixtureSimulator.targets_by_time(local_ban)
    assert "rose" in local_ban.desired_dimensions
    assert "rose" not in local_ban.avoided_dimensions
    assert "rose" in targets[0][2]
    assert "rose" not in targets[-1][2]
