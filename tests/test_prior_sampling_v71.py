import hashlib

import numpy as np
import pytest

from fragrance_ai.recommender.prior_sampling import coupled_blocks, material_draws, suppression_draws, VERSION
from fragrance_ai.recommender.dose_refinement import DoseModel
from fragrance_ai.recommender.science import TemporalMixtureSimulator
from tests.test_dose_refinement import setup_case


def collect(identities, draws, block_size=64):
    blocks = list(coupled_blocks(identities, draws, block_size))
    return np.concatenate([block[1] for block in blocks]), np.concatenate([block[2] for block in blocks])


@pytest.mark.parametrize('draws', [32, 200, 257, 2100])
def test_identity_samples_are_order_subset_prefix_and_batch_invariant(draws):
    all_rows, strength = collect(['a','b','c'], draws)
    other, other_strength = collect(['c','a'], draws, 17)
    np.testing.assert_array_equal(all_rows[:, [2, 0]], other)
    np.testing.assert_array_equal(strength, other_strength)
    np.testing.assert_array_equal(all_rows[:, 0], material_draws('a', draws))
    short, _ = collect(['a'], 23)
    np.testing.assert_array_equal(all_rows[:23, :1], short)


def test_search_prior_samples_are_exactly_preserved():
    seed = int(hashlib.sha256(b'dose-prior-search-1:example').hexdigest()[:16], 16)
    np.testing.assert_array_equal(material_draws('example', 32), np.random.default_rng(seed).normal(size=(32,3)))
    np.testing.assert_array_equal(suppression_draws(32), np.clip(np.random.default_rng(98173).normal(.20,.05,32),.08,.40))
    with pytest.raises(ValueError):
        material_draws('example', 32)[0, 0] = 5.


def test_forward_physics_matches_search_on_the_same_latent_scenarios():
    items, brief, lines, _ = setup_case()
    engine = TemporalMixtureSimulator()
    prepared = engine._prepare(lines, {i.ingredient_id:i for i in items}, {})
    interaction = engine._interaction_matrix(prepared)
    actual = engine._sampled_temporal_arrays(prepared, interaction, engine.targets_by_time(brief), 200,
                                            np.random.default_rng(0), coupled=True)
    model = DoseModel(items, {}, brief.constraints.product_concentration_percent, draws=200)
    by_id = {line.ingredient_id: line.concentrate_percent for line in lines}
    state = model.evaluate(np.asarray([by_id[item.ingredient_id] for item in items])/100.)
    np.testing.assert_allclose(actual[1].mean(0), state.signal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[2].mean(0), state.temporal, rtol=1e-12, atol=1e-12)


def test_tiny_dose_changes_do_not_redraw_the_entire_physics_prior():
    items, brief, lines, _ = setup_case()
    from dataclasses import replace
    changed = [replace(line, concentrate_percent=line.concentrate_percent+delta,
        finished_product_percent=line.finished_product_percent+delta*brief.constraints.product_concentration_percent/100)
        for line, delta in zip(lines, [1e-4, -1e-4, 0.])]
    engine, catalog = TemporalMixtureSimulator(), {i.ingredient_id:i for i in items}
    before = engine.evaluate(lines, catalog, brief, {}, draws=200)
    after = engine.evaluate(changed, catalog, brief, {}, draws=200)
    repeated = engine.evaluate(lines, catalog, brief, {}, draws=200)
    assert before == repeated
    assert before.uncertainty_sampling_version == VERSION
    for a, b in zip(before.temporal_points, after.temporal_points):
        keys = set(a.scent_profile)|set(b.scent_profile)
        assert sum(abs(a.scent_profile.get(k,0)-b.scent_profile.get(k,0)) for k in keys) < 1e-4
        assert abs(a.total_relative_intensity-b.total_relative_intensity) < 1e-4


def test_explicit_seed_keeps_legacy_sampling_mode():
    items, brief, lines, _ = setup_case()
    value = TemporalMixtureSimulator().evaluate(lines, {i.ingredient_id:i for i in items}, brief, {}, draws=64, seed=55)
    assert value.uncertainty_sampling_version == 'legacy_explicit_seed_stream'


@pytest.mark.parametrize('identities,draws', [(['a','a'],20),([],20),(['a'],0),(['a'],True)])
def test_invalid_sampling_contract_is_rejected(identities, draws):
    with pytest.raises(ValueError):
        list(coupled_blocks(identities, draws))
