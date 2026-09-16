"""Recipe-independent prior innovations shared by search and forward physics.

Changing a dose must not redraw every uncertain molecular property. Stable
per-ingredient streams couple the same latent scenarios across compositions,
support permutations, batches and draw counts. Priors and equations do not
change, and the streams do not use request text, desired odor or recipe scores.
"""
from functools import lru_cache
import hashlib

import numpy as np

VERSION = 'ingredient-coupled-prior-sampling/v1'
MAX_CACHED_DRAWS = 2048


def material_seed(identifier):
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('nonempty ingredient identity required')
    # Preserve the original analytic dose optimizer's innovations exactly.
    return int(hashlib.sha256(('dose-prior-search-1:'+identifier).encode()).hexdigest()[:16], 16)


@lru_cache(maxsize=128)
def _material_draws(identifier, draws):
    output = np.random.default_rng(material_seed(identifier)).normal(size=(draws, 3))
    output.setflags(write=False)
    return output


def material_draws(identifier, draws):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError('positive integer draw count required')
    if draws <= MAX_CACHED_DRAWS:
        return _material_draws(identifier, draws)
    output = np.random.default_rng(material_seed(identifier)).normal(size=(draws, 3))
    output.setflags(write=False)
    return output


def suppression_draws(draws):
    return np.clip(np.random.default_rng(98173).normal(.20, .05, draws), .08, .40)


def coupled_blocks(identifiers, draws, block_size=64):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1 or block_size < 1:
        raise ValueError('positive draw and block counts required')
    if len(set(identifiers)) != len(identifiers) or not identifiers:
        raise ValueError('unique nonempty ingredient identities required')
    if draws <= MAX_CACHED_DRAWS:
        rows = [material_draws(key, draws) for key in identifiers]
        strengths = suppression_draws(draws)
        for start in range(0, draws, block_size):
            stop = min(draws, start+block_size)
            yield start, np.stack([row[start:stop] for row in rows], axis=1), strengths[start:stop]
    else:
        # Large user-requested integrations stream bounded blocks, not a
        # draw_count x entire_catalog tensor held in RAM.
        generators = [np.random.default_rng(material_seed(key)) for key in identifiers]
        suppression = np.random.default_rng(98173)
        for start in range(0, draws, block_size):
            size = min(block_size, draws-start)
            noise = np.stack([generator.normal(size=(size, 3)) for generator in generators], axis=1)
            yield start, noise, np.clip(suppression.normal(.20, .05, size), .08, .40)
