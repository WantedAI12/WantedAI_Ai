"""Source-observed, full-odor references for post-application lotion design.

The target is frozen before recipe search and independent of the candidate
catalogue. These are conditional human descriptor profiles, NOT measured
lotion targets or a calibrated probability of user satisfaction.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .lotion_atlas import ATLAS_PROJECTION
from .profile_match import compare_profiles

VERSION = 'lotion-observed-reference/v1'
MODE = 'observed_reference'
# Named language routing, not fabricated additional measured endpoints.
ROUTES = {**ATLAS_PROJECTION,
    'fresh': ('COOL,COOLING', 'LIGHT'),
    'fatty': ('OILY, FATTY',), 'tropical': ('PINEAPPLE', 'BANANA', 'COCONUT'),
    'nutty': ('NUTTY, WALNUT ETC.', 'ALMOND', 'PEANUT BUTTER'),
    'vegetable': ('FRESH GREEN VEGETABLES',), 'roasted': ('COFFEE', 'BURNT,SMOKY'),
    'caramellic': ('CARAMEL',), 'winey': ('OAK WOOD,COGNAC',),
    'pineapple': ('PINEAPPLE',), 'musty': ('MUSTY, EARTHY, MOLDY',),
    'pungent': ('SHARP, PUNGENT, ACID',), 'creamy': ('BUTTERY, FRESH BUTTER',),
    'minty': ('MINTY, PEPPERMINT',), 'phenolic': ('DISINFECTANT, CARBOLIC',),
    'burnt': ('BURNT,SMOKY',), 'camphoreous': ('CAMPHOR',), 'honey': ('HONEY',),
    'melon': ('CANTALOUPE, HONEYDEW MELON',), 'buttery': ('BUTTERY, FRESH BUTTER',),
    'metallic': ('METALLIC',), 'leafy': ('HERBAL, GREEN,CUTGRASS',),
    'animal': ('ANIMAL',), 'cocoa': ('CHOCOLATE',), 'caraway': ('CARAWAY',),
    'banana': ('BANANA',), 'coconut': ('COCONUT',), 'rooty': ('RAW POTATO',),
    'malty': ('MALTY',), 'lemon': ('LEMON',), 'orange': ('ORANGE',),
    'grapefruit': ('GRAPEFRUIT',), 'lavender': ('LAVENDER',),
    'vanilla': ('VANILLA',), 'cedar': ('CEDARWOOD',), 'cinnamon': ('CINNAMON',)}
DETAIL_ALIASES = {'lemon': ('lemon', '레몬'), 'orange': ('orange', '오렌지'),
    'grapefruit': ('grapefruit', '자몽'), 'lavender': ('lavender', '라벤더'),
    'vanilla': ('vanilla', '바닐라'), 'cedar': ('cedarwood', 'cedar', '시더우드'),
    'cinnamon': ('cinnamon', '시나몬', '계피')}


def exposure_groups(brief, rows, schedule=None):
    """Define fixed intent windows and legacy display-quadrature diagnostics.

    Unscoped briefs describe the whole exposure. Explicit phase requirements
    get separate windows. Trapezoidal weights are retained for compatibility,
    but V65 search uses analytic transport integrals over these window bounds,
    NOT these sparse display weights. Both are independent of recipe outcomes.
    """
    times = np.asarray([0., *[r['minutes'] for r in rows]],float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('ordered positive final-curve times required')
    end = float(times[-1])
    opening = 15. if schedule is None else schedule.opening_until_minutes
    heart = 240. if schedule is None else schedule.heart_until_minutes
    explicit = set(brief.phase_target_profiles) | set(brief.phase_avoided_dimensions)
    windows = [('overall',0.,end,0)]
    if explicit:
        windows = []
        for phase,start,stop in (('opening',0.,opening),('heart',opening,heart),('drydown',heart,end)):
            if phase not in explicit and not brief.avoided_dimensions:
                continue
            indices = [i for i,r in enumerate(rows) if r['phase'] == phase]
            stop = min(stop,end)
            if not indices or stop <= start:
                if phase in explicit:
                    raise ValueError('requested phase has no modeled exposure interval')
                continue
            windows.append((phase,start,stop,indices[0]))
    groups = []
    for phase,start,stop,index in windows:
        weights = np.zeros(len(times))
        for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
            left,right = max(a,start),min(b,stop)
            if right <= left:
                continue
            toward_right = ((right-a)**2-(left-a)**2)/(2*(b-a))
            weights[i] += right-left-toward_right
            weights[i+1] += toward_right
        groups.append({'phase':phase,'window_start_minutes':start,'minutes':stop,
            'weights':weights[1:]/(stop-start),'target_index':index,
            'aggregation':'trapezoidal_airborne_odor_activity_exposure_not_human_intensity'})
    return groups


def normalize(values):
    x = np.asarray(values, float)
    if not np.isfinite(x).all() or np.any(x < 0) or np.any(x.sum(axis=-1) <= 0):
        raise ValueError('finite nonnegative nonempty reference profiles required')
    return x / x.sum(axis=-1, keepdims=True)


def fit_references(rows, endpoints):
    """Fixed weighted conditional means. No recipe outcomes or predictions.

    Weighting by squared observed descriptor fraction favors exemplars with
    the named characteristic without deleting their other observed notes.
    Both source measurement heads remain separate sensitivity scenarios.
    """
    raw = np.asarray([[r[head] for r in rows] for head in ('applicability', 'use')])
    shapes = normalize(raw)
    profiles, metadata = {}, {}
    for concept, names in ROUTES.items():
        ids = [endpoints.index(name) for name in names]
        salience = shapes[0, :, ids].sum(axis=0)
        weights = salience**2
        positive = weights > 0
        effective = float(weights.sum()**2 / np.square(weights).sum()) if weights.sum() else 0.
        groups = {r['graph'] or r['id'] for r, ok in zip(rows, positive) if ok}
        if effective < 3 or len(groups) < 3:
            continue
        weights /= weights.sum()
        profiles[concept] = normalize(np.einsum('r,hrd->hd', weights, shapes)).tolist()
        metadata[concept] = {'source_endpoint_names': list(names), 'positive_stimuli': int(positive.sum()),
            'effective_stimuli': effective, 'distinct_identity_groups': len(groups),
            'semantic_route_is_project_authored': True}
    return {'profiles': profiles, 'concept_metadata': metadata,
        'background': normalize(shapes.mean(axis=1)).tolist()}


class ObservedReferenceBank:
    def __init__(self, path, sha256):
        self.path, self.sha256 = Path(path), sha256
        raw = self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('lotion target reference hash mismatch')
        v = json.loads(raw)
        if v.get('schema') != VERSION or v.get('recipe_outcomes_used') is not False:
            raise ValueError('invalid independently frozen lotion target references')
        self.endpoints = tuple(v['endpoints'])
        self.profiles = {k: normalize(x) for k, x in v['profiles'].items()}
        self.background = normalize(v['background'])
        self.metadata = v['concept_metadata']
        self.parent_sha256 = v['parent_atlas_sha256']
        if len(set(self.endpoints)) != len(self.endpoints) or not self.profiles:
            raise ValueError('invalid reference endpoint identity')
        if any(x.shape != (2, len(self.endpoints)) for x in [self.background, *self.profiles.values()]):
            raise ValueError('reference profile shape mismatch')

    def assert_current(self):
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError('lotion target reference changed during request')

    def contract(self):
        return {'version': VERSION, 'evaluation_version': 'lotion-observed-exposure/v2', 'product': 'body_lotion',
            'prediction_model': 'finite_dose_ow_film_transport_with_atlas_reference_shapes',
            'aggregation': 'exposure_integral_or_explicit_phase_integrals_then_min_over_scenarios_and_reference_heads',
            'profile_representation': 'learned_146_axis_observed_reference_not_19_axis_purity',
            'cross_product_scores_comparable': False, 'matrix_calibrated': False,
            'reference_sha256': self.sha256,
            'target_basis': 'conditional_observed_full_146_descriptor_profiles',
            'predicted_basis': 'transport_weighted_learned_146_descriptor_mixture',
            'recipe_outcomes_used_to_define_target': False, 'unmentioned_notes_forced_to_zero': False,
            'absolute_intensity_calibrated': False, 'lotion_mixture_interactions_calibrated': False,
            'human_similarity_percent': None, 'legacy_scores_directly_comparable': False}

    def targets(self, brief, target_rows):
        from .brief_parser import _is_negated, _match_weight
        from .odor_descriptors import load_builtin_odor_descriptor_lexicon
        lexicon = {r.descriptor: r for r in load_builtin_odor_descriptor_lexicon().descriptors}
        output, unsupported = [], set()
        for row in target_rows:
            phase = row['phase']
            concepts = {k: v for k, v in row['target_profile'].items() if v > 0}
            detailed = list(brief.phase_recognized_descriptors.get(phase, brief.recognized_descriptors))
            negative_details = list(brief.phase_avoided_descriptors.get(phase, brief.avoided_descriptors))
            # For phase briefs use parser-owned scoped descriptors; never spread
            # an opening-only lemon descriptor into the requested drydown.
            text = brief.phase_brief_texts.get(phase, '' if brief.phase_target_profiles else brief.original_text).casefold()
            if text:
                for concept, aliases in DETAIL_ALIASES.items():
                    for alias in aliases:
                        pattern = re.escape(alias) if not alias.isascii() else r'(?<![a-z])'+re.escape(alias)+r'(?![a-z])'
                        for match in re.finditer(pattern, text):
                            (negative_details if _is_negated(text, *match.span()) else detailed).append(concept)
            detailed = sorted(set(detailed))
            # Replace only the detailed descriptor's coarse projection. Do not
            # count "pineapple" as three independent invented requirements.
            detail_profiles, factors = {}, {}
            for concept in detailed:
                detail_profiles[concept] = lexicon[concept].profile if concept in lexicon else {
                    'lemon': {'citrus': 1}, 'orange': {'citrus': 1}, 'grapefruit': {'citrus': 1},
                    'lavender': {'aromatic': 1}, 'vanilla': {'gourmand': 1},
                    'cedar': {'woody': 1}, 'cinnamon': {'spicy': 1}}.get(concept, {})
                aliases = lexicon[concept].aliases if concept in lexicon else DETAIL_ALIASES.get(concept, (concept,))
                matches = [m for alias in aliases for m in re.finditer(re.escape(alias), text)]
                factors[concept] = max((_match_weight(text, *m.span()) for m in matches), default=1.)
            # Partition shared coarse mass jointly. Sequential replacement or
            # a minimum per-detail weight would silently turn 90:10 into 2:1.
            consumed = {axis for p in detail_profiles.values() for axis in p}
            fine_weights = dict.fromkeys(detailed, 0.)
            for axis in consumed:
                total = sum(p.get(axis, 0.)*factors[c] for c,p in detail_profiles.items())
                for c,p in detail_profiles.items():
                    fine_weights[c] += concepts.get(axis, 0.)*p.get(axis, 0.)*factors[c]/total
            for axis in consumed:
                concepts.pop(axis, None)
            for concept, weight in fine_weights.items():
                if weight <= 0:
                    unsupported.add('unresolved_detail_weight:'+concept)
                else:
                    concepts[concept] = weight
            unsupported.update(set(concepts)-set(self.profiles))
            negatives = sorted(set(row['avoided']) | set(negative_details))
            unsupported.update(set(negatives)-set(ROUTES))
            available = {k: v for k, v in concepts.items() if k in self.profiles}
            if not available:
                output.append(None)
                continue
            vector = normalize(sum(self.profiles[k]*w for k, w in available.items()))
            avoid = sorted({name for k in negatives for name in ROUTES.get(k, ())})
            # Explicit bans remove only the explicitly banned endpoint mass
            # from the target. Companion notes otherwise remain intact.
            if avoid:
                vector[:, [self.endpoints.index(k) for k in avoid]] = 0.
                vector = normalize(vector)
            output.append({'profiles': vector, 'concepts': concepts, 'avoided': avoid,
                'source': 'observed_conditional_reference_not_measured_user_target'})
        if brief.target_profile_source == 'explicit_structured_relative_weights':
            unsupported.add('explicit_19_axis_profile_requires_legacy_profile_mode')
        if brief.intensity != 'medium':
            unsupported.add('requested_absolute_intensity_not_calibrated')
        if not brief.phase_target_profiles and brief.temporal_emphasis != {'opening':.25,'heart':.40,'drydown':.35}:
            unsupported.add('requested_persistence_not_calibrated')
        return output, sorted(unsupported)

    def compare(self, target, predicted, head):
        wanted = dict(zip(self.endpoints, target['profiles'][head]))
        actual = normalize(predicted)
        result = compare_profiles(wanted, actual, avoided=target['avoided'], dimensions=self.endpoints).to_dict()
        cosine = result['cosine_score']/100
        background = self.background[head]
        bg_cosine = float(np.dot(background, actual)/(np.linalg.norm(background)*np.linalg.norm(actual)))
        # A broad mean profile must not win just because every scent shares
        # common descriptors. This guard does not rescale the agreement score.
        result.update(version=VERSION, score_kind='observed_reference_profile_agreement_not_user_similarity',
            background_cosine_score=100*bg_cosine,
            reference_more_specific_than_background=cosine > bg_cosine+1e-8)
        return result


@lru_cache(maxsize=4)
def _load(path, digest, size, mtime):
    return ObservedReferenceBank(path, digest)


def load_configured_reference_bank():
    from .local_runtime import local_profile
    profile = local_profile()
    pair = profile.get('lotion_target_reference') if profile else None
    if pair is None:
        return None
    path, digest = pair
    stat = Path(path).stat()
    bank = _load(path, digest, stat.st_size, stat.st_mtime_ns)
    bank.assert_current()
    return bank


def configured_reference(request, predictor):
    if request.evaluation_mode == 'legacy_profile':
        return None
    bank = load_configured_reference_bank()
    if bank is None:
        if request.evaluation_mode == MODE:
            raise ValueError('observed reference mode requires a pinned local reference bank')
        return None
    if request.evaluation_mode == 'auto' and (request.target_profile is not None or request.phase_target_profiles is not None):
        return None
    if (predictor is None or tuple(predictor.provider.endpoints) != bank.endpoints
            or predictor.provider.component_model_sha256 != bank.parent_sha256):
        raise ValueError('observed references require their pinned Atlas component predictor')
    return bank
