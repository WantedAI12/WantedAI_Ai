"""Source-pinned open odor identities and one target compiler for all products.

Lexical coverage, annotation support and full-profile reference support are
different facts. Unknown detail never inherits an ancestor's measured status.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = 'hierarchical-odor-space/v77'
COMPILER = 'hierarchical-target-compiler/v77'


class OdorSpace:
    def __init__(self, path, digest):
        self.path, self.sha256 = Path(path), digest
        raw = self.path.read_bytes()
        if len(raw) > 15_000_000 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('odor space hash/size mismatch')
        self.value = json.loads(raw)
        v = self.value
        if v.get('schema') != SCHEMA or v.get('recipe_outcomes_used') is not False:
            raise ValueError('invalid source-only odor space')
        self.rows = {r['id']: r for r in v['concepts']}
        if len(self.rows) != len(v['concepts']):
            raise ValueError('duplicate odor identity')
        self.aliases = v['aliases']
        if any(k not in self.rows for k in self.aliases.values()):
            raise ValueError('alias without odor identity')

    def contract(self):
        rows = list(self.rows.values())
        result = {'version': SCHEMA, 'target_compiler': COMPILER, 'sha256': self.sha256,
            'concepts': len(rows), 'aliases': len(self.aliases),
            'source_reference_connected': sum(r['reference_key'] is not None for r in rows),
            'compositional_reference_connected': len(self.value.get('compositional_bindings',{})),
            'annotation_model_connected': sum(r['source_annotation_model_connected'] for r in rows),
            'contextual_vocabulary_only': sum(bool(r.get('contextual_only')) for r in rows),
            'reference_missing': sum(r['reference_key'] is None for r in rows)-len(self.value.get('compositional_bindings',{})),
            'source_languages': sorted({r.get('language', 'en/ko') for r in rows}),
            'coverage_scope': self.value['coverage_scope'], 'supports_composition': True,
            'unmentioned_notes_forced_to_zero': False, 'unknown_details_silently_replaced_by_family': False,
            'new_measured_intensity_axes': False, 'source': self.value['source'],
            'base_source':self.value.get('base_source'), 'base_attribution':self.value.get('base_attribution'),
            'worldwide_complete_odor_inventory_claimed': False}
        if self.value.get('resolution_extension'):
            result['reference_resolution']=self.value['resolution_extension']
            result['data_version']=self.value['version']
            result['reference_counts_include_labelled_model_estimates']=True
        if self.value.get('release_scope'):
            result['generation_scope'] = self.value['release_scope']['summary']
            result['inactive_generation_targets_retained_for_audit'] = True
        return result


@lru_cache(maxsize=4)
def _load(path, digest, size, mtime):
    return OdorSpace(path, digest)


def load_space(path, digest):
    stat = Path(path).stat()
    return _load(str(path), digest, stat.st_size, stat.st_mtime_ns)


def configured_odor_space():
    from .local_runtime import local_profile
    profile = local_profile()
    pair = profile.get('odor_space') if profile else None
    return load_space(*pair) if pair else None


def contextual_spans(text, alias):
    """Historical/general words require an explicit smell naming context."""
    if alias.casefold() not in text.casefold():
        return []
    escaped = re.escape(alias)
    patterns = (r'(?<!\w)'+escaped+r'(?=\s+(?:scent|smell|odor|odour)\b)',
                r'(?<=향: )'+escaped+r'(?!\w)')
    return sorted({m.span() for p in patterns for m in re.finditer(p, text, re.I)})


def unresolved_named_odors(text, matches):
    """Keep explicitly named, unknown scents visible, not all ordinary prose."""
    occupied = [(m['start'], m['end']) for m in matches]
    result = []
    patterns = (r'(?P<name>[A-Za-z][A-Za-z-]{2,})\s+(?:scent|smell|odou?r)\b',
                r'(?P<name>[가-힣]{2,})\s*(?:향|냄새)(?!수)')
    generic = {'desired','target','a','the','any','fragrance','perfume','original','원하는','좋은','어떤','이런','그런','전체','같은'}
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.I):
            start, end = m.span('name')
            word = m.group('name')
            if word.casefold() in generic or any(a < end and start < b for a,b in occupied):
                continue
            from .brief_parser import _is_negated
            result.append({'text': word, 'start': start, 'end': end, 'status': 'unresolved_named_odor',
                'polarity': 'avoid' if _is_negated(text.casefold(), start, end) else 'want'})
    return result


def compile_targets(bank, brief, target_rows):
    """Compile before search; never inspect recipe, candidates or predictions."""
    space = bank.odor_space
    output, unsupported = [], set()
    for row in target_rows:
        phase = row['phase']
        expression = brief.phase_expressions.get(phase, {})
        wanted = dict(brief.expression_targets)
        wanted.update(expression.get('wanted', {}))
        if phase == 'overall' and brief.phase_expressions:
            for name, local in brief.phase_expressions.items():
                weight = brief.temporal_emphasis.get(name, 0.)
                for concept, value in local.get('wanted', {}).items():
                    wanted[concept] = wanted.get(concept, 0.) + weight*value
        if any(not np.isfinite(v) or v < 0 for v in wanted.values()):
            raise ValueError('finite nonnegative odor intent weights required')
        negative = set(brief.expression_avoided) | set(expression.get('avoided', {}))
        inherited_phase_intent = False
        if not wanted and negative and brief.phase_expressions:
            # Same explicit contract as effective_phase_target: a prohibition-
            # only phase inherits the remaining overall positive intent. The
            # specific named notes must survive that inheritance as well.
            for name, local in brief.phase_expressions.items():
                for concept, value in local.get('wanted', {}).items():
                    if concept not in negative:
                        wanted[concept] = wanted.get(concept,0.) + value*brief.temporal_emphasis.get(name,0.)
            inherited_phase_intent = bool(wanted)
        # Only parser-owned explicit concepts consume family mass. No fixed
        # seven-note allowlist and no current recipe-dependent target fitting.
        concepts = {k: float(v) for k,v in wanted.items() if v > 0}
        controls = {}
        for concept in list(concepts):
            if concept in ('soft','subtle','strong','intense','diffusive','longlasting'):
                continue
            resolution=space.rows[concept].get('resolution',{})
            if resolution.get('semantic_role') == 'absence_condition':
                unsupported.add('absence_constraint_requires_product_background:'+concept)
            if resolution.get('semantic_role') in ('context','taxonomy','modifier'):
                # Keep the need for clarification explicit: the word is not a
                # measured smell merely because it appeared in a document.
                unsupported.add(resolution['status']+':'+concept)
        # These words already have a separate parser-owned physical/style
        # channel. Do not demand a new measured odor identity for "soft".
        for concept in ('soft','subtle','strong','intense','diffusive','longlasting'):
            if concept in concepts and not space.value['reference_bindings'].get(concept):
                controls[concept] = concepts.pop(concept)
        compositions = {}
        for concept, weight in list(concepts.items()):
            parts = space.value.get('compositional_bindings', {}).get(concept)
            if parts:
                concepts.pop(concept)
                compositions[concept] = list(parts)
                for part in parts:
                    concepts[part] = concepts.get(part, 0.) + weight/len(parts)
        consumed = {a for k in concepts for a in space.rows[k]['coarse_projection']}
        consumed.update(a for k in controls for a in space.rows[k]['coarse_projection'])
        # Rose/white-flower implies floral in the old display parser, not a
        # second independently requested reference with an invented weight.
        if consumed & {'rose','white_floral'}:
            consumed.add('floral')
        for family, weight in row['target_profile'].items():
            if weight > 0 and family not in consumed:
                concepts[family] = concepts.get(family, 0.) + float(weight)
        negative.update(row['avoided'])
        unsupported.update('conflicting_concept:'+c for c in set(concepts) & negative)
        if negative & consumed:
            unsupported.update('excluded_required_family:'+c for c in negative & consumed)
        resolved = {}
        for c, weight in concepts.items():
            ref = space.value['reference_bindings'].get(c)
            if ref is None and c in bank.profiles:
                ref = c
            if ref is None or ref not in bank.profiles:
                unsupported.add('reference_missing:'+c)
            else:
                resolved[ref] = resolved.get(ref, 0.) + weight
                if bank.metadata.get(ref, {}).get('complete_source_attribute_coverage') is False:
                    unsupported.add('source_reference_incomplete:'+c)
        avoided = set()
        for c in negative:
            names = space.value['endpoint_routes'].get(c)
            if not names:
                ref = space.value['reference_bindings'].get(c, c)
                names = bank.metadata.get(ref, {}).get('source_endpoint_names')
            if not names:
                unsupported.add('exclusion_endpoint_missing:'+c)
            else:
                avoided.update(names)
        if not resolved:
            output.append(None)
            continue
        for ref in resolved:
            core = set(space.value['endpoint_routes'].get(ref, bank.metadata.get(ref, {}).get('source_endpoint_names', ())))
            if core and core <= avoided:
                unsupported.add('excluded_required_reference:'+ref)
        vector = sum(bank.profiles[k]*w for k,w in resolved.items())
        for name in avoided:
            vector[:, bank.endpoints.index(name)] = 0.
        if np.any(vector.sum(-1) <= 0):
            unsupported.add('exclusions_remove_entire_reference')
            output.append(None)
            continue
        vector /= vector.sum(-1, keepdims=True)
        weights = {k:v/sum(resolved.values()) for k,v in resolved.items()}
        from .odor_resolution import reference_evidence
        output.append({'profiles': vector, 'concepts': concepts, 'reference_weights': weights,
            'avoided': sorted(avoided), 'source': 'frozen_hierarchical_source_reference_not_measured_user_target',
            'annotation_conditioned_concepts': [k for k in resolved if 'annotation' in bank.metadata.get(k, {}).get('reference_kind','')],
            'reference_identity_support': {k:bank.metadata.get(k,{}).get('distinct_identity_groups') for k in resolved},
            'reference_evidence':{k:reference_evidence(bank.metadata.get(k,{})) for k in resolved},
            'explicit_lexical_compositions': compositions,
            'separate_request_controls': controls,
            'positive_intent_inherited_for_prohibition_only_phase': inherited_phase_intent,
            'composition_weight_semantics':'relative_language_weights_not_recipe_mass_fractions',
            'target_compiler': COMPILER, 'space_sha256': space.sha256,
            'unmentioned_endpoints_forced_to_zero': False})
    unsupported.update(('unresolved_excluded_odor:' if r.get('polarity') == 'avoid' else
                        'unresolved_named_odor:')+r['text'] for r in brief.unresolved_odor_terms)
    if brief.target_profile_source == 'explicit_structured_relative_weights':
        unsupported.add('explicit_19_axis_profile_requires_legacy_profile_mode')
    for match in brief.expression_matches:
        if match.get('interpretation_alternatives', {}).get('requires_interpretation'):
            unsupported.add(('unresolved_excluded_odor:' if match.get('polarity') == 'avoid' else
                             'ambiguous_odor_interpretation:')+match['text'])
    return output, sorted(unsupported)


def target_coverage(targets, unsupported):
    """Missing positive detail limits a claim; it need not discard a candidate.

    Exclusions and contradictions cannot be relaxed into a partial match. An
    entirely unknown target/phase has no objective and is not fabricated.
    """
    hard_prefixes = ('conflicting_', 'excluded_', 'exclusion_endpoint_missing:',
        'exclusions_remove_', 'unresolved_excluded_odor:',
        'explicit_19_axis_profile_requires_', 'absence_constraint_requires_')
    hard = sorted(x for x in unsupported if x.startswith(hard_prefixes))
    available = bool(targets) and all(t is not None for t in targets)
    complete = available and not unsupported
    return {'searchable': available and not hard, 'complete': complete,
        'hard_blockers': hard, 'unresolved_requirements': list(unsupported),
        'score_scope': 'complete_requested_profile' if complete else 'resolved_positive_intent_only',
        'partial_score_may_satisfy_full_target': False}


def target_report(bank, brief, target_rows):
    targets, unsupported = compile_targets(bank, brief, target_rows)
    coverage = target_coverage(targets, unsupported)
    rendered = [{**r, 'profiles':r['profiles'].tolist()} if r else None for r in targets]
    identity = {'compiler': COMPILER, 'space_sha256': bank.odor_space.sha256,
        'reference_sha256': bank.sha256, 'targets': rendered, 'unsupported': unsupported}
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return {**identity, 'sha256': hashlib.sha256(raw).hexdigest(),
        'status': 'resolved_source_reference' if coverage['complete'] else 'partial_source_reference' if coverage['searchable'] else 'unresolved_reference',
        'coverage': coverage, 'searchable': coverage['searchable'],
        'interpretation_alternatives': [m['interpretation_alternatives'] for m in brief.expression_matches
            if m.get('interpretation_alternatives')],
        'recipe_outcomes_used': False, 'product_independent_intent': True,
        'background_policy': 'observed_companion_facets_retained_explicit_exclusions_only',
        'unknown_detail_policy': 'reported_missing_not_replaced_by_broad_parent'}
