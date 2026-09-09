"""Open, compositional odor language beside the fixed quantitative backbone.

The registry contains source identities, qualities and taxonomy nodes. None
of its binary annotations is reinterpreted as a measured intensity axis.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .catalog import find_text_spans
from .odor_descriptors import exclusive_odor_spans

PATH = Path(__file__).resolve().parents[1]/'data/odor_expression_v61.json'


def registry():
    stat = PATH.stat()
    return _registry(stat.st_size,stat.st_mtime_ns)


@lru_cache(maxsize=2)
def _registry(size, mtime):
    raw = PATH.read_bytes()
    value = json.loads(raw)
    if value.get('schema') != 'odor-expression-registry/v1' or value.get('quantitative_axes_changed') is not False:
        raise ValueError('invalid odor expression registry')
    rows, aliases = {}, {}
    for row in value['concepts']:
        key = row['id']
        if key in rows or row['kind'] not in ('odor', 'quality', 'family'):
            raise ValueError('duplicate/invalid expression identity')
        rows[key] = row
        for alias in row['aliases']:
            if not alias or alias in aliases:
                raise ValueError('ambiguous expression alias: '+alias)
            aliases[alias] = key
    return {'payload': value, 'rows': rows, 'aliases': aliases, 'sha256': hashlib.sha256(raw).hexdigest()}


def expression_contract():
    r = registry()
    rows = list(r['rows'].values())
    return {'version': 'odor-expression-v61', 'registry_sha256': r['sha256'],
        'canonical_concepts': len(rows), 'odor_concepts': sum(x['kind']=='odor' for x in rows),
        'quality_concepts': sum(x['kind']=='quality' for x in rows),
        'family_nodes': sum(x['kind']=='family' for x in rows), 'aliases': len(r['aliases']),
        'annotated_concepts': sum(bool(x['source_terms']) for x in rows),
        'quantitative_backbone_axes': 146, 'quantitative_axes_changed': False,
        'supports_composition': True, 'unknown_terms_guaranteed_understood': False,
        'claim_boundary': r['payload']['claim_boundary'],
        'attribution': r['payload']['attribution'], 'source_repository': r['payload']['source']['repository']}


def language_extensions(existing_aliases):
    """New aliases only; the original material projection registry stays fixed."""
    r = registry()
    return {alias: r['rows'][key] for alias, key in r['aliases'].items()
            if alias not in existing_aliases and r['rows'][key]['coarse_projection']
            and r['rows'][key]['kind'] != 'family'}


def parse_expression(text, *, excluded_material_spans=()):
    from .brief_parser import _is_negated, _match_weight, _list_scopes
    r = registry()
    lowered = text.casefold()
    spans = {a: find_text_spans(lowered, a) for a in r['aliases']}
    scopes = _list_scopes(lowered, {span for values in spans.values() for span in values})
    spans = exclusive_odor_spans(spans, set(r['aliases']))
    numeric = {}
    for values in spans.values():
        for start,end in values:
            left = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*%\s*$',lowered[:start])
            right = re.match(r'\s*(\d+(?:\.\d+)?)\s*%',lowered[end:])
            if left and not re.search(r'(?:농도|concentration|intensity)\s*$',lowered[:left.start()]):
                numeric[(start,end)] = float(left.group(1))/100.
            elif right:
                numeric[(start,end)] = float(right.group(1))/100.
    # Paired descriptor percentages are relative intent weights. A solitary
    # product concentration is never assumed to be a descriptor proportion.
    if len(numeric)<2:
        numeric = {}
    positive, negative, matches = {}, {}, []
    for alias, values in spans.items():
        key = r['aliases'][alias]
        for start, end in values:
            if any(left <= start and end <= right and (start,end)!=(left,right)
                   for left,right in excluded_material_spans):
                continue
            negated = _is_negated(lowered, start, end, scopes)
            weight = numeric.get((start,end),float(_match_weight(lowered, start, end, scopes)))
            if weight<0 or weight>1 and (start,end) in numeric:
                raise ValueError('descriptor percentage must be in [0,100]')
            destination = negative if negated else positive
            destination[key] = max(destination.get(key, 0.), weight)
            matches.append({'concept_id': key, 'text': text[start:end], 'start':start,
                'end':end, 'polarity':'avoid' if negated else 'want', 'weight':weight,
                'kind': r['rows'][key]['kind'],
                'support': 'source_annotation' if r['rows'][key]['source_terms'] else 'vocabulary_only'})
    # Conflicting clauses remain visible. Do not silently erase either one.
    return {'version': 'odor-expression-v61', 'wanted':positive, 'avoided':negative,
        'matches':sorted(matches, key=lambda x:x['start']),
        'conflicting_concepts':sorted(set(positive)&set(negative)),
        'composition_semantics':'separate_concepts_with_scopes_not_new_measured_compound_odor',
        'unknown_text_is_not_a_zero_odor':True}


def brief_expression(brief):
    return {'version':'odor-expression-v61', 'wanted':brief.expression_targets,
        'avoided':brief.expression_avoided, 'matches':brief.expression_matches,
        'phases':brief.phase_expressions, 'human_similarity_percent':None}


def expression_utility(items, brief, model=None):
    """Small search preference, not an acceptance score or material permission.

    Exact linked source support is used where present. Missing annotations use
    the separate molecular head, not zero-valued sensory absence. No source is
    allowed to activate a blocked/unavailable material.
    """
    requested = {k:v for k,v in brief.expression_targets.items()
                 if registry()['rows'][k]['kind']=='odor'}
    avoided = {k:v for k,v in brief.expression_avoided.items()
               if registry()['rows'][k]['kind']=='odor'}
    for expression in brief.phase_expressions.values():
        # Phase-positive facets must survive initial candidate discovery. Local
        # phase exclusions are handled in temporal diagnostics, not as all-day bans.
        for key, weight in expression['wanted'].items():
            if registry()['rows'][key]['kind']=='odor':
                requested[key] = max(requested.get(key,0.), weight)
    if not requested and not avoided:
        return np.zeros(len(items)), {'status':'not_requested'}
    if model is None:
        from .fine_odor_model import configured_fine_odor
        model = configured_fine_odor()
    if model is None:
        return np.zeros(len(items)), {'status':'model_not_configured', 'unmodeled':sorted(set(requested)|set(avoided))}
    values, evidence = model.materials(items)
    missing = np.asarray([row['status']=='missing_or_multicomponent_structure' for row in evidence])
    index = {key:i for i,key in enumerate(model.endpoints)}
    utility = np.zeros(len(items))
    for terms, sign in ((requested,1.),(avoided,-1.)):
        denominator = sum(terms.values()) or 1.
        for key, weight in terms.items():
            if key in index:
                scores = values[:,index[key]].copy()
                # Unknown identity cannot win an avoidance request by looking
                # like a measured zero. This is an uncertainty penalty only.
                scores[missing] = 1. if sign<0 else 0.
                utility += sign*weight/denominator*scores
    return utility, {'status':'fine_annotation_search_preference',
        'unmodeled':sorted((set(requested)|set(avoided))-set(index)),
        'model_sha256':model.sha256, 'material_evidence':evidence,
        'prediction_correction_sha256':getattr(model,'prediction_correction_sha256',None),
        'missing_identity_avoidance_policy':'pessimistic_unknown_not_verified_absence',
        'acceptance_score_changed':False, 'human_similarity_percent':None}


def summarize_expression(values, endpoints, *, top_k=12, requested=None):
    if values is None:
        return {'status':'unavailable', 'descriptors':[]}
    x = np.asarray(values, float)
    if x.shape != (len(endpoints),) or not np.isfinite(x).all():
        raise ValueError('invalid fine expression vector')
    r = registry()['rows']
    wanted = set(requested or ())
    positions = sorted(range(len(x)), key=lambda i:(-x[i], endpoints[i]))[:top_k]
    positions = sorted(set(positions)|{i for i,k in enumerate(endpoints) if k in wanted}, key=lambda i:(-x[i],endpoints[i]))
    return {'status':'annotation_profile_proxy',
        'value_kind':'weighted_source_annotation_propensity_not_sensory_intensity',
        'descriptors':[{'concept_id':endpoints[i], 'label_en':r[endpoints[i]]['label_en'],
            'kind':r[endpoints[i]]['kind'], 'value':float(x[i]), 'requested':endpoints[i] in wanted} for i in positions],
        'unmodeled_requested':sorted(wanted-set(endpoints)), 'human_similarity_percent':None}


def recipe_expression(lines, brief, catalog):
    from .fine_odor_model import configured_fine_odor
    model = configured_fine_odor()
    if model is None or not lines:
        return {'status':'model_not_configured' if model is None else 'no_recipe',
                'intent':brief_expression(brief),'human_similarity_percent':None}
    known = {i.ingredient_id:i for i in catalog.ingredients}
    items = [known[line.ingredient_id] for line in lines]
    values,evidence = model.materials(items)
    w = np.asarray([line.concentrate_percent for line in lines])
    missing = any(row['status']=='missing_or_multicomponent_structure' for row in evidence)
    summary = summarize_expression(w@values/w.sum() if w.sum()>0 and not missing else None,
        model.endpoints,requested=set(brief.expression_targets)|set(brief.expression_avoided))
    return {**summary,'intent':brief_expression(brief),'material_evidence':evidence,
        'weighting':'concentrate_mass_proxy_not_finished_product_headspace',
        'model_sha256':model.sha256,'quantitative_acceptance_score_changed':False,
        'prediction_correction_sha256':getattr(model,'prediction_correction_sha256',None),
        'fine_expression_target_met':None,'fine_expression_similarity_calibrated':False}
