"""Run enhanced search only after an unchanged baseline yields no usable recipe.

Successful baseline objects are returned verbatim: no second computation, no
alias correction, no result rewriting, and no case-ID or benchmark lookup.
The context is request-local, so a failed request cannot change other requests.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from functools import wraps
import math
from time import monotonic

_ACTIVE = ContextVar('perfumery_failure_recovery', default=False)
_DOSE_CORRECTION = ContextVar('perfumery_dose_correction', default=False)
_DOSE_BASELINE = ContextVar('perfumery_dose_baseline', default=None)
VERSION = 'failed-only-recovery/v87'
_DEADLINE=ContextVar('perfumery_request_deadline',default=None)
# Leave 30 seconds inside the existing 300-second web function for request
# preparation, exact final validation and serialization outside the core.
TOTAL_REQUEST_SECONDS=270.


def request_seconds_remaining():
    deadline=_DEADLINE.get()
    return None if deadline is None else max(0.,deadline-monotonic())


@contextmanager
def repair_budget(product,reserve=0.):
    from .search_budget import ACTIVE,SearchBudget
    deadline=_DEADLINE.get()
    remaining=180. if deadline is None else deadline-monotonic()-reserve
    if remaining<1.:
        yield False
        return
    budget=SearchBudget(product,total_seconds=min(120.,remaining))
    token=ACTIVE.set(budget)
    try:
        yield True
    finally:
        ACTIVE.reset(token)


def excluded_by_representation(value):
    pe=_get(value,'perceptual_evaluation',{}) or {}
    if pe.get('profile_coverage',{}).get('target_excluded') is True:
        return True
    def inspect(item):
        if isinstance(item,dict):
            if item.get('target_excluded_by_representation') is True:
                return True
            return any(inspect(v) for v in item.values() if isinstance(v,(dict,list)))
        if isinstance(item,list):
            return any(inspect(v) for v in item if isinstance(v,(dict,list)))
        return False
    return any(inspect(_get(value,name,{})) for name in ('full_profile_assessment','perception_guidance','score_contract'))


def recovery_active():
    return _ACTIVE.get()


def dose_correction_active():
    return _DOSE_CORRECTION.get()


def dose_correction_seed():
    baseline = _DOSE_BASELINE.get()
    return _get(baseline, 'closest_candidate', ()) if baseline is not None else ()


@contextmanager
def dose_correction_context(baseline=None):
    token = _DOSE_CORRECTION.set(True)
    seed_token = _DOSE_BASELINE.set(baseline)
    try:
        with recovery_context():
            yield
    finally:
        _DOSE_BASELINE.reset(seed_token)
        _DOSE_CORRECTION.reset(token)


@contextmanager
def recovery_context():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _score(value):
    result = _get(value, 'calculated_profile_similarity')
    if result is None:
        result = _get(value, 'score')
    return float(result) if isinstance(result, (int, float)) and not isinstance(result, bool) and math.isfinite(result) else None


def _needs_recovery(value):
    # Preserve any existing usable result, including non-strict legacy modes.
    if _get(value, 'recipe'):
        return False
    if excluded_by_representation(value):
        return False
    passed = _get(value, 'full_profile_target_met', _get(value, 'profile_target_met', False))
    if passed is True or not _get(value, 'closest_candidate'):
        return False
    return _score(value) is not None


def _annotate(value, detail):
    if isinstance(value, dict):
        return {**value, 'failure_recovery': detail}
    result = copy(value)
    result.score_contract = {**(_get(value, 'score_contract') or {}), 'failure_recovery': detail}
    return result


def perfume_recovery_available(baseline, engine, *args, **kwargs):
    from .hierarchical_perfume import active
    return active(engine.perception_guidance, baseline.brief)


def failed_only_recovery(product, *, eligibility=None):
    if product not in ('perfume', 'body_lotion'):
        raise ValueError('unsupported recovery product')
    def decorate(function):
        @wraps(function)
        def v87_result(*args, **kwargs):
            if recovery_active():
                return function(*args, **kwargs)
            baseline = function(*args, **kwargs)
            if not _needs_recovery(baseline):
                return baseline
            if eligibility is not None and not eligibility(baseline, *args, **kwargs):
                return baseline
            retry_kwargs = dict(kwargs)
            if product == 'body_lotion':
                from .alias_consistency import EQUIVALENTS
                from .catalog import find_text_spans
                request = args[0] if args else kwargs.get('request')
                text = _get(request, 'brief', '')
                alias_correction = isinstance(text, str) and any(find_text_spans(text, alias) for alias in EQUIVALENTS)
                if not alias_correction:
                    return baseline
            # Keep cancellation/error observers alive during recovery. The SSE
            # transport already suppresses repeated or decreasing progress.
            with repair_budget(product,reserve=45.) as allowed:
                if not allowed:
                    return baseline
                with recovery_context():
                    candidate = function(*args, **retry_kwargs)
            before, after = _score(baseline), _score(candidate)
            usable = bool(_get(candidate, 'recipe'))
            selected = usable or (after is not None and after > before+1e-8)
            detail = {'version': VERSION, 'product': product,
                'trigger': 'baseline_has_no_usable_recipe_and_has_scored_closest_candidate',
                'baseline_score': before, 'recovery_score': after,
                'baseline_status': _get(baseline, 'status'), 'recovery_status': _get(candidate, 'status'),
                'recovery_selected': selected, 'baseline_usable_results_bypass_recovery': True,
                'request_body_changed': False, 'benchmark_case_lookup_used': False,
                'core_evaluations': 2}
            return _annotate(candidate if selected else baseline, detail)

        @wraps(function)
        def recover(*args, **kwargs):
            # The complete V87 path, including its recovered successes, runs
            # first. V88 never takes computation away from that successful path.
            baseline = v87_result(*args, **kwargs)
            if recovery_active() or not _needs_recovery(baseline):
                return baseline
            if eligibility is not None and not eligibility(baseline, *args, **kwargs):
                return baseline
            retry_kwargs = dict(kwargs)
            if product == 'body_lotion':
                reference = _get(baseline, 'product_model', {}) or {}
                if reference.get('evaluation_version') not in ('lotion-observed-exposure/v2','lotion-observed-exposure/v3'):
                    return baseline
                # Warm-start from actual ingredient amounts, never old scores.
                retry_kwargs['_incumbent_recipe'] = _get(baseline, 'closest_candidate')
            with repair_budget(product) as allowed:
                if not allowed:
                    return baseline
                with dose_correction_context(baseline):
                    candidate = function(*args, **retry_kwargs)
            before, after = _score(baseline), _score(candidate)
            selected = bool(_get(candidate, 'recipe')) or (after is not None and after > before+1e-8)
            previous = (_get(baseline, 'failure_recovery') or
                        (_get(baseline, 'score_contract', {}) or {}).get('failure_recovery'))
            detail = {'version':'failed-only-dose-correction/v88', 'product':product,
                'trigger':'complete_v87_path_has_no_usable_recipe',
                'baseline_score':before, 'recovery_score':after, 'recovery_selected':selected,
                'previous_recovery':previous, 'score_offset':0.,
                'core_evaluations':(previous or {}).get('core_evaluations',1)+1,
                'baseline_usable_results_bypass_recovery':True,
                'request_body_changed':False, 'benchmark_case_lookup_used':False}
            return _annotate(candidate if selected else baseline, detail)

        @wraps(function)
        def wrapped(*args, **kwargs):
            if _DEADLINE.get() is not None:
                return recover(*args, **kwargs)
            token=_DEADLINE.set(monotonic()+TOTAL_REQUEST_SECONDS)
            try:
                return recover(*args, **kwargs)
            finally:
                _DEADLINE.reset(token)
        return wrapped
    return decorate
