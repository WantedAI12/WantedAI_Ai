"""Request-scoped, cooperative quality/latency budgets for physical search.

These govern computation, never score definitions, target vectors or safety.
There is no universal golden ratio: allocation depends on problem size and
verified progress. Values below are a versioned policy, not a quality claim.
"""
from contextvars import ContextVar
from dataclasses import dataclass,field
from functools import wraps
from time import monotonic
import math

ACTIVE = ContextVar('perfumery_search_budget',default=None)


@dataclass
class SearchBudget:
    product: str
    total_seconds: float = 180.
    started: float = field(default_factory=monotonic)
    calls: dict = field(default_factory=dict)
    best_scores: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.product not in ('perfume','body_lotion') or isinstance(self.total_seconds,bool)
                or not math.isfinite(self.total_seconds) or not 0<self.total_seconds<=240):
            raise ValueError('valid product and positive search budget at most 240 seconds required')

    @property
    def remaining(self):
        return max(0.,self.total_seconds-(monotonic()-self.started))

    def limit(self,stage,default,materials=0):
        from .failure_recovery import recovery_active
        preferred = {
            'linear':min(4.,1.5+materials/1000),
            'linear_retry':8.,'nominal_seed':10.,'cone':24.,
            'corrective':12.,'profile_cuts':18.,'feedback':24.,
            'feedback_corrective':5.,'fractional_inverse':16.,'physical_polish':30. if recovery_active() else 18.,
            'quantization':2.,
        }.get(stage,default)
        value = max(.001,min(preferred,self.remaining))
        record = self.calls.setdefault(stage,{'allocations':0,'maximum_slice_seconds':0.})
        record['allocations'] += 1
        record['maximum_slice_seconds'] = max(record['maximum_slice_seconds'],value)
        return value

    def observe(self,stage,score):
        if score is not None:
            self.best_scores[stage] = max(score,self.best_scores.get(stage,-float('inf')))

    def report(self):
        from .failure_recovery import recovery_active
        return {'version':'balanced-search-budget/v87' if recovery_active() else 'balanced-search-budget/v82','product':self.product,
            'request_budget_seconds':self.total_seconds,
            'budget_exhausted':self.remaining<=0,'cooperative_not_hard_preemption':True,
            'stages':self.calls,'verified_best_scores':self.best_scores,
            'score_threshold_changed':False}


def allowance(stage,default,materials=0):
    budget = ACTIVE.get()
    return budget.limit(stage,default,materials) if budget is not None else default


def exhausted():
    budget = ACTIVE.get()
    return budget is not None and budget.remaining<=0


def governed(product):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            if ACTIVE.get() is not None:
                return function(*args,**kwargs)
            # A multi-base design may enter several governed solvers. They
            # share the outer request deadline rather than each receiving a
            # fresh 180-second allowance.
            from .failure_recovery import request_seconds_remaining
            remaining=request_seconds_remaining()
            budget = SearchBudget(product,total_seconds=180. if remaining is None else max(.001,min(180.,remaining)))
            token = ACTIVE.set(budget)
            try:
                result = function(*args,**kwargs)
                score = result.get('score') if isinstance(result,dict) else getattr(result,'calculated_profile_similarity',None)
                budget.observe('returned_profile',score)
                report = budget.report()
                if isinstance(result,dict):
                    result.setdefault('material_column_search',{})['work_budget'] = report
                elif getattr(result,'score_contract',None) is not None:
                    result.score_contract['search_work_budget'] = report
                return result
            finally:
                ACTIVE.reset(token)
        return wrapped
    return decorate
