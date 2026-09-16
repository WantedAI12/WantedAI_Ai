"""Distinguish diagnostic reference policy from explicit operating constraints."""
from pydantic import Field
from .rd_evidence import Input, EvidencePolicy


class ReviewEvidencePolicy(Input):
    finished_batch_mass_g: float | None = Field(default=None, strict=True, gt=0, le=1e12)
    maximum_lead_time_days: int | None = Field(default=None, strict=True, ge=0, le=3650)
    maximum_purchase_cost_usd: float | None = Field(default=None, strict=True, gt=0, le=1e12)


DIAGNOSTIC_REFERENCE = {'finished_batch_mass_g': 1000., 'maximum_lead_time_days': 3650,
                        'maximum_purchase_cost_usd': 1e12}
POLICY_QUESTIONS = {
    'finished_batch_mass_g': ('완제품 배치 질량', 'g', 1e12),
    'maximum_lead_time_days': ('최대 납기', 'day', 3650),
    'maximum_purchase_cost_usd': ('구매 예산 상한', 'USD', 1e12),
}


def resolve_policy(policy, diagnostic_only):
    values = policy.model_dump(mode='json', exclude_none=True)
    missing = [key for key in DIAGNOSTIC_REFERENCE if key not in values]
    if missing and not diagnostic_only:
        return None, missing, None
    context = None
    if missing:
        context = {'basis': 'diagnostic_reference_not_user_purchase_requirements',
                   'defaulted_fields': {key: DIAGNOSTIC_REFERENCE[key] for key in missing},
                   'operational_recommendation_allowed': False,
                   'scope': 'one_kg_reference_and_nonbinding_diagnostic_purchase_caps'}
        values = {**DIAGNOSTIC_REFERENCE, **values}
    return EvidencePolicy.model_validate(values), missing, context


def policy_question(key):
    label, unit, maximum = POLICY_QUESTIONS[key]
    path = 'evidence_policy.' + key
    return {'id': path, 'field': path, 'required': True, 'input_type': 'number',
            'unit': unit, 'maximum': maximum, 'reason_code': 'explicit_value_missing',
            'question': label + '를 입력해 주세요.'}
