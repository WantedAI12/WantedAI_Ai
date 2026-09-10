from copy import deepcopy

import pytest

from scripts.verify_modal_v62 import verify_assistant_result


def valid():
    proposal={'desired':['rose'],'avoided':['musky'],'product':'body_lotion','clarification':'none'}
    return {'source':'deterministic_grounding_after_model_mismatch','intent_proposal':{**proposal,'desired':['floral','rose']},
        'requires_confirmation':True,'formula_generated':False,'scientific_score_generated':False,
        'grounding':{'status':'explicit_parser_recovery','original_model_proposal':proposal}}


def test_valid_parent_category_recovery_proves_model_execution():
    assert verify_assistant_result(valid())


@pytest.mark.parametrize('kind',['model_timeout','missing_model_evidence','wrong_product','lost_exclusion','invented_formula','invented_score'])
def test_verifier_cannot_confuse_parser_fallback_or_wrong_semantics_with_success(kind):
    value=deepcopy(valid())
    if kind=='model_timeout':value['source']='deterministic_fallback_after_model_failure'
    elif kind=='missing_model_evidence':value['grounding'].pop('original_model_proposal')
    elif kind=='wrong_product':value['intent_proposal']['product']='perfume'
    elif kind=='lost_exclusion':value['intent_proposal']['avoided']=[]
    elif kind=='invented_formula':value['formula_generated']=True
    else:value['scientific_score_generated']=True
    with pytest.raises((AssertionError,KeyError)):verify_assistant_result(value)
