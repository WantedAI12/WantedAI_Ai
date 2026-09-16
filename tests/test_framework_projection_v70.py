from copy import deepcopy
import pytest
from fragrance_ai.platform.rd_evidence import EvidenceAssessment
from tests.test_rd_contract import data as data  # noqa: F401


@pytest.mark.parametrize('framework,region', [('IFRA','EU'),('EU_REACH','EU'),('K_REACH','KR'),('FDA','US')])
@pytest.mark.parametrize('state,expected', [('supported','reviewed_supported'),('restricted','reviewed_with_limits'),('prohibited','blocked'),('unknown','not_assessed')])
def test_each_framework_is_assessed_not_just_displayed(data, framework, region, state, expected):
    catalog, factory, bundle, request, _ = data
    value = deepcopy(bundle)
    for snapshot in value['snapshots']:
        for row in snapshot['records']:
            row['region'] = region
            row['frameworks'] = dict.fromkeys(('IFRA','EU_REACH','K_REACH','FDA'),'supported')
            row['frameworks'][framework] = state
    request = {**request,'target_region':region}
    result = factory(value).assess(EvidenceAssessment.model_validate(request),catalog)
    checks = {row['id']:row for row in result['framework_checks']}
    assert checks[framework]['status'] == expected
    assert checks[framework]['registered_review_passed'] == (state in ('supported','restricted'))
    assert not checks[framework]['regulatory_certificate_issued']


def test_supply_price_failure_is_not_mislabelled_as_regulatory_failure(data):
    catalog, factory, bundle, request, _ = data
    bundle['snapshots'][-1]['records'][0]['price_per_kg'] = 500.
    result = factory(bundle).assess(EvidenceAssessment.model_validate(request),catalog)
    assert not result['gate_passed']
    assert all(row['registered_review_passed'] for row in result['framework_checks'] if row['required_for_region'])


def test_registered_review_cannot_hide_an_embedded_ifra_violation():
    from tests.test_regulatory_status import payload
    from fragrance_ai.recommender.regulatory_status import regulatory_summary
    value = payload()
    value['safety']['ifra_screen']['compliant'] = False
    evidence = {'gate_passed':True,'required_frameworks':['IFRA'],'blockers':[],
        'framework_checks':[{'id':'IFRA','status':'reviewed_supported','registered_review_passed':True}]}
    result = regulatory_summary(value,evidence_assessment=evidence)
    assert result['tabs'][0]['status'] == 'blocked' and result['status'] == 'blocked'
