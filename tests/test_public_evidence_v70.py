from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from fragrance_ai.platform.public_evidence import PublicEvidenceStore
from fragrance_ai.platform.rd_evidence import EvidenceStore, EvidenceAssessment
from fragrance_ai.recommender.regulatory_status import regulatory_summary
from tests.test_rd_contract import data as data, client_for, reviewed_request  # noqa: F401


@pytest.fixture
def public(tmp_path):
    documents = []
    for label in ('SUPPLIER', 'IFRA', 'EU_REACH', 'K_REACH', 'FDA'):
        path = tmp_path/(label+'.txt')
        raw = ('explicit test fixture '+label).encode()
        path.write_bytes(raw)
        documents.append({'id': label, 'path': path.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                          'url': 'https://example.invalid/'+label, 'retrieved_at': (datetime.now(timezone.utc)-timedelta(days=1)).isoformat()})
    material = {'ingredient_ids': ['phenethyl_alcohol'], 'cas_number': '60-12-8', 'supplier': 'fixture', 'sku': 'fixture',
                'source_id': 'SUPPLIER', 'price_per_kg': 70., 'frameworks_verified': [],
                'available_kg': None, 'lead_time_days': None, 'minimum_order_kg': None,
                'supplier_ifra_limits': [{'product_category': 'body_lotion', 'source_id': 'IFRA', 'maximum_finished_product_percent': 1.}]}
    previous = deepcopy(material)
    previous['price_per_kg'] = 60.
    body = {'schema_version': 'public-regulatory-supply-1', 'version': 'current',
            'scope': 'public_source_facts_not_operator_reviewed_evidence', 'manufacturing_approval': False,
            'documents': documents, 'materials': [material], 'missing_operator_fields': ['quantitative_available_stock'],
            'previous_snapshots': [{'version': 'old', 'documents': documents, 'materials': [previous]}]}
    path = tmp_path/'public.json'
    path.write_text(json.dumps(body), encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return PublicEvidenceStore(path, digest), path, digest


def test_public_facts_reach_all_four_tabs_without_fabricating_approval(public, monkeypatch):
    store, path, digest = public
    monkeypatch.setenv('PERFUMERY_AI_PUBLIC_EVIDENCE_PATH', str(path))
    monkeypatch.setenv('PERFUMERY_AI_PUBLIC_EVIDENCE_SHA256', digest)
    from tests.test_regulatory_status import payload
    item = payload()
    item['brief']['constraints'].update(product_category='body_lotion', product_concentration_percent=2.)
    item['recipe'] = [{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}]
    result = regulatory_summary(item)
    assert all(tab['source_evidence_connected'] for tab in result['tabs'][:4])
    checked = result['public_source_screen']
    assert checked['frameworks'][0]['findings'][0]['source_limit_exceeded']
    assert not checked['gate_passed'] and not any(row['compliance_verified'] for row in checked['frameworks'])
    assert checked['materials'][0]['source_observations'][0]['available_kg'] is None


def test_changed_raw_document_is_rejected(public):
    store, path, _ = public
    (path.parent/'SUPPLIER.txt').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        store.assert_current()


def test_real_prior_public_snapshot_comparison_preserves_missing_inventory(public):
    public_store, _, _ = public
    store = EvidenceStore(public_store=public_store)
    from fragrance_ai.recommender.catalog import IngredientCatalog
    request = EvidenceAssessment(lines=[{'ingredient_id': 'phenethyl_alcohol', 'concentrate_percent': 100.}],
        target_region='EU', product_category='eau_de_parfum', product_concentration_percent=15.,
        max_formula_cost_per_kg=180., policy={'finished_batch_mass_g': 1000., 'maximum_lead_time_days': 10, 'maximum_purchase_cost_usd': 100.})
    result = store.change_impact(request, IngredientCatalog.load_builtin(), 'old')
    assert result['affected_material_count'] == 1 and result['review_required']
    assert not result['state_changed'] and not result['after']['gate_passed']
    assert result['changes'][0]['previous'][0]['price_per_kg'] == 60.
    with pytest.raises(ValueError, match='unknown'):
        store.change_impact(request, IngredientCatalog.load_builtin(), 'invented-history')


def test_public_data_requires_explicit_diagnostic_mode_and_cannot_approve(data, public):
    catalog, _, _, _, _ = data
    store = EvidenceStore(public_store=public[0])
    client, calls = client_for(catalog, store)
    with client:
        body = reviewed_request()
        reviewed = client.post('/v2/briefs/prepare', json=body).json()
        blocked = client.post('/v2/formulas/evaluate', json={**body, 'confirmed_review_id': reviewed['review_id']})
        assert blocked.status_code == 422 and not calls
        body['diagnostic_only'] = True
        diagnostic = client.post('/v2/briefs/prepare', json=body).json()
        assert diagnostic['review_id'] != reviewed['review_id'] and diagnostic['diagnostic_only']
        evaluated = client.post('/v2/formulas/evaluate', json={**body, 'confirmed_review_id': diagnostic['review_id']})
        assert evaluated.status_code == 200, evaluated.text
        result = evaluated.json()
        assert not result['candidates'] and result['diagnostic_candidates']
        assert result['diagnostic_only'] and not result['manufacturing_approval']
        assert not result['diagnostic_candidates'][0]['recommendation_allowed']
        assert client.get('/v2/evidence/status').json()['public_sources_registered']
