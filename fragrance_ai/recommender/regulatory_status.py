"""Evidence-backed UI projection, not a new regulatory approval engine.

Uses the already executed local IFRA screen. Other frameworks remain explicitly
unassessed until scoped, current substance and business evidence is connected.
No network calls or repeat inference are performed during serialization.
"""
from copy import deepcopy


SOURCES = {
    "IFRA": "https://ifrafragrance.org/using-the-standards",
    "EU_REACH": "https://echa.europa.eu/substances-restricted-under-reach",
    "K_REACH": "https://kreach.me.go.kr/repwrt/portal/chmclsRegistProcss.do",
    "FDA": "https://www.fda.gov/cosmetics/cosmetic-ingredients/fragrances-cosmetics",
}


def regulatory_summary(payload, *, evidence_assessment=None):
    safety = payload.get("safety") or {}
    constraints = payload["brief"]["constraints"]
    recipe = payload.get("recipe") or []
    candidate = payload.get("closest_candidate") or []
    subject = "returned_recipe" if recipe else "closest_candidate_only" if candidate else "no_formula"
    screen = safety.get("ifra_screen") or {}
    has_ifra = bool(screen) and subject != "no_formula"
    ifra_blocked = has_ifra and screen.get("compliant") is False
    tabs = [{
        "id": "IFRA", "label": "IFRA",
        "status": "blocked" if ifra_blocked else "partial_screen_only" if has_ifra else "not_assessed",
        "status_label": "내장 규칙 위반" if ifra_blocked else "부분 규칙 검사" if has_ifra else "미평가",
        "evidence_scope": "embedded_subset_not_full_IFRA_certification",
        "rule_version": screen.get("coverage", {}).get("rule_set"),
        "embedded_amendment": screen.get("coverage", {}).get("embedded_amendment"),
        "coverage": deepcopy(screen.get("coverage", {})),
        "findings": deepcopy(screen.get("details", [])),
        "standards_checked_on": safety.get("standards_checked_on"),
        "standards_review_due": safety.get("standards_review_due"),
        "required_evidence": ["현행 전체 IFRA 규칙과 제품 카테고리별 사용량 검토", "향료 혼합물의 적합성 선언 및 별도의 안전성 평가"],
        "note": "내장 부분 규칙 검사이며 IFRA 인증 또는 전체 규제 적합성 승인이 아닙니다.",
        "source_url": SOURCES["IFRA"],
    }]
    for identifier, label, requirements, note in (
        ("EU_REACH", "EU REACH", ["CAS/EC 식별자 및 혼합물 구성", "현행 제한·허가 조건과 용도 검토", "공급망 등록·면제 및 사업자별 의무 근거"],
         "현재 REACH 물질별 규칙·등록 근거가 연결되지 않았습니다. EU 알레르겐 표시 검사는 REACH 적합성 검사를 대체하지 않습니다."),
        ("K_REACH", "K-REACH", ["물질 식별자 및 국내 제조·수입 주체", "연간 제조·수입량과 용도", "등록·신고·면제 및 제한 조건의 해당 여부와 증빙"],
         "레시피만으로 사업자별 화평법 이행 여부를 판정할 수 없으며 관련 증빙이 아직 연결되지 않았습니다."),
        ("FDA", "FDA", ["제품의 의도된 용도 및 표시·광고 문구", "안전성 입증 및 제한 원료 검토", "해당하는 시설 등록·제품 리스팅 등 의무 검토"],
         "향수 등 화장품은 일반적으로 FDA 사전 승인 대상이 아닙니다. 'FDA 승인'으로 표시하지 않으며, 방향제 등은 제품 분류부터 별도 확인해야 합니다."),
    ):
        tabs.append({"id": identifier, "label": label, "status": "not_assessed", "status_label": "확인 필요",
                     "evidence_scope": "no_scoped_rule_or_registration_evidence_connected", "findings": [],
                     "required_evidence": requirements, "note": note, "source_url": SOURCES[identifier]})
    from ..platform.public_evidence import PublicEvidenceStore
    public = PublicEvidenceStore.configured()
    public_check = (evidence_assessment or {}).get('public_source_screen')
    if public_check is None and public and subject != 'no_formula':
        public_check = public.screen(recipe or candidate, category=constraints['product_category'],
                                     concentration=constraints['product_concentration_percent'])
    if public_check:
        for tab, checked in zip(tabs, public_check['frameworks']):
            tab['public_source_screen'] = checked
            if checked['status'] != 'not_assessed':
                tab['source_evidence_connected'] = True
                tab['required_evidence'] = list(dict.fromkeys(tab['required_evidence'] + ['공급 품목·배치·사업자 범위 확인']))
                if tab['status'] == 'not_assessed':
                    tab.update(status='reference_available', status_label='공개자료 연결·검토 필요')
                if checked.get('public_registry_check') and tab['status'] != 'blocked':
                    tab.update(status='public_registry_review_required', status_label='공개 물질목록 대조·검토 필요',
                        evidence_scope='public_registry_identity_and_findings_not_operating_approval')
                if checked.get('public_registry_check', {}).get('formula_rule_checks', {}).get('source_limit_exceeded'):
                    tab.update(status='blocked', status_label='공개 IFRA 기준 초과·검토 필요')
    if evidence_assessment is not None:
        reviewed_checks = {row['id']:row for row in evidence_assessment.get('framework_checks', [])}
        for tab in tabs:
            tab['registered_assessment'] = {
                'assessment_id': evidence_assessment.get('result_id'),
                'required_for_region': tab['id'] in evidence_assessment.get('required_frameworks', []),
                'gate_passed': evidence_assessment.get('gate_passed', False),
                'blockers': deepcopy(evidence_assessment.get('blockers', []))}
            checked = reviewed_checks.get(tab['id'])
            if checked:
                tab['framework_assessment'] = deepcopy(checked)
                if checked['status'] == 'blocked':
                    tab.update(status='blocked',status_label='등록 근거 검사 차단')
                elif checked['registered_review_passed'] and tab['status'] != 'blocked':
                    tab.update(status='registered_review_supported',status_label='등록 근거 검사 통과',
                        evidence_scope='scoped_registered_review_not_regulatory_certificate')
                elif checked['status'] == 'partially_reviewed':
                    tab.update(status='partial_screen_only',status_label='일부 원료 근거만 확인')
    registered_blocked = any(tab['status'] == 'blocked' for tab in tabs)
    status = "not_assessed" if subject == "no_formula" else "blocked" if registered_blocked or not safety.get("internal_gate_passed", False) else "review_required"
    tabs.append({"id": "REGULATORY_STATUS", "label": "규제 상태", "status": status,
                 "status_label": {"not_assessed": "미평가", "blocked": "검토 차단", "review_required": "검토 필요"}[status],
                 "internal_gate_passed": safety.get("internal_gate_passed", False),
                 "internal_blockers": list(safety.get("violations") or []),
                 "missing_documents": list(safety.get("missing_documents") or []),
                 "unassessed_frameworks": [row["id"] for row in tabs if row["status"] in ("not_assessed",'reference_available','public_registry_review_required')],
                 "commercial_release_authorized_by_this_summary": False})
    return {"schema_version": "regulatory-tabs-1", "subject": subject,
            "material_count": len(recipe or candidate), "target_region": constraints["target_region"],
            "product_category": constraints["product_category"],
            "product_concentration_percent": constraints["product_concentration_percent"],
            "status": status, "tabs": tabs, "live_regulatory_lookup_performed": False,
            **({'public_source_screen': public_check} if public_check else {})}
