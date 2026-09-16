"""Exercise actual whole-catalog public-data wiring and explicit diagnostic APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    prepared = json.loads(args.preparation.read_text(encoding='utf-8'))
    os.environ.update(PERFUMERY_AI_ENV='research',PERFUMERY_AI_LOCAL_PROFILE=prepared['profile'])
    args.output.mkdir(parents=True,exist_ok=False)
    from fastapi.testclient import TestClient
    from deploy.shared_runtime_v69 import create_release_app
    from scripts.audit_modal_v69_full import Audit, require
    language_calls = []
    def backend(message):
        language_calls.append(message)
        raise AssertionError('literal request must not invoke language worker')
    with TestClient(create_release_app(language_backend=backend),raise_server_exceptions=False) as client:
        a = Audit(args.output,client,{},local=True)
        health = a.call('health','GET','/health')[0]
        require(health['wheel_sha256'] == prepared['wheel_sha256'],'wrong candidate package')
        status = a.call('status','GET','/v2/evidence/status')[0]
        require(status['public_sources_registered'] and not status['operator_bundle_registered'],'wrong evidence class')
        require(status['coverage']['active_material_count'] == 3830,'partial denominator')
        rows = []
        for offset in range(0,3830,500):
            value = a.call(f'coverage_{offset}','GET',f'/v2/evidence/coverage?offset={offset}&limit=500')[0]
            rows.extend(value['items'])
        require(len(rows) == len({row['ingredient_id'] for row in rows}) == 3830,'missing or duplicate material coverage')
        require(sum(row['public_price_connected'] for row in rows) == 243,'wrong source coverage')
        assistant = a.post('literal_assistant','/v1/ai/assistant',{'message':'머스크 없이 장미 향 바디로션으로 만들어줘'})
        require(not language_calls and assistant['source'] == 'deterministic_complete_literal_intent','unnecessary language call')
        metaphor = a.post('english_wood','/v1/briefs/prepare',{'formula':{'brief':'the last smoke of a dying campfire over charred timber'}})
        require(metaphor['intent']['target_profile']['woody'] > 0,'wood intent lost')
        policy = {'finished_batch_mass_g':1000.,'maximum_lead_time_days':10,'maximum_purchase_cost_usd':100.}
        request = {'lines':[{'ingredient_id':'phenethyl_alcohol','concentrate_percent':100.}],
                   'target_region':'EU','product_category':'body_lotion','product_concentration_percent':2.,
                   'max_formula_cost_per_kg':180.,'policy':policy}
        assessment = a.post('actual_public_assessment','/v2/formulas/assess-evidence',request)
        require(not assessment['gate_passed'],'public facts were promoted to operating approval')
        frameworks = assessment['public_source_screen']['frameworks']
        require([row['id'] for row in frameworks] == ['IFRA','EU_REACH','K_REACH','FDA'],'missing framework')
        require(any(row.get('source_limit_exceeded', False) for row in frameworks[0]['findings']),'supplier lotion limit not checked')
        previous = status['contract']['public_sources']['previous_versions'][-1]
        impact = a.post('actual_observation_change','/v2/formulas/change-impact',{**request,'previous_evidence_version':previous})
        require(impact['review_required'] and not impact['state_changed'],'public change implied approval')
        formula = {'brief':'피오니와 청사과 향','max_risk_tier':2,'enable_registry_trace_candidates':True,
                   'product_category':'eau_de_parfum','target_region':'EU',
                   'product_concentration_percent':15.,'max_formula_cost_per_kg':180.}
        original = {'request':{'formula':formula},'evidence_policy':policy}
        reviewed = a.post('operating_prepare','/v2/briefs/prepare',original)
        a.post('operating_still_blocked','/v2/formulas/evaluate',{**original,'confirmed_review_id':reviewed['review_id']},expected=(422,))
        diagnostic = {**original,'diagnostic_only':True}
        checked = a.post('diagnostic_prepare','/v2/briefs/prepare',diagnostic)
        a.post('mode_change_requires_reconfirmation','/v2/formulas/evaluate',{**diagnostic,'confirmed_review_id':reviewed['review_id']},expected=(409,))
        result = a.post('actual_diagnostic_evaluate','/v2/formulas/evaluate',{**diagnostic,'confirmed_review_id':checked['review_id']})
        require(result['diagnostic_only'] and not result['candidates'] and result['diagnostic_candidates'],'wrong diagnostic gate')
        candidate = result['diagnostic_candidates'][0]['result']
        lines = [{'ingredient_id':row['ingredient_id'],'concentrate_percent':row['concentrate_percent']}
                 for row in (candidate.get('recipe') or candidate['closest_candidate'])]
        fixed = {**diagnostic,'lines':lines}
        fixed_review = a.post('fixed_diagnostic_prepare','/v2/briefs/prepare',fixed)
        reassessed = a.post('actual_diagnostic_reassess','/v2/formulas/reassess',{**fixed,'confirmed_review_id':fixed_review['review_id']})
        require(not reassessed['candidates'] and reassessed['diagnostic_candidates'],'fixed public assessment implied approval')
        a.save('report.json',{'passed':all(row['passed'] for row in a.calls),'calls':len(a.calls),
            'active_materials_checked':len(rows),'public_price_connected':243,'operator_approved':0,
            'four_frameworks_connected':True,'literal_llm_calls':len(language_calls),
            'profile_sha256':hashlib.sha256(Path(prepared['profile']).read_bytes()).hexdigest(),
            'deployment_changed':False})
        print((args.output/'report.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
