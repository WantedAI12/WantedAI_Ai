"""Export project-authored instruction examples; never fit sensory model weights."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fragrance_ai.platform.lotion_reference import lotion_reference
from fragrance_ai.recommender.formulation_workflow import formulation_workflow, knowledge_contract


def curriculum():
    context = lotion_reference()['application_context']
    context['fragrance_concentration_percent'] = .5
    cases = [
        ('perfume_roles', '향수 조향 과정을 알려줘', {'product_type': 'perfume'}),
        ('lotion_missing_base', '로션 만드는 과정을 알려줘', {'product_type': 'body_lotion'}),
        ('reference_batch', '이 CCT 로션 400 g의 공정과 계량을 정리해줘',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'batch_mass_g': 400}}),
        ('scale_not_time', '같은 로션 2 kg이면 혼합 시간도 다섯 배인가?',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'batch_mass_g': 2000}}),
        ('wrong_hot_method', '이 Simulgel EG 기준 로션을 가열 공정으로 처리해줘',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'method': 'hot'}}),
        ('manual_not_shear', '같은 로션을 손으로 저으면 동일한가?',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'mixer_kind': 'manual'}}),
        ('ph_not_stability', 'pH 5.5이면 이 로션의 안정성과 보존력이 입증되나?',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'measured_ph': 5.5}}),
        ('ph_outside_component', 'PE 9010 포함 로션의 측정 pH가 2인 경우 검토해줘',
         {'product_type': 'body_lotion', 'application_context': context, 'process': {'measured_ph': 2}}),
    ]
    changed = deepcopy(context)
    changed['emulsion_type'] = 'water_in_oil'
    cases.append(('wrong_emulsion', 'W/O 로션에 같은 공정을 쓸 수 있나?',
                  {'product_type': 'body_lotion', 'application_context': changed}))
    changed = deepcopy(context)
    changed['base_components'][0]['mass_percent'] -= 1
    changed['base_components'].append({'name': 'Sodium Lactate', 'role': 'humectant', 'mass_percent': 1})
    cases.append(('electrolyte', '이 Simulgel 로션에 sodium lactate를 추가했을 때 검토해줘',
                  {'product_type': 'body_lotion', 'application_context': changed}))
    for identifier, instruction, request in cases:
        plan = formulation_workflow(request)
        # Link the shared knowledge once; do not duplicate supplier summaries
        # into every instruction or turn them into generated sensory labels.
        target = {key: plan[key] for key in ('status', 'selected_method', 'batch',
            'manufacturing_approved', 'human_similarity_percent', 'process_constraints_satisfied')}
        target['check_codes'] = [row['code'] for row in plan['checks']]
        target['step_ids'] = [row['id'] for row in plan['steps']]
        target['source_ids'] = [row['id'] for row in plan['sources']]
        yield {'id': identifier, 'instruction': instruction, 'input': request,
            'output': target, 'knowledge_sha256': plan['knowledge']['sha256'],
            'evidence_kind': 'synthetic_instruction_target_not_measurement',
            'numeric_sensory_training_eligible': False, 'holdout_evaluation_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve an existing curriculum; use a new output directory')
    rows = list(curriculum())
    payload = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows).encode('utf-8')
    args.output.mkdir(parents=True)
    (args.output / 'instructions.jsonl').write_bytes(payload)
    report = {'schema_version': 'formulation-curriculum/1', 'records': len(rows),
        'instructions_sha256': hashlib.sha256(payload).hexdigest(), 'knowledge': knowledge_contract(),
        'workflow_source_sha256': hashlib.sha256((ROOT / 'fragrance_ai/recommender/formulation_workflow.py').read_bytes()).hexdigest(),
        'training_executed': False, 'measured_observations_added': 0,
        'scope': 'instruction curriculum for grounded procedural tasks; not new sensory data or independent evaluation'}
    (args.output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
