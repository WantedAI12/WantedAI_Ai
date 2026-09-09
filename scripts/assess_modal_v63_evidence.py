"""Classify retained remote results; never disguise an offline check as HTTP."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_latency_v63 import differences
from scripts.verify_modal_v62 import verify_assistant_result

RELEASE_FIELDS = {
    '/deployment/wheel_sha256', '/deployment/catalog_snapshot/wheel_sha256',
    '/deployment/catalog_snapshot/catalog_sha256',
    '/deployment/catalog_snapshot/catalog_manifest_sha256',
}
TRACE_FIELDS = {
    '/candidate_variants_evaluated', '/ingredient_sets_evaluated',
    '/perception_guidance/exact_formula_evaluations',
    '/full_profile_assessment/search/full_pool_search/attempts',
    '/full_profile_assessment/search/full_pool_search/dose_refinement/attempts',
}


def assess_formula_equivalence(before, after):
    if set(before) != set(after):
        raise AssertionError('recipe response fields changed')
    changes = differences(before, after)
    counts = {'release_binding': 0, 'search_trace': 0, 'perception_roundoff': 0, 'fine_float32_roundoff': 0}
    maxima = {'perception': 0., 'fine': 0.}
    for row in changes:
        path = row['path']
        if path in RELEASE_FIELDS:
            counts['release_binding'] += 1
            continue
        if path in TRACE_FIELDS:
            if path.endswith('/attempts'):
                if not isinstance(row['before'], list) or not isinstance(row['after'], list):
                    raise AssertionError('invalid search trace shape')
            elif any(type(v) is not int or v < 0 for v in (row['before'], row['after'])):
                raise AssertionError('invalid search trace count')
            counts['search_trace'] += 1
            continue
        a, b = row['before'], row['after']
        numeric = (type(a) in (int, float) and type(b) in (int, float)
                   and math.isfinite(a) and math.isfinite(b))
        # Diagnostic model outputs may differ by roundoff between hosts. Their
        # labels/order/model binding stay exact, as do the final recipe and score.
        if path.startswith('/perception_guidance/') and numeric and abs(a-b) <= 1e-12:
            counts['perception_roundoff'] += 1
            maxima['perception'] = max(maxima['perception'], abs(a-b))
            continue
        if (re.fullmatch(r'/score_contract/fine_odor_expression/descriptors/\d+/value', path)
                and numeric and 0 <= a <= 1 and 0 <= b <= 1 and abs(a-b) <= 2**-23):
            counts['fine_float32_roundoff'] += 1
            maxima['fine'] = max(maxima['fine'], abs(a-b))
            continue
        raise AssertionError('non-diagnostic recipe change: ' + path)
    required = ('recipe', 'closest_candidate', 'calculated_profile_similarity',
                'full_profile_target_met', 'status', 'brief', 'safety',
                'temporal_profile', 'ingredient_temporal_profile', 'simulation_draws')
    for field in required:
        if field not in before or field not in after or before[field] != after[field]:
            raise AssertionError('required exact recipe output changed: ' + field)
    return {'required_decision_formula_and_temporal_fields_exact': True,
        'full_response_exact_except_release_hashes': all(row['path'] in RELEASE_FIELDS for row in changes),
        'difference_counts': counts, 'maximum_diagnostic_absolute_difference': maxima,
        'search_trace_differences_retained': counts['search_trace'] > 0,
        'changes': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new evidence assessment output required')
    def read(name):
        return json.loads((args.evidence / name).read_text(encoding='utf-8'))
    original = read('report.json')
    equivalence = assess_formula_equivalence(
        json.loads(args.previous.read_text(encoding='utf-8')), read('formula.json'))
    assert read('formula.json') == read('formula_repeat.json')
    verify_assistant_result(read('assistant.json'))
    assert read('assistant.json') == read('assistant_repeat.json')
    assert original['checks']['assistant_cache']['repeat_llm_calls'] == 0
    assert original['checks']['assistant_cache']['repeat_cache'] == 'hit'
    assert original['temporary_token_revoked']
    assert original['checks']['all_selected_models_connected']
    assert all(v == 200 for k, v in original['checks'].items() if k.endswith('_http_status'))
    assert original['checks']['unauthenticated_status'] in (401, 403)
    result = {'status': 'software_contracts_and_final_recipe_equivalence_verified',
        'evidence_kind': 'offline_assessment_of_retained_actual_authenticated_http_responses',
        'new_network_requests': 0, 'source_report': str(args.evidence / 'report.json'),
        'original_report_passed': original['passed'], 'original_report_preserved': True,
        'comparison': equivalence, 'assistant_repeat_body_exact': True,
        'assistant_repeat_llm_calls': 0, 'temporary_token_revoked': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({**result, 'comparison': {k: v for k, v in equivalence.items() if k != 'changes'}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
