"""Transparent full-suite plus targeted error-replay evidence; not a new full run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from compare_lotion_benchmarks import read_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    fm, fs, full = read_run(args.full)
    rm, rs, replay = read_run(args.replay)
    error_ids = {key for key, row in full.items() if row['status'] == 'error'}
    if len(full) != 400 or set(replay) != error_ids:
        raise ValueError('replay must cover every original error, without extra or omitted requests')
    for key in ('target', 'pool', 'auto_base_design', 'adaptive_oil_refinement_steps', 'perception_model'):
        if fm[key] != rm[key]:
            raise ValueError('replay changed evaluation settings: '+key)
    original_cases = {row['id']: row for row in fm['cases']}
    if any(row != original_cases[row['id']] for row in rm['cases']):
        raise ValueError('replay changed a request')
    before, after = fm['source_sha256'], rm['source_sha256']
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    if changed != ['fragrance_ai/recommender/accord_trials.py']:
        raise ValueError('this targeted replay only supports the isolated accord roundoff fix')
    combined = {**full, **replay}
    failed = [row for row in combined.values() if not row.get('profile_target_met')]
    reasons = Counter('execution_error' if row['status'] == 'error' else
        'current_static_profile_upper_below_95' if (row.get('profile_coverage') or {}).get('target_excluded') else
        'search_incomplete' if row.get('search_incomplete') else
        'transport_composition_or_search_limit' for row in failed)
    report = {
        'schema': 'full-suite-with-targeted-repair-evidence/v1',
        'scope': '400-request initial run plus all error-case replays; not a fresh final-source full400 run',
        'initial_complete_run': fs, 'targeted_replay': rs, 'changed_source_files': changed,
        'same_final_source_full400_rerun': False,
        'unchanged_model_data_target_and_nonaccord_source': True,
        'requests_accounted_for': len(combined), 'requests_recomputed_after_fix': len(replay),
        'requests_retained_from_initial_run': len(full)-len(replay),
        'combined_passed95': sum(bool(row.get('profile_target_met')) for row in combined.values()),
        'combined_failed95': len(failed),
        'combined_execution_errors': sum(row['status'] == 'error' for row in combined.values()),
        'combined_failure_reason_counts': dict(reasons),
        'human_accuracy_measured': False, 'deployed': False,
        'replayed_cases': [replay[key] for key in sorted(replay)],
        'case_evidence': [{'id': key, 'score': row.get('score'), 'status': row['status'],
            'profile_target_met': bool(row.get('profile_target_met')),
            'source': 'after_roundoff_fix_replay' if key in replay else 'before_roundoff_fix_complete_run'}
            for key, row in sorted(combined.items())],
        'input_sha256': {str(folder/name): hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for folder in (args.full, args.replay) for name in ('manifest.json','summary.json','results.jsonl')}}
    report['combined_pass_rate_percent'] = report['combined_passed95']/len(combined)*100
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in (
        'initial_complete_run','targeted_replay','replayed_cases','case_evidence','input_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
