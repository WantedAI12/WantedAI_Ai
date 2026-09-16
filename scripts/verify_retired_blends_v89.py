"""Replay stored compositions to verify the operator's ten exact exclusions.

This performs no model inference and never changes the benchmark denominator.
Prior response artifacts and their original pass/fail decisions stay intact.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--installed', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.installed.resolve() if args.installed else ROOT))
    from fragrance_ai.recommender.retired_blends import (
        RetiredBlendFilter, SOURCE_RECORDS, SOURCE_AUDIT_SHA256, POLICY_VERSION)
    audit_path = ROOT/'.benchmarks/v89_backend_handoff/failure-audit-final-03.json'
    assert sha(audit_path) == SOURCE_AUDIT_SHA256
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    latest = {}
    baseline = ROOT/'.benchmarks/v86_target90/full800-01'
    for line in (baseline/'results.jsonl').read_text(encoding='utf8').splitlines():
        row = json.loads(line)
        latest[row['product'], row['case_id']] = {
            **row, 'passed': row['target_met'], 'response_path': str(baseline/'responses'/row['response_file'])}
    assert len(latest) == 800
    for path in (ROOT/'.benchmarks/v87_failed_recovery/final-summary-01/report.json',
                 ROOT/'.benchmarks/v88_blend_correction/result-audit-02/report.json'):
        report = json.loads(path.read_text(encoding='utf8'))
        for row in report['cases']:
            latest[row['product'], row['case_id']] = row
    for row in audit['cases']:
        latest[row['product'], row['case_id']] = {
            **row, 'passed': row['target_met'], 'response_path': row['response_file']}
    configured = {(r['product'], r['case_id']): r for r in SOURCE_RECORDS}
    excluded, untouched = [], []
    for key, row in sorted(latest.items()):
        path = Path(row['response_path'])
        before = sha(path)
        if row.get('response_sha256'):
            assert before == row['response_sha256']
        payload = json.loads(gzip.decompress(path.read_bytes()))
        result = payload['response']
        lines = result.get('recipe') or result.get('closest_candidate') or []
        assert lines
        first = next(r for r in lines if r['concentrate_percent'] > 0)
        dose = round(first['finished_product_percent']/first['concentrate_percent']*100, 6)
        predicate = RetiredBlendFilter(key[0], dose)
        retired = predicate.reject_lines(lines)
        assert sha(path) == before
        item = {'product': key[0], 'case_id': key[1], 'formula_id': result['formula_id'],
                'source_response_sha256': before, 'original_passed': row['passed']}
        if retired:
            assert key in configured and not row['passed']
            assert before == configured[key]['source_response_sha256']
            excluded.append(item)
        else:
            assert key not in configured and row['passed']
            untouched.append(item)
    assert len(excluded) == 10 and len(untouched) == 790
    report = {'policy': POLICY_VERSION, 'recorded_request_results': 800,
              'excluded_recorded_blends': len(excluded), 'previously_passed_untouched': len(untouched),
              'score_changes': 0, 'ingredient_or_scent_deletions': 0,
              'fresh_model_inference_calls': 0, 'original_evaluation_records_preserved': True,
              'scope': 'stored_multi_version_composition_selection_not_fresh_accuracy_evaluation',
              'deployment_changed': False, 'excluded': excluded, 'kept': untouched}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('excluded','kept')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
