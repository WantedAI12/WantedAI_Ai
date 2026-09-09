"""Compare complete frozen request suites, retaining failures and regressions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read_run(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if manifest.get('product', 'body_lotion') != 'body_lotion' or summary.get('product', 'body_lotion') != 'body_lotion':
        raise ValueError('perfume and lotion benchmark results cannot be combined')
    rows = [json.loads(line) for line in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    expected = {case['id'] for case in manifest['cases']}
    if (len(rows) != len(expected) or {row['id'] for row in rows} != expected
            or not summary['source_unchanged'] or summary['count'] != len(rows)):
        raise ValueError('incomplete, duplicated, or changed-source benchmark')
    return manifest, summary, {row['id']: row for row in rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('choose a new output file')
    bm, bs, baseline = read_run(args.baseline)
    cm, cs, candidate = read_run(args.candidate)
    if bm['case_sha256'] != cm['case_sha256'] or bm['target'] != cm['target'] or bm['pool'] != cm['pool']:
        raise ValueError('different cases, targets or candidate policies cannot be paired')
    pairs = []
    for case in bm['cases']:
        old, new = baseline[case['id']], candidate[case['id']]
        delta = new['score']-old['score'] if old.get('score') is not None and new.get('score') is not None else None
        pairs.append({'id': case['id'], 'brief': case['brief'], 'baseline_score': old.get('score'),
            'candidate_score': new.get('score'), 'delta': delta,
            'baseline_passed95': bool(old.get('profile_target_met')),
            'candidate_passed95': bool(new.get('profile_target_met')),
            'baseline_incomplete': bool(old.get('search_incomplete')),
            'candidate_incomplete': bool(new.get('search_incomplete'))})
    defined = [row for row in pairs if row['delta'] is not None]
    report = {'scope': 'paired_frozen_400_software_profiles_not_human_accuracy',
        'target': bm['target'], 'requests': len(pairs),
        'baseline_passed95': bs['passed95'], 'candidate_passed95': cs['passed95'],
        'baseline_errors': bs['errors'], 'candidate_errors': cs['errors'],
        'baseline_incomplete': sum(p['baseline_incomplete'] for p in pairs),
        'candidate_incomplete': sum(p['candidate_incomplete'] for p in pairs),
        'new_pass_ids': [p['id'] for p in pairs if p['candidate_passed95'] and not p['baseline_passed95']],
        'lost_pass_ids': [p['id'] for p in pairs if p['baseline_passed95'] and not p['candidate_passed95']],
        'defined_score_pairs': len(defined),
        'mean_score_delta_on_defined_pairs': statistics.mean(p['delta'] for p in defined),
        'improved_by_more_than_0_025': sum(p['delta'] > .025 for p in defined),
        'regressed_by_more_than_0_025': sum(p['delta'] < -.025 for p in defined),
        'baseline_suite_seconds': bs['seconds'], 'candidate_suite_seconds': cs['seconds'],
        'baseline_workers': bm.get('workers'), 'candidate_workers': cm.get('workers'),
        'timing_scope': 'local_suite_measurements_not_controlled_production_latency',
        'universal_95_proven': False, 'pairs': pairs,
        'evidence_sha256': {str(folder/name): hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for folder in (args.baseline, args.candidate) for name in ('manifest.json', 'summary.json', 'results.jsonl')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs', 'evidence_sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
