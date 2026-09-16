"""Auditable comparison of complete experiments without cherry-picked cases."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_run(root, *, verify_responses=True):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    cases = protocol['cases']
    if len({c['id'] for c in cases}) != len(cases) or not finite(protocol['target']):
        raise ValueError('invalid fixed cases or target')
    case_map = {case['id']: case for case in cases}
    raw = (root/'results.jsonl').read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    expected = {(version, product, c['id']) for version in protocol['versions']
                for product in protocol['products'] for c in cases}
    found = {}
    for row in rows:
        key = row['version'], row['product'], row['case_id']
        if key in found or key not in expected:
            raise ValueError('duplicate or out-of-protocol result')
        case = case_map[row['case_id']]
        if row['brief'] != case['brief'] or row['group'] != case['group'] or row['target'] != protocol['target']:
            raise ValueError('case identity or threshold changed')
        score = row.get('score')
        if score is not None and (not finite(score) or not 0 <= score <= 100.+1e-7):
            raise ValueError('invalid numeric score')
        if row.get('target_met') is True and ('error' in row or row.get('http') != 200
                or score is None or score+1e-7 < protocol['target']):
            raise ValueError('false pass or failed execution counted as success')
        if verify_responses and row.get('response_file'):
            path = (root/'responses'/row['response_file']).resolve()
            if not path.is_relative_to((root/'responses').resolve()):
                raise ValueError('response path escapes run')
            if sha(path) != row.get('response_sha256'):
                raise ValueError('response hash missing or changed')
            artifact = json.loads(gzip.decompress(path.read_bytes()))
            value = artifact['response']
            actual = value.get('calculated_profile_similarity', value.get('score'))
            met = value.get('full_profile_target_met', value.get('profile_target_met', False))
            if actual != score or met != row.get('target_met') or artifact['request']['brief'] != row['brief']:
                raise ValueError('summary differs from raw response')
        elif verify_responses and 'error' not in row:
            raise ValueError('response evidence missing')
        found[key] = row
    return protocol, found, expected, {'path': str(root.resolve()), 'protocol_sha256': sha(root/'protocol.json'),
        'results_sha256': hashlib.sha256(raw).hexdigest()}


def summarize(roots, *, allow_partial=False, verify_responses=True):
    rows, expected, sources, bindings = {}, set(), [], {}
    selected = None
    for root in roots:
        protocol, found, keys, evidence = load_run(root, verify_responses=verify_responses)
        contract = {k: protocol[k] for k in ('cases', 'products', 'target')}
        if selected is not None and contract != selected:
            raise ValueError('comparison cases, products or target do not match')
        selected = contract
        if set(bindings) & set(protocol['versions']):
            raise ValueError('comparison version labels are not unique')
        bindings.update(protocol['versions'])
        rows.update(found)
        expected |= keys
        sources.append(evidence)
    if selected is None:
        raise ValueError('at least one run required')
    complete = set(rows) == expected
    if not complete and not allow_partial:
        raise ValueError(f'incomplete comparison: {len(rows)}/{len(expected)}')
    summaries, pairs = {}, {}
    count = len(selected['cases'])
    for version in bindings:
        for product in selected['products']:
            subset = [r for (v,p,_),r in rows.items() if (v,p) == (version,product)]
            scores = [r['score'] for r in subset if finite(r.get('score'))]
            passes = sum(r.get('target_met') is True for r in subset)
            summaries[f'{version}:{product}'] = {'evaluated': len(subset), 'expected': count,
                'passed': passes, 'pass_rate_of_full_suite_percent': 100*passes/count,
                'missing': count-len(subset), 'nonpassing_including_missing': count-passes,
                'mean_of_scored_cases': statistics.mean(scores) if scores else None,
                'score_denominator': len(scores), 'no_score': len(subset)-len(scores),
                'errors': sum('error' in r or r.get('http',500) >= 500 for r in subset),
                'statuses': dict(Counter(str(r.get('status')) for r in subset))}
    for left,right in combinations(bindings,2):
        for product in selected['products']:
            paired = [(rows.get((left,product,c['id'])),rows.get((right,product,c['id'])))
                      for c in selected['cases']]
            paired = [(a,b) for a,b in paired if a is not None and b is not None]
            differences = [b['score']-a['score'] for a,b in paired
                           if finite(a.get('score')) and finite(b.get('score'))]
            pairs[f'{left}->{right}:{product}'] = {'paired_cases': len(paired), 'expected': count,
                'pass_gains': sum(a.get('target_met') is not True and b.get('target_met') is True for a,b in paired),
                'pass_losses': sum(a.get('target_met') is True and b.get('target_met') is not True for a,b in paired),
                'improved': sum(d > 1e-6 for d in differences), 'regressed': sum(d < -1e-6 for d in differences),
                'mean_score_delta': statistics.mean(differences) if differences else None,
                'score_pair_denominator': len(differences),
                'lost_case_ids': [a['case_id'] for a,b in paired
                    if a.get('target_met') is True and b.get('target_met') is not True]}
    leaders = {}
    if complete:
        for product in selected['products']:
            most = max(summaries[f'{v}:{product}']['passed'] for v in bindings)
            leaders[product] = [v for v in bindings if summaries[f'{v}:{product}']['passed'] == most]
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'complete': complete,
        'evaluated': len(rows), 'expected': len(expected), 'target': selected['target'],
        'human_accuracy_measured': False, 'raw_responses_verified': verify_responses,
        'selection_allowed': complete and verify_responses, 'leaders_by_pass_count': leaders,
        'automatic_promotion_performed': False, 'scope': 'fixed_suite_not_all_possible_requests',
        'source_runs': sources, 'versions': bindings, 'summaries': summaries, 'pairs': pairs,
        'missing_results': [list(key) for key in sorted(expected-set(rows))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = summarize(args.input, allow_partial=args.allow_partial)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('complete','evaluated','expected','leaders_by_pass_count','summaries','pairs')}))


if __name__ == '__main__':
    main()
