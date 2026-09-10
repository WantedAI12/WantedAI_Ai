"""Fixed actual-checkpoint API cases; compare one learned metric before/after refinement."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ('floral fruity woody', 'green aromatic woody', 'gourmand woody', 'clean fresh citrus woody')


def main():
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--brief', action='append', choices=CASES,
                   help='check only named affected cases; default retains all four')
    args = p.parse_args()
    if args.output.exists():
        p.error('preserve earlier evidence; use a new output filename')
    source = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (ROOT/'fragrance_ai').rglob('*.py')}
    rows = []
    with TestClient(create_app()) as client:
        for brief in args.brief or CASES:
            body = {'brief': brief, 'registry_pool': 'conditional_research', 'max_risk_tier': 2}
            start = time.perf_counter()
            response = client.post('/v1/applications/body-lotion/design', json=body)
            assert response.status_code == 200, response.text
            result = response.json()
            elapsed = time.perf_counter()-start
            cached = client.post('/v1/applications/body-lotion/design', json=body)
            assert cached.json() == result and cached.headers['X-Perfumery-Lotion-Cache'] == 'hit'
            report = result['learned_optimization']
            balance = report.get('profile_balance')
            if balance and balance.get('recipe_changed'):
                assert report['fresh_transport_verified'] and balance['fresh_transport_verified']
                assert balance['selected_score'] > balance['baseline_score']+1e-5
            row = {'brief': brief, 'seconds': elapsed, 'strict_score': result['score'],
                   'strict_passed95': result['profile_target_met'], 'status': report['status'],
                   'balance_before': balance.get('baseline_score') if balance else None,
                   'balance_after': balance.get('selected_score') if balance else None,
                   'balance_recipe_changed': bool(balance and balance.get('recipe_changed')),
                   'balance_incomplete': bool(balance and balance.get('solver_incomplete'))}
            print(json.dumps(row), flush=True)
            rows.append({**row, 'result': result})
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name, digest in source.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'selected_actual_model_API_cases_not_full400_or_human_accuracy' if args.brief else
        'four_predeclared_actual_model_API_cases_not_full400_or_human_accuracy',
        'source_sha256': source, 'source_unchanged': True, 'product': 'body_lotion',
        'new_training_performed': False, 'deployed': False, 'rows': rows,
        'balance_improved_cases': sum(row['balance_recipe_changed'] for row in rows),
        'strict_passed95': sum(row['strict_passed95'] for row in rows),
        'all_failures_retained': True}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
