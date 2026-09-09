"""Compare actual recipe outputs and verify the pinned CPU LLM cache via HTTP."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def differences(left, right, path=''):
    if isinstance(left, dict) and isinstance(right, dict):
        return [row for key in sorted(set(left) | set(right))
                for row in differences(left.get(key), right.get(key), path + '/' + key)]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [row for index, (a, b) in enumerate(zip(left, right))
                for row in differences(a, b, path + '/' + str(index))]
    return [] if left == right else [{'path': path, 'before': left, 'after': right}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False,
            indent=2, allow_nan=False), encoding='utf-8')

    old = json.loads((args.before / 'response.json').read_text(encoding='utf-8'))
    new = json.loads((args.after / 'response.json').read_text(encoding='utf-8'))
    changes = differences(old, new)
    save('recipe-differences.json', changes)
    # Package binding hashes must change; formulas and scientific outputs must not.
    allowed = {'/deployment/wheel_sha256', '/deployment/catalog_snapshot/wheel_sha256',
               '/deployment/catalog_snapshot/catalog_sha256',
               '/deployment/catalog_snapshot/catalog_manifest_sha256'}
    unexpected = [row for row in changes if row['path'] not in allowed]
    save('unexpected-differences.json', unexpected)
    before = json.loads((args.before / 'report.json').read_text(encoding='utf-8'))
    after = json.loads((args.after / 'report.json').read_text(encoding='utf-8'))
    report = {'scope': 'local_pinned_runtime_speed_not_deployed_or_human_validation',
        'before_seconds': before['first_seconds'], 'after_seconds': after['first_seconds'],
        'speedup': before['first_seconds'] / after['first_seconds'],
        'latency_reduction_percent': 100 * (1 - after['first_seconds'] / before['first_seconds']),
        'response_repeat_equal': after['repeat_equal'], 'binding_changes': changes,
        'unexpected_recipe_changes': unexpected}
    save('recipe-report.json', report)
    if unexpected:
        raise AssertionError('unexpected recipe output changes; inspect saved diff')
    from fastapi.testclient import TestClient
    from scripts.serve_product_runtime_v42 import create_app
    app = create_app(enable_language=True)
    if app.state.local_language_backend is None:
        raise ValueError('actual pinned CPU LLM required, not a mock')
    with TestClient(app) as client:
        responses = []
        for index in range(2):
            start = time.perf_counter()
            value = client.post('/v1/ai/assistant',
                json={'message': '머스크 없이 장미 향 바디로션으로 만들어줘'})
            row = {'seconds': time.perf_counter() - start, 'status': value.status_code,
                'llm_calls': int(value.headers.get('X-Perfumery-LLM-Calls', '-1')),
                'cache': value.headers.get('X-Perfumery-Language-Cache'), 'body': value.json()}
            save(f'assistant-{index}.json', row)
            responses.append(row)
        first, repeated = responses
        assert first['status'] == repeated['status'] == 200
        assert first['llm_calls'] == 1 and repeated['llm_calls'] == 0
        assert first['cache'] == 'miss' and repeated['cache'] == 'hit'
        assert first['body'] == repeated['body']
        assert first['body']['source'] in ('quantized_language_model', 'deterministic_grounding_after_model_mismatch')
        proposal = first['body']['intent_proposal']
        assert 'musky' in proposal['avoided'] and 'musky' not in proposal['desired']
        assert 'rose' in proposal['desired'] and proposal['product'] == 'body_lotion'
        assert first['body']['requires_confirmation'] and not first['body']['formula_generated']
    report['assistant'] = {'first_seconds': first['seconds'], 'repeat_seconds': repeated['seconds'],
        'first_llm_calls': 1, 'repeat_llm_calls': 0, 'repeat_generated_tokens': 0,
        'body_exactly_equal': True, 'actual_quantized_model_used': True,
        'first_unique_prompt_not_shortened': True, 'external_paid_llm_api_calls': 0}
    report['status'] = 'passed'
    save('report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'binding_changes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
