"""Persist read-only QA of the release, including every retired-blend request."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-wheel', required=True)
    parser.add_argument('--local-preparation', type=Path)
    parser.add_argument('--suites', nargs='+', choices=('handoff','diagnostic','retired','rate_limit'),
                        default=['handoff','diagnostic','retired','rate_limit'])
    parser.add_argument('--case-indices', nargs='+', type=int, choices=range(10), default=list(range(10)),
                        help='Retired-blend cases to replay; defaults to all ten.')
    args = parser.parse_args()
    if len(set(args.case_indices)) != len(args.case_indices):
        parser.error('--case-indices must not contain duplicates; use separate outputs for repeated runs')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.local_preparation:
        meta = json.loads(args.local_preparation.read_text(encoding='utf8'))
        assert meta['wheel_sha256'] == args.expected_wheel
        sys.path.insert(0, str(ROOT))
        from deploy.qa_release_v89 import run_suite
        sys.path.insert(0, meta['installed'])
        os.environ.update(PERFUMERY_AI_LOCAL_PROFILE=meta['profile'], PERFUMERY_AI_ENV='research',
                          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        from deploy.target_runtime_v87 import create_release_app
        from fastapi.testclient import TestClient
        def invoke(suite, index):
            with TestClient(create_release_app(registry_path=str(ROOT/'benchmarks/industrial_ingredient_registry_v1.db')),
                            raise_server_exceptions=False) as client:
                result = run_suite(client, args.expected_wheel, suite, index)
            result['wheel_sha256'] = args.expected_wheel
            for row in result['calls']:
                row['verification_scope'] = 'isolated_local_ASGI'
            return result
    else:
        import modal
        function = modal.Function.from_name('perfumery-ai-core', 'verify_release')
        def invoke(suite, index):
            with modal.enable_output():
                return function.remote(suite=suite, case_index=index)
    jobs = [(suite, index) for suite in args.suites for index in (args.case_indices if suite == 'retired' else [None])]
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'expected_wheel_sha256': args.expected_wheel,
        'deployed_image_exercised': not bool(args.local_preparation), 'backend_proxy_token_tested': False,
        'credentials_changed': False, 'new_proxy_tokens_created': False,
        'expected_jobs': len(jobs), 'jobs': [], 'complete': False,
        'scope': 'API_software_QA_not_full_800_accuracy_evaluation'}
    def save():
        write(args.output/'qa.json', report)
    save()
    for suite, index in jobs:
        name = suite + (f'_{index:02}' if index is not None else '')
        directory = args.output/name
        directory.mkdir()
        print(json.dumps({'starting': name}, ensure_ascii=False), flush=True)
        try:
            result = invoke(suite, index)
            if result['wheel_sha256'] != args.expected_wheel:
                raise ValueError('deployed wheel differs from the approved release')
            calls = []
            for original in result.pop('calls', []):
                row = dict(original)
                label = row['name']
                if not label.replace('_', '').isalnum():
                    raise ValueError('invalid artifact name')
                raw = row.pop('response_body').encode('utf8')
                assert hashlib.sha256(raw).hexdigest() == row['response_sha256']
                suffix = 'sse' if 'event-stream' in row['response_headers'].get('content-type','') else 'json'
                response_file = label+'.response.'+suffix
                (directory/response_file).write_bytes(raw)
                row['response_file'] = response_file
                request = row.pop('request', None)
                if request is not None:
                    row['request_file'] = label+'.request.json'
                    write(directory/row['request_file'], request)
                calls.append(row)
            result['calls'] = calls
        except Exception as error:
            result = {'passed': False, 'suite': suite, 'case_index': index,
                      'error_type': type(error).__name__, 'error': str(error)[:1500]}
        write(directory/'result.json', result)
        summary = {key: value for key, value in result.items() if key != 'calls'}
        summary.update(name=name, result_file=str((directory/'result.json').relative_to(args.output)),
                       api_calls=len(result.get('calls', [])))
        report['jobs'].append(summary)
        save()
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    report.update(complete=True, passed=all(row['passed'] for row in report['jobs']),
                  completed_at=datetime.now(timezone.utc).isoformat())
    report['api_calls'] = sum(row['api_calls'] for row in report['jobs'])
    retired = [row for row in report['jobs'] if row['suite'] == 'retired']
    report['retired_cases'] = {'checked': len(retired),
        'different_composition_returned': sum(row.get('replacement_composition_returned') is True for row in retired),
        'target_met': sum(row.get('target_met') is True for row in retired),
        'fresh_requests_not_full_benchmark': True}
    save()
    print(json.dumps({key:value for key,value in report.items() if key != 'jobs'}, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
