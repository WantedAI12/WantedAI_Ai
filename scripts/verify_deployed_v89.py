"""Verify the deployed image, leaving actual backend Proxy Token testing to backend."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import httpx
import modal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-wheel',required=True)
    parser.add_argument('--expected-release',default='v89-language-backend-contract-20260916')
    parser.add_argument('--suite',default='contracts',choices=('contracts','backend_contracts'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report={'started_at':datetime.now(timezone.utc).isoformat(),
        'url':'https://junseong2im--perfumery-ai-core-web.modal.run',
        'public_authenticated_requests_tested':False,'backend_proxy_token_tested':False,
        'backend_token_verification_owner':'backend_team_per_user_instruction',
        'new_proxy_credentials_created':False,'existing_backend_credentials_modified':False}
    def save():
        (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    save()
    try:
        response=httpx.get(report['url']+'/health',timeout=45,follow_redirects=False)
        report['unauthenticated_http_status']=response.status_code
        assert response.status_code in (401,403)
        save()
        with modal.enable_output():
            result=modal.Function.from_name('perfumery-ai-core','verify_release').remote(suite=args.suite)
        assert result['passed'] and result['release_id']==args.expected_release
        assert result['wheel_sha256']==args.expected_wheel
        calls=[]
        for row in result.pop('calls'):
            row=dict(row)
            name=row['name']
            assert name.replace('_','').isalnum()
            body=row.pop('response_body').encode('utf8')
            assert hashlib.sha256(body).hexdigest()==row['response_sha256']
            extension='sse' if 'event-stream' in row['response_headers'].get('content-type','') else 'json'
            filename=f'{name}.response.{extension}'
            (args.output/filename).write_bytes(body)
            row['response_file']=filename
            payload=row.pop('request',None)
            if payload is not None:
                request_file=name+'.request.json'
                (args.output/request_file).write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
                row['request_file']=request_file
            calls.append(row)
        history=subprocess.run([sys.executable,'-m','modal','app','history','perfumery-ai-core','--json'],
            capture_output=True,text=True,encoding='utf8',check=True,timeout=45)
        latest=json.loads(history.stdout)[0]
        report.update(result,calls=calls,deployed=True,modal_version=latest['version'],
            time_deployed=latest['time_deployed'],completed_at=datetime.now(timezone.utc).isoformat())
        save()
        print(json.dumps({k:v for k,v in report.items() if k!='calls'},ensure_ascii=False),flush=True)
    except Exception as error:
        report.update(passed=False,error_type=type(error).__name__,error=str(error)[:1500],
                      completed_at=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__=='__main__':
    main()
