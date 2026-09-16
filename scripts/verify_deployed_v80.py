"""Check the deployed image using existing Modal admin authentication only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import httpx
import modal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-wheel',default='fa0e9f39462a79fa37816f5920ce463920275759548aa3f2c4dce50ab69d767e')
    p.add_argument('--snapshot-audit',type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('new deployment verification output required')
    base = 'https://junseong2im--perfumery-ai-core-web.modal.run'
    report = {'deployed':True,'url':base,'new_proxy_credentials_created':False,
        'existing_backend_credentials_modified':False,'public_authenticated_requests_tested':False,
        'started_at':datetime.now(timezone.utc).isoformat()}
    status = httpx.get(base+'/health',timeout=45,follow_redirects=False).status_code
    if status not in (401,403):
        raise ValueError('public proxy auth boundary missing')
    report['unauthenticated_http_status'] = status
    payload = None
    if a.snapshot_audit:
        payload = {key:json.loads((a.snapshot_audit/name).read_text(encoding='utf8'))['request']
            for key,name in [('comparison','rd_saved_compare.json'),('revision','rd_saved_revise.json')]}
    with modal.enable_output():
        result = modal.Function.from_name('perfumery-ai-core','verify_release').remote(payload)
    if (not result.get('passed') or result.get('release_id')!='v80-supported-scope-20260915'
            or result.get('wheel_sha256')!=a.expected_wheel):
        raise ValueError('deployed release identity or contract mismatch')
    listed = subprocess.run([sys.executable,'-m','modal','app','history','perfumery-ai-core','--json'],
        capture_output=True,text=True,encoding='utf8',check=True,timeout=45)
    history = json.loads(listed.stdout)
    report.update(result,modal_version=history[0]['version'],time_deployed=history[0]['time_deployed'],
        completed_at=datetime.now(timezone.utc).isoformat())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('checks','language_assistant')},ensure_ascii=False))


if __name__=='__main__':
    main()
