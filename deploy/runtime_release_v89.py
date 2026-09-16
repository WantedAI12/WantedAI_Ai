"""V89 source-bound release with explicit language and backend contracts."""
import argparse
import json
from pathlib import Path
from deploy import runtime_release_v78 as base

ROOT=base.ROOT
RELEASE_ID='v89-language-backend-contract-20260916'
registry_path=base.registry_path


def verify(root,expected):
    value=base.verify(root,expected,release_id=RELEASE_ID)
    root=Path(root)
    profile=json.loads((root/'perfumery.local.json').read_text(encoding='utf8'))
    space=json.loads((root/profile['odor_space']['path']).read_text(encoding='utf8'))
    reference=json.loads((root/profile['lotion_target_reference']['path']).read_text(encoding='utf8'))
    from fragrance_ai.recommender.odor_release_scope import validate_scope
    validate_scope(space,reference['profiles'])
    return value


def prepare(preparation,output):
    result=base.prepare(preparation,output,release_id=RELEASE_ID)
    verify(output,result['bundle_sha256'])
    return result


def check_installed(root,expected):
    manifest=verify(root,expected)
    result=base.check_installed(root,expected,release_id=RELEASE_ID,data_version='v80')
    from deploy.target_runtime_v87 import create_release_app,verify_target_contract
    from fastapi.testclient import TestClient
    app=create_release_app(registry_path=registry_path(root,manifest))
    result['target_contract']=verify_target_contract(app)
    with TestClient(app) as client:
        for path in ('/v2/evidence/versions','/v1/applications/unified/coefficients/contract','/v1/ai/odor-language?limit=1'):
            response=client.get(path)
            assert response.status_code==200,response.text
        schema=client.get('/openapi.json').json()
        assert 'ChangeImpactResponse' in schema['components']['schemas']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preparation',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-installed',type=Path)
    parser.add_argument('--sha256')
    args=parser.parse_args()
    if args.preparation and args.output:
        prepare(args.preparation,args.output)
    elif args.check_installed:
        print(json.dumps(check_installed(args.check_installed,args.sha256)))
    else:
        parser.error('select preparation/output or installed check')
