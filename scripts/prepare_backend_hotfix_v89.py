"""Overlay backend-only fixes on the deployed wheel, excluding model experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OVERLAYS=(
    'fragrance_ai/platform/backend_wire_contract.py',
    'fragrance_ai/platform/ai_extensions.py',
    'fragrance_ai/platform/lotion_inputs.py',
    'fragrance_ai/platform/rd_api.py',
    'fragrance_ai/platform/operation_contracts.py',
    'fragrance_ai/recommender/lotion_estimation.py',
    'deploy/web_app.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-output',type=Path,required=True)
    a=p.parse_args()
    base=json.loads(a.base.read_text(encoding='utf8'))
    assert sha(base['wheel'])==base['wheel_sha256']
    installed=Path(base['installed'])
    from zipfile import ZipFile
    with ZipFile(base['wheel']) as archive:
        for row in archive.infolist():
            if not row.is_dir():
                assert (installed/row.filename).read_bytes()==archive.read(row),row.filename
    a.stage.mkdir(parents=True,exist_ok=False)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    for folder in ('fragrance_ai','deploy'):
        shutil.copytree(installed/folder,a.stage/folder,ignore=ignore)
    for filename in ('pyproject.toml','README.md','LICENSE','LICENSE_POLICY.md'):
        shutil.copyfile(ROOT/filename,a.stage/filename)
    for name in OVERLAYS:
        shutil.copyfile(ROOT/name,a.stage/name)
    # The default change must be the only estimation algorithm source change.
    before=(installed/'fragrance_ai/recommender/lotion_estimation.py').read_text(encoding='utf8')
    after=(a.stage/'fragrance_ai/recommender/lotion_estimation.py').read_text(encoding='utf8')
    expected=before.replace('    registry_pool: Literal["core", "conditional_research"] = "core"',
        '    registry_pool: Literal["core", "conditional_research"] = Field(default="conditional_research",\n'
        '        description="Optional; omission uses the extended screened registry. Explicit core selects the former limited pool; risk and safety constraints still apply.")')
    assert after==expected,'unexpected estimation algorithm changes'
    protected=[]
    for path in (installed/'fragrance_ai/recommender').glob('*.py'):
        name=path.relative_to(installed).as_posix()
        if name not in OVERLAYS:
            assert sha(path)==sha(a.stage/name),name
            protected.append(name)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_source_package_v86.py'),
        '--parent-profile',base['profile'],'--source-root',str(a.stage.resolve()),
        '--output',str(a.output),'--profile-output',str(a.profile_output)],cwd=ROOT,check=True)
    proof={'base_wheel_sha256':base['wheel_sha256'],'overlay_files':list(OVERLAYS),
           'protected_recommender_modules':len(protected),'unaccepted_model_experiments_included':False,
           'changed_lotion_default':'core_to_conditional_research','risk_and_safety_defaults_changed':False,
           'deployed':False}
    (a.output/'hotfix-provenance.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()
