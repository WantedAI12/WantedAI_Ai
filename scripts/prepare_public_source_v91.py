"""Copy the deployed source snapshot into a clean publication worktree.

No private models, material bundles, raw QA responses or credentials are copied.
The publication worktree must already exist and remain separate from the source.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    target = args.destination.resolve()
    if target == ROOT or not (target/'.git').is_file():
        raise ValueError('a separate git worktree is required')
    meta = json.loads(args.preparation.read_text(encoding='utf8'))
    installed = Path(meta['installed'])
    if hashlib.sha256(Path(meta['wheel']).read_bytes()).hexdigest() != meta['wheel_sha256']:
        raise ValueError('deployed wheel hash mismatch')
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=target, text=True).splitlines())
    copied = []

    def copy(source, name):
        destination = target/name
        if not destination.resolve().is_relative_to(target):
            raise ValueError('path outside publication worktree')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(name)

    sources, ast_sources = {}, {}
    for path in sorted((installed/'fragrance_ai').rglob('*.py')):
        name = path.relative_to(installed).as_posix()
        copy(path, name)
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        ast_sources[name] = hashlib.sha256(ast.dump(ast.parse(path.read_text(encoding='utf8')),
            include_attributes=False).replace(', type_params=[]', '').encode()).hexdigest()
    # Non-code wheel assets must be already public and unchanged.
    for folder in ('fragrance_ai/data', 'fragrance_ai/ui'):
        for path in (installed/folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.py', '.pyc'):
                continue
            name = path.relative_to(installed).as_posix()
            left, right = path.read_bytes(), (target/name).read_bytes() if name in tracked else b''
            if path.suffix in ('.json', '.html', '.js', '.css'):
                left, right = left.replace(b'\r\n', b'\n'), right.replace(b'\r\n', b'\n')
            if name not in tracked or left != right:
                raise ValueError('new or changed data requires separate publication review: ' + name)
            copy(path, name)
    # Packaging settings were used for the released wheel; production entry and
    # verification helpers are the exact files used for the latest deployment.
    for name in ('.gitignore', 'pyproject.toml', 'README.md', 'MODAL_DEPLOYMENT.md', '.github/workflows/release.yml'):
        copy(ROOT/name, name)
    for path in (ROOT/'deploy').glob('*.py'):
        copy(path, 'deploy/'+path.name)
    selected_tests = {name for name in tracked if name.startswith('tests/') and name.endswith('.py')}
    selected_tests.update(re.findall(r'^!(tests/[^\r\n]+\.py)$', (ROOT/'.gitignore').read_text(encoding='utf8'), re.M))
    for name in sorted(selected_tests):
        if (ROOT/name).is_file():
            copy(ROOT/name, name)
    workflow = (ROOT/'.github/workflows/release.yml').read_text(encoding='utf8')
    scripts = set(re.findall(r'scripts/[\w.-]+\.py', workflow))
    # Approved training, acquisition and release tooling. Exploratory probes,
    # candidate comparisons and unaccepted model experiments remain local.
    for path in (ROOT/'scripts').glob('*.py'):
        if re.match(r'(acquire|assemble|build|collect|expand|extend|extract|index|merge|recover|resolve|train|verify|prepare|qa|report_deployed|export_training)', path.name):
            if re.search(r'v(?:7\d|8\d|9[01])|recipe_delivery|training_data_bundle|public_source', path.name):
                scripts.add('scripts/'+path.name)
    scripts.update({'scripts/verify_junit_floor.py', 'scripts/calibrate_concentration_headspace_v1.py',
                    'scripts/train_formulation_v69.py'})
    for name in sorted(scripts):
        copy(ROOT/name, name)
    copy(ROOT/'scripts/java/BackendContractProbe.java', 'scripts/java/BackendContractProbe.java')
    for path in ROOT.glob('*.md'):
        if re.search(r'V(?:7\d|8\d|9[01])|V7[0-9]_|V76_CONTINUE|REMAINING_ODOR', path.name):
            copy(path, path.name)
    proof = {'release_id': 'v91-backend-compatibility-20260917',
        'deployed_wheel_sha256': meta['wheel_sha256'], 'runtime_python_files': sources,
        'runtime_ast_sha256': ast_sources,
        'model_weights_published': False, 'raw_training_or_qa_data_published': False,
        'credentials_published': False, 'model_weights_sha256': meta['model_sha256']}
    output = target/'release/V91/source_manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'copied_source_and_support_files': len(copied), 'runtime_python_files': len(sources),
        'private_artifacts_copied': False, 'destination': str(target)}))


if __name__ == '__main__':
    main()
