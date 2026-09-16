"""Stage allowlisted selection or equivalent transport-inference changes."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
OVERLAYS = ('fragrance_ai/recommender/retired_blends.py',
            'fragrance_ai/recommender/service.py',
            'fragrance_ai/recommender/lotion_reference_search.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-output', type=Path, required=True)
    parser.add_argument('--overlay-group', choices=('retired_blends','transport_fast_path','recipe_delivery','backend_alignment'), default='retired_blends')
    args = parser.parse_args()
    groups = {'retired_blends': OVERLAYS,
              'transport_fast_path': ('fragrance_ai/recommender/formulation_core.py',),
              'recipe_delivery': ('fragrance_ai/platform/backend_wire_contract.py',
                                  'fragrance_ai/platform/ai_extensions.py',
                                  'fragrance_ai/platform/operation_contracts.py', 'deploy/web_app.py'),
              'backend_alignment': ('fragrance_ai/platform/backend_wire_contract.py',
                  'fragrance_ai/platform/ai_extensions.py', 'fragrance_ai/platform/operation_contracts.py',
                  'fragrance_ai/platform/response_transport.py', 'fragrance_ai/platform/rd_policy.py',
                  'fragrance_ai/platform/rd_api.py', 'fragrance_ai/platform/rd_clarification.py',
                  'fragrance_ai/recommender/fixed_assessment.py', 'fragrance_ai/recommender/optimizer.py')}
    overlays = groups[args.overlay_group]
    base = json.loads(args.base.read_text(encoding='utf8'))
    installed = Path(base['installed'])
    assert sha(base['wheel']) == base['wheel_sha256']
    with ZipFile(base['wheel']) as wheel:
        for member in wheel.infolist():
            if not member.is_dir():
                assert (installed/member.filename).read_bytes() == wheel.read(member)
    for name, expected in base['deploy_sources'].items():
        assert sha(installed/name) == expected
    if args.overlay_group == 'transport_fast_path':
        def unchanged_tree(path):
            tree = ast.parse(path.read_text(encoding='utf8'))
            tree.body = [node for node in tree.body if not (
                isinstance(node, ast.FunctionDef) and node.name == 'transport_forward_arrays')]
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == 'FormulationCore':
                    node.body = [item for item in node.body if not (
                        isinstance(item, ast.FunctionDef) and item.name == 'kernel')]
            return ast.dump(tree, include_attributes=False)
        name = overlays[0]
        assert unchanged_tree(installed/name) == unchanged_tree(ROOT/name), 'unrelated shared-model changes detected'
    args.stage.mkdir(parents=True, exist_ok=False)
    for folder in ('fragrance_ai', 'deploy'):
        shutil.copytree(installed/folder, args.stage/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'LICENSE_POLICY.md'):
        shutil.copyfile(ROOT/name, args.stage/name)
    for name in overlays:
        shutil.copyfile(ROOT/name, args.stage/name)
    protected = []
    for path in (installed/'fragrance_ai').rglob('*.py'):
        name = path.relative_to(installed).as_posix()
        if name not in overlays:
            assert sha(path) == sha(args.stage/name)
            protected.append(name)
    subprocess.run([sys.executable, str(ROOT/'scripts/prepare_source_package_v86.py'),
                    '--parent-profile', base['profile'], '--source-root', str(args.stage.resolve()),
                    '--output', str(args.output), '--profile-output', str(args.profile_output)],
                   cwd=ROOT, check=True)
    prepared = json.loads((args.output/'preparation.json').read_text(encoding='utf8'))
    assert prepared['model_sha256'] == base['model_sha256']
    assert prepared['target_reference_sha256'] == base['target_reference_sha256']
    assert prepared['material_rows'] == base['material_rows']
    report = {'base_wheel_sha256': base['wheel_sha256'], 'overlay_files': list(overlays), 'overlay_group': args.overlay_group,
              'protected_source_modules': len(protected), 'model_weights_unchanged': True,
              'target_references_unchanged': True, 'material_rows': base['material_rows'],
              'unaccepted_model_experiments_included': False, 'deployed': False}
    filename = {'retired_blends': 'retired-blends-provenance.json',
                'transport_fast_path': 'lotion-latency-provenance.json',
                'recipe_delivery': 'recipe-delivery-provenance.json',
                'backend_alignment': 'backend-alignment-provenance.json'}[args.overlay_group]
    (args.output/filename).write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
