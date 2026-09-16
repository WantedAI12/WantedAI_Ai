"""Build a new source-bound local candidate; preserve all deployed material rows."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--public-evidence', type=Path)
    p.add_argument('--target-reference', type=Path)
    p.add_argument('--profile-name')
    p.add_argument('--component-observations',type=Path)
    p.add_argument('--formulation-core',type=Path)
    p.add_argument('--parent-profile',type=Path)
    p.add_argument('--physical-evidence',type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'
    subprocess.run([sys.executable, '-m', 'build', '--wheel', '--outdir', str(output/'wheel')], cwd=ROOT, check=True, capture_output=True)
    wheel = next((output/'wheel').glob('*.whl'))
    with zipfile.ZipFile(wheel) as package:
        sources = {name: hashlib.sha256(package.read(name)).hexdigest() for name in package.namelist() if name.startswith('fragrance_ai/') and name.endswith('.py')}
        data = {name: hashlib.sha256(package.read(name)).hexdigest() for name in package.namelist() if name.startswith('fragrance_ai/data/') and Path(name).suffix in ('.json', '.db', '.npz')}
    from fragrance_ai.recommender.runtime import _source_snapshot, _material_digest
    from fragrance_ai.recommender.registry_activation import load_runtime_catalog, write_runtime_catalog
    assert sources == _source_snapshot(), 'wheel does not contain current source'
    parent_profile=(args.parent_profile.resolve() if args.parent_profile else ROOT/'tmp/modal-runtime-v69/release-02/perfumery.local.json')
    root=parent_profile.parent
    profile = json.loads(parent_profile.read_text(encoding='utf-8'))
    original_manifest = root/profile['catalog']['path']
    assert sha(original_manifest) == profile['catalog']['sha256']
    manifest = json.loads(original_manifest.read_text(encoding='utf-8'))
    binding = manifest['runtime_catalog']
    catalog, activation, stats = load_runtime_catalog(original_manifest.parent/binding['path'],
        expected_sha256=binding['sha256'], expected_wheel_sha256=binding['wheel_sha256'], expected_registry_sha256=binding['registry_sha256'])
    output_catalog = output/'catalog'
    output_catalog.mkdir()
    runtime = output_catalog/'runtime_catalog_v3.json.gz'
    digest = write_runtime_catalog(runtime, catalog, activation, stats, wheel_sha256=sha(wheel))
    rebuilt, _, _ = load_runtime_catalog(runtime, expected_sha256=digest, expected_wheel_sha256=sha(wheel), expected_registry_sha256=binding['registry_sha256'])
    assert _material_digest(rebuilt) == _material_digest(catalog), 'material evidence changed'
    manifest = deepcopy(manifest)
    manifest.update(status='local_candidate_not_deployed', runtime_source_sha256=sources, runtime_data_sha256=data)
    manifest['runtime_catalog'].update(path=runtime.name, sha256=digest, wheel_sha256=sha(wheel))
    manifest.setdefault('provenance', {})['v70_source_rebinding'] = {'parent_manifest_sha256': sha(original_manifest), 'material_values_unchanged': True}
    target = output_catalog/'catalog_manifest.json'
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for value in profile.values():
        if isinstance(value, dict) and 'path' in value:
            value['path'] = (root/value['path']).relative_to(ROOT).as_posix()
    profile['catalog'] = {'path': target.relative_to(ROOT).as_posix(), 'sha256': sha(target)}
    if args.formulation_core:
        from fragrance_ai.recommender.formulation_core import FormulationCore
        model_path=args.formulation_core.resolve()
        selected=FormulationCore(model_path,sha(model_path))
        selected.assert_current()
        profile['formulation_core']={'path':model_path.relative_to(ROOT).as_posix(),'sha256':selected.sha256}
    if args.public_evidence:
        from fragrance_ai.platform.public_evidence import PublicEvidenceStore
        public_path = args.public_evidence.resolve()
        PublicEvidenceStore(public_path, sha(public_path)).assert_current()
        profile['public_evidence'] = {'path': public_path.relative_to(ROOT).as_posix(), 'sha256': sha(public_path)}
    if args.target_reference:
        from fragrance_ai.recommender.lotion_reference_objective import ObservedReferenceBank
        path = args.target_reference.resolve()
        bank = ObservedReferenceBank(path, sha(path))
        if bank.parent_sha256 != profile['formulation_core']['sha256']:
            raise ValueError('reference/model parent mismatch')
        profile['lotion_target_reference'] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    if args.component_observations:
        from fragrance_ai.recommender.reference_observations import ComponentReferenceObservations
        path=args.component_observations.resolve()
        ComponentReferenceObservations(path,sha(path)).assert_current()
        profile['component_reference_observations']={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
    if args.physical_evidence:
        from fragrance_ai.recommender.physical_evidence_v76 import load
        path=args.physical_evidence.resolve();stat=path.stat()
        load(str(path),sha(path),stat.st_size,stat.st_mtime_ns)
        profile['physical_evidence']={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}
    name = args.profile_name or ('perfumery.v70-'+output.name+'.local.json')
    if Path(name).name != name or not name.endswith('.local.json'):
        raise ValueError('local profile requires a plain file name')
    profile_path = ROOT/name
    if profile_path.exists():
        raise ValueError('use an unused candidate name')
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = {'wheel': str(wheel), 'wheel_sha256': sha(wheel), 'profile': str(profile_path),
              'profile_sha256': sha(profile_path), 'catalog_sha256': sha(target),
              'material_digest': _material_digest(catalog), 'material_rows': len(catalog.ingredients),
              'model_sha256': profile['formulation_core']['sha256'], 'deployed': False}
    report['target_reference_sha256'] = profile['lotion_target_reference']['sha256']
    (output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
