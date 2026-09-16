"""Private inference closure including pinned public regulatory documents.

Preparing this directory is local only. It neither deploys Modal nor grants
redistribution rights, supplier approval, or business registration.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from deploy.runtime_release_v69 import CORE_SHA256, ROLES, ROOT, _digest, _member, sha

RELEASE_ID = 'v70-repair-20260914'
PUBLIC_SUFFIXES = ('.json', '.html', '.txt', '.pdf', '.xlsx')


def closure(root, source, wheel_rel, wheel_sha):
    root = Path(root).resolve()
    if (source.get('schema') != 'perfumery-local-runtime/v1' or source.get('scope') != 'local_research'
            or source['formulation_core']['sha256'] != CORE_SHA256):
        raise ValueError('shared research checkpoint/profile required')
    profile = {key: deepcopy(source[key]) for key in ('schema', 'scope', 'lotion_reference', *ROLES, 'public_evidence')}
    profile['language'] = None
    files = {}

    def add(path, digest, *, public=False):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or not _digest(digest) or sha(path) != digest:
            raise ValueError('dependency path or digest mismatch')
        name = path.relative_to(root).as_posix()
        suffixes = PUBLIC_SUFFIXES if public else ('.json', '.json.gz', '.npz', '.db', '.whl')
        if not name.endswith(suffixes) or (public and path.stat().st_size > 16 * 1024 * 1024):
            raise ValueError('unexpected runtime dependency kind or size')
        if name in files and files[name] != digest:
            raise ValueError('conflicting dependency binding')
        files[name] = digest
        return path

    add(root / wheel_rel, wheel_sha)
    for role in ROLES:
        item = profile[role]
        path = add(root / item['path'], item['sha256'])
        document = json.loads(path.read_text(encoding='utf-8'))
        if role == 'catalog':
            binding = document['runtime_catalog']
            if binding['wheel_sha256'] != wheel_sha:
                raise ValueError('catalog/wheel binding mismatch')
            add(path.parent / binding['path'], binding['sha256'])
        elif role in ('perfume', 'body_lotion'):
            for key in ('base_model', 'component_model', 'registry'):
                binding = document[key]
                add(path.parent / binding['path'], binding['sha256'])
        elif role == 'formulation_core':
            if document.get('schema') != 'shared-formulation-core/v69' or document.get('accepted_for_local_inference') is not True:
                raise ValueError('unaccepted shared neural checkpoint')
            binding = document['weights']
            add(path.parent / binding['path'], binding['sha256'])
    from fragrance_ai.platform.public_evidence import PublicEvidenceStore
    public = profile['public_evidence']
    store = PublicEvidenceStore(root / public['path'], public['sha256'])
    for path, digest in store.members:
        add(path, digest, public=True)
    return profile, files


def prepare(preparation, output, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    prepared = json.loads(Path(preparation).read_text(encoding='utf-8'))
    selected, wheel = Path(prepared['profile']).resolve(), Path(prepared['wheel']).resolve()
    if not selected.is_relative_to(root) or not wheel.is_relative_to(root) or sha(selected) != prepared['profile_sha256']:
        raise ValueError('source preparation/profile mismatch')
    source = json.loads(selected.read_text(encoding='utf-8'))
    wheel_rel = wheel.relative_to(root).as_posix()
    profile, files = closure(root, source, wheel_rel, prepared['wheel_sha256'])
    if output.exists():
        raise ValueError('preserve previous bundle; choose a new directory')
    output.mkdir(parents=True)
    for name, digest in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if sha(target) != digest:
            raise ValueError('copied dependency changed')
    profile_path = output / 'perfumery.local.json'
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    files['perfumery.local.json'] = sha(profile_path)
    manifest = {'release_id': RELEASE_ID, 'scope': 'authenticated_noncommercial_research_service',
        'source_profile_sha256': prepared['profile_sha256'], 'profile_sha256': sha(profile_path),
        'wheel_path': wheel_rel, 'wheel_sha256': prepared['wheel_sha256'], 'shared_checkpoint_sha256': CORE_SHA256,
        'public_evidence_sha256': profile['public_evidence']['sha256'], 'files': files,
        'public_data_redistribution_authorized': False, 'raw_training_corpus_included': False,
        'public_regulatory_source_documents_included': True, 'production_deployed': False}
    manifest_path = output / 'bundle.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    checked = verify(output, sha(manifest_path))
    print(json.dumps({'bundle': str(output), 'bundle_sha256': sha(manifest_path), 'files': len(files),
        'bytes': sum((output / name).stat().st_size for name in files), 'wheel_sha256': checked['wheel_sha256'],
        'public_evidence_sha256': checked['public_evidence_sha256'], 'production_deployed': False}))
    return checked


def verify(root, expected):
    root = Path(root).resolve()
    path = root / 'bundle.json'
    if not _digest(expected) or sha(path) != expected:
        raise ValueError('trusted bundle hash is required and must match')
    manifest = json.loads(path.read_text())
    if (manifest.get('release_id') != RELEASE_ID or manifest.get('shared_checkpoint_sha256') != CORE_SHA256
            or manifest.get('scope') != 'authenticated_noncommercial_research_service'
            or manifest.get('public_data_redistribution_authorized') is not False
            or manifest.get('raw_training_corpus_included') is not False):
        raise ValueError('wrong release/distribution scope')
    files = manifest['files']
    for name, digest in files.items():
        if not _digest(digest) or sha(_member(root, name)) != digest:
            raise ValueError('runtime member changed')
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} != set(files) | {'bundle.json'}:
        raise ValueError('unlisted runtime file')
    profile = json.loads((root / 'perfumery.local.json').read_text(encoding='utf-8'))
    if files.get('perfumery.local.json') != manifest['profile_sha256'] or profile.get('language') is not None:
        raise ValueError('unbound profile or executable selection')
    _, required = closure(root, profile, manifest['wheel_path'], manifest['wheel_sha256'])
    if required != {key: digest for key, digest in files.items() if key != 'perfumery.local.json'}:
        raise ValueError('incomplete or unexpected transitive runtime closure')
    if profile['public_evidence']['sha256'] != manifest['public_evidence_sha256']:
        raise ValueError('public evidence binding mismatch')
    return manifest


def check_installed(root, expected):
    root = Path(root).resolve()
    manifest = verify(root, expected)
    if os.environ.get('PERFUMERY_AI_ENV') != 'research' or Path(os.environ.get('PERFUMERY_AI_LOCAL_PROFILE', '')).resolve() != root / 'perfumery.local.json':
        raise ValueError('installed check requires the bundled research profile')
    from fragrance_ai.platform.public_evidence import PublicEvidenceStore
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from fragrance_ai.recommender.formulation_core import configured_formulation_core
    from deploy.shared_runtime_v69 import create_release_app
    from fastapi.testclient import TestClient
    catalog, _ = load_configured_catalog()
    public = PublicEvidenceStore.configured()
    assert public.digest == manifest['public_evidence_sha256']
    assert public.coverage(catalog)['active_material_count'] == 3830
    assert configured_formulation_core().sha256 == CORE_SHA256
    with TestClient(create_release_app(registry_path=registry_path(root, manifest))) as client:
        assert client.get('/health').json()['wheel_sha256'] == manifest['wheel_sha256']
        assert client.get('/v2/evidence/status').json()['public_sources_registered']
    verify(root, expected)
    import fragrance_ai
    result = {'installed_bundle_valid': True, 'catalog': 3830,
              'four_frameworks': len(public.registry_indexes[public.value['version']].frameworks),
              'imported_from': str(Path(fragrance_ai.__file__).resolve()),
              'bundle_sha256': expected, 'wheel_sha256': manifest['wheel_sha256'],
              'production_deployed': False}
    print(json.dumps(result))
    return result


def registry_path(root, manifest):
    """Resolve the actual profile-bound registry, not a repository-only alias."""
    root = Path(root).resolve()
    profile = json.loads(_member(root, 'perfumery.local.json').read_text(encoding='utf-8'))
    view_path = _member(root, profile['perfume']['path'])
    binding = json.loads(view_path.read_text(encoding='utf-8'))['registry']
    path = (view_path.parent / binding['path']).resolve()
    if not path.is_relative_to(root) or manifest['files'].get(path.relative_to(root).as_posix()) != binding['sha256'] or sha(path) != binding['sha256']:
        raise ValueError('profile registry is not bound to the verified bundle')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--check-installed', type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.check_installed:
        result = check_installed(args.check_installed, args.sha256)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    elif args.verify:
        print(json.dumps(verify(args.verify, args.sha256)))
    elif args.preparation and args.output:
        prepare(args.preparation, args.output)
    else:
        parser.error('provide --preparation/--output, --verify, or --check-installed')
