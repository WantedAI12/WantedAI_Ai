"""Tiny deployment-data closure fixtures; these are not regulatory evidence."""
from datetime import datetime, timedelta, timezone
import ast
import hashlib
import json
from pathlib import Path

import pytest

from deploy import runtime_release_v69 as previous
from deploy import runtime_release_v70 as release
from tests.test_runtime_release_v69 import source as source  # noqa: F401


@pytest.fixture
def prepared_source(source, monkeypatch):
    monkeypatch.setattr(release, 'CORE_SHA256', previous.CORE_SHA256)
    document = source / 'source.txt'
    document.write_text('explicit source fixture, not legal evidence')
    public = source / 'public.json'
    body = {'schema_version': 'public-regulatory-supply-1', 'version': 'fixture',
        'scope': 'public_source_facts_not_operator_reviewed_evidence', 'manufacturing_approval': False,
        'documents': [{'id': 'IFRA', 'path': document.name, 'sha256': release.sha(document),
            'url': 'https://example.invalid/fixture', 'retrieved_at': (datetime.now(timezone.utc)-timedelta(days=1)).isoformat()}],
        'materials': [], 'missing_operator_fields': ['all_real_operator_evidence']}
    public.write_text(json.dumps(body))
    profile_path = source / 'perfumery.local.json'
    profile = json.loads(profile_path.read_text())
    profile['public_evidence'] = {'path': public.name, 'sha256': release.sha(public)}
    profile_path.write_text(json.dumps(profile))
    preparation = source / 'preparation.json'
    preparation.write_text(json.dumps({'profile': str(profile_path), 'profile_sha256': release.sha(profile_path),
        'wheel': str(source / previous.WHEEL_REL), 'wheel_sha256': previous.WHEEL_SHA256}))
    return source, preparation


def test_public_source_documents_and_profile_are_in_the_private_closure(prepared_source, tmp_path):
    source, preparation = prepared_source
    target = tmp_path / 'release'
    manifest = release.prepare(preparation, target, root=source)
    assert release.verify(target, release.sha(target / 'bundle.json')) == manifest
    assert 'source.txt' in manifest['files'] and 'public.json' in manifest['files']
    assert not manifest['public_data_redistribution_authorized'] and not manifest['production_deployed']
    assert json.loads((target / 'perfumery.local.json').read_text())['language'] is None
    assert release.registry_path(target, manifest) == target / 'dependency.json'


@pytest.mark.parametrize('tamper', ['source', 'unlisted', 'missing_binding', 'rights', 'wrong_digest'])
def test_private_evidence_bundle_fails_closed(prepared_source, tmp_path, tamper):
    source, preparation = prepared_source
    target = tmp_path / 'release'
    release.prepare(preparation, target, root=source)
    path = target / 'bundle.json'
    manifest = json.loads(path.read_text())
    if tamper == 'source':
        (target / 'source.txt').write_text('changed')
    elif tamper == 'unlisted':
        (target / 'accidental.env').write_text('fixture only')
    elif tamper == 'missing_binding':
        del manifest['files']['source.txt']
        path.write_text(json.dumps(manifest))
    elif tamper == 'rights':
        manifest['public_data_redistribution_authorized'] = True
        path.write_text(json.dumps(manifest))
    expected = '0' * 64 if tamper == 'wrong_digest' else release.sha(path)
    with pytest.raises(ValueError):
        release.verify(target, expected)


def test_executable_cannot_enter_through_public_document_role(prepared_source, tmp_path):
    source, preparation = prepared_source
    raw = b'not an executable, but forbidden suffix'
    (source / 'bad.exe').write_bytes(raw)
    public = source / 'public.json'
    value = json.loads(public.read_text())
    value['documents'][0].update(path='bad.exe', sha256=hashlib.sha256(raw).hexdigest())
    public.write_text(json.dumps(value))
    profile_path = source / 'perfumery.local.json'
    profile = json.loads(profile_path.read_text())
    profile['public_evidence']['sha256'] = release.sha(public)
    profile_path.write_text(json.dumps(profile))
    record = json.loads(preparation.read_text())
    record['profile_sha256'] = release.sha(profile_path)
    preparation.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='kind'):
        release.prepare(preparation, tmp_path / 'release', root=source)


def test_prepared_modal_entrypoint_does_not_change_url_auth_or_allocated_resources():
    path = Path(__file__).resolve().parents[1] / 'deploy/modal_release_v70.py'
    calls = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.Call)]
    app = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'App')
    assert ast.literal_eval(app.args[0]) == 'perfumery-ai-core'
    auth = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'asgi_app')
    assert {k.arg: ast.literal_eval(k.value) for k in auth.keywords} == {'requires_proxy_auth': True}
    function = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'function')
    resources = {k.arg: ast.literal_eval(k.value) for k in function.keywords if k.arg != 'image'}
    assert resources['cpu'] == 1. and resources['memory'] == 1024
    assert resources['min_containers'] == 0 and resources['max_containers'] == 1
    assert 'gpu' not in resources
