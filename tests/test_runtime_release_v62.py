import json
from pathlib import Path

import pytest

from deploy.runtime_release_v62 import RELEASE_ID,WHEEL_SHA256,collect,sha,verify


def test_release_requires_the_exact_selected_profile(tmp_path):
    (tmp_path/'perfumery.local.json').write_text('{}')
    with pytest.raises(ValueError,match='selected V62 profile changed'):collect(tmp_path)


def test_bundle_verifies_every_byte_and_rejects_path_escape(tmp_path):
    (tmp_path/'model.json').write_text('model')
    m={'release_id':RELEASE_ID,'wheel_sha256':WHEEL_SHA256,'files':{'model.json':sha(tmp_path/'model.json')}}
    p=tmp_path/'bundle.json';p.write_text(json.dumps(m))
    assert verify(tmp_path,sha(p))['release_id']==RELEASE_ID
    (tmp_path/'model.json').write_text('drift')
    with pytest.raises(ValueError,match='file changed'):verify(tmp_path)
    m['files']={'../outside.json':'a'*64};p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='file changed'):verify(tmp_path)


def test_deployment_keeps_auth_endpoint_identity_and_no_production_promotion():
    text=(Path(__file__).resolve().parents[1]/'deploy/modal_release_v62.py').read_text()
    assert "modal.App('perfumery-ai-core')" in text
    assert 'def web():' in text and '@modal.asgi_app(requires_proxy_auth=True)' in text
    assert "'PERFUMERY_AI_ENV':'research'" in text
    assert 'min_containers=0,max_containers=1' in text
    assert "'OPENBLAS_NUM_THREADS':'1'" in text


def test_cross_python_worker_uses_importable_class_and_explicit_package_mount():
    import ast
    root=Path(__file__).resolve().parents[1]
    source=(root/'deploy/compact_language_worker.py').read_text()
    module=ast.parse(source)
    assert any(isinstance(n,ast.ClassDef) and n.name=='CompactLanguage' for n in module.body)
    assert 'serialized=True' not in source
    assert ".add_local_python_source('deploy')" in source
    assert 'max_tokens\': 160' in source and 'timeout=150' in source
    assert ".add_local_python_source('deploy')" in (root/'deploy/modal_release_v62.py').read_text()
