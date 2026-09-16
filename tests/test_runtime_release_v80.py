"""Release compatibility: same public URL, unchanged auth, complete current wheel."""
import ast
from pathlib import Path

from deploy import runtime_release_v80


def test_release_uses_one_shared_trained_checkpoint_and_scoped_validator():
    assert runtime_release_v80.CORE_SHA256 == '158b9f82ff9fb784eeec114ef1f7bbfc4d613badeb98f46e0f60435d5af38d2f'
    text = Path(runtime_release_v80.__file__).read_text(encoding='utf8')
    assert 'validate_scope(space' in text and 'base.verify(' in text


def test_existing_public_service_limits_and_proxy_auth_are_preserved():
    path = Path(__file__).resolve().parents[1]/'deploy/modal_release_v80.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    app = next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='App')
    assert ast.literal_eval(app.args[0]) == 'perfumery-ai-core'
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='web')
    decorators = [n for n in function.decorator_list if isinstance(n,ast.Call)]
    service = next(n for n in decorators if n.func.attr=='function')
    values = {k.arg:ast.literal_eval(k.value) for k in service.keywords if k.arg!='image'}
    assert values == {'cpu':1.,'memory':1024,'min_containers':0,'max_containers':1,'scaledown_window':300,'timeout':300}
    auth = next(n for n in decorators if n.func.attr=='asgi_app')
    assert {k.arg:ast.literal_eval(k.value) for k in auth.keywords} == {'requires_proxy_auth':True}
    assert 'backend_for(CompactLanguage)' in path.read_text(encoding='utf8')
