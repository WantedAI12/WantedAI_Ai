from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "modal_app.py"


def test_modal_web_function_requires_proxy_auth_without_embedded_credentials():
    text = SOURCE.read_text(encoding="utf-8")
    assert "@modal.asgi_app(requires_proxy_auth=True)" in text
    assert "Modal-Key" not in text
    assert "Modal-Secret" not in text
    assert re.search(r"\bwk-[A-Za-z0-9]", text) is None
    assert re.search(r"\bws-[A-Za-z0-9]", text) is None
