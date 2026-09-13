from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "modal_app.py"


def test_modal_web_function_requires_proxy_auth_without_embedded_credentials():
    text = SOURCE.read_text(encoding="utf-8")
    assert "@modal.asgi_app(requires_proxy_auth=True)" in text
    assert "Modal-Key" not in text
    assert "Modal-Secret" not in text
    assert re.search(r"\bwk-[A-Za-z0-9]", text) is None
    assert re.search(r"\bws-[A-Za-z0-9]", text) is None


def test_common_api_import_never_reads_private_deployment_artifacts():
    result = subprocess.run(
        [sys.executable, '-W', 'error', '-c',
         "import builtins\nfrom unittest.mock import patch\n"
         "original_import = builtins.__import__\n"
         "def without_cloud_sdk(name, *args, **kwargs):\n"
         "    if name == 'modal' or name.startswith('modal.'):\n"
         "        raise AssertionError('cloud SDK imported by common API')\n"
         "    return original_import(name, *args, **kwargs)\n"
         "with patch('pathlib.Path.open', side_effect=AssertionError('artifact read on import')), \\\n"
         "     patch('builtins.__import__', side_effect=without_cloud_sdk):\n"
         "    from deploy.web_app import create_web_app\n"
         "    assert callable(create_web_app)\n"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
