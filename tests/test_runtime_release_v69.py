"""Private release assembly guards with tiny, isolated artifact fixtures."""

import ast
import json
from pathlib import Path

import pytest

from deploy import runtime_release_v69 as release


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()

    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": name, "sha256": release.sha(path)}

    wheel = put("runtime.whl", "fixture-not-an-installable-wheel")
    monkeypatch.setattr(release, "WHEEL_REL", wheel["path"])
    monkeypatch.setattr(release, "WHEEL_SHA256", wheel["sha256"])
    dependency = put("dependency.json", {"fixture": True})
    material = put("materials.json.gz", {"fixture": True})
    core_weights = put("weights.npz", "fixture-not-a-trained-network")
    core = put(
        "core.json",
        {
            "schema": "shared-formulation-core/v69",
            "accepted_for_local_inference": True,
            "weights": core_weights,
        },
    )
    monkeypatch.setattr(release, "CORE_SHA256", core["sha256"])
    profile = {
        "schema": "perfumery-local-runtime/v1",
        "scope": "local_research",
        "lotion_reference": "atlas",
        "formulation_core": core,
        "language": {"executable": {"path": "private.exe"}},
        "odor_calibration": {"path": "unused-private-bank.json", "sha256": "0" * 64},
    }
    for role in release.ROLES:
        if role == "formulation_core":
            continue
        value = {"fixture": role}
        if role == "catalog":
            value["runtime_catalog"] = {**material, "wheel_sha256": wheel["sha256"]}
        elif role in ("perfume", "body_lotion"):
            value.update(
                {k: dependency for k in ("base_model", "component_model", "registry")}
            )
        profile[role] = put(role + ".json", value)
    selected = root / "perfumery.local.json"
    selected.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(release, "PROFILE_SHA256", release.sha(selected))
    return root


def test_only_required_inference_closure_is_copied(source, tmp_path):
    output = tmp_path / "bundle"
    original = (source / "perfumery.local.json").read_bytes()
    manifest = release.prepare(output, root=source)
    assert release.verify(output, release.sha(output / "bundle.json")) == manifest
    profile = json.loads((output / "perfumery.local.json").read_text())
    assert profile["language"] is None
    assert "odor_calibration" not in profile
    assert (source / "perfumery.local.json").read_bytes() == original
    assert not any(name.endswith((".exe", ".pt", ".pkl")) for name in manifest["files"])


def test_source_profile_or_core_drift_is_rejected(source, monkeypatch):
    with (source / "perfumery.local.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="selected V69 profile changed"):
        release.collect(source)
    monkeypatch.setattr(
        release, "PROFILE_SHA256", release.sha(source / "perfumery.local.json")
    )
    (source / "core.json").write_text("changed")
    with pytest.raises(ValueError, match="dependency path or hash"):
        release.collect(source)


@pytest.mark.parametrize(
    "kind", ["changed", "unlisted", "missing", "escape", "scope", "profile"]
)
def test_private_bundle_is_sealed(source, tmp_path, kind):
    output = tmp_path / "bundle"
    release.prepare(output, root=source)
    path = output / "bundle.json"
    manifest = json.loads(path.read_text())
    if kind == "changed":
        (output / "core.json").write_text("changed")
    elif kind == "unlisted":
        (output / "accidental.env").write_text("fixture")
    elif kind == "missing":
        del manifest["files"][release.WHEEL_REL]
    elif kind == "escape":
        manifest["files"]["../outside.json"] = "0" * 64
    elif kind == "scope":
        manifest["public_data_redistribution_authorized"] = True
    else:
        manifest["profile_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        release.verify(output)


def test_new_release_keeps_existing_authenticated_app_and_cpu_limits():
    path = Path(__file__).resolve().parents[1] / "deploy/modal_release_v69.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    app = next(
        n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "App"
    )
    assert ast.literal_eval(app.args[0]) == "perfumery-ai-core"
    auth = next(
        n
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "asgi_app"
    )
    assert {k.arg: ast.literal_eval(k.value) for k in auth.keywords} == {
        "requires_proxy_auth": True
    }
    function = next(
        n
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "function"
    )
    values = {
        k.arg: ast.literal_eval(k.value) for k in function.keywords if k.arg != "image"
    }
    assert values["min_containers"] == 0 and values["max_containers"] == 1
    assert values["cpu"] == 1.0 and values["memory"] == 1024
    assert "gpu" not in values
    environment = next(
        n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "env"
    )
    assert any(
        isinstance(k, ast.Constant)
        and k.value == "PYTHONPATH"
        and isinstance(v, ast.Constant)
        and v.value == "/root"
        for k, v in zip(environment.args[0].keys, environment.args[0].values)
    )


def test_remote_stream_verifier_preserves_exact_json_and_progress():
    from scripts.verify_modal_v69 import verify_stream

    expected = {"recipe": [], "closest_candidate": [{"ingredient_id": "test"}]}
    lines = [
        "event: progress",
        'data: {"percent": 0}',
        "",
        "event: result",
        "data: " + json.dumps(expected),
        "",
        "event: progress",
        'data: {"percent": 100}',
        "",
    ]
    assert verify_stream(lines, expected)["exact_result_match"] is True
    with pytest.raises(ValueError):
        verify_stream(lines, {"recipe": []})
    with pytest.raises(ValueError):
        verify_stream(
            ["event: error", 'data: {"code":"INFERENCE_FAILED"}', ""], expected
        )
    with pytest.raises(ValueError):
        verify_stream(
            [
                "event: progress",
                'data: {"percent": 50}',
                "",
                "event: progress",
                'data: {"percent": 10}',
                "",
                *lines,
            ],
            expected,
        )


def test_audit_stream_verifier_rejects_missing_or_error_frames():
    from scripts.verify_modal_v69 import audit_frames

    value = {"sequence": 0, "data": {"status": "completed"}}
    assert audit_frames("event: done\r\ndata: " + json.dumps(value) + "\r\n\r\n") == [
        {"event": "done", "data": value}]
    for invalid in ("", ": heartbeat\n\n", 'event: error\ndata: {"code":"FAILED"}\n\n'):
        with pytest.raises(ValueError):
            audit_frames(invalid)
