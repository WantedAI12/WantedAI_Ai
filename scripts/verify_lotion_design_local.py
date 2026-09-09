"""Exercise the packaged API in-process, with outbound connections forbidden."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-auto-base", action="store_true")
    p.add_argument('--candidate-manifest', type=Path)
    p.add_argument('--candidate-wheel', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("choose a new output file")
    if bool(args.candidate_manifest) != bool(args.candidate_wheel):
        p.error('candidate verification requires both manifest and wheel')
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.package_root.resolve()))
    import fragrance_ai
    assert Path(fragrance_ai.__file__).resolve().is_relative_to(args.package_root.resolve())
    def guard(event, values):
        if event == "socket.connect":
            # Windows asyncio builds its self-pipe with a loopback socketpair.
            host = values[1][0] if isinstance(values[1], tuple) else ""
            try:
                if ipaddress.ip_address(host).is_loopback:
                    return
            except ValueError:
                pass
            raise RuntimeError("outbound network forbidden in packaged verification")
        if event == "socket.getaddrinfo":
            raise RuntimeError("network forbidden in packaged verification")
    sys.addaudithook(guard)
    import deploy.modal_app as deployment
    if args.candidate_manifest:
        # Configure only this verification process. Never edit deployment
        # defaults or relax a digest check to exercise a separate candidate.
        artifact = json.loads(args.candidate_manifest.read_text(encoding='utf-8'))['runtime_catalog']
        assert artifact['registry_sha256'] == deployment.REGISTRY_SHA256
        assert hashlib.sha256(args.candidate_wheel.read_bytes()).hexdigest() == artifact['wheel_sha256']
        deployment.WHEEL = args.candidate_wheel.resolve()
        deployment.WHEEL_SHA256 = artifact['wheel_sha256']
        deployment.RUNTIME_CATALOG = args.candidate_manifest.resolve().parent / artifact['path']
        deployment.RUNTIME_CATALOG_SHA256 = artifact['sha256']
    from deploy.modal_app import create_web_app, REGISTRY, WHEEL, WHEEL_SHA256
    from fastapi.testclient import TestClient
    report = {"scope": "packaged_local_ASGI_not_deployed_HTTP", "wheel_sha256": WHEEL_SHA256,
              "package_module": str(fragrance_ai.__file__), "external_network_forbidden": True}
    if args.candidate_manifest:
        report['candidate_manifest_sha256'] = hashlib.sha256(args.candidate_manifest.read_bytes()).hexdigest()
    assert hashlib.sha256(WHEEL.read_bytes()).hexdigest() == WHEEL_SHA256
    checked = 0
    for source in (ROOT / "fragrance_ai").rglob("*.py"):
        packaged = args.package_root / source.relative_to(ROOT)
        assert packaged.exists(), f"missing packaged source: {source}"
        assert source.read_bytes() == packaged.read_bytes(), f"stale packaged source: {source}"
        checked += 1
    report["source_files_byte_verified"] = checked
    with TestClient(create_web_app(str(REGISTRY))) as client:
        initial = {"formula": {"brief": "..."}}
        ready = client.post("/v1/briefs/prepare", json=initial)
        assert ready.status_code == 200
        answer = client.post("/v1/briefs/clarify", json={"request": initial,
            "request_id": ready.json()["request_id"], "answers": {"formula.brief": "citrus woody scent"}})
        assert answer.status_code == 200 and answer.json()["prepared"]["status"] == "ready"
        report["clarification"] = "passed"
        start = time.perf_counter()
        designed = client.post("/v1/applications/body-lotion/design", json={"brief": "citrus woody scent",
            "registry_pool": "conditional_research", "max_risk_tier": 2})
        report["design_seconds"] = time.perf_counter()-start
        assert designed.status_code == 200
        r = designed.json()
        assert r["candidate_recipe"]
        assert abs(sum(row["concentrate_percent"] for row in r["candidate_recipe"])-100) < 1e-6
        assert r["simulation"]["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-10
        report["design"] = {key: r.get(key) for key in ("status", "score", "profile_target_met", "solver_calls")}
        report["catalog_candidates"] = r["estimation"]["candidate_count"]
        report["human_similarity_percent"] = r["human_similarity_percent"]
        report["manufacturing_approved"] = r["manufacturing_approved"]
        report['odor_projection_versions'] = r['preparation'].get('odor_projection_versions')
        bad = client.post("/v1/applications/body-lotion/design", json={"brief": "woody scent",
             "storage": {"ph": 5.5, "days": 30, "minimum_parent_retention_percent": 95}})
        assert bad.status_code == 422
        report["missing_kinetics_not_assumed_stable"] = True
        # The frozen 400-case baseline is kept intact. Only this negative-only
        # phase case was affected by the subsequent target-inheritance fix.
        corrected = client.post("/v1/applications/body-lotion/design", json={
            "brief": "opening no sweetness, drydown woody musk",
            "registry_pool": "conditional_research", "max_risk_tier": 2})
        assert corrected.status_code == 200
        correction = corrected.json()
        assert correction["score"] is not None
        assert all(row["score"] is not None for row in correction["timepoint_assessments"])
        report["targeted_correction_extra_7"] = {key: correction.get(key) for key in
            ("status", "score", "profile_target_met", "solver_calls")}
        if args.verify_auto_base:
            automatic = client.post("/v1/applications/body-lotion/design", json={
                "brief": "white floral, earthy scent", "registry_pool": "conditional_research",
                "max_risk_tier": 2, "base_design": {}})
            assert automatic.status_code == 200
            a = automatic.json()
            assert a["profile_target_met"] and a["score"] + 1e-8 >= 95
            selected_oil = a['base_design']['selected_oil_base_percent']
            if a['base_design']['fixed_reference_passed95']:
                assert selected_oil == 10 and not a['base_design']['reference_formula_modified']
            else:
                assert selected_oil == 30 and a['base_design']['reference_formula_modified']
            assert not a["base_design"]["stability_verified"]
            total = sum(c["finished_product_percent"] for c in a["base_design"]["finished_product_base"])
            total += sum(c["finished_product_percent"] for c in a["candidate_recipe"])
            assert abs(total-100.) < 1e-8
            context = a["estimation"]["application_context"]
            assert next(c for c in context["base_components"] if c["role"] == "oil")["mass_percent"] == selected_oil
            for s in [a["simulation"], *a["scenario_simulations"]]:
                assert s["diagnostics"]["mass_balance_max_abs_error_mg_cm2"] < 1e-10
            report["auto_base_design"] = {"score": a["score"],
                "fixed_reference_score": a["base_design"]["fixed_reference_score"],
                "selected_oil_base_percent": a["base_design"]["selected_oil_base_percent"],
                "finished_product_mass_percent_total": total,
                "manufacturing_approved": a["manufacturing_approved"]}
            # Keep testing an actual base change even if repaired source data
            # makes the earlier example pass directly on the fixed base.
            variant = client.post('/v1/applications/body-lotion/design', json={
                'brief':'citrus, aromatic scent', 'registry_pool':'conditional_research',
                'max_risk_tier':2, 'base_design':{}})
            assert variant.status_code == 200
            v = variant.json()
            assert v['profile_target_met'] and v['score']+1e-8 >= 95
            assert not v['base_design']['fixed_reference_passed95']
            changed_oil = v['base_design']['selected_oil_base_percent']
            # Both 20 and 30 are explicitly permitted default variants. New
            # candidates may reach the target at 20 instead of requiring 30.
            assert changed_oil in (20, 30) and v['base_design']['reference_formula_modified']
            changed_context = v['estimation']['application_context']
            assert next(c for c in changed_context['base_components'] if c['role'] == 'oil')['mass_percent'] == changed_oil
            report['changed_base_path'] = {'score':v['score'], 'selected_oil_base_percent':changed_oil,
                'fixed_reference_score':v['base_design']['fixed_reference_score'],
                'basis_cache_status':v.get('basis_cache_status')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
