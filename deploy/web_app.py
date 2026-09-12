"""Shared API factory; importing it never builds or validates a deployment."""

import hashlib
import json
import threading
from datetime import date
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
WHEEL = (
    ROOT
    / "dist"
    / "body-lotion-v32"
    / "perfumery_ai_core-1.4.0-py3-none-any.whl"
)
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REMOTE_WHEEL = "/opt/perfumery/perfumery_ai_core-1.4.0-py3-none-any.whl"
REMOTE_REGISTRY = "/opt/perfumery/industrial_ingredient_registry_v1.db"
WHEEL_SHA256 = "0d0205af3db8ca9dbcdfe9cbb0c117ba6be3ae4c64c448f9c742c864a9ed3f1c"
REGISTRY_SHA256 = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"
RUNTIME_CATALOG = ROOT / "dist" / "body-lotion-v32" / "profile-extension" / "runtime_catalog_v3.json.gz"
REMOTE_CATALOG = "/opt/perfumery/runtime_catalog.json.gz"
RUNTIME_CATALOG_SHA256 = "78410c521d637a2fb2737b1ae707f991a96667a188b09d95482864ce8b541a7c"
_RUNTIME_CATALOG_CACHE = {}
_RUNTIME_CATALOG_CACHE_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


INDEX_HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Perfumery AI Core</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f3ff;color:#111827}
main{max-width:1100px;margin:auto;padding:28px}.hero{padding:24px;border-radius:20px;color:white;
background:linear-gradient(135deg,#111827,#3730a3,#991b1b)}textarea,input,select,button{font:inherit}
textarea{width:100%;box-sizing:border-box;padding:12px;border-radius:12px;border:1px solid #c7d2fe}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}
label{display:grid;gap:6px;background:white;padding:12px;border-radius:12px}button{padding:13px 20px;
border:0;border-radius:12px;background:#4338ca;color:white;font-weight:700;cursor:pointer}
pre{white-space:pre-wrap;background:#111827;color:#e5e7eb;padding:16px;border-radius:12px;overflow:auto}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left}
.note{color:#4b5563}.status{font-weight:700;margin:14px 0}</style></head><body><main>
<section class="hero"><h1>Perfumery AI Core</h1><p>CPU 자연어 조향 · 안전/가격/가용성 제약 · 29,240개 산업 레지스트리</p></section>
<p class="note">원하는 향을 입력하면 안전 후보 pool에서 정량 조향식을 생성합니다. 계산 점수는 사람 후각 정확도나 제조 승인이 아닙니다.</p>
<textarea id="brief" rows="4">깨끗하고 시원한 시트러스 우디 향, 은은한 머스크와 드라이한 잔향</textarea>
<div class="grid"><label>위험등급<select id="risk"><option value="1">1 · 기본 안전</option><option value="2">2 · 조건부 전체 레지스트리</option></select></label>
<label>시장<select id="region"><option>EU</option><option>KR</option><option>US</option></select></label>
<label>제품군<select id="category"><option>eau_de_parfum</option><option>eau_de_toilette</option><option>shampoo</option><option>candle</option><option>room_spray</option></select></label>
<label>원료 최대 $/kg<input id="price" type="number" value="180" min="10" max="300"></label>
<label>최대 원료 수<input id="count" type="number" value="12" min="6" max="20"></label></div>
<button id="run">조향식 생성</button><div id="status" class="status"></div>
<table><thead><tr><th>원료</th><th>노트</th><th>농축액 %</th><th>위험</th><th>$/kg</th></tr></thead><tbody id="formula"></tbody></table>
<h2>시간별 향 변화</h2>
<table><thead><tr><th>시간</th><th>구간</th><th>오프닝 대비 강도</th><th>주요 향축</th></tr></thead><tbody id="temporal"></tbody></table>
<h2>원료별 잔존 농도 예측</h2>
<p class="note">도포 표면의 1차 증발 프록시이며 밀폐 용기 실측 농도가 아닙니다.</p>
<table><thead><tr><th>원료</th><th>0분</th><th>15분</th><th>60분</th><th>240분</th><th>480분</th></tr></thead><tbody id="concentration"></tbody></table>
<details><summary>전체 계산 결과</summary><pre id="raw"></pre></details></main><script>
const q=id=>document.getElementById(id);q('run').onclick=async()=>{q('status').textContent='계산 중...';for(const id of ['formula','temporal','concentration'])q(id).replaceChildren();
try{const risk=Number(q('risk').value);const response=await fetch('/v1/formulas',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({brief:q('brief').value,max_risk_tier:risk,enable_registry_trace_candidates:risk===2,target_region:q('region').value,product_category:q('category').value,max_ingredient_price_per_kg:Number(q('price').value),max_ingredients:Number(q('count').value)})});
const data=await response.json();if(!response.ok)throw new Error(data.detail||'요청 실패');q('status').textContent=`${data.status} · 안전 게이트 ${data.safety.internal_gate_passed?'PASS':'BLOCK'} · 원료 ${data.recipe.length}개`;
for(const line of data.recipe){const tr=document.createElement('tr');for(const value of [line.name,line.pyramid,line.concentrate_percent,line.risk_tier,line.price_per_kg]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)}q('formula').appendChild(tr)}
for(const point of data.temporal_profile||[]){const tr=document.createElement('tr');for(const value of [`${point.minutes}분`,point.phase,`${point.relative_to_opening_intensity_percent}%`,(point.dominant_dimensions||[]).join(', ')]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)}q('temporal').appendChild(tr)}
for(const profile of data.ingredient_temporal_profile||[]){const tr=document.createElement('tr');const values=[profile.name,...profile.points.map(point=>`${point.estimated_remaining_concentrate_percent}%`)];for(const value of values){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)}q('concentration').appendChild(tr)}
q('raw').textContent=JSON.stringify(data,null,2)}catch(error){q('status').textContent=error.message}};
</script></body></html>"""


def create_web_app(registry_path: str = REMOTE_REGISTRY, *, runtime_catalog_path: str | None = None, allow_legacy_research: bool = False, language_backend=None, perception_guidance=None, lotion_perception_guidance=None, stock_mixture_predictor=None, catalog_manifest_path=None, catalog_manifest_sha256=None):
    """Build the exact FastAPI application used locally and on Modal."""

    from collections import deque
    import time

    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, ConfigDict, Field

    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.catalog import IngredientCatalog
    from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry
    from fragrance_ai.recommender.registry_activation import (
        activate_registry_conditionals,
        load_runtime_catalog,
    )
    from fragrance_ai.recommender.runtime_cache import InferenceCache, InferenceBusy
    from fragrance_ai.recommender.runtime import RuntimeAIFactory, load_verified_catalog_bundle, MANIFEST_ENV, MANIFEST_HASH_ENV
    from fragrance_ai.recommender.models import MAX_FORMULA_INGREDIENTS

    class FormulaRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        brief: str = Field(min_length=1, max_length=2_000)
        max_risk_tier: int = Field(default=1, ge=1, le=2)
        max_ingredient_price_per_kg: float = Field(default=300.0, gt=0, le=10_000_000)
        max_formula_cost_per_kg: float = Field(default=180.0, gt=0, le=10_000_000)
        min_availability: float = Field(default=0.75, ge=0, le=1.0)
        target_similarity: float = Field(default=95.0, gt=0, le=100)
        product_concentration_percent: float = Field(default=15.0, gt=0, le=30)
        max_ingredients: int = Field(default=MAX_FORMULA_INGREDIENTS, ge=3, le=MAX_FORMULA_INGREDIENTS)
        enable_registry_trace_candidates: bool = False
        require_full_profile_match: bool = True
        experimental_disable_safety: bool = False
        target_region: str = Field(default="EU", pattern=r"^(EU|KR|US)$")
        product_category: str = Field(
            default="eau_de_parfum",
            pattern=(
                r"^(eau_de_parfum|eau_de_toilette|eau_de_cologne|shampoo|"
                r"body_wash|candle|room_spray|diffuser)$"
            ),
        )

    web = FastAPI(
        title="Perfumery AI Core",
        version="1.4.0",
        description="CPU natural-language perfumery formulation API",
    )
    web.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://ppwk-perfumery-ai-core.hf.space",
            "http://127.0.0.1:7860",
            "http://localhost:7860",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type"],
    )
    registry_file = Path(registry_path).expanduser().resolve()
    if _sha256_file(registry_file) != REGISTRY_SHA256:
        raise ValueError("industrial registry hash mismatch")
    # The optional path is server configuration, never an HTTP request field.
    # Explicit partial configuration must not borrow a hash from the environment.
    import os
    from fragrance_ai.recommender.local_runtime import configured_pair
    environment_binding = configured_pair(MANIFEST_ENV, MANIFEST_HASH_ENV, 'catalog')
    explicit_binding = catalog_manifest_path is not None or catalog_manifest_sha256 is not None
    if not explicit_binding:
        catalog_manifest_path, catalog_manifest_sha256 = environment_binding
    bundle = None
    if explicit_binding or catalog_manifest_path or catalog_manifest_sha256:
        if runtime_catalog_path is not None:
            raise ValueError('choose one runtime catalog binding, not both path and manifest')
        bundle = load_verified_catalog_bundle(catalog_manifest_path, catalog_manifest_sha256)
        if bundle['binding']['registry_sha256'] != REGISTRY_SHA256:
            raise ValueError('manifest catalog registry differs from the API registry')
    wheel_sha = bundle['binding']['wheel_sha256'] if bundle else WHEEL_SHA256
    catalog_sha = bundle['binding']['sha256'] if bundle else RUNTIME_CATALOG_SHA256
    manifest_sha = bundle['manifest_sha256'] if bundle else None
    prepared_path = Path(runtime_catalog_path) if runtime_catalog_path else (RUNTIME_CATALOG if modal.is_local() else Path(REMOTE_CATALOG))
    if bundle:
        prepared_path = bundle['catalog_path']
    prepared = bool(catalog_sha and prepared_path.is_file())
    if (runtime_catalog_path or catalog_sha) and not prepared:
        raise ValueError("required runtime catalog is unavailable")
    if prepared and not bundle and _sha256_file(prepared_path) != catalog_sha:
        raise ValueError("runtime catalog hash mismatch")
    cache_key = (str(registry_file), wheel_sha, catalog_sha if prepared else 'dynamic', manifest_sha)
    with _RUNTIME_CATALOG_CACHE_LOCK:
        cached = _RUNTIME_CATALOG_CACHE.get(cache_key)
        if cached is None:
            if bundle:
                runtime_catalog, activation_report, registry_stats = (
                    bundle['catalog'], bundle['activation_report'], bundle['registry_stats'])
            elif prepared:
                runtime_catalog, activation_report, registry_stats = load_runtime_catalog(
                    prepared_path, expected_sha256=catalog_sha,
                    expected_wheel_sha256=wheel_sha, expected_registry_sha256=REGISTRY_SHA256,
                )
                # A minimal evidence catalog need not duplicate the full
                # registry dashboard. Preserve existing API coverage fields by
                # reading them once at startup, never once per formula request.
            else:
                with IndustrialIngredientRegistry(registry_path) as registry:
                    registry_stats = registry.stats()
                runtime_catalog, activation_report = activate_registry_conditionals(
                    IngredientCatalog.load_builtin(), registry_path, expected_sha256=REGISTRY_SHA256,
                )
            if "safety_screened" not in registry_stats:
                with IndustrialIngredientRegistry(registry_file) as registry:
                    registry_stats = registry.stats()
            _RUNTIME_CATALOG_CACHE[cache_key] = (
                runtime_catalog,
                activation_report,
                registry_stats,
            )
        else:
            runtime_catalog, activation_report, registry_stats = cached
    catalog_snapshot = {
        **registry_stats,
        **runtime_catalog.stats(),
        **activation_report.to_dict(),
        "registry_sha256": activation_report.registry_sha256,
    }
    strict_factory = RuntimeAIFactory(catalog=runtime_catalog, minimum_profile_target=95.,
        require_full_profile_match=True, allow_experimental_safety=False, perception_guidance=perception_guidance,
        manifest_sha256=manifest_sha)
    binding_contract = {'catalog_manifest_sha256': manifest_sha, 'catalog_sha256': catalog_sha,
                        'wheel_sha256': wheel_sha, 'registry_sha256': REGISTRY_SHA256,
                        'material_snapshot_sha256': strict_factory.runtime_contract['material_snapshot_sha256']}
    catalog_snapshot['runtime_binding'] = dict(binding_contract)
    bound_files = (bundle['manifest_path'], bundle['catalog_path']) if bundle else ()
    def file_metadata():
        return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in bound_files)
    bound_metadata = file_metadata()
    def assert_catalog_current():
        strict_factory.assert_current_snapshot()
        if environment_binding != configured_pair(MANIFEST_ENV, MANIFEST_HASH_ENV, 'catalog'):
            raise ValueError('runtime catalog configuration changed; reload API')
        try:
            current = file_metadata()
        except OSError as error:
            raise ValueError('runtime catalog snapshot unavailable; reload API') from error
        if current != bound_metadata:
            raise ValueError('runtime catalog snapshot changed; reload API')
    request_times: deque[float] = deque()
    rate_lock = threading.Lock()
    response_cache = InferenceCache(max_pending=4, max_followers=8)
    web.state.formula_cache = response_cache

    def enforce_formula_rate_limit() -> None:
        now = time.monotonic()
        with rate_lock:
            while request_times and now - request_times[0] >= 60.0:
                request_times.popleft()
            if len(request_times) >= 30:
                raise HTTPException(
                    status_code=429,
                    detail="formula request limit exceeded",
                    headers={"Retry-After": "60"},
                )
            request_times.append(now)

    @web.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

    @web.get("/health")
    async def health() -> dict:
        assert_catalog_current()
        return {
            "status": "ok",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": wheel_sha,
            "registry_sha256": REGISTRY_SHA256,
        }

    @web.get("/v1/catalog")
    def catalog() -> dict:
        assert_catalog_current()
        return dict(catalog_snapshot)

    def generate_formula(request: FormulaRequest, response: Response, *, target_profile_override=None, explicit_bans=(), rate_limited=False, fixed_formula_weights=None, intent_controls=None, progress_callback=None) -> dict:
        if not rate_limited:
            enforce_formula_rate_limit()
        constraints = RecipeConstraints(
            max_risk_tier=request.max_risk_tier,
            max_ingredient_price_per_kg=request.max_ingredient_price_per_kg,
            max_formula_cost_per_kg=request.max_formula_cost_per_kg,
            min_availability=request.min_availability,
            target_similarity=request.target_similarity,
            product_concentration_percent=request.product_concentration_percent,
            max_ingredients=request.max_ingredients,
            explicit_bans=set(explicit_bans),
            allow_rare=False,
            enable_registry_trace_candidates=(
                request.enable_registry_trace_candidates
            ),
            experimental_disable_safety=(
                request.experimental_disable_safety
                and request.enable_registry_trace_candidates
            ),
            target_region=request.target_region,
            product_category=request.product_category,
            simulation_draws=200 if request.require_full_profile_match else 64,
            physics_search_population=6 if request.require_full_profile_match else 7,
            minimum_realism_score=65.0 if request.require_full_profile_match else 50.0,
        )
        as_of = date.today()
        key_data = {
            "wheel": wheel_sha, "registry": REGISTRY_SHA256,
            "runtime_contract": strict_factory.runtime_contract,
            "catalog": catalog_sha, "as_of": as_of.isoformat(),
            "request": request.model_dump(mode="json"),
        }
        if target_profile_override is not None or explicit_bans:
            key_data["intent_edits"] = {"target_profile": target_profile_override, "explicit_bans": sorted(explicit_bans)}
        if fixed_formula_weights is not None:
            key_data["fixed_formula_weights"] = fixed_formula_weights
        if intent_controls:
            key_data["intent_controls"] = intent_controls
        request_key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()

        def compute():
            # Explicit legacy research requests remain compatible. Production
            # and the default path cannot downgrade the 95-point quality floor.
            production = os.environ.get("PERFUMERY_AI_ENV", "development").strip().lower() == "production"
            if not request.require_full_profile_match and (production or not allow_legacy_research):
                raise ValueError("this server requires the strict 95-point profile gate; legacy research must be explicitly enabled by its operator")
            engine = strict_factory() if request.require_full_profile_match else NaturalLanguagePerfumeryAI(
                catalog=runtime_catalog, perception_guidance=strict_factory.perception_guidance)
            with engine as ai:
                if fixed_formula_weights is not None:
                    from fragrance_ai.recommender.fixed_assessment import assess_fixed_formula
                    return assess_fixed_formula(ai, request.brief.strip(), constraints, fixed_formula_weights,
                                                as_of=as_of, target_profile_override=target_profile_override,
                                                intent_controls=intent_controls).to_dict()
                return ai.create_recipe(request.brief.strip(), constraints, as_of=as_of,
                                        target_profile_override=target_profile_override,
                                        **({"intent_controls": intent_controls} if intent_controls else {}),
                                        **({"progress_callback": progress_callback} if progress_callback is not None else {})).to_dict()

        try:
            # Optional operator-provided promotion, continual or language-model
            # state may change independently of the immutable deployment image.
            assert_catalog_current()
            tracked = {'PERFUMERY_AI_PERCEPTION_MANIFEST', 'PERFUMERY_AI_PERCEPTION_MANIFEST_SHA256', 'PERFUMERY_AI_ENV'} if strict_factory.perception_guidance is not None else set()
            # Lotion model state cannot affect this perfume-only cache. Its
            # own lane independently checks model drift and cache identity.
            tracked.update({'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST', 'PERFUMERY_AI_LOTION_PERCEPTION_MANIFEST_SHA256'})
            tracked.update({MANIFEST_ENV, MANIFEST_HASH_ENV})
            # The local profile hash is part of runtime_contract/cache identity
            # and its snapshot is checked on both sides of the cached call.
            from fragrance_ai.recommender.local_runtime import PROFILE_ENV, REFERENCE_ENV
            tracked.update({PROFILE_ENV, REFERENCE_ENV})
            cacheable = not any(name.startswith("PERFUMERY_AI_") and value and name not in tracked for name, value in os.environ.items())
            payload, cache_status = response_cache.run(request_key, compute, cacheable=cacheable)
            assert_catalog_current()
        except (InferenceBusy, TimeoutError) as error:
            raise HTTPException(status_code=503, detail=str(error), headers={"Retry-After": "5"}) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response.headers["X-Perfumery-Cache"] = cache_status
        payload["deployment"] = {
            "provider": "modal",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": wheel_sha,
            "registry_sha256": REGISTRY_SHA256,
            "catalog_snapshot": dict(binding_contract),
            "perception_model": strict_factory.runtime_contract['perception_model'],
            "product_model": {**strict_factory.runtime_contract['product_model'],
                              "requested_product": request.product_category,
                              "scope": "perfume_or_fragrance_concentrate_not_finished_lotion"},
            "registry_connected_total": (
                activation_report.reference_molecules_connected
            ),
            "registry_conditional_trace_active": (
                activation_report.conditional_trace_candidates_active
            ),
            "registry_activation_mode": activation_report.activation_mode,
        }
        return payload

    @web.post("/v1/formulas")
    def formulas(request: FormulaRequest, response: Response) -> dict:
        return generate_formula(request, response)

    from deploy.formula_stream import register_formula_stream
    register_formula_stream(web, FormulaRequest, generate_formula, enforce_formula_rate_limit, assert_catalog_current)

    # Audit/report routes transform backend-owned history only. They cannot
    # invoke generate_formula and do not mutate the recipe JSON contract.
    from deploy.audit_api import register_audit_routes
    register_audit_routes(web, enforce_formula_rate_limit)

    from fragrance_ai.platform.ai_extensions import register_ai_extensions
    register_ai_extensions(web, FormulaRequest, runtime_catalog, generate_formula, enforce_formula_rate_limit,
                           language_backend=language_backend, perception_guidance=strict_factory.perception_guidance,
                           lotion_perception_guidance=lotion_perception_guidance, stock_mixture_predictor=stock_mixture_predictor,
                           catalog_contract=binding_contract, runtime_guard=assert_catalog_current)
    return web
