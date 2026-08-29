"""Modal CPU deployment for Perfumery AI Core.

Deploy with::

    python -m modal deploy deploy/modal_app.py
"""

import hashlib
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
WHEEL = (
    ROOT
    / "dist"
    / "automatic-safe-activation-v1"
    / "perfumery_ai_core-1.4.0-py3-none-any.whl"
)
REGISTRY = ROOT / "benchmarks" / "industrial_ingredient_registry_v1.db"
REMOTE_WHEEL = "/opt/perfumery/perfumery_ai_core-1.4.0-py3-none-any.whl"
REMOTE_REGISTRY = "/opt/perfumery/industrial_ingredient_registry_v1.db"
WHEEL_SHA256 = "416a6eec32fb2c484aaf6bb2cc8a6a7cc6b661ccc18ff70674ad250a9ad8a120"
REGISTRY_SHA256 = "d837ccde2146a67d616a821dd926ff67dcc6bbb550b26da6599f72989a3c6765"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if modal.is_local():
    if not WHEEL.is_file() or not REGISTRY.is_file():
        raise RuntimeError(
            "Modal deployment requires the sealed wheel and registry artifacts"
        )
    if (
        _sha256_file(WHEEL) != WHEEL_SHA256
        or _sha256_file(REGISTRY) != REGISTRY_SHA256
    ):
        raise RuntimeError("Modal deployment artifact hash mismatch")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "fastapi[standard]==0.116.1",
        "cryptography==46.0.3",
    )
    .add_local_file(WHEEL, REMOTE_WHEEL, copy=True)
    .run_commands(f"python -m pip install --no-cache-dir {REMOTE_WHEEL}")
    .add_local_file(REGISTRY, REMOTE_REGISTRY, copy=True)
)

app = modal.App("perfumery-ai-core")


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
<div class="grid"><label>위험등급<select id="risk"><option value="1">1 · 기본 안전</option><option value="2">2 · 조건부 허용</option></select></label>
<label>시장<select id="region"><option>EU</option><option>KR</option><option>US</option></select></label>
<label>제품군<select id="category"><option>eau_de_parfum</option><option>eau_de_toilette</option><option>shampoo</option><option>candle</option><option>room_spray</option></select></label>
<label>원료 최대 $/kg<input id="price" type="number" value="180" min="10" max="300"></label>
<label>최대 원료 수<input id="count" type="number" value="12" min="6" max="20"></label></div>
<button id="run">조향식 생성</button><div id="status" class="status"></div>
<table><thead><tr><th>원료</th><th>노트</th><th>농축액 %</th><th>위험</th><th>$/kg</th></tr></thead><tbody id="formula"></tbody></table>
<details><summary>전체 계산 결과</summary><pre id="raw"></pre></details></main><script>
const q=id=>document.getElementById(id);q('run').onclick=async()=>{q('status').textContent='계산 중...';q('formula').replaceChildren();
try{const response=await fetch('/v1/formulas',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({brief:q('brief').value,max_risk_tier:Number(q('risk').value),target_region:q('region').value,product_category:q('category').value,max_ingredient_price_per_kg:Number(q('price').value),max_ingredients:Number(q('count').value)})});
const data=await response.json();if(!response.ok)throw new Error(data.detail||'요청 실패');q('status').textContent=`${data.status} · 안전 게이트 ${data.safety.internal_gate_passed?'PASS':'BLOCK'} · 원료 ${data.recipe.length}개`;
for(const line of data.recipe){const tr=document.createElement('tr');for(const value of [line.name,line.pyramid,line.concentrate_percent,line.risk_tier,line.price_per_kg]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td)}q('formula').appendChild(tr)}q('raw').textContent=JSON.stringify(data,null,2)}catch(error){q('status').textContent=error.message}};
</script></body></html>"""


def create_web_app(registry_path: str = REMOTE_REGISTRY):
    """Build the exact FastAPI application used locally and on Modal."""

    from collections import deque
    import threading
    import time

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, ConfigDict, Field

    from fragrance_ai import NaturalLanguagePerfumeryAI, RecipeConstraints
    from fragrance_ai.recommender.industrial_catalog import IndustrialIngredientRegistry

    class FormulaRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        brief: str = Field(min_length=1, max_length=2_000)
        max_risk_tier: int = Field(default=1, ge=1, le=2)
        max_ingredient_price_per_kg: float = Field(default=180.0, gt=0, le=300)
        max_formula_cost_per_kg: float = Field(default=160.0, gt=0, le=500)
        min_availability: float = Field(default=0.75, ge=0.5, le=1.0)
        target_similarity: float = Field(default=90.0, ge=50, le=95)
        product_concentration_percent: float = Field(default=15.0, gt=0, le=30)
        max_ingredients: int = Field(default=12, ge=6, le=20)
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
    with IndustrialIngredientRegistry(registry_path) as registry:
        catalog_snapshot = {**registry.stats(), "registry_sha256": registry.sha256}
    request_times: deque[float] = deque()
    rate_lock = threading.Lock()

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
        return {
            "status": "ok",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": WHEEL_SHA256,
            "registry_sha256": REGISTRY_SHA256,
        }

    @web.get("/v1/catalog")
    def catalog() -> dict:
        return dict(catalog_snapshot)

    @web.post("/v1/formulas")
    def formulas(request: FormulaRequest) -> dict:
        enforce_formula_rate_limit()
        constraints = RecipeConstraints(
            max_risk_tier=request.max_risk_tier,
            max_ingredient_price_per_kg=request.max_ingredient_price_per_kg,
            max_formula_cost_per_kg=request.max_formula_cost_per_kg,
            min_availability=request.min_availability,
            target_similarity=request.target_similarity,
            product_concentration_percent=request.product_concentration_percent,
            max_ingredients=request.max_ingredients,
            allow_rare=False,
            target_region=request.target_region,
            product_category=request.product_category,
            simulation_draws=64,
            physics_search_population=2,
            minimum_realism_score=50.0,
        )
        try:
            with NaturalLanguagePerfumeryAI() as ai:
                result = ai.create_recipe(request.brief.strip(), constraints)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        payload = result.to_dict()
        payload["deployment"] = {
            "provider": "modal",
            "runtime": "cpu",
            "gpu_required": False,
            "wheel_sha256": WHEEL_SHA256,
            "registry_sha256": REGISTRY_SHA256,
        }
        return payload

    return web


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=120,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(requires_proxy_auth=True)
def web():
    return create_web_app()
