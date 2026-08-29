"use strict";

const state = {
  token: sessionStorage.getItem("perfumery.token") || "",
  tenant: sessionStorage.getItem("perfumery.tenant") || "",
  projects: [],
  activeProject: null,
  formulas: [],
  activeFormula: null,
  activeVersion: null,
  versions: [],
  catalog: [],
  jobs: new Map(),
  pollers: new Map(),
};

const $ = (id) => document.getElementById(id);
const refs = Object.fromEntries([
  "tenantInput", "tokenInput", "connectButton", "disconnectButton", "serviceDot",
  "serviceText", "backendBadge", "projectForm", "projectName", "projectList",
  "refreshProjects", "formulaList", "recipeForm", "briefInput", "formulaName",
  "maxCost", "maxIngredients", "simulationDraws", "activeProjectLabel", "jobList",
  "queueCount", "formulaWorkspace", "welcomePanel", "activeFormulaName",
  "activeFormulaMeta", "formulaStatus", "versionBadge", "similarityMetric",
  "proxyMetric", "realismMetric", "costMetric", "ingredientMetric", "formulaTable",
  "formulaTotal", "totalBar", "normalizeButton", "saveVersionButton", "editNote",
  "addIngredientSelect", "profileChart", "pyramidChart", "revisionForm",
  "revisionInput", "accordForm", "accordName", "accordBrief", "refreshVersions",
  "versionTimeline", "compareLeft", "compareRight", "compareButton", "compareResult",
  "toastRegion",
].map((id) => [id, $(id)]));

refs.tenantInput.value = state.tenant;
refs.tokenInput.value = state.token;

function node(tag, options = {}, children = []) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined) element.textContent = String(options.text);
  if (options.type) element.type = options.type;
  if (options.value !== undefined) element.value = String(options.value);
  if (options.title) element.title = options.title;
  if (options.dataset) Object.assign(element.dataset, options.dataset);
  for (const child of children) element.append(child);
  return element;
}

function toast(message, type = "info") {
  const item = node("div", {className: `toast ${type}`, text: message});
  refs.toastRegion.append(item);
  setTimeout(() => item.remove(), 4200);
}

function resetTenantWorkspace() {
  state.projects = [];
  state.activeProject = null;
  state.formulas = [];
  state.activeFormula = null;
  state.activeVersion = null;
  state.versions = [];
  state.catalog = [];
  state.jobs.clear();
  state.pollers.clear();
  refs.activeProjectLabel.textContent = "프로젝트 미선택";
  renderProjects();
  renderFormulaList();
  renderJobs();
  showWelcome();
}

async function api(path, options = {}) {
  if (!state.token || !state.tenant) throw new Error("테넌트와 토큰을 먼저 연결하세요.");
  const headers = new Headers(options.headers || {});
  headers.set("Authorization", `Bearer ${state.token}`);
  headers.set("X-Tenant-ID", state.tenant);
  headers.set("X-Request-ID", `ui-${crypto.randomUUID()}`);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, {...options, headers});
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail;
    throw new Error(message || `HTTP ${response.status}`);
  }
  return payload;
}

async function checkHealth() {
  try {
    const response = await fetch("/health/ready", {cache: "no-store"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "not ready");
    refs.serviceDot.className = "state-dot online";
    refs.serviceText.textContent = "서비스 준비됨";
    refs.backendBadge.textContent = payload.horizontal_scaling ? "POSTGRES · SCALE" : payload.workspace_backend;
  } catch (_) {
    refs.serviceDot.className = "state-dot offline";
    refs.serviceText.textContent = "서비스 연결 실패";
    refs.backendBadge.textContent = "OFFLINE";
  }
}

async function connect() {
  const nextTenant = refs.tenantInput.value.trim();
  const nextToken = refs.tokenInput.value.trim();
  if (!nextTenant || !nextToken) return toast("테넌트와 토큰이 필요합니다.", "error");
  if (nextTenant !== state.tenant || nextToken !== state.token) resetTenantWorkspace();
  state.tenant = nextTenant;
  state.token = nextToken;
  sessionStorage.setItem("perfumery.tenant", state.tenant);
  sessionStorage.setItem("perfumery.token", state.token);
  try {
    const capabilities = await api("/v1/system/capabilities");
    refs.backendBadge.textContent = capabilities.horizontal_scaling ? "POSTGRES · SCALE" : capabilities.workspace_backend;
    await Promise.all([loadProjects(), loadCatalog(), loadJobs()]);
    toast("워크스페이스에 연결했습니다.", "success");
  } catch (error) {
    toast(`연결 실패: ${error.message}`, "error");
  }
}

function disconnect() {
  sessionStorage.removeItem("perfumery.tenant");
  sessionStorage.removeItem("perfumery.token");
  state.token = "";
  state.tenant = "";
  refs.tokenInput.value = "";
  refs.tenantInput.value = "";
  resetTenantWorkspace();
  toast("연결 정보를 해제했습니다.");
}

async function loadProjects() {
  const payload = await api("/v1/projects");
  state.projects = payload.items;
  if (state.activeProject) {
    state.activeProject = state.projects.find((item) => item.project_id === state.activeProject.project_id) || null;
  }
  renderProjects();
}

function renderProjects() {
  refs.projectList.replaceChildren();
  if (!state.projects.length) return refs.projectList.append(node("p", {className: "empty", text: "프로젝트가 없습니다."}));
  for (const project of state.projects) {
    const button = node("button", {
      className: `nav-item ${state.activeProject?.project_id === project.project_id ? "active" : ""}`,
      type: "button",
    }, [node("strong", {text: project.name}), node("small", {text: project.project_id})]);
    button.addEventListener("click", () => selectProject(project));
    refs.projectList.append(button);
  }
}

async function createProject(event) {
  event.preventDefault();
  try {
    const project = await api("/v1/projects", {
      method: "POST",
      body: JSON.stringify({name: refs.projectName.value.trim(), description: ""}),
    });
    refs.projectName.value = "";
    await loadProjects();
    await selectProject(project);
    toast("프로젝트를 생성했습니다.", "success");
  } catch (error) { toast(error.message, "error"); }
}

async function selectProject(project) {
  state.activeProject = project;
  state.activeFormula = null;
  state.activeVersion = null;
  refs.activeProjectLabel.textContent = project.name;
  renderProjects();
  showWelcome();
  await loadFormulas();
}

async function loadFormulas() {
  if (!state.activeProject) return;
  const payload = await api(`/v1/projects/${encodeURIComponent(state.activeProject.project_id)}/formulas`);
  state.formulas = payload.items;
  renderFormulaList();
}

function renderFormulaList() {
  refs.formulaList.replaceChildren();
  if (!state.activeProject) return refs.formulaList.append(node("p", {className: "empty", text: "프로젝트를 선택하세요."}));
  if (!state.formulas.length) return refs.formulaList.append(node("p", {className: "empty", text: "아직 처방이 없습니다."}));
  for (const formula of state.formulas) {
    const button = node("button", {
      className: `nav-item ${state.activeFormula?.formula_id === formula.formula_id ? "active" : ""}`,
      type: "button",
    }, [
      node("strong", {text: formula.name}),
      node("small", {text: `${formula.kind.toUpperCase()} · v${formula.latest_version.version_number}`}),
    ]);
    button.addEventListener("click", () => loadFormula(formula.project_id, formula.formula_id));
    refs.formulaList.append(button);
  }
}

async function enqueue(path, body, label) {
  const job = await api(path, {method: "POST", body: JSON.stringify(body)});
  state.jobs.set(job.job_id, job);
  renderJobs();
  toast(`${label} 작업을 큐에 등록했습니다.`, "success");
  pollJob(job.job_id);
  return job;
}

async function generateRecipe(event) {
  event.preventDefault();
  if (!state.activeProject) return toast("먼저 프로젝트를 선택하세요.", "error");
  try {
    await enqueue("/v1/jobs/recipes", {
      project_id: state.activeProject.project_id,
      brief: refs.briefInput.value.trim(),
      name: refs.formulaName.value.trim() || "Generated formula",
      constraints: {
        max_formula_cost_per_kg: Number(refs.maxCost.value),
        max_ingredients: Number(refs.maxIngredients.value),
        simulation_draws: Number(refs.simulationDraws.value),
      },
    }, "처방 생성");
  } catch (error) { toast(error.message, "error"); }
}

async function pollJob(jobId) {
  if (state.pollers.has(jobId)) return;
  state.pollers.set(jobId, true);
  const poll = async () => {
    try {
      const job = await api(`/v1/jobs/${encodeURIComponent(jobId)}`);
      if (!state.pollers.has(jobId)) return;
      state.jobs.set(jobId, job);
      renderJobs();
      if (job.status === "succeeded") {
        state.pollers.delete(jobId);
        const outcome = job.result?.result;
        if (outcome?.status === "no_safe_match") {
          toast("안전·유사도 조건을 통과한 처방이 없어 저장하지 않았습니다.", "error");
        } else {
          toast("작업이 완료됐습니다.", "success");
        }
        await loadFormulas();
        const formula = job.result?.workspace_formula;
        const version = job.result?.workspace_version;
        const projectId = formula?.project_id || version?.project_id || job.payload?.project_id;
        const formulaId = formula?.formula_id || version?.formula_id || job.payload?.formula_id;
        if (projectId && formulaId) await loadFormula(projectId, formulaId);
      } else if (["failed", "cancelled"].includes(job.status)) {
        state.pollers.delete(jobId);
        toast(`작업 실패: ${job.error_code || job.status}`, "error");
      }
    } catch (error) {
      if (!state.pollers.has(jobId)) return;
      state.pollers.delete(jobId);
      toast(`작업 조회 실패: ${error.message}`, "error");
    }
  };
  await poll();
  while (state.pollers.has(jobId)) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (state.pollers.has(jobId)) await poll();
  }
}

async function loadJobs() {
  const payload = await api("/v1/jobs?limit=100");
  state.jobs = new Map(payload.items.map((job) => [job.job_id, job]));
  renderJobs();
  for (const job of payload.items) {
    if (["queued", "running"].includes(job.status)) pollJob(job.job_id);
  }
}

function renderJobs() {
  refs.jobList.replaceChildren();
  const jobs = [...state.jobs.values()].sort((a, b) => b.created_at.localeCompare(a.created_at));
  refs.queueCount.textContent = String(jobs.filter((job) => !["succeeded", "failed", "cancelled"].includes(job.status)).length);
  if (!jobs.length) return refs.jobList.append(node("p", {className: "empty", text: "대기 중인 작업이 없습니다."}));
  for (const job of jobs.slice(0, 12)) {
    const outcome = job.status === "succeeded" ? job.result?.result?.status : null;
    refs.jobList.append(node("div", {className: `job-card ${job.status}`}, [
      node("div", {className: "job-head"}, [node("strong", {text: job.kind}), node("span", {text: outcome || job.status})]),
      node("small", {text: `${job.job_id} · attempt ${job.attempts}/${job.max_attempts}`}),
    ]));
  }
}

async function loadCatalog() {
  const payload = await api("/v1/catalog");
  state.catalog = payload.ingredients;
  renderAddIngredientOptions();
}

function renderAddIngredientOptions() {
  const current = new Set((activePayload()?.recipe || []).map((line) => line.ingredient_id));
  refs.addIngredientSelect.replaceChildren(node("option", {value: "", text: "원료 추가…"}));
  for (const ingredient of state.catalog.filter((item) => !current.has(item.ingredient_id)).sort((a, b) => a.name.localeCompare(b.name))) {
    refs.addIngredientSelect.append(node("option", {value: ingredient.ingredient_id, text: `${ingredient.name} · ${ingredient.pyramid}`}));
  }
}

async function loadFormula(projectId, formulaId) {
  try {
    state.activeFormula = await api(`/v1/projects/${encodeURIComponent(projectId)}/formulas/${encodeURIComponent(formulaId)}`);
    state.activeVersion = state.activeFormula.latest_version;
    refs.welcomePanel.classList.add("hidden");
    refs.formulaWorkspace.classList.remove("hidden");
    renderFormulaList();
    renderActiveFormula();
    await loadVersions();
  } catch (error) { toast(error.message, "error"); }
}

function activePayload() { return state.activeVersion?.payload || null; }
function metric(value, suffix = "") { return Number.isFinite(Number(value)) ? `${Number(value).toFixed(2)}${suffix}` : "—"; }

function renderActiveFormula() {
  const payload = activePayload();
  if (!payload || !state.activeFormula) return;
  refs.activeFormulaName.textContent = state.activeFormula.name;
  refs.activeFormulaMeta.textContent = `${state.activeFormula.kind.toUpperCase()} · ${state.activeFormula.formula_id}`;
  refs.formulaStatus.textContent = payload.status || "draft";
  refs.formulaStatus.className = `status-pill ${payload.status === "prototype_ready" ? "good" : payload.status === "no_safe_match" ? "bad" : "warn"}`;
  refs.versionBadge.textContent = `v${state.activeVersion.version_number}`;
  const analysisInvalidated = payload.simulation_status === "not_run_after_manual_edit";
  refs.similarityMetric.textContent = metric(payload.similarity_score);
  refs.proxyMetric.textContent = analysisInvalidated ? "—" : metric(payload.simulation_p05);
  refs.realismMetric.textContent = analysisInvalidated ? "—" : metric(payload.realism_score);
  refs.costMetric.textContent = metric(payload.estimated_concentrate_cost_per_kg, " /kg");
  refs.ingredientMetric.textContent = String((payload.recipe || []).length);
  renderFormulaTable();
  renderProfile(payload.achieved_profile || {});
  renderPyramid();
  renderAddIngredientOptions();
}

function renderFormulaTable() {
  refs.formulaTable.replaceChildren();
  const lines = activePayload()?.recipe || [];
  for (const line of lines) {
    const catalogItem = state.catalog.find((item) => item.ingredient_id === line.ingredient_id);
    const cap = Math.max(.0001, Number(catalogItem?.max_concentrate_percent) || 100);
    const numberInput = node("input", {className: "percent-input", type: "number", value: Number(line.concentrate_percent).toFixed(4)});
    numberInput.min = "0.0001"; numberInput.max = String(cap); numberInput.step = "0.01";
    const range = node("input", {type: "range"});
    range.min = "0.01"; range.max = String(cap); range.step = "0.01";
    range.value = String(Math.min(cap, Math.max(.01, Number(line.concentrate_percent))));
    const update = (value) => {
      line.concentrate_percent = Math.min(cap, Math.max(.0001, Number(value) || .0001));
      numberInput.value = Number(line.concentrate_percent).toFixed(4);
      range.value = String(line.concentrate_percent);
      updateFormulaTotal(); recomputeDraftAnalysis();
    };
    numberInput.addEventListener("input", () => update(numberInput.value));
    numberInput.addEventListener("change", () => update(numberInput.value));
    range.addEventListener("input", () => update(range.value));
    const remove = node("button", {className: "remove-line", type: "button", text: "×", title: "원료 제거"});
    remove.addEventListener("click", () => {
      activePayload().recipe = lines.filter((item) => item !== line);
      recomputeDraftAnalysis();
      renderActiveFormula();
    });
    const row = node("tr", {}, [
      node("td", {text: line.name}),
      node("td", {}, [node("span", {className: "pyramid-tag", text: line.pyramid || "—"})]),
      node("td", {}, [numberInput]),
      node("td", {}, [range]),
      node("td", {}, [remove]),
    ]);
    refs.formulaTable.append(row);
  }
  updateFormulaTotal();
}

function updateFormulaTotal() {
  const total = (activePayload()?.recipe || []).reduce((sum, line) => sum + Number(line.concentrate_percent || 0), 0);
  refs.formulaTotal.textContent = `${total.toFixed(4)}%`;
  refs.totalBar.style.width = `${Math.min(100, Math.abs(total))}%`;
  refs.totalBar.classList.toggle("invalid", Math.abs(total - 100) > .02);
  refs.saveVersionButton.disabled = Math.abs(total - 100) > .02 || total <= 0;
}

function recomputeDraftAnalysis() {
  const payload = activePayload();
  if (!payload) return;
  const profile = {};
  let cost = 0;
  for (const line of payload.recipe || []) {
    const ingredient = state.catalog.find((item) => item.ingredient_id === line.ingredient_id);
    if (!ingredient) continue;
    const percent = Math.max(0, Number(line.concentrate_percent) || 0);
    const parsedImpact = Number(ingredient.odor_impact);
    const impact = Number.isFinite(parsedImpact) && parsedImpact >= 0 ? parsedImpact : 1;
    cost += percent / 100 * Number(ingredient.price_per_kg || 0);
    for (const [dimension, intensity] of Object.entries(ingredient.profile || {})) {
      profile[dimension] = (profile[dimension] || 0) + percent * impact * Number(intensity || 0);
    }
  }
  const profileTotal = Object.values(profile).reduce((sum, value) => sum + value, 0);
  payload.achieved_profile = Object.fromEntries(
    Object.entries(profile).map(([dimension, value]) => [dimension, profileTotal > 0 ? value / profileTotal : 0]),
  );
  payload.estimated_concentrate_cost_per_kg = cost;
  const target = payload.brief?.target_profile || {};
  const dimensions = new Set([...Object.keys(target), ...Object.keys(payload.achieved_profile)]);
  let dot = 0; let targetNorm = 0; let achievedNorm = 0;
  for (const dimension of dimensions) {
    const left = Number(target[dimension] || 0);
    const right = Number(payload.achieved_profile[dimension] || 0);
    dot += left * right; targetNorm += left * left; achievedNorm += right * right;
  }
  payload.similarity_score = targetNorm > 0 && achievedNorm > 0 ? dot / Math.sqrt(targetNorm * achievedNorm) * 100 : 0;
  payload.status = "draft_unsaved";
  refs.formulaStatus.textContent = payload.status;
  refs.formulaStatus.className = "status-pill warn";
  refs.similarityMetric.textContent = metric(payload.similarity_score);
  refs.proxyMetric.textContent = "—";
  refs.realismMetric.textContent = "—";
  refs.costMetric.textContent = metric(cost, " /kg");
  refs.ingredientMetric.textContent = String((payload.recipe || []).length);
  renderProfile(payload.achieved_profile);
  renderPyramid();
}

function normalizeFormula() {
  const lines = activePayload()?.recipe || [];
  const entries = lines.map((line) => {
    const ingredient = state.catalog.find((item) => item.ingredient_id === line.ingredient_id);
    return {
      line,
      weight: Math.max(.0001, Number(line.concentrate_percent) || .0001),
      cap: Math.max(.0001, Number(ingredient?.max_concentrate_percent) || 100),
    };
  });
  if (!entries.length || entries.reduce((sum, item) => sum + item.cap, 0) < 100 - 1e-8) {
    return toast("현재 원료 구성의 허용 상한으로는 100%를 만들 수 없습니다. 원료를 추가하세요.", "error");
  }

  let remaining = 100;
  let active = entries.slice();
  while (active.length) {
    const weightTotal = active.reduce((sum, item) => sum + item.weight, 0);
    const scale = remaining / weightTotal;
    const saturated = active.filter((item) => item.weight * scale > item.cap + 1e-10);
    if (!saturated.length) {
      for (const item of active) item.line.concentrate_percent = item.weight * scale;
      remaining = 0;
      break;
    }
    for (const item of saturated) {
      item.line.concentrate_percent = item.cap;
      remaining -= item.cap;
    }
    const saturatedSet = new Set(saturated);
    active = active.filter((item) => !saturatedSet.has(item));
  }
  if (Math.abs(remaining) > 1e-6) return toast("배합 정규화에 실패했습니다.", "error");
  recomputeDraftAnalysis();
  renderFormulaTable();
}

function renderProfile(profile) {
  refs.profileChart.replaceChildren();
  const ranked = Object.entries(profile).filter(([, value]) => Number(value) > .005).sort((a, b) => b[1] - a[1]).slice(0, 10);
  if (!ranked.length) return refs.profileChart.append(node("p", {className: "empty", text: "프로필 계산 전입니다."}));
  const max = Math.max(...ranked.map(([, value]) => Number(value)));
  for (const [name, value] of ranked) {
    const fill = node("span"); fill.style.width = `${Number(value) / max * 100}%`;
    refs.profileChart.append(node("div", {className: "profile-row"}, [
      node("span", {text: name.replaceAll("_", " ")}), node("div", {className: "bar"}, [fill]), node("strong", {text: `${(Number(value) * 100).toFixed(1)}%`}),
    ]));
  }
}

function renderPyramid() {
  refs.pyramidChart.replaceChildren();
  const totals = {top: 0, heart: 0, base: 0};
  for (const line of activePayload()?.recipe || []) totals[line.pyramid] = (totals[line.pyramid] || 0) + Number(line.concentrate_percent || 0);
  const max = Math.max(1, ...Object.values(totals));
  for (const level of ["top", "heart", "base"]) {
    const fill = node("span"); fill.style.width = `${totals[level] / max * 100}%`;
    refs.pyramidChart.append(node("div", {className: `pyramid-row ${level}`}, [
      node("span", {text: level}), node("div", {className: "pyramid-bar"}, [fill]), node("strong", {text: `${totals[level].toFixed(1)}%`}),
    ]));
  }
}

function addIngredient() {
  const ingredient = state.catalog.find((item) => item.ingredient_id === refs.addIngredientSelect.value);
  if (!ingredient || !activePayload()) return;
  activePayload().recipe.push({
    ingredient_id: ingredient.ingredient_id, name: ingredient.name, pyramid: ingredient.pyramid,
    concentrate_percent: .1, price_per_kg: ingredient.price_per_kg, availability: ingredient.availability,
    risk_tier: ingredient.risk_tier,
  });
  recomputeDraftAnalysis();
  renderActiveFormula();
  refs.addIngredientSelect.value = "";
}

async function saveVersion() {
  if (!state.activeFormula || !state.activeVersion) return;
  const lines = activePayload().recipe.map((line) => ({ingredient_id: line.ingredient_id, concentrate_percent: Number(line.concentrate_percent)}));
  try {
    const version = await api(`/v1/projects/${encodeURIComponent(state.activeFormula.project_id)}/formulas/${encodeURIComponent(state.activeFormula.formula_id)}/versions`, {
      method: "POST",
      body: JSON.stringify({base_version_id: state.activeVersion.version_id, change_note: refs.editNote.value.trim(), lines}),
    });
    refs.editNote.value = "";
    state.activeVersion = version;
    state.activeFormula.latest_version = version;
    renderActiveFormula();
    await Promise.all([loadVersions(), loadFormulas()]);
    toast("수정 내용을 새 버전으로 저장했습니다.", "success");
  } catch (error) { toast(error.message, "error"); }
}

async function submitRevision(event) {
  event.preventDefault();
  if (!state.activeFormula || !state.activeVersion) return;
  try {
    await enqueue("/v1/jobs/revisions", {
      project_id: state.activeFormula.project_id,
      formula_id: state.activeFormula.formula_id,
      base_version_id: state.activeVersion.version_id,
      instruction: refs.revisionInput.value.trim(),
      constraints: {},
    }, "자연어 수정");
    refs.revisionInput.value = "";
  } catch (error) { toast(error.message, "error"); }
}

async function submitAccord(event) {
  event.preventDefault();
  if (!state.activeProject) return toast("프로젝트를 먼저 선택하세요.", "error");
  try {
    await enqueue("/v1/jobs/accords", {
      project_id: state.activeProject.project_id,
      brief: refs.accordBrief.value.trim(),
      name: refs.accordName.value.trim() || "New accord",
      constraints: {},
    }, "어코드 생성");
    refs.accordBrief.value = "";
  } catch (error) { toast(error.message, "error"); }
}

async function loadVersions() {
  if (!state.activeFormula) return;
  const payload = await api(`/v1/projects/${encodeURIComponent(state.activeFormula.project_id)}/formulas/${encodeURIComponent(state.activeFormula.formula_id)}/versions`);
  state.versions = payload.items;
  renderVersions();
}

function renderVersions() {
  refs.versionTimeline.replaceChildren();
  refs.compareLeft.replaceChildren(); refs.compareRight.replaceChildren();
  for (const version of state.versions) {
    const item = node("div", {className: `version-item ${state.activeVersion?.version_id === version.version_id ? "active" : ""}`});
    const button = node("button", {type: "button", text: `v${version.version_number} · ${version.change_kind}`});
    button.addEventListener("click", () => loadVersion(version.version_id));
    item.append(button, node("small", {text: `${version.created_at} · ${version.content_sha256.slice(0, 12)}`}));
    refs.versionTimeline.append(item);
    refs.compareLeft.append(node("option", {value: version.version_id, text: `v${version.version_number}`}));
    refs.compareRight.append(node("option", {value: version.version_id, text: `v${version.version_number}`}));
  }
  if (state.versions.length > 1) {
    refs.compareLeft.value = state.versions[state.versions.length - 1].version_id;
    refs.compareRight.value = state.versions[0].version_id;
  }
}

async function loadVersion(versionId) {
  try {
    state.activeVersion = await api(`/v1/projects/${encodeURIComponent(state.activeFormula.project_id)}/formulas/${encodeURIComponent(state.activeFormula.formula_id)}/versions/${encodeURIComponent(versionId)}`);
    renderActiveFormula(); renderVersions();
  } catch (error) { toast(error.message, "error"); }
}

async function compareVersions() {
  if (!state.activeFormula || !refs.compareLeft.value || !refs.compareRight.value) return;
  try {
    const path = `/v1/projects/${encodeURIComponent(state.activeFormula.project_id)}/formulas/${encodeURIComponent(state.activeFormula.formula_id)}/compare?left=${encodeURIComponent(refs.compareLeft.value)}&right=${encodeURIComponent(refs.compareRight.value)}`;
    const diff = await api(path);
    refs.compareResult.replaceChildren();
    const metrics = node("div", {className: "metric-diffs"});
    for (const [name, values] of Object.entries(diff.metric_changes)) {
      const className = values.delta > 0 ? "delta-up" : values.delta < 0 ? "delta-down" : "";
      metrics.append(node("p", {className, text: `${name}: ${values.before.toFixed(2)} → ${values.after.toFixed(2)} (${values.delta >= 0 ? "+" : ""}${values.delta.toFixed(2)})`}));
    }
    refs.compareResult.append(metrics);
    for (const change of diff.ingredient_changes) {
      refs.compareResult.append(node("div", {className: "diff-row"}, [
        node("strong", {text: change.name}), node("span", {text: change.before_percent.toFixed(3)}), node("span", {text: "→"}), node("span", {text: change.after_percent.toFixed(3)}),
        node("span", {className: change.delta_percent > 0 ? "delta-up" : "delta-down", text: `${change.delta_percent > 0 ? "+" : ""}${change.delta_percent.toFixed(3)}`}),
      ]));
    }
    if (!diff.ingredient_changes.length) refs.compareResult.append(node("p", {className: "empty", text: "원료 배합 차이가 없습니다."}));
  } catch (error) { toast(error.message, "error"); }
}

function showWelcome() {
  refs.formulaWorkspace.classList.add("hidden");
  refs.welcomePanel.classList.remove("hidden");
}

refs.connectButton.addEventListener("click", connect);
refs.disconnectButton.addEventListener("click", disconnect);
refs.projectForm.addEventListener("submit", createProject);
refs.refreshProjects.addEventListener("click", () => loadProjects().catch((error) => toast(error.message, "error")));
refs.recipeForm.addEventListener("submit", generateRecipe);
refs.normalizeButton.addEventListener("click", normalizeFormula);
refs.saveVersionButton.addEventListener("click", saveVersion);
refs.addIngredientSelect.addEventListener("change", addIngredient);
refs.revisionForm.addEventListener("submit", submitRevision);
refs.accordForm.addEventListener("submit", submitAccord);
refs.refreshVersions.addEventListener("click", () => loadVersions().catch((error) => toast(error.message, "error")));
refs.compareButton.addEventListener("click", compareVersions);

checkHealth();
if (state.token && state.tenant) connect();
