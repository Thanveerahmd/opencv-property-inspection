"use strict";
// Inspection console (the Property Inspection Platform's UI, OpenCV 5 edition): dashboard over every walkthrough,
// the walkthrough list, the upload form and live progress while a video is processed. Each finished walkthrough opens
// in its own 3D viewer (/walkthroughs/<name>/, web/walkthrough.html).
const $ = (id) => document.getElementById(id);
const API = "/api/v1";
const LOST = "Lost connection to the server — this job is no longer being tracked. Reload the page once the server is reachable again.";
let clockTimer = null, lastHealth = null;
// The progress card shows the walkthrough this page is following (its state survives moving between views).
const runState = { walk: { t0: 0 } };
let panelOwner = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = (name, size) => `<svg${size ? ` width="${size}" height="${size}"` : ""} aria-hidden="true"><use href="#i-${name}"/></svg>`;

// ---- API key: only needed when the server has auth on (health.auth) --------------------------
const KEY_LS = "pi_api_key";
let apiKey = "";
try { apiKey = localStorage.getItem(KEY_LS) || ""; } catch { /* storage blocked */ }
function setApiKey(k) {
  apiKey = String(k || "").trim();
  try { if (apiKey) localStorage.setItem(KEY_LS, apiKey); else localStorage.removeItem(KEY_LS); } catch { /* storage blocked */ }
}
// URL for <img src> / <a href> / <video src>: these cannot send headers, so the key goes in the query string.
function withKey(url) {
  url = String(url);
  if (!apiKey || !url.startsWith("/api/")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}key=${encodeURIComponent(apiKey)}`;
}
// fetch() for the backend: `path` is relative to API ("/health") or absolute ("/api/..."); adds X-API-Key.
function api(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (apiKey) headers.set("X-API-Key", apiKey);
  return fetch(path.startsWith("/api/") ? path : API + path, { cache: "no-store", ...opts, headers });
}
function errText(r, j) {
  let d = j && j.detail;
  if (d && typeof d !== "string") d = JSON.stringify(d);
  if (r.status === 401 || r.status === 403) { $("key-box").hidden = false; return "API key required or invalid — enter it in the sidebar (System status)."; }
  return d || r.statusText || `HTTP ${r.status}`;
}
async function apiJson(path, opts) {
  const r = await api(path, opts);
  let j = null;
  try { j = await r.json(); } catch { /* empty or non-JSON body */ }
  if (!r.ok) throw Object.assign(new Error(errText(r, j)), { status: r.status });
  return j;
}
const listOf = (j) => Array.isArray(j) ? j : (j && (j.items || j.rows)) || [];

// ---- formatting helpers (risk, money, dates) -----------------------------------------------------
const GRADES = ["A", "B", "C", "D", "E"];
const gcls = (g) => GRADES.includes(g) ? g : "x";
const gradeBadge = (g) => g ? `<span class="gb g-${gcls(g)}" title="Condition grade ${esc(g)}">${esc(g)}</span>` : `<span class="gb g-x" title="Not graded">–</span>`;
const decisionCls = (d) => /^PROCEED WITH|CONDITION/i.test(d || "") ? "cond" : /^PROCEED|^ACCEPT/i.test(d || "") ? "go" : /^REFER/i.test(d || "") ? "refer" : "cond";
const decShort = (d) => String(d || "—").replace(/\s+-\s+/, " — ");
function money(v, cur) {
  if (v == null || v === "" || isNaN(v)) return "—";
  try { return new Intl.NumberFormat(undefined, { style: "currency", currency: cur || "USD", maximumFractionDigits: 0 }).format(v); }
  catch { return `${Math.round(v).toLocaleString()} ${cur || ""}`.trim(); }
}
const moneyRange = (a, b, cur) => a == null && b == null ? "—" : (b == null || a === b) ? money(a ?? b, cur) : `${money(a, cur)} – ${money(b, cur)}`;
const pctRange = (r) => Array.isArray(r) ? (r[0] === r[1] ? `${r[0]}%` : `${r[0]}% to ${r[1]}%`) : r == null ? "—" : `${r}%`;
function fmtDate(v, short) {
  if (v == null || v === "") return "";
  const d = new Date(typeof v === "number" && v < 1e12 ? v * 1000 : v);
  return isNaN(d) ? String(v) : d.toLocaleString(undefined, short ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { dateStyle: "medium", timeStyle: "short" });
}
const covPct = (c) => c == null ? null : typeof c === "object" ? c.inspectable_pct ?? null : c;

function toast(html, ms = 8000) {
  const t = $("toast"); t.innerHTML = html; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

// ---- theme ---------------------------------------------------------------------------------------
const THEME_LS = "pi_theme";
const darkMQ = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
function effectiveTheme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t === "dark" || t === "light" ? t : darkMQ && darkMQ.matches ? "dark" : "light";
}
function paintThemeBtn() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  $("theme-btn").innerHTML = `${icon(next === "dark" ? "moon" : "sun")}<span>${next === "dark" ? "Dark" : "Light"} mode</span>`;
  $("theme-btn").setAttribute("aria-label", `Switch to ${next} theme`);
}
function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem(THEME_LS, next); } catch { /* storage blocked */ }
  paintThemeBtn();
}

// ---- init ------------------------------------------------------------------------------------------
async function init() {
  $("api-key").value = apiKey;
  $("api-key").onchange = () => { setApiKey($("api-key").value); refreshAll(); };
  paintThemeBtn();
  $("theme-btn").onclick = toggleTheme;
  if (darkMQ && darkMQ.addEventListener) darkMQ.addEventListener("change", paintThemeBtn);
  health(); setInterval(health, 15000);
  walks(); setInterval(walks, 20000);

  const wd = $("walk-drop");
  wd.onclick = () => $("walk-file").click();
  wd.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("walk-file").click(); } };
  wd.ondragover = (e) => { e.preventDefault(); wd.classList.add("over"); };
  wd.ondragleave = () => wd.classList.remove("over");
  wd.ondrop = (e) => {
    e.preventDefault(); wd.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) { const dt = new DataTransfer(); dt.items.add(f); $("walk-file").files = dt.files; pickWalk(); }
  };
  $("walk-file").onchange = pickWalk;
  $("walk-form").onsubmit = submitWalk;
  document.querySelectorAll('input[name="walk-mode"]').forEach((x) => x.addEventListener("change", () => { modeTouched = true; }));
  // Scanning view readout: which moment of the walkthrough video is on screen.
  const rv = $("run-vid"), mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  rv.addEventListener("timeupdate", () => {
    $("scan-hud").hidden = rv.hidden || !rv.duration;
    if (rv.duration) $("scan-hud").textContent = `${mmss(rv.currentTime)} / ${mmss(rv.duration)}`;
  });
  $("pf-refresh").onclick = portfolio;
  window.onhashchange = route;
  route();
}

function refreshAll() { health(); walks(); if (currentView === "dashboard") portfolio(); }

// ---- routing: #/dashboard, #/walkthroughs, #/walkthroughs/new -------------------------------------
let currentView = null, currentRouteId = null;
function parseRoute() {
  let h = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
  if (h === "portfolio") h = "dashboard";
  if (h === "inspect") h = "walkthroughs";
  const [view, id] = h.split("/");
  return { view: ["dashboard", "walkthroughs"].includes(view) ? view : "dashboard", id: id || null };
}
function go(hash) { if (location.hash !== hash) location.hash = hash; else route(); }

function route() {
  const r = parseRoute();
  const changed = r.view !== currentView || r.id !== currentRouteId;
  currentView = r.view; currentRouteId = r.id;
  for (const v of ["dashboard", "walkthroughs"]) {
    $("view-" + v).hidden = v !== r.view;
    $("nav-" + v).classList.toggle("on", v === r.view);
    if (v === r.view) $("nav-" + v).setAttribute("aria-current", "page"); else $("nav-" + v).removeAttribute("aria-current");
  }
  if (changed) window.scrollTo(0, 0);
  hideError();
  if (r.view === "dashboard") { portfolio(); renderWalkCards(); }
  if (r.view === "walkthroughs") { renderWalkCards(); if (r.id === "new") showUploadForm(); }
  document.title = { dashboard: "Dashboard", walkthroughs: "Video walkthroughs" }[r.view] + " · Property Inspection AI";
  placeRunning();
}

// "New walkthrough" (#/walkthroughs/new): bring the upload form into view and put focus on the drop zone,
// instead of leaving the visitor on the list of finished walkthroughs.
function showUploadForm() {
  requestAnimationFrame(() => {
    $("walk-form").scrollIntoView({ behavior: "smooth", block: "start" });
    $("walk-drop").focus({ preventScroll: true });
  });
}

// The progress card sits on the walkthroughs page while a walkthrough is followed.
function placeRunning() {
  const run = $("running");
  const slot = !watching ? null : currentView === "walkthroughs" ? "walk-run-slot"
    : currentView === "dashboard" ? "dash-run-slot" : null;
  $("nav-walk-live").hidden = !watching;
  if (!slot) { run.hidden = true; panelOwner = null; return; }
  if (panelOwner !== "walk") paintOwner("walk");
  if (run.parentElement.id !== slot) $(slot).appendChild(run);
  run.hidden = false;
}

// Sidebar "System status": the models this edition runs, the GPU and the integrations.
let modeTouched = false;
async function health() {
  const el = $("health");
  try {
    const r = await api("/health");
    const h = await r.json();
    lastHealth = h;
    el.className = "health ok"; el.textContent = "Online";
    const busy = (walkItems || []).some((w) => w.status === "running");
    const gpu = h.cuda ? (busy ? "GPU · in use" : "GPU") : "no GPU";
    const rows = [
      ["Depth Anything 3", gpu, !!h.cuda, "Depth: Depth Anything 3 in the 3D engine"],
      [`OpenCV ${h.opencv || "5"}`, "SLAM · defects", true, `OpenCV ${h.opencv || "?"}: camera tracking (SLAM), coverage, evidence integrity and defect analysis`],
      ["Inspection agent", h.bedrock_model ? "Bedrock" : "Bedrock / rules", true,
        h.bedrock_model ? `Agent: Amazon Bedrock ${h.bedrock_model} (rule-based policy if it is unavailable)`
          : "Agent: Amazon Bedrock when AWS credentials are set, otherwise the rule-based policy"],
    ];
    $("models").innerHTML = rows.map(([n, where, on, tip]) =>
      `<li title="${esc(tip)}"><span class="dot ${on ? "on" : ""}"></span><span class="nm">${esc(n)}</span><span class="where">${esc(where)}</span></li>`).join("");
    if (h.vram_total_mb) {
      $("vram").hidden = false;
      $("gpu-name").textContent = String(h.gpu || "GPU").replace(/^NVIDIA\s+(GeForce\s+)?/i, "").replace(/\s+Laptop GPU$/i, "");
      $("vram-txt").textContent = `${((h.vram_used_mb || 0) / 1024).toFixed(1)} / ${(h.vram_total_mb / 1024).toFixed(1)} GB`;
      $("vram-bar").style.width = `${Math.min(100, 100 * (h.vram_used_mb || 0) / h.vram_total_mb).toFixed(1)}%`;
    } else $("vram").hidden = true;
    // The server's default mode (SI_MODE) preselects Lite / Advanced until the visitor picks one.
    if (!modeTouched && (h.mode === "lite" || h.mode === "advanced")) {
      const radio = document.querySelector(`input[name="walk-mode"][value="${h.mode}"]`);
      if (radio) radio.checked = true;
    }
    const running = Array.isArray(h.running) ? h.running.length : h.running || 0;
    paintVideoQueue();
    el.title = ["AWS integrations", `• Amazon S3: ${h.s3 ? "on" : "off"}`,
      `• Amazon Bedrock agent: ${h.bedrock_model || "default model if AWS credentials are set"}`,
      `OpenCV ${h.opencv || "?"}`, h.gpu ? `GPU: ${h.gpu}` : "GPU: none found",
      `API key: ${h.auth ? "required" : "not required"}`, running ? `Running jobs: ${running}` : null].filter(Boolean).join("\n");
    if (h.auth) $("key-box").hidden = false; else if (!apiKey) $("key-box").hidden = true;
    $("pf-bq").textContent = h.s3 ? " (on)" : " (currently off)";
  } catch { el.className = "health bad"; el.textContent = "Offline"; el.title = "Backend not reachable"; }
}

// ---- progress card ---------------------------------------------------------------------------------
function paintOwner(kind) {
  panelOwner = kind;
  const st = runState.walk, v = $("run-vid");
  $("run-title").textContent = `Processing “${st.name || "walkthrough"}”`;
  $("run-note").textContent = WALK_NOTE;
  $("run-log").textContent = st.log || "starting…";
  // A stored video that can't be loaded (file removed, format the browser can't play) hides the preview.
  v.onerror = () => { if (panelOwner === "walk" && runState.walk.videoUrl) { runState.walk.videoUrl = null; paintOwner("walk"); } };
  if (st.videoUrl) { if (v.getAttribute("src") !== st.videoUrl) v.src = st.videoUrl; v.hidden = false; v.play().catch(() => {}); }
  else { v.hidden = true; v.pause(); }
  $("run-scan").hidden = false;           // the scan animation runs over the video, or over the dark frame without one
  $("run-stage").textContent = st.stageText || "Starting";
  $("scan-tag").textContent = st.stageText || "Processing";
  if (st.prog) paintProgress(...st.prog); else resetProgress();
  tickClock();
}

function setStageText(kind, txt) {
  runState[kind].stageText = txt;
  if (panelOwner === kind) { $("run-stage").textContent = txt; $("scan-tag").textContent = txt; }
}

const fmtDur = (s) => {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
};

// Fill the progress bar with a real percentage + elapsed / remaining time.
function setProgress(kind, pct, label, elapsedS, etaS) {
  runState[kind].prog = [pct, label, elapsedS, etaS];
  if (panelOwner === kind) paintProgress(pct, label, elapsedS, etaS);
}
function paintProgress(pct, label, elapsedS, etaS) {
  pct = Math.max(0, Math.min(100, pct));
  const bar = $("run-bar");
  bar.classList.add("det"); bar.querySelector("span").style.width = pct.toFixed(1) + "%";
  const eta = etaS == null ? "" : etaS > 0 ? `<span>about <b>${fmtDur(etaS)}</b> left</span>` : "";
  $("run-progress").innerHTML = `<span><b>${Math.floor(pct)}%</b></span>` + (label ? `<span>${esc(label)}</span>` : "") +
    `<span>${fmtDur(elapsedS)} elapsed</span>` + eta;
}

function resetProgress() {
  const bar = $("run-bar");
  bar.classList.remove("det"); bar.querySelector("span").style.width = "";
  $("run-progress").innerHTML = "";
}

function tickClock() {
  if (panelOwner) $("run-elapsed").textContent = fmtDur((Date.now() - runState[panelOwner].t0) / 1000) + " elapsed";
}
function startClock() { if (!clockTimer) clockTimer = setInterval(tickClock, 1000); tickClock(); }
function stopClockIfIdle() { if (!watching) { clearInterval(clockTimer); clockTimer = null; } }

function busy(b, on, label) {
  b.disabled = on; b.classList.toggle("busy", on);
  const svg = b.querySelector("svg");
  b.textContent = label; if (svg && !on) b.prepend(svg);
}

// ---- Portfolio / dashboard -------------------------------------------------------------------------
let pf = null, pfSort = { key: "created_at", dir: -1 }, pfBusy = false;
const PF_COLS = [["label", "Walkthrough"], ["property_id", "Property"], ["created_at", "Date"], ["grade", "Grade"],
  ["condition_score", "Score"], ["risk_level", "Risk"], ["decision", "Decision"], ["repair_high", "Repair cost"],
  ["valuation_pct", "Value impact"], ["coverage_pct", "Coverage"]];

async function portfolio() {
  if (pfBusy) return;
  pfBusy = true;
  $("pf-err").hidden = true;
  if (!pf) $("pf-kpis").innerHTML = ["Walkthroughs", "Repair exposure", "Referred", "Average coverage"].map((k) =>
    `<div class="kpi"><div class="k">${k}</div><div class="v muted">…</div></div>`).join("");
  busy($("pf-refresh"), true, "Refresh");
  try { pf = await apiJson("/portfolio"); renderPortfolio(); }
  catch (e) { $("pf-err").textContent = "Cannot load portfolio: " + e.message; $("pf-err").hidden = false; }
  finally { busy($("pf-refresh"), false, "Refresh"); pfBusy = false; }
}

function renderPortfolio() {
  const p = pf || {}, rows = p.rows || [], ex = p.repair_exposure || {}, cur = ex.currency || "USD";
  const byDec = p.by_decision || {}, byGrade = p.by_grade || {}, byRisk = p.by_risk_level || {};
  const n = p.n ?? rows.length;
  const referred = Object.entries(byDec).filter(([k]) => /^REFER/i.test(k)).reduce((a, [, c]) => a + c, 0);
  const nProps = new Set(rows.map((r) => r.property_id || r.id)).size;
  const covs = rows.map((r) => covPct(r.coverage_pct)).filter((c) => c != null);
  const avgCov = covs.length ? Math.round(covs.reduce((a, c) => a + c, 0) / covs.length) : null;
  const pct = (a) => n ? `${Math.round(100 * a / n)}% of records` : "";
  const tile = (ico, tone, k, v, s) => `<div class="kpi"><div class="k"><span class="ico ${tone}">${icon(ico)}</span>${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  $("pf-kpis").innerHTML = tile("video", "", "Walkthroughs", esc(n), `${nProps} ${nProps === 1 ? "property" : "properties"} inspected by video`) +
    tile("coin", "amber", "Repair exposure", moneyRange(ex.low, ex.high, cur), "low – high estimate, whole book") +
    tile("refer", "red", "Referred", esc(referred), pct(referred) || "decision = REFER") +
    tile("scan", "", "Average coverage", avgCov != null ? `${avgCov}%` : "—", "of surfaces filmed close enough to inspect");

  const max = Math.max(1, ...GRADES.map((g) => byGrade[g] || 0));
  $("pf-grades").innerHTML = GRADES.map((g) => `<div class="gbar"><span class="gb g-${g}">${g}</span>
    <div class="track" role="img" aria-label="Grade ${g}: ${byGrade[g] || 0}"><i class="g-${g}" style="width:${(100 * (byGrade[g] || 0) / max).toFixed(1)}%"></i></div><b>${byGrade[g] || 0}</b></div>`).join("");

  const LVL = [["Low", "c-ok"], ["Moderate", "c-rev"], ["Medium", "c-rev"], ["High", "c-det"], ["Critical", "c-det"]];
  const riskKeys = Object.keys(byRisk).sort((a, b) => LVL.findIndex((x) => x[0] === a) - LVL.findIndex((x) => x[0] === b));
  const tot = riskKeys.reduce((a, k) => a + byRisk[k], 0) || 1;
  const lc = (k) => (LVL.find((x) => x[0] === k) || [0, "c-mute"])[1];
  $("pf-risk").innerHTML = riskKeys.length ? `<div class="stackbar" role="img" aria-label="Risk level breakdown">${riskKeys.map((k) =>
      `<i class="${lc(k)}" style="width:${(100 * byRisk[k] / tot).toFixed(1)}%" title="${esc(k)}: ${byRisk[k]}"></i>`).join("")}</div>
    <ul class="legend-list">${riskKeys.map((k) => `<li><span class="sw ${lc(k)}"></span>${esc(k)}<b>${byRisk[k]}</b><span class="pct">${Math.round(100 * byRisk[k] / tot)}%</span></li>`).join("")}</ul>`
    : `<div class="empty-state">No graded records yet.</div>`;

  const decs = Object.entries(byDec).sort((a, b) => b[1] - a[1]);
  const dmax = Math.max(1, ...decs.map(([, c]) => c));
  const dc = (k) => ({ go: "c-ok", cond: "c-rev", refer: "c-det" }[decisionCls(k)]);
  $("pf-dec").innerHTML = decs.map(([k, c]) => `<div class="dec-bar"><div class="top"><span title="${esc(k)}">${esc(decShort(k))}</span><b>${c}</b></div>
      <div class="track"><i class="${dc(k)}" style="width:${(100 * c / dmax).toFixed(1)}%"></i></div></div>`).join("")
    || `<div class="empty-state">No decisions yet.</div>`;

  const { key, dir } = pfSort;
  const val = (r) => { const v = key === "coverage_pct" ? covPct(r[key]) : r[key]; return Array.isArray(v) ? v[0] : v; };
  const sorted = rows.slice().sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  $("pf-head").innerHTML = `<tr>${PF_COLS.map(([k, l]) =>
    `<th data-k="${k}" tabindex="0" aria-sort="${k === key ? (dir > 0 ? "ascending" : "descending") : "none"}" class="${k === key ? "sorted" + (dir > 0 ? " asc" : "") : ""}${["condition_score", "repair_high", "valuation_pct", "coverage_pct"].includes(k) ? " num" : ""}">${l}</th>`).join("")}</tr>`;
  $("pf-head").querySelectorAll("th").forEach((th) => {
    th.onclick = () => {
      pfSort = { key: th.dataset.k, dir: pfSort.key === th.dataset.k ? -pfSort.dir : (th.dataset.k === "created_at" ? -1 : 1) };
      renderPortfolio();
    };
    th.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); th.click(); } };
  });
  $("pf-rows").innerHTML = sorted.map((r) => {
    const cov = covPct(r.coverage_pct);
    return `<tr class="click" tabindex="0" data-id="${esc(r.id)}">
    <td><div class="cell-main" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.label || r.id)}</div></td>
    <td>${esc(r.property_id || "—")}</td>
    <td class="nowrap muted">${esc(fmtDate(r.created_at, true))}</td>
    <td>${gradeBadge(r.grade)}</td><td class="num">${esc(r.condition_score ?? "—")}</td>
    <td>${r.risk_level ? `<span class="lvl ${esc(r.risk_level)}">${esc(r.risk_level)}</span>` : "—"}</td>
    <td><span class="dec-txt ${decisionCls(r.decision)}" title="${esc(r.decision || "")}${r.approval ? ` · approved: ${esc(r.approval)}` : ""}">${esc(String(r.decision || "—").split(" - ")[0])}</span></td>
    <td class="num">${moneyRange(r.repair_low, r.repair_high, cur)}</td><td class="num">${pctRange(r.valuation_pct)}</td>
    <td class="num">${cov != null ? esc(cov) + "%" : "—"}</td></tr>`;
  }).join("") || `<tr><td colspan="${PF_COLS.length}" class="empty-state">No walkthroughs yet. Upload a video to inspect a property.</td></tr>`;
  $("pf-rows").querySelectorAll("tr[data-id]").forEach((tr) => {
    tr.onclick = () => window.open(`/walkthroughs/${encodeURIComponent(tr.dataset.id)}/`, "_blank", "noopener");
    tr.onkeydown = (e) => { if (e.key === "Enter") tr.click(); };
  });
}

// ---- Walkthroughs ----------------------------------------------------------------------------------
let watching = null, walkItems = null;
// Every step the server reports (status.json: progress.key = engine / analysis / agent / viewer, progress.phase = the
// step's text), grouped as the pipeline runs: the 3D engine, the OpenCV analysis, then the agent and the viewer.
const WALK_STEPS = [
  { id: "upload", g: "Upload", t: "Uploading video", s: "To the inspection server" },
  { id: "e1", g: "3D engine", k: "engine", t: "Depth Anything 3", s: "Depth, camera pose and intrinsics for every frame · GPU", re: /depth anything|frames \+/i },
  { id: "e2", k: "engine", t: "Floor plane and metric scale", s: "RANSAC floor; the camera height sets metres", re: /floor plane|metric scale/i },
  { id: "e3", k: "engine", t: "OpenCV 5 camera tracking", s: "LK optical flow, SIFT + MAGSAC pose checks, trajectory refinement", re: /tracking|sift|magsac/i },
  { id: "e4", k: "engine", t: "Depth filtering", s: "Flying pixels removed, joint-bilateral upsampling", re: /depth filtering|bilateral/i },
  { id: "e5", k: "engine", t: "ICP alignment and TSDF mesh", s: "Point-to-plane ICP per frame, 1.5 cm TSDF, free-space carving", re: /alignment|tsdf/i },
  { id: "e6", k: "engine", t: "Gravity and Manhattan frame", s: "Walls, floor and ceiling made flat", re: /model frame|manhattan|regularis/i },
  { id: "e7", k: "engine", t: "Room layout", s: "Walls, doors, openings, floor and ceiling", re: /structural layout/i },
  { id: "e8", k: "engine", t: "Furniture, named by Grounding DINO", s: "Objects separated from the room structure", re: /objects \(|grounding/i },
  { id: "e9", k: "engine", t: "Real-world texturing", s: "Walls, floor and ceiling textured from the video", re: /textured|shell/i },
  { id: "e10", k: "engine", t: "Floor plan and navigation map", s: "Occupancy grid and A* path", re: /2d maps|occupancy|navigation/i },
  { id: "e11", k: "engine", t: "3D exports and self-check", s: "GLB models, previews, reconstruction metrics", re: /exports|previews|evaluation|model ready/i },
  { id: "a1", g: "OpenCV analysis", k: "analysis", t: "Quality gate and evidence integrity", s: "Sharpness; reused, spliced, looped or filmed-off-a-screen footage", re: /quality|integrity/i },
  { id: "a2", k: "analysis", t: "Privacy", s: "Faces (YuNet) and private text blurred before analysis", re: /privacy|blurring/i },
  { id: "a3", k: "analysis", t: "OpenCV TSDF and room planes", s: "cv2.Volume fusion on the engine's depth and poses", re: /fusion|fusing|layout|fitting walls/i },
  { id: "a4", k: "analysis", t: "Surface coverage and keyframes", s: "Every 10 cm of wall, floor and ceiling against every frame", re: /coverage/i },
  { id: "a5", k: "analysis", t: "Defect candidates", s: "OpenCV detectors; Grounding DINO and MobileNetV2 rule out objects", re: /defect|finding/i },
  { id: "a6", k: "analysis", t: "Labels and rectified walls", s: "Rating-plate text; each wall as one image", re: /label|visual|rectif/i },
  { id: "ag", g: "Agent and viewer", t: "Inspection agent", s: "Better views, re-film requests, risk and a proposed decision" },
  { id: "vw", t: "3D viewer and report", s: "Finding pins, walk-inside path, live map video, GLBs" },
];
function paintWalkSteps() {
  $("walk-steps").innerHTML = WALK_STEPS.map((x) => (x.g ? `<li class="grp">${esc(x.g)}</li>` : "") +
    `<li data-step="${x.id}"><span class="st">${esc(x.t)}<small>${esc(x.s)}</small></span></li>`).join("");
}
const WALK_NOTE = "Depth Anything 3 + OpenCV 5 SLAM build the 3D model on the GPU, then OpenCV analyses the keyframes and the agent proposes a decision — usually several minutes. You can leave this page open; the results open automatically when finished.";

// Walkthroughs uploaded from this browser are followed first; otherwise the page follows whatever the GPU is
// processing, so a job started elsewhere (another tab, the command line) still shows its live progress.
const MY_WALKS_LS = "pi_my_walks";
function myWalks() { try { return JSON.parse(localStorage.getItem(MY_WALKS_LS) || "[]"); } catch { return []; } }
function rememberWalk(name) {
  try { localStorage.setItem(MY_WALKS_LS, JSON.stringify([...myWalks().filter((n) => n !== name), name].slice(-20))); }
  catch { /* storage blocked */ }
}

async function walks() {
  try {
    walkItems = listOf(await apiJson("/walkthroughs"));
    renderWalkCards();
    // Re-attach to this browser's walkthrough that is still queued or processing (page reload, second tab, ...).
    const mine = myWalks();
    const run = walkItems.find((w) => w.status === "running" && mine.includes(w.name))
      || walkItems.find((w) => w.status === "running")
      || walkItems.find((w) => w.status === "queued" && mine.includes(w.name));
    if (run && !watching) watchWalk(run.name);
  } catch {                             // offline: don't leave stale "running" spinners in the list
    document.querySelectorAll("#walks .mini-spin, #dash-walks .mini-spin").forEach((x) => x.remove());
    document.querySelectorAll(".wcard.running .wstat-status").forEach((x) => { x.textContent = "server offline"; });
  }
}

function walkCardHtml(w) {
  if (!w.url) {
    const thumb = w.status === "running" ? `<span><span class="mini-spin"></span>Processing…</span>`
      : w.status === "queued" ? `<span>${icon("clock", 16)} Waiting for the GPU…</span>` : icon("video", 28);
    const note = { running: "processing — click to follow progress",
                   queued: "queued: waits for the walkthrough ahead of it — click to follow",
                   interrupted: "interrupted by a server restart — click for details",
                   failed: "failed — click for details" }[w.status] || `${w.status} — click for details`;
    return `<button type="button" class="wcard running" data-walk="${esc(w.name)}" style="text-align:left;font:inherit;padding:0;background:var(--surface);color:var(--text)">
      <div class="wthumb">${thumb}</div>
      <div class="wbody"><h3>${esc(w.name)}</h3><span class="muted small wstat-status">${esc(note)}</span></div></button>`;
  }
  const cov = covPct(w.coverage);
  const nf = Array.isArray(w.findings) ? w.findings.length : w.findings ?? 0;
  return `<a class="wcard" href="${esc(w.url)}" target="_blank" rel="noopener" aria-label="Open 3D walkthrough ${esc(w.name)}">
    <div class="wthumb"><img src="${esc(w.url)}coverage_map.png" alt="Coverage map of ${esc(w.name)}" loading="lazy" onerror="this.remove()">
      ${w.grade ? gradeBadge(w.grade) : ""}<span class="badge3d">3D · ${esc(w.duration_s != null ? Math.round(w.duration_s) + " s" : "video")}</span></div>
    <div class="wbody">
      <div style="display:flex;justify-content:space-between;gap:8px;align-items:center"><h3>${esc(w.name)}${w.mode ? ` <span class="mode-tag">${w.mode === "lite" ? "Lite" : "Advanced"}</span>` : ""}</h3>${icon("ext", 15)}</div>
      ${w.decision ? `<span class="dec-txt ${decisionCls(w.decision)}" style="align-self:flex-start">${esc(decShort(w.decision))}</span>` : ""}
      <div class="wstats"><div><div class="k">Coverage</div><div class="v">${cov != null ? esc(cov) + "%" : "—"}</div></div>
        <div><div class="k">Findings</div><div class="v">${esc(nf)}</div></div>
        <div><div class="k">Rooms</div><div class="v">${Array.isArray(w.rooms) ? w.rooms.length : "—"}</div></div></div>
      ${Array.isArray(w.rooms) && w.rooms.length ? `<span class="muted small">${esc(w.rooms.join(" · "))}</span>` : ""}
    </div></a>`;
}

// Sidebar status line: what the GPU is doing with videos (the only job queue).
function paintVideoQueue() {
  const items = walkItems || [];
  const nRun = items.filter((w) => w.status === "running").length, nWait = items.filter((w) => w.status === "queued").length;
  const txt = nRun ? `Processing ${nRun} video${nRun > 1 ? "s" : ""}${nWait ? ` · ${nWait} waiting` : ""}`
    : nWait ? `${nWait} video${nWait > 1 ? "s" : ""} waiting` : lastHealth && !lastHealth.cuda ? "No GPU found" : "GPU idle";
  $("queue-txt").textContent = txt;
  $("queue-txt").classList.toggle("busy", nRun > 0);
}

function renderWalkCards() {
  paintVideoQueue();
  if (!walkItems) {
    $("walks").innerHTML = $("dash-walks").innerHTML = `<div class="card empty-state">Loading walkthroughs…</div>`;
    return;
  }
  const done = walkItems.filter((w) => w.url).length;
  $("nav-walk-count").textContent = done || "";
  const html = walkItems.map(walkCardHtml).join("");
  $("walks").innerHTML = html || `<div class="card empty-state">No walkthroughs yet — upload a video to build the first 3D inspection.</div>`;
  $("dash-walks").innerHTML = walkItems.filter((w) => w.url).slice(0, 6).map(walkCardHtml).join("")
    || `<div class="card empty-state">No walkthroughs yet. <a href="#/walkthroughs/new">Upload a video</a>.</div>`;
  document.querySelectorAll("[data-walk]").forEach((b) => b.onclick = () => { watchWalk(b.dataset.walk); go("#/walkthroughs"); });
}

function pickWalk() {
  const f = $("walk-file").files[0];
  $("walk-go").disabled = !f || !!watching;
  $("walk-drop-text").textContent = f ? f.name : "Drop a video here or browse";
  if (f) $("walk-drop").classList.add("has-file");
}

function setWalkStep(step) {
  if (!$("walk-steps").querySelector("li[data-step]")) paintWalkSteps();
  const idx = WALK_STEPS.findIndex((x) => x.id === step);
  document.querySelectorAll("#walk-steps li[data-step]").forEach((li, i) => {
    li.classList.toggle("done", i < idx); li.classList.toggle("active", i === idx);
  });
  const cur = WALK_STEPS[idx];
  setStageText("walk", cur ? cur.t : step);
  // the scan overlay names the part of the pipeline that is running: "3D engine · 5 / 11"
  let start = idx;
  while (start > 0 && !WALK_STEPS[start].g) start--;
  let end = start + 1;
  while (end < WALK_STEPS.length && !WALK_STEPS[end].g) end++;
  const hud = $("scan-hud");
  hud.hidden = !cur;
  hud.textContent = cur ? `${WALK_STEPS[start].g} · ${idx - start + 1} / ${end - start}` : "";
}

// The current pipeline step from status.json (progress.key = engine / analysis / agent / viewer, progress.phase = the
// stage text), else from the log lines.
function walkStep(p, lines) {
  const k = (p && p.key) || "", ph = (p && p.phase) || "";
  if (k === "viewer" || k === "done") return "vw";
  if (k === "agent" || (k === "analysis" && /agent/i.test(ph))) return "ag";
  const pool = WALK_STEPS.filter((x) => x.k === (k === "analysis" ? "analysis" : "engine"));
  const hit = pool.find((x) => x.re.test(ph));
  if (hit) return hit.id;
  if (k === "analysis") return "a1";
  const txt = lines.join("\n");                     // no status yet: read the log
  const seen = WALK_STEPS.filter((x) => x.k === "engine" && x.re.test(txt));
  return seen.length ? seen[seen.length - 1].id : "e1";
}

const cleanLine = (l) => l.replace(/\u001b\[[0-9;]*m/g, "").trim();
const isNoise = (l) => !l || /triton|gsplat|Traceback|File "|^\s*\^|import triton|ModuleNotFoundError|warnings\.warn|it\/s\]|s\/it\]/i.test(l);

function startWalkPanel(name, videoUrl) {
  runState.walk = { t0: Date.now(), name, videoUrl: videoUrl || null, log: "starting…", prog: null, stageText: "" };
  $("walk-go").disabled = true; $("walk-go").classList.add("busy");
  setProgress("walk", 0, "Uploading video", 0, null);
  if (panelOwner === "walk") paintOwner("walk");
  placeRunning();
  startClock();
}

function stopWalkPanel() {
  watching = null; stopClockIfIdle();
  $("walk-go").disabled = !$("walk-file").files.length; $("walk-go").classList.remove("busy");
  runState.walk.prog = null;
  if (panelOwner === "walk") $("run-vid").pause();
  placeRunning();
}

function watchWalk(name, videoUrl) {
  if (watching === name) return;
  watching = name;
  // No local copy of the video (page reloaded, other tab, started elsewhere): play the stored upload.
  startWalkPanel(name, videoUrl || withKey(`${API}/walkthrough/${encodeURIComponent(name)}/video`));
  setWalkStep("e1");
  const view = `/walkthroughs/${name}/`;
  let misses = 0, fails = 0;
  const tick = async () => {
    if (watching !== name) return;
    let l;
    try {
      const r = await api(`/walkthrough/${name}/log`);
      if (r.status === 404) {                       // status not written yet: job is still starting
        if (++misses < 60) { setTimeout(tick, 3000); return; }
        throw new Error("no progress log");
      }
      if (!r.ok) throw new Error(r.statusText);
      l = await r.json();
      fails = 0;
    } catch (err) {                     // transient blip: retry; server gone: stop the animation
      if (++fails < 4) { setTimeout(tick, 4000); return; }
      stopWalkPanel(); $("walk-status").textContent = LOST; return;
    }
    const lines = (l.tail || []).map(cleanLine);
    const useful = lines.filter((x) => !isNoise(x));
    runState.walk.log = useful.slice(-6).join("\n") || "starting…";
    if (panelOwner === "walk") $("run-log").textContent = runState.walk.log;
    $("walk-status").innerHTML = `<span class="mini-spin"></span>${esc(name)}: ${esc(l.status)}`;
    if (l.status === "queued") {                  // another walkthrough has the GPU: say so instead of a silent wait
      const msg = queueText(l.queue || {});
      setWalkStep("e1"); setStageText("walk", "Waiting for the GPU");
      $("walk-status").innerHTML = `<span class="mini-spin"></span>${esc(name)}: ${esc(msg)}`;
      const waited = (l.queue && l.queue.waiting_s) || 0;
      runState.walk.t0 = Date.now() - waited * 1000;
      setProgress("walk", 0, msg, waited, null);
      setTimeout(tick, 4000); return;
    }
    if (l.status === "running") {
      const p = l.progress;
      setWalkStep(walkStep(p, lines));
      if (p) {
        runState.walk.t0 = Date.now() - p.elapsed_s * 1000;   // server clock: survives page reloads
        setProgress("walk", p.percent, p.steps ? `Step ${p.step}/${p.steps}: ${p.phase}` : p.phase, p.elapsed_s,
                    p.elapsed_s < 20 && !p.step ? null : p.eta_s);
      }
      setTimeout(tick, 4000); return;
    }
    if (l.progress) setProgress("walk", 100, "Done", l.progress.elapsed_s, 0);
    stopWalkPanel(); walks();
    if (l.status === "done") {
      $("walk-status").innerHTML = `${esc(name)}: done — <a href="${esc(view)}" target="_blank" rel="noopener">` +
        `<strong>Open results (3D model, live map, risk report) ↗</strong></a>`;
      if (currentView === "walkthroughs") {
        $("walk-status").innerHTML += " — opening…";
        setTimeout(() => { location.href = view; }, 1500);   // a popup would be blocked after the long wait
      } else toast(`Walkthrough <b>${esc(name)}</b> is ready — <a href="${esc(view)}">open 3D results</a>`, 20000);
    } else if (l.status === "interrupted" || l.status === "failed") {
      $("walk-status").innerHTML = (l.status === "interrupted"
        ? `${esc(name)}: stopped before finishing (the server was restarted). `
        : `${esc(name)}: failed — ${esc(useful.slice(-1)[0] || "")} `) +
        `<a href="#" data-retry="${esc(name)}"><strong>Process it again</strong></a> (no re-upload needed).`;
      $("walk-status").querySelector("[data-retry]").onclick = (e) => { e.preventDefault(); retryWalk(name); };
    } else {
      $("walk-status").textContent = l.status === "unknown"
        ? `${name}: stopped before finishing (the server was restarted). Upload the video again.`
        : `${name}: ${l.status} — ${useful.slice(-1)[0] || ""}`;
    }
  };
  tick();
}

// "Waiting: another walkthrough is processing, started 7 min ago (about 4 min left)"
function queueText(q) {
  if (!q.running) return "Waiting for the GPU — starting shortly";
  const el = q.running_elapsed_s || 0;
  let s = `Waiting: another walkthrough is processing, started ${el < 60 ? "just now" : `${Math.round(el / 60)} min ago`}`;
  if (q.running_eta_s > 0) s += ` (about ${Math.max(1, Math.round(q.running_eta_s / 60))} min left)`;
  if (q.ahead > 1) s += ` · ${q.ahead - 1} more queued before yours`;
  return s;
}

async function retryWalk(name) {
  $("walk-status").innerHTML = `<span class="mini-spin"></span>Restarting ${esc(name)}…`;
  try {
    const r = await api(`/walkthrough/${encodeURIComponent(name)}/retry`, { method: "POST" });
    let j = null; try { j = await r.json(); } catch { /* non-JSON body */ }
    if (!r.ok && r.status !== 409) throw new Error(errText(r, j));
    rememberWalk(name); watchWalk(name); go("#/walkthroughs");
  } catch (err) { $("walk-status").textContent = "Failed: " + err.message; }
}

async function submitWalk(e) {
  e.preventDefault();
  const f = $("walk-file").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("video", f);
  const prop = $("walk-property").value.trim();
  if (prop) fd.append("property_id", prop);
  const mode = (document.querySelector('input[name="walk-mode"]:checked') || {}).value || "advanced";
  fd.append("mode", mode);
  const videoUrl = URL.createObjectURL(f);
  // the server names the walkthrough <file name>_<date-time>; this name is only shown while the video uploads
  const name = (f.name.replace(/\.[^.]*$/, "").replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 40)) || "walkthrough";
  watching = null;
  startWalkPanel(name, videoUrl);
  watching = "upload:" + name; placeRunning();
  setWalkStep("upload");
  $("walk-status").innerHTML = `<span class="mini-spin"></span>Uploading…`;
  $("running").scrollIntoView({ behavior: "smooth", block: "start" });   // the progress card sits above the form
  try {
    const r = await api("/walkthrough", { method: "POST", body: fd });
    let j = null; try { j = await r.json(); } catch { /* non-JSON body */ }
    watching = null;
    if (!r.ok) throw new Error(errText(r, j));
    rememberWalk(j.name);
    watchWalk(j.name, videoUrl);
  } catch (err) { stopWalkPanel(); $("walk-status").textContent = "Failed: " + err.message; }
}

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("error").hidden = true; }
init();
