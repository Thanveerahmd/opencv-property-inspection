// Inspection console: a dashboard over every inspection, live progress while a video is processed, and for each
// walkthrough what OpenCV measured, how the evidence was checked, what the agent decided and the person's approval.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (v, cur = "USD") => `${cur ? `${cur} ` : ""}${Math.round(v || 0).toLocaleString()}`;
const KIND = { crack: "Crack", stain: "Stain", mould: "Mould", rust: "Rust", peeling: "Peeling paint", water_damage: "Water damage" };
const STATUS = { confirmed: "Confirmed", needs_review: "Needs review", candidate: "Not reviewed", dismissed: "Dismissed" };
let current = null;
let poll = null;
let viewer = null;

async function api(path, opts) {
  const r = await fetch(path, { cache: "no-store", ...opts });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function show(view) {
  for (const id of ["dash", "progress", "empty", "detail"]) $(id).hidden = id !== view;
  $("navDash").classList.toggle("on", view === "dash");
  $("navInsp").classList.toggle("on", view !== "dash");
}

async function health() {
  try {
    const h = await api("/api/health");
    $("badges").innerHTML = [
      `<span class="badge on">OpenCV ${esc(h.opencv)}</span>`,
      `<span class="badge ${h.s3 ? "on" : ""}">Amazon S3 ${h.s3 ? "on" : "off"}</span>`,
      `<span class="badge ${h.bedrock_model ? "on" : ""}">Amazon Bedrock ${h.bedrock_model ? "agent" : "off"}</span>`,
    ].join("");
  } catch { /* the badges are decoration */ }
}

const decisionOf = (w) => w.decision?.decision || w.decision?.proposed;

async function refreshList() {
  const items = await api("/api/walkthroughs");
  $("list").innerHTML = items.map((w) => {
    const d = decisionOf(w);
    const tag = w.state !== "done"
      ? `<span class="small muted">${esc(w.state)}${w.progress != null ? ` · ${Math.round(w.progress * 100)}%` : ""}${w.stage ? ` · ${esc(w.stage)}` : ""}</span>`
      : `<span class="small muted">${w.grade ? `Grade ${esc(w.grade)} · ` : ""}${w.inspected_pct ?? "–"}% inspected ${d ? `· <span class="pill d-${esc(d)}">${esc(d)}</span>` : ""}</span>`;
    return `<li class="item ${w.name === current ? "on" : ""}" data-name="${esc(w.name)}"><b>${esc(w.property || w.name)}</b>${w.property ? `<span class="small muted">${esc(w.name)}</span>` : ""}${tag}</li>`;
  }).join("") || `<li class="muted small">None yet.</li>`;
  for (const li of $("list").querySelectorAll("li.item")) li.onclick = () => { location.hash = encodeURIComponent(li.dataset.name); };
  return items;
}

// ---------------------------------------------------------------- dashboard

async function dashboard() {
  current = null;
  clearTimeout(poll);
  show("dash");
  await refreshList();
  const d = await api("/api/dashboard");
  const ex = d.repair_exposure;
  const tile = (label, value, note = "") => `<div class="card kpi"><span class="lbl">${label}</span><span class="big">${value}</span><span class="small muted">${note}</span></div>`;
  $("kpis").innerHTML = [
    tile("Inspections", d.walkthroughs, `${d.analysed} analysed${d.processing ? ` · ${d.processing} processing` : ""}`),
    tile("Awaiting approval", d.awaiting_approval, "agent proposals a person has not signed off"),
    tile("Surface inspected", d.mean_inspected_pct != null ? `${d.mean_inspected_pct}%` : "–", "mean per walkthrough (rule: 60%)"),
    tile("Repair exposure", ex.high ? `${money(ex.low, "")}–${money(ex.high, "")}` : "0", `${esc(ex.currency)}, indicative`),
    tile("Findings", d.confirmed, `confirmed · ${d.review} awaiting review`),
  ].join("");

  // one series (count per grade): single hue, value labels on the bars, hover tooltip; the table below is the table view
  const max = Math.max(1, ...Object.values(d.grades));
  const label = { A: "Good", B: "Fair", C: "Below average", D: "Poor", E: "Very poor" };
  $("gradeChart").setAttribute("aria-label", `Inspections by condition grade: ${Object.entries(d.grades).map(([g, n]) => `${g} ${n}`).join(", ")}`);
  $("gradeChart").innerHTML = Object.entries(d.grades).map(([g, n]) => `<div class="gcol" tabindex="0">
      <span class="tip">Grade ${g} (${label[g]}): ${n} inspection${n === 1 ? "" : "s"}</span>
      <span class="gval">${n}</span><span class="gbar" style="height:${(n / max) * 100}%"></span><span class="glab">${g}</span></div>`).join("");

  const list = (title, obj) => `<div><h4>${title}</h4>${Object.keys(obj).length ? Object.entries(obj).map(([k, v]) => `<div class="kv"><span>${esc(k)}</span><b>${v}</b></div>`).join("") : "<p class='muted small'>None yet</p>"}</div>`;
  $("dashSplit").innerHTML = list("Decisions", d.decisions) + list("Risk level", d.risk_levels) + list("Evidence integrity", d.integrity);

  $("dashTable").innerHTML = `<tr><th>Property</th><th>Grade</th><th>Risk</th><th>Decision</th><th>Inspected</th><th>Repair cost</th><th>Integrity</th><th>Updated</th></tr>`
    + d.recent.map((w) => {
      const dec = decisionOf(w);
      return `<tr class="click" data-name="${esc(w.name)}"><td><b>${esc(w.property || w.name)}</b>${w.property ? `<br><span class="small muted">${esc(w.name)}</span>` : ""}</td>
        <td>${w.grade ? `<span class="grade g${esc(w.grade)}">${esc(w.grade)}</span>` : "–"}</td><td>${esc(w.risk_level || (w.state !== "done" ? w.state : "–"))}</td>
        <td>${dec ? `<span class="pill d-${esc(dec)}">${esc(dec)}</span> <span class="small muted">${esc(w.decision?.state === "awaiting approval" ? "proposed" : "approved")}</span>` : "–"}</td>
        <td>${w.inspected_pct != null ? `${w.inspected_pct}%` : "–"}</td>
        <td>${w.repair_cost ? `${money(w.repair_cost.low, "")}–${money(w.repair_cost.high, "")}` : "–"}</td>
        <td>${w.integrity ? `<span class="lvl ${esc(w.integrity)}">${esc(w.integrity)}</span>` : "–"}</td><td class="small muted">${esc(w.updated)}</td></tr>`;
    }).join("");
  for (const tr of $("dashTable").querySelectorAll("tr.click")) tr.onclick = () => { location.hash = encodeURIComponent(tr.dataset.name); };
}

// ---------------------------------------------------------------- one walkthrough

const STEPS = [
  ["frames", "Read frames, check sharpness and exposure"], ["tracking", "Depth (cv2.dnn) and camera tracking"],
  ["levelling", "Level the floor, fix metric scale"], ["analysis", "Integrity, privacy, 3D fusion, layout, coverage, defects, walls"],
  ["agent", "Agent reviews the evidence and proposes a decision"],
];

function showProgress(name, st) {
  show("progress");
  $("pName").textContent = `${st.property ? `${st.property} · ` : ""}${name}`;
  $("pStage").textContent = st.state === "failed" ? "Processing failed" : st.stage || st.state;
  const p = Math.round((st.progress || 0) * 100);
  $("pBar").style.width = `${p}%`;
  const fmt = (s) => (s == null ? "–" : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`);
  $("pTimes").innerHTML = st.state === "failed" ? `<span class="lvl high">${esc(st.error)}</span>`
    : `<b>${p}%</b> · elapsed ${fmt(st.elapsed_s)} · about ${fmt(st.eta_s)} left`;
  const at = STEPS.findIndex(([k]) => k === st.step);
  $("pSteps").innerHTML = STEPS.map(([k, label], i) => `<li class="${st.state === "queued" ? "" : i < at ? "done" : i === at ? "now" : ""}">${label}</li>`).join("");
  const v = $("pVideo");
  if (st.video && !v.dataset.src) { v.dataset.src = st.video; v.src = `/uploads/${encodeURIComponent(st.video)}`; }
}

async function open(name) {
  current = name;
  await refreshList();
  let data;
  try { data = await api(`/api/walkthroughs/${encodeURIComponent(name)}`); } catch { show("empty"); return; }
  clearTimeout(poll);
  if (!data.result || ["queued", "processing"].includes(data.status?.state)) {
    showProgress(name, data.status || {});
    if (data.status?.state !== "failed") poll = setTimeout(() => current === name && open(name), 2000);
    return;
  }
  $("pVideo").removeAttribute("src");
  delete $("pVideo").dataset.src;
  show("detail");
  render(name, data.result, data.agent);
}

function render(name, r, a) {
  const base = `/files/${encodeURIComponent(name)}/`;
  const dec = a?.decision;
  const rk = a?.risk;
  $("dName").textContent = `${r.property ? `${r.property} · ` : ""}${r.name} · ${r.frames} frames · ${r.source}`;
  const final = a?.approval?.decision;
  $("dDecision").innerHTML = dec ? `<span class="d-${esc(final || dec.proposed)}">${esc(final || dec.proposed)}</span>`
    + (final ? ` <span class="small muted">approved by ${esc(a.approval.by)}</span>` : ` <span class="small muted">proposed, awaiting approval</span>`) : "No decision yet";
  $("dSummary").textContent = dec?.summary || "";
  $("dEngine").textContent = a ? `Decided by ${a.engine} in ${a.seconds}s · ${a.trace.length} steps` : "";
  const q = encodeURIComponent(name);
  $("exports").innerHTML = `<a class="btn" href="/api/walkthroughs/${q}/report.html" target="_blank" rel="noopener">Inspection report</a>
    <a class="btn ghost" href="/api/walkthroughs/${q}/report.md">Markdown</a><a class="btn ghost" href="/api/walkthroughs/${q}/report.json">JSON</a>`;
  $("approved").innerHTML = a?.approval ? `Recorded: <b>${esc(a.approval.decision)}</b> ${a.approval.agreed ? "(agrees with the agent)" : "(overrides the agent)"} ${esc(a.approval.note)} · ${esc(a.approval.at)}` : "";
  $("apDecision").value = dec?.proposed || "REFER";

  renderRisk(rk, r);
  renderIntegrity(r.integrity);

  const c = r.coverage;
  const bar = (label, k) => `<div class="bar"><span>${label}</span><span class="track"><i class="s" style="width:${k.seen_pct}%"></i><i class="g" style="width:${k.inspectable_pct}%"></i><i class="x" style="width:${k.inspected_pct}%"></i></span><span class="v">${k.inspected_pct}%</span></div>`;
  $("covBars").innerHTML = bar("Overall", c.overall) + Object.entries(c.by_kind).map(([k, v]) => bar(k[0].toUpperCase() + k.slice(1), v)).join("")
    + `<p class="small muted">Bars: seen · inspectable · <b>inspected</b> (number)</p>`;
  $("gaps").innerHTML = (r.gaps || []).slice(0, 6).map((g) => `<li>${esc(g.label)}: ${g.length_m ? `${g.length_m} m of wall` : `${g.uncovered_m2} m² uncovered`}</li>`).join("") || "<li class='muted'>None</li>";
  $("refilm").innerHTML = (a?.refilm || []).map((x) => `<li><b>${esc(x.surface)}</b>: ${esc(x.instruction)}</li>`).join("") || "<li class='muted'>None</li>";

  renderFindings(name, base, r, a);

  const walls = r.visuals?.walls || [];
  $("walls").innerHTML = (r.visuals?.floor_plan ? `<figure><img src="${base}${r.visuals.floor_plan}" alt="Floor plan"><figcaption>Floor plan with the camera path and findings</figcaption></figure>` : "")
    + walls.filter((w) => w.image).map((w) => `<figure><img src="${base}${esc(w.image)}" alt="Wall ${esc(w.id)}" loading="lazy"><figcaption>${esc(w.id)} · ${w.length_m} m${w.seen_pct != null ? ` · ${w.seen_pct}% seen` : ""}</figcaption></figure>`).join("");

  $("trace").innerHTML = (a?.trace || []).map((s) => {
    const cls = s.kind === "thought" ? "thought" : s.kind === "human" ? "human" : s.kind === "fallback" ? "fallback" : "";
    const body = s.kind === "thought" ? esc(s.text)
      : s.kind === "fallback" ? `Model unavailable, rule-based policy used: ${esc(s.error)}`
      : `<span class="tool">${esc(s.tool)}</span> <span class="muted">${esc(JSON.stringify(s.input))}</span><pre>${esc(JSON.stringify(s.output, null, 1))}</pre>${s.image ? `<img src="${base}${esc(s.image)}" alt="view">` : ""}`;
    return `<li><span class="n">${s.step}</span><div class="b ${cls}">${body}</div></li>`;
  }).join("");

  $("timings").innerHTML = Object.entries(r.timings_s).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v} s</td></tr>`).join("")
    + `<tr><td><b>total</b></td><td><b>${Object.values(r.timings_s).reduce((x, y) => x + y, 0).toFixed(1)} s</b></td></tr>`;
  $("evidence").innerHTML = `<p>OpenCV ${esc(r.opencv)} · ${r.frames} frames · quality: ${Object.entries(r.quality).map(([k, v]) => `${k} ${v}`).join(", ")}</p>
    <p>Privacy: ${r.privacy.faces ?? 0} faces and ${r.privacy.text ?? r.privacy.text_regions ?? 0} text regions blurred before analysis</p>
    <p>3D: ${r.fusion.points.toLocaleString()} surface points from ${r.fusion.frames} depth maps at ${r.fusion.voxel_m * 100} cm voxels</p>
    ${(r.labels || []).length ? `<p>Labels read: ${r.labels.map((l) => l.lines.map((x) => esc(x.text)).join(" ")).join(" · ")}</p>` : ""}`;

  showPoints(base, r, a?.findings || r.findings || []);
}

function renderRisk(rk, r) {
  if (!rk) { $("riskTiles").innerHTML = ""; return; }
  const cur = rk.repair_cost.currency;
  const t = (label, value, note, cls = "") => `<div class="card kpi ${cls}"><span class="lbl">${label}</span><span class="big">${value}</span><span class="small muted">${note}</span></div>`;
  $("riskTiles").innerHTML = [
    t("Condition grade", `<span class="grade g${esc(rk.grade)}">${esc(rk.grade)}</span>`, `${esc(rk.grade_label)} · score ${rk.condition_score}`),
    t("Risk level", esc(rk.risk_level), `premium loading ${rk.insurance.premium_loading_pct}%`),
    t("Repair cost", `${money(rk.repair_cost.low, "")}–${money(rk.repair_cost.high, "")}`, `${esc(cur)}, indicative`),
    t("Collateral risk", esc(rk.mortgage.collateral_risk), rk.mortgage.repair_retention ? `retain ${money(rk.mortgage.repair_retention, cur)}` : "no repair retention"),
    t("Surface inspected", `${r.coverage.overall.inspected_pct}%`, rk.evidence.complete ? "enough evidence" : "below the 60% rule"),
  ].join("");
  $("underwriting").innerHTML = `Underwriting rules: <b>${esc(rk.decision)}</b>`;
  $("reasons").innerHTML = rk.reasons.map((x) => `<li>${esc(x)}</li>`).join("") || "<li class='muted'>No rule fired.</li>";
  $("perils").innerHTML = `<tr><th>Category</th><th>Insurance peril</th><th>Findings</th><th>Repair cost</th></tr>`
    + (rk.insurance.perils.map((p) => `<tr><td>${esc(p.label)}</td><td>${esc(p.peril)}</td><td>${p.findings}</td><td>${money(p.repair_cost[0], "")}–${money(p.repair_cost[1], "")}</td></tr>`).join("")
      || `<tr><td colspan="4" class="muted">No defects counted</td></tr>`);
  const v = rk.valuation;
  $("valuation").textContent = `Valuation impact ${v.impact_pct[0]}% to ${v.impact_pct[1]}% (${v.basis}). ${rk.disclaimer}`;
}

function renderIntegrity(ev) {
  ev = ev || { level: "low", flags: [], checks: {} };
  $("intLevel").innerHTML = `<span class="lvl ${esc(ev.level)}">${esc(ev.level)}</span>`;
  $("intFlags").innerHTML = ev.flags.map((f) => `<li class="flag ${esc(f.level)}"><span class="lvl ${esc(f.level)}">${esc(f.level)}</span> ${esc(f.message)}</li>`).join("")
    || `<li class="flag low"><span class="lvl low">ok</span> One continuous take, not seen before, no sign of a screen being filmed.</li>`;
  const ch = ev.checks || {};
  const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  $("intChecks").innerHTML = [
    row("Frames hashed", ch.frames ?? "–"), row("Matches with earlier walkthroughs", (ch.reused || []).length),
    row("Hard cuts", (ch.hard_cuts || []).length), row("Frozen footage", `${ch.frozen_s ?? 0} s`),
    row("Looped stretches", (ch.loops || []).length), row("Screen moiré (median score)", ch.moire_median ?? "–"),
  ].join("");
}

function renderFindings(name, base, r, a) {
  const fs = (a?.findings || r.findings || []).slice().sort((x, y) => ({ confirmed: 0, needs_review: 1, candidate: 2, dismissed: 3 }[x.status] - { confirmed: 0, needs_review: 1, candidate: 2, dismissed: 3 }[y.status]) || y.score - x.score);
  const n = (s) => fs.filter((f) => f.status === s).length;
  $("fCount").textContent = `${n("confirmed")} confirmed · ${n("needs_review")} for review · ${n("candidate")} not reviewed · ${n("dismissed")} dismissed`;
  const why = {};
  for (const x of r.rejected || []) {
    const k = /in front|stands out|looks like/.test(x.reason) ? "on furniture or objects" : /door/.test(x.reason) ? "on doors" : /beyond/.test(x.reason) ? "seen through an opening" : "outside the room model";
    why[k] = (why[k] || 0) + 1;
  }
  $("ruledOut").textContent = (r.rejected || []).length
    ? `${r.rejected.length} candidates were ruled out before review using the 3D model and object recognition: ${Object.entries(why).map(([k, v]) => `${v} ${k}`).join(", ")}.` : "";
  $("findings").innerHTML = fs.map((f) => {
    const s = f.size || {};
    const dims = [s.length_m ? `${s.length_m.toFixed(2)} m long` : "", f.kind === "crack" && s.width_mm ? `${s.width_mm.toFixed(1)} mm wide` : "", s.area_m2 ? `${s.area_m2.toFixed(2)} m²` : ""].filter(Boolean).join(" · ");
    const open = f.status !== "confirmed" && f.status !== "dismissed";
    return `<div class="finding ${esc(f.status)}">
      <img src="${base}${esc(f.crop)}" alt="${esc(f.kind)} candidate" loading="lazy">
      <div class="body"><span class="status ${esc(f.status)}">${esc(STATUS[f.status] || f.status)}${f.severity ? ` · ${esc(f.severity)}` : ""}</span>
      <b>${esc(f.id)} · ${esc(KIND[f.defect || f.kind] || f.kind)}</b>
      <span class="muted">${esc(f.surface || "")} ${esc(f.group || "")}${dims ? ` · ${dims}` : ""} · score ${f.score}</span>
      ${f.evidence || f.reason ? `<span>${esc(f.evidence || f.reason)}</span>` : ""}
      ${f.reviewed_by ? `<span class="small muted">reviewed by ${esc(f.reviewed_by)}</span>` : ""}
      <div class="row review">${open ? `<select data-sev="${esc(f.id)}"><option>minor</option><option selected>moderate</option><option>major</option></select>
        <button data-id="${esc(f.id)}" data-status="confirmed">Confirm</button><button class="ghost" data-id="${esc(f.id)}" data-status="dismissed">Dismiss</button>`
        : `<button class="ghost" data-id="${esc(f.id)}" data-status="needs_review">Reopen</button>`}</div></div></div>`;
  }).join("") || "<p class='muted'>No defects found on the inspected surfaces.</p>";
  for (const b of $("findings").querySelectorAll("button[data-id]")) {
    b.onclick = async () => {
      b.disabled = true;
      const sev = $("findings").querySelector(`select[data-sev="${b.dataset.id}"]`)?.value || "moderate";
      await api(`/api/walkthroughs/${encodeURIComponent(name)}/findings/${encodeURIComponent(b.dataset.id)}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: b.dataset.status, severity: sev }),
      });
      await open(name);
    };
  }
}

async function showPoints(base, r, findings) {
  const el = $("viewer");
  if (viewer) { viewer.dispose(); viewer = null; }
  el.innerHTML = "";
  let buf;
  try { buf = await (await fetch(`${base}points.bin`, { cache: "no-store" })).arrayBuffer(); } catch { return; }
  const n = new Uint32Array(buf, 0, 1)[0];
  const pos = new Float32Array(buf, 4, n * 3);
  const col = new Uint8Array(buf, 4 + n * 12, n * 3);
  $("ptsInfo").textContent = `${n.toLocaleString()} points`;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(el.clientWidth, el.clientHeight);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1211);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
  // hide the ceiling so the room can be seen from above
  const ceil = r.layout?.ceiling_height_m ?? 2.6;
  const keep = [];
  for (let i = 0; i < n; i++) if (pos[i * 3 + 1] < ceil - 0.25) keep.push(i);
  g.setIndex(keep);
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ size: 0.03, vertexColors: true })));
  const traj = (r.trajectory || []).map((p) => new THREE.Vector3(p[0], p[1], p[2]));
  if (traj.length > 1) scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(traj), new THREE.LineBasicMaterial({ color: 0x8be3d3 })));
  for (const f of findings) {
    if (!f.xyz || f.status === "dismissed") continue;
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: f.status === "confirmed" ? 0xef5b48 : 0xf3c04f }));
    m.position.set(...f.xyz);
    scene.add(m);
  }
  g.computeBoundingBox();
  const c = g.boundingBox.getCenter(new THREE.Vector3());
  const size = g.boundingBox.getSize(new THREE.Vector3()).length();
  const cam = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.05, 100);
  cam.position.set(c.x - size * 0.35, c.y + size * 0.75, c.z + size * 0.55);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.target.copy(c);
  controls.update();
  let raf = 0;
  const loop = () => { controls.update(); renderer.render(scene, cam); raf = requestAnimationFrame(loop); };
  loop();
  viewer = { dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); } };
}

// ---------------------------------------------------------------- actions and routing

$("video").onchange = () => { $("fileName").textContent = $("video").files[0]?.name || "Choose a video"; };
$("upload").onsubmit = async (e) => {
  e.preventDefault();
  const f = $("video").files[0];
  if (!f) return;
  $("go").disabled = true;
  $("uploadMsg").textContent = "Uploading…";
  try {
    const fd = new FormData();
    fd.append("video", f);
    fd.append("property", $("property").value);
    const res = await api("/api/walkthroughs", { method: "POST", body: fd });
    $("uploadMsg").textContent = "Uploaded. Processing on the server.";
    location.hash = encodeURIComponent(res.name);
  } catch (err) {
    $("uploadMsg").textContent = `Upload failed: ${err.message}`;
  } finally {
    $("go").disabled = false;
  }
};

async function approve(useProposal) {
  const data = await api(`/api/walkthroughs/${encodeURIComponent(current)}`);
  const decision = useProposal ? data.agent?.decision?.proposed : $("apDecision").value;
  await api(`/api/walkthroughs/${encodeURIComponent(current)}/approve`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, note: $("apNote").value }),
  });
  await open(current);
}
$("apYes").onclick = () => approve(true);
$("apOverride").onclick = () => approve(false);
$("navInsp").onclick = async (e) => {
  e.preventDefault();
  const items = await api("/api/walkthroughs");
  const first = items.find((w) => w.state === "done") || items[0];
  if (first) location.hash = encodeURIComponent(first.name); else show("empty");
};

async function route() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (!h || h === "dashboard") await dashboard();
  else await open(h);
}
window.addEventListener("hashchange", route);
await health();
await route();
