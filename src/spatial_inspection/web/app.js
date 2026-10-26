// Inspection console: lists walkthroughs, shows what OpenCV measured and what the agent decided, and records the
// human approval.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
let current = null;
let poll = null;
let viewer = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
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

async function refreshList() {
  const items = await api("/api/walkthroughs");
  $("list").innerHTML = items.map((w) => {
    const d = w.decision?.decision || w.decision?.proposed;
    const tag = w.state !== "done" ? `<span class="small muted">${esc(w.state)}${w.stage ? ` · ${esc(w.stage)}` : ""}</span>`
      : `<span class="small muted">${w.inspected_pct ?? "–"}% inspected · ${w.findings ?? 0} candidates ${d ? `· <span class="pill d-${esc(d)}">${esc(d)}</span>` : ""}</span>`;
    return `<li class="item ${w.name === current ? "on" : ""}" data-name="${esc(w.name)}"><b>${esc(w.name)}</b>${tag}</li>`;
  }).join("") || `<li class="muted small">None yet.</li>`;
  for (const li of $("list").querySelectorAll("li.item")) li.onclick = () => open(li.dataset.name);
  return items;
}

async function open(name) {
  current = name;
  location.hash = encodeURIComponent(name);
  await refreshList();
  const data = await api(`/api/walkthroughs/${encodeURIComponent(name)}`);
  if (!data.result) {
    $("empty").hidden = false;
    $("detail").hidden = true;
    $("empty").innerHTML = `<h2>${esc(name)}</h2><p class="muted">${esc(data.status?.state)}: ${esc(data.status?.stage || data.status?.error || "")}</p>`;
    clearTimeout(poll);
    if (data.status?.state !== "failed") poll = setTimeout(() => open(name), 3000);
    return;
  }
  clearTimeout(poll);
  $("empty").hidden = true;
  $("detail").hidden = false;
  render(name, data.result, data.agent);
}

function render(name, r, a) {
  const base = `/files/${encodeURIComponent(name)}/`;
  const dec = a?.decision;
  $("dName").textContent = `${r.name} · ${r.frames} frames · ${r.source}`;
  const final = a?.approval?.decision;
  $("dDecision").innerHTML = dec ? `<span class="d-${esc(final || dec.proposed)}">${esc(final || dec.proposed)}</span>`
    + (final ? ` <span class="small muted">approved by ${esc(a.approval.by)}</span>` : ` <span class="small muted">proposed, awaiting approval</span>`) : "No decision yet";
  $("dSummary").textContent = dec?.summary || "";
  $("dEngine").textContent = a ? `Decided by ${a.engine} in ${a.seconds}s · ${a.trace.length} steps` : "";
  $("approved").innerHTML = a?.approval ? `Recorded: <b>${esc(a.approval.decision)}</b> ${a.approval.agreed ? "(agrees with the agent)" : "(overrides the agent)"} ${esc(a.approval.note)} · ${esc(a.approval.at)}` : "";
  $("apDecision").value = dec?.proposed || "REFER";

  const c = r.coverage;
  const bar = (label, k) => `<div class="bar"><span>${label}</span><span class="track"><i class="s" style="width:${k.seen_pct}%"></i><i class="g" style="width:${k.inspectable_pct}%"></i><i class="x" style="width:${k.inspected_pct}%"></i></span><span class="v">${k.inspected_pct}%</span></div>`;
  $("covBars").innerHTML = bar("Overall", c.overall) + Object.entries(c.by_kind).map(([k, v]) => bar(k[0].toUpperCase() + k.slice(1), v)).join("")
    + `<p class="small muted">Bars: seen · inspectable · <b>inspected</b> (number)</p>`;
  $("gaps").innerHTML = (r.gaps || []).slice(0, 6).map((g) => `<li>${esc(g.label)}: ${g.length_m ? `${g.length_m} m of wall` : `${g.uncovered_m2} m² uncovered`}</li>`).join("") || "<li class='muted'>None</li>";
  $("refilm").innerHTML = (a?.refilm || []).map((x) => `<li><b>${esc(x.surface)}</b>: ${esc(x.instruction)}</li>`).join("") || "<li class='muted'>None</li>";

  const fs = a?.findings || r.findings || [];
  $("findings").innerHTML = fs.map((f) => `<div class="finding ${esc(f.status)}">
      <img src="${base}${esc(f.crop)}" alt="${esc(f.kind)} candidate" loading="lazy">
      <div class="body"><span class="status ${esc(f.status)}">${esc(f.status)}${f.severity ? ` · ${esc(f.severity)}` : ""}</span>
      <b>${esc(f.id)} · ${esc(f.defect || f.kind)}</b>
      <span class="muted">OpenCV score ${f.score} · ${esc(f.surface || "")}${f.size?.length_m ? ` · ${f.size.length_m} m long` : ""}${f.size?.area_m2 ? ` · ${f.size.area_m2} m²` : ""}</span>
      ${f.evidence || f.reason ? `<span>${esc(f.evidence || f.reason)}</span>` : ""}</div></div>`).join("") || "<p class='muted'>No candidates.</p>";

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

  showPoints(base, r, fs);
}

async function showPoints(base, r, findings) {
  const el = $("viewer");
  if (viewer) { viewer.dispose(); viewer = null; }
  el.innerHTML = "";
  let buf;
  try { buf = await (await fetch(`${base}points.bin`)).arrayBuffer(); } catch { return; }
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
    if (!f.xyz) continue;
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.08, 16, 12), new THREE.MeshBasicMaterial({ color: f.status === "confirmed" ? 0xef5b48 : f.status === "dismissed" ? 0x777777 : 0xf3c04f }));
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
    const res = await api("/api/walkthroughs", { method: "POST", body: fd });
    $("uploadMsg").textContent = "Uploaded. Processing on the server.";
    await open(res.name);
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

await health();
const items = await refreshList();
const first = decodeURIComponent(location.hash.slice(1)) || items.find((w) => w.state === "done")?.name;
if (first) open(first);
