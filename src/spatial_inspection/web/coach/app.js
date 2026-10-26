// Coverage coach page (frontend/coach.html?w=<walkthrough>): replay a walkthrough with its coverage drawn over the
// video, then keep filming in a simulated phone until the inspectable surface passes the target.
// Geometry, coverage and guidance come from engine.js; this file only draws and handles input.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import * as E from "./engine.js";

const $ = (id) => document.getElementById(id);
const NAME = new URLSearchParams(location.search).get("w") || "";
const BASE = `/walkthroughs/${encodeURIComponent(NAME)}/`;
const FILM_EVERY_MS = 150, TIP_EVERY_MS = 350, MOVE_MPS = 1.1, TURN_DPS = 75, TILT_DPS = 55;

const app = {
  mode: "replay", scene: null, frame: 0, last: -1, shown: null,      // shown = coverage state on screen
  base: null, sim: null, pos: null, yaw: 0, pitch: 0, filming: true,
  factor: 1, calib: null, route: null, routeNext: 0, layer: "floor", tip: null, keys: new Set(),
};

// ------------------------------------------------------------------------------------------------- 3D
const renderer = new THREE.WebGLRenderer({ canvas: $("gl"), antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const world = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, 16 / 9, 0.05, 60);
cam.matrixAutoUpdate = false;
let room = null, dots = null, dirty = true;
const tmpM = new THREE.Matrix4();

function token(name) {
  return new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888");
}
let COL = {};
function readColors() {
  COL = { todo: token("--todo"), seen: token("--seen"), good: token("--good"), stage: token("--stage") };
}

function setCamera(pose) {
  const R = pose.R, p = pose.pos;
  // OpenCV camera (x right, y down, z forward) -> three.js camera (x right, y up, looks along -z)
  tmpM.set(R[0], -R[1], -R[2], p[0], R[3], -R[4], -R[5], p[1], R[6], -R[7], -R[8], p[2], 0, 0, 0, 1);
  cam.matrixWorld.copy(tmpM);
  cam.matrixWorldInverse.copy(tmpM).invert();
  dirty = true;
}

function buildDots(scene) {
  const s = scene.step * 0.86;
  const geo = new THREE.PlaneGeometry(s, s);
  const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide,
                                            polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
  dots = new THREE.InstancedMesh(geo, mat, scene.n);
  const q = new THREE.Quaternion(), z = new THREE.Vector3(0, 0, 1), nrm = new THREE.Vector3();
  const one = new THREE.Vector3(1, 1, 1), at = new THREE.Vector3(), m = new THREE.Matrix4();
  for (let i = 0; i < scene.n; i++) {
    nrm.set(scene.N[3 * i], scene.N[3 * i + 1], scene.N[3 * i + 2]);
    q.setFromUnitVectors(z, nrm);
    at.set(scene.X[3 * i], scene.X[3 * i + 1], scene.X[3 * i + 2]).addScaledVector(nrm, 0.015);
    dots.setMatrixAt(i, m.compose(at, q, one));
    dots.setColorAt(i, COL.todo);
  }
  dots.renderOrder = 2;
  world.add(dots);
  roomLook(app.mode);
}

function paint(st, only = null) {
  if (!dots) return;
  const set = (i) => dots.setColorAt(i, st.good[i] ? COL.good : st.seen[i] ? COL.seen : COL.todo);
  if (only) for (const i of only) set(i);
  else for (let i = 0; i < app.scene.n; i++) set(i);
  dots.instanceColor.needsUpdate = true;
  dirty = true;
}

/** Replay: the room model only writes depth (the video shows through, hidden surfaces stay hidden).
 *  Simulation: the textured model is drawn. */
function roomLook(mode) {
  room?.traverse((o) => {
    if (!o.isMesh) return;
    for (const m of [].concat(o.material)) { m.colorWrite = mode !== "replay"; m.needsUpdate = true; }
  });
  world.background = mode === "replay" ? null : COL.stage;
  if (dots) dots.material.opacity = mode === "replay" ? 0.4 : 0.5;     // the video needs to stay readable
  dirty = true;
}

function resize() {
  const el = $("phone");
  const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  dirty = true;
}
new ResizeObserver(resize).observe($("phone"));

// ------------------------------------------------------------------------------------------------- load
async function load() {
  readColors();
  if (!NAME) return fail("Open this page from a walkthrough (it needs <code>?w=&lt;walkthrough name&gt;</code>).");
  $("back").href = BASE;
  let data;
  try {
    const r = await fetch(BASE + "coach.json", { cache: "no-store" });
    if (r.status === 401) return fail("This server needs the access key: open the console once, enter it, then reload.");
    if (!r.ok) {
      return fail(`No coach data for <b>${esc(NAME)}</b> yet. Create it with<br><code>python scripts/export_coach.py ${esc(NAME)}</code>`);
    }
    data = await r.json();
  } catch (e) {
    return fail("Could not load the coach data: " + esc(String(e)));
  }
  const scene = (app.scene = E.loadScene(data));
  const [W, H] = scene.cam.image_size;
  $("phone").style.setProperty("--ar", `${W} / ${H}`);
  cam.fov = (2 * Math.atan(H / 2 / scene.cam.fy) * 180) / Math.PI;
  cam.aspect = W / H;
  cam.updateProjectionMatrix();
  const frames = scene.frames.t.length;
  $("sub").textContent = `${data.name} · ${fmtTime(scene.frames.t[frames - 1])} video · ${scene.n.toLocaleString()} surface points every ${Math.round(scene.step * 100)} cm`;
  $("scrub").max = frames - 1;
  $("mark").style.left = `${scene.target}%`;
  $("mark-lbl").textContent = `target ${scene.target}%`;
  const ai = data.measured.overall.inspected_pct;
  $("ai-note").textContent = `The underwriting decision counts the surface the AI keyframes inspect: ${ai}% in this video. ` +
    "Keyframes are picked from what is filmed, so aim a little past the target.";
  buildDots(scene);
  resize();
  new GLTFLoader().load(BASE + "inspection_3d_inside.glb", (g) => {
    room = g.scene;
    world.add(room);
    roomLook(app.mode);
  }, undefined, () => { /* no model: the overlay still works, without occlusion by furniture */ });
  const v = $("video");
  v.src = BASE + (data.video || "source_video.mp4");
  v.addEventListener("loadeddata", () => showFrame(frameAt(v.currentTime)));
  v.addEventListener("seeked", () => showFrame(frameAt(v.currentTime)));
  v.addEventListener("timeupdate", () => showFrame(frameAt(v.currentTime)));
  v.addEventListener("play", () => { $("play").textContent = "Pause"; trackVideo(); });
  v.addEventListener("pause", () => { $("play").textContent = "Play"; });
  v.addEventListener("ended", () => { $("play").textContent = "Replay"; });
  v.addEventListener("error", () => { $("tip-detail").textContent = "The source video is missing; the overlay still follows the scrubber."; });
  showFrame(0);
  background();
  requestAnimationFrame(loop);
}

function fail(html) {
  $("msg").innerHTML = `<div>${html}</div>`;
  $("msg").hidden = false;
  $("tip").hidden = true;
  $("sub").textContent = NAME || "no walkthrough";
}

const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

/** Calibration (how optimistic the prediction is for this video) and the planner's standing-spot views, in
 *  small slices so the page stays responsive. */
function background() {
  const s = app.scene;
  if (!app.calib?.done) {
    app.calib = E.calibrate(s, 25);
    if (app.calib.done) {
      app.factor = app.calib.precision;
      updateMeter();
    }
    $("route-note").textContent = `Preparing predictions… ${Math.round(50 * app.calib.progress)}%`;
    return setTimeout(background, 20);
  }
  const p = E.warmSpots(s, 25);
  if (p < 1) {
    $("route-note").textContent = `Preparing predictions… ${Math.round(50 + 50 * p)}%`;
    return setTimeout(background, 20);
  }
  $("route-note").textContent = `Predicts the fewest extra views that bring the inspectable surface past ${s.target}%. ` +
    `New views count ×${app.factor.toFixed(2)}: on this video the room model saw ${Math.round(100 * (1 - app.factor))}% more ` +
    "than the depth camera did.";
}

// ------------------------------------------------------------------------------------------------- replay
function frameAt(t) {
  const T = app.scene.frames.t;
  let lo = 0, hi = T.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (T[m] <= t + 1e-3) lo = m; else hi = m - 1; }
  return lo;
}

function trackVideo() {
  const v = $("video");
  if (v.paused || app.mode !== "replay") return;
  showFrame(frameAt(v.currentTime));
  if ("requestVideoFrameCallback" in v) v.requestVideoFrameCallback(trackVideo);
  else requestAnimationFrame(trackVideo);
}

function showFrame(f) {
  if (app.mode !== "replay" || !app.scene) return;
  if (f === app.last) return;
  app.frame = app.last = f;
  const s = app.scene;
  app.shown = E.stateAtFrame(s, f);
  paint(app.shown);
  setCamera(E.poseFromFrame(s, f));
  $("scrub").value = f;
  const T = s.frames.t;
  $("tlabel").textContent = `${fmtTime(T[f])} / ${fmtTime(T[T.length - 1])}`;
  $("time-chip").textContent = `frame ${f + 1} of ${T.length}`;
  updateMeter();
  tipSoon();
}

// ------------------------------------------------------------------------------------------------- simulation
function enterSim(fromFrame = app.scene.frames.t.length - 1) {
  const s = app.scene;
  $("video").pause();
  app.mode = "sim";
  app.base = E.stateAtFrame(s, fromFrame);
  app.sim = { seen: app.base.seen.slice(), good: app.base.good.slice() };
  app.shown = app.sim;
  const p = E.poseFromFrame(s, fromFrame);
  const a = E.anglesOf(p);
  app.pos = p.pos.slice();
  app.yaw = a.yaw;
  app.pitch = a.pitch;
  app.route = null;
  app.tip = null;                      // a new mode starts with a fresh instruction, not the held one
  renderRoute();
  setTabs();
  roomLook("sim");
  paint(app.sim);
  applyPose();
  updateMeter();
  tipSoon(true);
  $("phone").focus?.();
}

function enterReplay() {
  app.mode = "replay";
  app.last = -1;
  app.route = null;
  app.tip = null;
  renderRoute();
  setTabs();
  roomLook("replay");
  showFrame(app.frame);
}

function setTabs() {
  const sim = app.mode === "sim";
  $("tab-replay").setAttribute("aria-selected", String(!sim));
  $("tab-sim").setAttribute("aria-selected", String(sim));
  $("video").hidden = sim;
  $("ctl-replay").hidden = sim;
  $("ctl-sim").hidden = !sim;
  $("mode-chip").textContent = sim ? "Try it yourself · predicted coverage" : "Replay · measured coverage";
  $("mode-chip").classList.toggle("live", sim && app.filming);
  $("time-chip").hidden = sim;
}

function pose() {
  return E.poseFromAngles(app.pos, app.yaw, app.pitch);
}

function applyPose() {
  setCamera(pose());
}

let lastFilm = 0;
function filmNow(force = false) {
  if (app.mode !== "sim" || (!app.filming && !force)) return;
  const v = E.viewOf(app.scene, pose());
  const changed = [];
  for (const i of v.seen) if (!app.sim.seen[i]) { app.sim.seen[i] = 1; changed.push(i); }
  for (const i of v.good) if (!app.sim.good[i]) { app.sim.good[i] = 1; changed.push(i); }
  if (changed.length) {
    paint(app.sim, changed);
    updateMeter();
  }
}

function step(dt) {
  const k = app.keys;
  let fwd = 0, side = 0, turn = 0, tilt = 0;
  if (k.has("w") || k.has("fwd")) fwd += 1;
  if (k.has("s") || k.has("back")) fwd -= 1;
  if (k.has("a")) side -= 1;
  if (k.has("d")) side += 1;
  if (k.has("arrowleft") || k.has("left")) turn += 1;
  if (k.has("arrowright") || k.has("right")) turn -= 1;
  if (k.has("arrowup") || k.has("up")) tilt += 1;
  if (k.has("arrowdown") || k.has("down")) tilt -= 1;
  if (!fwd && !side && !turn && !tilt) return false;
  app.yaw = E.wrap180(app.yaw + turn * TURN_DPS * dt);
  app.pitch = Math.max(-80, Math.min(85, app.pitch + tilt * TILT_DPS * dt));
  if (fwd || side) {
    const y = (app.yaw * Math.PI) / 180;
    const f = [-Math.sin(y), -Math.cos(y)], r = [Math.cos(y), -Math.sin(y)];
    const d = MOVE_MPS * dt;
    const dx = (f[0] * fwd + r[0] * side) * d, dz = (f[1] * fwd + r[1] * side) * d;
    const p = app.pos;
    for (const [mx, mz] of [[dx, dz], [dx, 0], [0, dz]]) {        // slide along walls
      const n = [p[0] + mx, p[1], p[2] + mz];
      if (E.canMove(app.scene, p, n)) { app.pos = n; break; }
    }
  }
  applyPose();
  return true;
}

// ------------------------------------------------------------------------------------------------- guidance
let tipTimer = 0, tipHeld = 0;
function tipSoon(now = false) {
  clearTimeout(tipTimer);
  tipTimer = setTimeout(updateTip, now ? 0 : 120);
}

function currentPose() {
  return app.mode === "sim" ? pose() : E.poseFromFrame(app.scene, app.frame);
}

function updateTip() {
  const s = app.scene;
  if (!s || !app.shown) return;
  const tip = E.suggest(s, app.shown, currentPose(), { expectedPct: currentPct() });
  const t = performance.now();
  // hold a message for a moment so the coach does not flicker between similar options
  if (app.tip && tip.action !== "keep" && app.tip.action === tip.action && t - tipHeld < 1200) return;
  app.tip = tip;
  tipHeld = t;
  const ico = { keep: "●", turn: "↻", move: "➜", done: "✓", stuck: "?" }[tip.action] || "•";
  $("tip-ico").textContent = ico;
  $("tip-ico").className = "ico " + tip.action;
  const reached = currentPct() >= app.scene.target && tip.action !== "done";
  $("tip-text").textContent = app.mode === "replay" && tip.action !== "done" ? `Coach: ${tip.text}` : tip.text;
  $("tip-detail").textContent = (reached ? `Target reached. To add more: ` : "") + tip.detail;
  drawArrow(tip);
  drawPlan();
}

function drawArrow(tip) {
  const el = $("arrow");
  if (!tip.view || tip.action === "keep" || tip.action === "done") { el.hidden = true; return; }
  const { yaw, pitch } = E.anglesOf(currentPose());
  let dyaw = E.wrap180(tip.view.yaw - yaw), dp = tip.view.pitch - pitch;
  if (tip.action === "move") {
    const to = Math.atan2(-(tip.view.pos[0] - currentPose().pos[0]), -(tip.view.pos[2] - currentPose().pos[2])) * 180 / Math.PI;
    dyaw = E.wrap180(to - yaw);
    dp = 0;
  }
  let cls, glyph;
  if (Math.abs(dyaw) >= 20) { cls = dyaw > 0 ? "l" : "r"; glyph = dyaw > 0 ? "←" : "→"; }
  else if (Math.abs(dp) >= 15) { cls = dp > 0 ? "u" : "d"; glyph = dp > 0 ? "↑" : "↓"; }
  else { el.hidden = true; return; }
  el.className = "edge " + cls;
  el.textContent = glyph;
  el.hidden = false;
}

// ------------------------------------------------------------------------------------------------- side panel
function currentPct() {
  const s = app.scene;
  if (!s || !app.shown) return 0;
  if (app.mode === "sim") return E.expectedPct(s, app.sim, app.base, app.factor);
  return E.stats(s, app.shown).overall.inspectable_pct;
}

let meterTimer = 0;
function updateMeter() {
  clearTimeout(meterTimer);
  meterTimer = setTimeout(renderMeter, 60);
}

function renderMeter() {
  const s = app.scene;
  if (!s || !app.shown) return;
  const st = E.stats(s, app.shown);
  const p = currentPct();
  $("pct").textContent = p.toFixed(1);
  $("fill").style.width = `${Math.min(100, p)}%`;
  $("fill").classList.toggle("ok", p >= s.target);
  if (app.mode === "sim") {
    const measured = E.stats(s, app.base).overall.inspectable_pct;
    $("pct-kind").textContent = "expected";
    $("pct-note").textContent = `${measured}% measured in the video + ${(p - measured).toFixed(1)}% predicted from your new views` +
      (app.calib?.done ? ` (counted ×${app.factor.toFixed(2)}).` : " (calibrating…).");
  } else {
    $("pct-kind").textContent = "measured";
    $("pct-note").textContent = `Measured from the video's depth up to ${fmtTime(s.frames.t[app.frame])}. ` +
      (p >= s.target ? "Target reached." : `${(s.target - p).toFixed(1)}% short of the ${s.target}% target.`);
  }
  $("kinds").innerHTML = ["walls", "floor", "ceiling"].map((k) => {
    const v = st.byKind[k].inspectable_pct;
    return `<span>${k[0].toUpperCase() + k.slice(1)}</span><span class="b"><i style="width:${v}%"></i></span><span class="v">${v}%</span>`;
  }).join("");
  $("gaps").innerHTML = E.gaps(s, app.shown).slice(0, 7).map((g) =>
    `<li><span>${esc(g.label)}</span><span>${g.length_m != null ? `${g.length_m} m of wall` : `${g.uncovered_m2} m²`}</span></li>`)
    .join("") || "<li><span>Nothing large left</span><span></span></li>";
  drawPlan();
}

// floor plan: x to the right, -z up (north), like the 3D engine's floor plan
const plan = { cv: $("plan"), ctx: $("plan").getContext("2d"), box: null, wallCols: null };

function planSetup() {
  const s = app.scene;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < s.n; i++) {
    const x = s.X[3 * i], z = s.X[3 * i + 2];
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
  }
  const pad = 0.4;
  x0 -= pad; x1 += pad; z0 -= pad; z1 += pad;
  const W = plan.cv.width;
  const sc = W / (x1 - x0);
  plan.cv.height = Math.round((z1 - z0) * sc);
  plan.box = { x0, z0, sc };
  // wall samples grouped by column along each wall
  plan.wallCols = [];
  s.walls.forEach((w) => {
    const g = s.groups.findIndex((gr) => gr.id === w.id);
    const cols = new Map();
    for (let i = 0; i < s.n; i++) {
      if (s.group[i] !== g) continue;
      const t = w.axis === "x" ? s.X[3 * i + 2] : s.X[3 * i];
      const k = Math.round(t / s.step);
      if (!cols.has(k)) cols.set(k, []);
      cols.get(k).push(i);
    }
    plan.wallCols.push({ w, cols: [...cols.entries()] });
  });
}

const px = (x) => (x - plan.box.x0) * plan.box.sc;
const pz = (z) => (z - plan.box.z0) * plan.box.sc;

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawPlan() {
  const s = app.scene;
  if (!s || !app.shown) return;
  if (!plan.box) planSetup();
  const c = plan.ctx, st = app.shown, sc = plan.box.sc, step = s.step;
  c.clearRect(0, 0, plan.cv.width, plan.cv.height);
  const col = { todo: css("--todo"), seen: css("--seen"), good: css("--good"), text: css("--text"), t3: css("--text-3"),
                plan: css("--plan"), surf: css("--surface") };
  const kind = app.layer === "floor" ? 1 : 2;
  c.globalAlpha = 0.55;
  for (let i = 0; i < s.n; i++) {
    if (s.kind[i] !== kind) continue;
    c.fillStyle = st.good[i] ? col.good : st.seen[i] ? col.seen : col.todo;
    c.fillRect(px(s.X[3 * i] - step / 2), pz(s.X[3 * i + 2] - step / 2), step * sc + 0.5, step * sc + 0.5);
  }
  c.globalAlpha = 1;
  // furniture
  c.strokeStyle = col.t3; c.lineWidth = 1;
  for (const b of s.boxes) {
    c.save();
    c.translate(px(b.center[0]), pz(b.center[1]));
    c.rotate((b.angle_deg * Math.PI) / 180);
    c.strokeRect(-b.size[0] / 2 * sc, -b.size[1] / 2 * sc, b.size[0] * sc, b.size[1] * sc);
    c.restore();
  }
  // walls, coloured per column; door openings drawn light
  c.lineWidth = 5; c.lineCap = "butt";
  for (const { w, cols } of plan.wallCols) {
    for (const [k, idx] of cols) {
      let gd = 0, sn = 0;
      for (const i of idx) { gd += st.good[i]; sn += st.seen[i]; }
      c.strokeStyle = gd / idx.length >= 0.5 ? col.good : sn / idx.length >= 0.5 ? col.seen : col.todo;
      const t0 = k * step - step / 2, t1 = k * step + step / 2;
      c.beginPath();
      if (w.axis === "x") { c.moveTo(px(w.offset), pz(t0)); c.lineTo(px(w.offset), pz(t1)); }
      else { c.moveTo(px(t0), pz(w.offset)); c.lineTo(px(t1), pz(w.offset)); }
      c.stroke();
    }
    c.strokeStyle = col.surf;
    c.lineWidth = 3;
    for (const h of w.holes) {
      if (h.y0 > 0.5) continue;                                     // windows: keep the wall line
      c.beginPath();
      if (w.axis === "x") { c.moveTo(px(w.offset), pz(h.t0)); c.lineTo(px(w.offset), pz(h.t1)); }
      else { c.moveTo(px(h.t0), pz(w.offset)); c.lineTo(px(h.t1), pz(w.offset)); }
      c.stroke();
    }
    c.lineWidth = 5;
  }
  // the recorded walk
  const P = s.frames.pos;
  c.strokeStyle = col.t3; c.lineWidth = 1.5; c.setLineDash([4, 4]);
  c.beginPath();
  P.forEach((q, f) => (f ? c.lineTo(px(q[0]), pz(q[2])) : c.moveTo(px(q[0]), pz(q[2]))));
  c.stroke();
  c.setLineDash([]);
  // planned spots
  (app.route?.spots || []).forEach((sp, k) => {
    const x = px(sp.pos[0]), z = pz(sp.pos[2]);
    const y = (sp.yaw * Math.PI) / 180;
    c.strokeStyle = col.plan; c.lineWidth = 2;
    c.beginPath(); c.moveTo(x, z); c.lineTo(x - Math.sin(y) * 16, z - Math.cos(y) * 16); c.stroke();
    c.fillStyle = k < app.routeNext ? col.t3 : col.plan;
    c.beginPath(); c.arc(x, z, 9, 0, 2 * Math.PI); c.fill();
    c.fillStyle = "#fff"; c.font = "bold 11px Archivo, sans-serif"; c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText(String(k + 1), x, z + 0.5);
  });
  // suggested spot
  if (app.tip?.action === "move" && app.tip.view) {
    c.strokeStyle = col.text; c.lineWidth = 2; c.setLineDash([3, 3]);
    c.beginPath(); c.arc(px(app.tip.view.pos[0]), pz(app.tip.view.pos[2]), 12, 0, 2 * Math.PI); c.stroke();
    c.setLineDash([]);
  }
  // the phone and its field of view
  const cp = currentPose();
  const { yaw } = E.anglesOf(cp);
  const half = Math.atan(s.cam.image_size[0] / 2 / s.cam.fx);
  const x = px(cp.pos[0]), z = pz(cp.pos[2]), y = (yaw * Math.PI) / 180, L = s.goodRange * sc;
  c.fillStyle = col.text; c.globalAlpha = 0.14;
  c.beginPath(); c.moveTo(x, z);
  c.lineTo(x - Math.sin(y + half) * L, z - Math.cos(y + half) * L);
  c.lineTo(x - Math.sin(y - half) * L, z - Math.cos(y - half) * L);
  c.closePath(); c.fill();
  c.globalAlpha = 1;
  c.beginPath(); c.arc(x, z, 6, 0, 2 * Math.PI); c.fill();
  c.strokeStyle = col.surf; c.lineWidth = 2; c.stroke();
}

$("plan").addEventListener("click", (ev) => {
  const s = app.scene;
  if (!s || !plan.box) return;
  const r = plan.cv.getBoundingClientRect();
  const x = ((ev.clientX - r.left) / r.width) * plan.cv.width / plan.box.sc + plan.box.x0;
  const z = ((ev.clientY - r.top) / r.height) * plan.cv.height / plan.box.sc + plan.box.z0;
  if (!E.walkable(s, x, z)) return;
  if (app.mode !== "sim") enterSim(app.frame);
  app.pos = [x, app.pos[1], z];
  applyPose();
  filmNow();
  tipSoon(true);
});

// ------------------------------------------------------------------------------------------------- route
function makeRoute() {
  const s = app.scene;
  if (!s || !app.shown) return;
  if (!app.calib?.done) { $("route-note").textContent = "Still preparing predictions; try again in a moment."; return; }
  const base = app.mode === "sim" ? app.base : app.shown;
  app.route = E.plan(s, app.shown, currentPose(), { base, gainFactor: app.factor });
  app.routeNext = 0;
  renderRoute();
  drawPlan();
}

function renderRoute() {
  const r = app.route;
  $("next-spot").disabled = !r || app.routeNext >= (r?.spots.length || 0);
  if (!r) { $("route").innerHTML = ""; return; }
  $("route").innerHTML = r.spots.map((sp, k) => `
    <li class="${k < app.routeNext ? "done" : k === app.routeNext ? "next" : ""}">
      <span class="n">${k + 1}</span>
      <span class="what"><b>${esc(sp.target || "Surface")}</b>
        <div>walk ${sp.walk_m} m, ${sp.how} · +${sp.gain_m2} m² → ${sp.pct_after}%</div></span>
      <button class="ghost" type="button" data-spot="${k}">Go</button>
    </li>`).join("");
  $("route-note").textContent = r.spots.length
    ? `${r.start_pct}% → ${r.final_pct}% in ${r.spots.length} ${r.spots.length === 1 ? "view" : "views"}` +
      (r.reached ? "." : ": the target is out of reach from where a person can stand.")
    : (r.reached ? "Already past the target." : "No view from a standing spot adds enough.");
}

$("route").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-spot]");
  if (b) goSpot(Number(b.dataset.spot));
});

function goSpot(k) {
  const r = app.route;
  if (!r || !r.spots[k]) return;
  if (app.mode !== "sim") {
    const keep = r;
    enterSim(app.frame);
    app.route = keep;
  }
  const sp = r.spots[k];
  app.pos = sp.pos.slice();
  app.yaw = sp.yaw;
  app.pitch = sp.pitch;
  app.routeNext = k + 1;
  applyPose();
  filmNow();
  renderRoute();
  tipSoon(true);
}

// ------------------------------------------------------------------------------------------------- input
$("tab-replay").addEventListener("click", () => app.mode !== "replay" && enterReplay());
$("tab-sim").addEventListener("click", () => app.mode !== "sim" && enterSim());       // from the end of the video
$("continue").addEventListener("click", () => enterSim(app.frame));
$("reset").addEventListener("click", () => enterSim());
$("plan-btn").addEventListener("click", makeRoute);
$("next-spot").addEventListener("click", () => goSpot(app.routeNext));
$("play").addEventListener("click", () => {
  const v = $("video");
  if (v.ended) v.currentTime = 0;
  if (v.paused) v.play().catch(() => {}); else v.pause();
});
$("scrub").addEventListener("input", (ev) => {
  const f = Number(ev.target.value);
  const v = $("video");
  v.pause();
  if (Number.isFinite(v.duration)) v.currentTime = app.scene.frames.t[f] + 0.01;
  showFrame(f);
});
$("film").addEventListener("click", toggleFilm);
function toggleFilm() {
  app.filming = !app.filming;
  $("film").setAttribute("aria-pressed", String(app.filming));
  $("film-lbl").textContent = app.filming ? "Filming" : "Not filming";
  $("mode-chip").classList.toggle("live", app.mode === "sim" && app.filming);
  if (app.filming) filmNow();
}
for (const [id, layer] of [["lay-floor", "floor"], ["lay-ceil", "ceiling"]]) {
  $(id).addEventListener("click", () => {
    app.layer = layer;
    $("lay-floor").setAttribute("aria-pressed", String(layer === "floor"));
    $("lay-ceil").setAttribute("aria-pressed", String(layer === "ceiling"));
    drawPlan();
  });
}
$("theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.getAttribute("data-theme") === "dark" ||
    (!document.documentElement.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
  const t = dark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("pi_theme", t); } catch (e) { /* private window */ }
  readColors();
  roomLook(app.mode);
  if (app.shown) paint(app.shown);
  drawPlan();
});

const KEYS = new Set(["w", "a", "s", "d", "arrowleft", "arrowright", "arrowup", "arrowdown"]);
addEventListener("keydown", (ev) => {
  if (app.mode !== "sim" || ev.target.closest?.("input, textarea")) return;
  const k = ev.key.toLowerCase();
  if (k === " ") { ev.preventDefault(); if (!ev.repeat) toggleFilm(); return; }
  if (KEYS.has(k)) {
    ev.preventDefault();
    app.keys.add(k);
    if (!ev.repeat && step(0.12)) { filmNow(); drawPlan(); tipSoon(); }   // a tap moves a little; holding keeps going
  }
});
addEventListener("keyup", (ev) => app.keys.delete(ev.key.toLowerCase()));
addEventListener("blur", () => app.keys.clear());
document.querySelectorAll(".pad button").forEach((b) => {
  const k = b.dataset.k;
  b.addEventListener("pointerdown", (ev) => { ev.preventDefault(); app.keys.add(k); });
  for (const e of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(e, () => app.keys.delete(k));
});

// drag on the view to look around (switches to "Try it yourself" from the replay)
let drag = null;
$("phone").addEventListener("pointerdown", (ev) => {
  if (!app.scene || ev.button > 0) return;
  drag = { x: ev.clientX, y: ev.clientY, moved: false };
  $("phone").setPointerCapture(ev.pointerId);
});
$("phone").addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  if (!drag.moved && app.mode !== "sim") enterSim(app.frame);
  drag.moved = true;
  drag.x = ev.clientX; drag.y = ev.clientY;
  const deg = cam.fov / $("phone").clientHeight;
  app.yaw = E.wrap180(app.yaw + dx * deg);
  app.pitch = Math.max(-80, Math.min(85, app.pitch + dy * deg));
  applyPose();
});
for (const e of ["pointerup", "pointercancel"]) $("phone").addEventListener(e, () => { drag = null; tipSoon(); });

// ------------------------------------------------------------------------------------------------- loop
let prev = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - prev) / 1000);
  prev = now;
  if (app.mode === "sim" && app.scene) {
    const moved = step(dt);
    if (now - lastFilm > FILM_EVERY_MS) { lastFilm = now; filmNow(); }
    if (moved && now - tipHeld > TIP_EVERY_MS) tipSoon();
    if (moved) drawPlan();
  }
  if (dirty) {
    renderer.render(world, cam);
    dirty = false;
  }
  requestAnimationFrame(loop);
}

load();
