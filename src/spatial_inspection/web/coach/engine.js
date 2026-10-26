// Coverage coach engine: visibility, coverage, guidance and route planning on coach.json
// (backend/spatial/coach.py). Pure functions, no DOM: frontend/coach/app.js draws, tests/js/ tests.
//
// Mirrors compute_coverage in backend/spatial/coverage.py: a surface sample counts as "seen" when it is in the
// camera frame within max_range, and "inspectable" (good) when it is also within good_range, at most
// max_angle_deg off the surface normal, in a usable frame. compute_coverage tests occlusion against each frame's
// depth map; here walls (minus their door openings) and furniture boxes from the layout stand in for it, so
// results for new camera poses are predictions, while the replay uses the measured first_good of every sample.
//
// Frames: model x right, y up, z back, floor y = 0, metres. A pose is {pos: [x, y, z], R: row-major 3x3
// camera-to-model rotation in OpenCV convention (x right, y down, z forward), ok: usable frame}.
// Angles: yaw 0 looks along -z, positive yaw turns left; pitch positive looks up.

export const KINDS = ["walls", "floor", "ceiling"];
const FLOOR = 1;
const DEG = Math.PI / 180;

export function loadScene(d) {
  const n = d.samples.n;
  const X = new Float32Array(3 * n), N = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) X[i] = d.samples.xyz_cm[i] / 100;
  for (let i = 0; i < n; i++) N.set(d.normals[d.samples.normal[i]], 3 * i);
  const kind = Uint8Array.from(d.samples.kind);
  const group = Uint16Array.from(d.samples.group);
  const p = d.params;
  const boxes = d.boxes.map((b) => {
    const a = (b.angle_deg || 0) * DEG;
    // shrink by 2 cm: a ray grazing a box edge should not hide the floor right next to it
    return { ...b, c: Math.cos(a), s: Math.sin(a), hx: Math.max(0.01, b.size[0] / 2 - 0.02),
             hz: Math.max(0.01, b.size[1] / 2 - 0.02) };
  });
  const kindTotal = [0, 0, 0];
  const groupTotal = new Array(d.groups.length).fill(0);
  for (let i = 0; i < n; i++) { kindTotal[kind[i]]++; groupTotal[group[i]]++; }
  // walkable floor: 10 cm cells that have a floor sample (floor under furniture has none)
  const floorCells = new Set();
  for (let i = 0; i < n; i++) if (kind[i] === FLOOR) floorCells.add(cellKey(X[3 * i], X[3 * i + 2]));
  const ys = d.frames.pos.map((q) => q[1]).sort((a, b) => a - b);
  return {
    n, X, N, kind, group, data: d, groups: d.groups, walls: d.walls, boxes, rooms: d.rooms,
    ceil: d.ceiling_m, frames: d.frames, cam: d.camera,
    firstSeen: Int32Array.from(d.samples.first_seen), firstGood: Int32Array.from(d.samples.first_good),
    inspected: Uint8Array.from(d.samples.inspected),
    maxRange: p.max_range_m, goodRange: p.good_range_m, cosMax: Math.cos(p.max_angle_deg * DEG),
    step: p.sample_step_m, margin: p.margin_px ?? 4, target: p.target_pct ?? 60,
    kindTotal, groupTotal, floorCells, height: ys.length ? ys[ys.length >> 1] : 1.5,
    _posCache: new Map(),
  };
}

const cellKey = (x, z) => `${Math.floor(x * 10)},${Math.floor(z * 10)}`;

// ---------------------------------------------------------------------------------------------- poses
export function poseFromFrame(scene, f) {
  return { pos: scene.frames.pos[f].slice(), R: scene.frames.rot[f].slice(), ok: scene.frames.ok[f] !== 0 };
}

export function poseFromAngles(pos, yawDeg, pitchDeg) {
  const y = yawDeg * DEG, p = pitchDeg * DEG;
  const f = [-Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)];
  let r = [-f[2], 0, f[0]];                                  // f x up, up = +y
  const rl = Math.hypot(r[0], r[2]) || 1;
  r = [r[0] / rl, 0, r[2] / rl];
  const dn = [f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0]];   // f x r
  return { pos: pos.slice(), R: [r[0], dn[0], f[0], r[1], dn[1], f[1], r[2], dn[2], f[2]], ok: true };
}

export function anglesOf(pose) {
  const R = pose.R;
  const f = [R[2], R[5], R[8]];
  return { yaw: Math.atan2(-f[0], -f[2]) / DEG, pitch: Math.asin(Math.max(-1, Math.min(1, f[1]))) / DEG };
}

export const wrap180 = (a) => ((((a + 180) % 360) + 360) % 360) - 180;

// ---------------------------------------------------------------------------------------------- visibility
/** Is the segment from camera C to sample point P blocked by a wall (outside its openings) or a furniture box?
 *  The last 5 cm before P are ignored, so the surface a sample lies on never hides it. */
export function occluded(scene, C, P) {
  const dx = P[0] - C[0], dy = P[1] - C[1], dz = P[2] - C[2];
  const dist = Math.hypot(dx, dy, dz);
  const sMax = 1 - 0.05 / Math.max(dist, 0.06);
  for (const w of scene.walls) {
    const ax = w.axis === "x";
    const den = ax ? dx : dz;
    if (Math.abs(den) < 1e-9) continue;
    const s = (w.offset - (ax ? C[0] : C[2])) / den;
    if (s <= 1e-6 || s >= sMax) continue;
    const t = ax ? C[2] + s * dz : C[0] + s * dx;
    const y = C[1] + s * dy;
    if (t < w.t0 || t > w.t1 || y < 0 || y > scene.ceil) continue;
    let open = false;
    for (const h of w.holes) if (t >= h.t0 && t <= h.t1 && y >= h.y0 && y <= h.y1) { open = true; break; }
    if (!open) return true;
  }
  for (const b of scene.boxes) {
    const ox = (C[0] - b.center[0]) * b.c + (C[2] - b.center[1]) * b.s;
    const oz = -(C[0] - b.center[0]) * b.s + (C[2] - b.center[1]) * b.c;
    const lx = dx * b.c + dz * b.s, lz = -dx * b.s + dz * b.c;
    let t0 = 1e-6, t1 = sMax;
    for (const [o, v, lo, hi] of [[ox, lx, -b.hx, b.hx], [oz, lz, -b.hz, b.hz], [C[1], dy, b.y0, b.y1]]) {
      if (Math.abs(v) < 1e-12) { if (o < lo || o > hi) { t1 = -1; break; } continue; }
      let a = (lo - o) / v, c = (hi - o) / v;
      if (a > c) [a, c] = [c, a];
      if (a > t0) t0 = a;
      if (c < t1) t1 = c;
      if (t0 > t1) break;
    }
    if (t0 <= t1) return true;
  }
  return false;
}

/** Samples visible from a camera position in any direction: {seen, good} index arrays (cached per position). */
export function fromPosition(scene, C, ok = true) {
  const key = `${C[0].toFixed(2)},${C[1].toFixed(2)},${C[2].toFixed(2)},${ok ? 1 : 0}`;
  const hit = scene._posCache.get(key);
  if (hit) return hit;
  const { X, N, n, maxRange, goodRange, cosMax } = scene;
  const seen = [], good = [];
  const P = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const dx = X[3 * i] - C[0], dy = X[3 * i + 1] - C[1], dz = X[3 * i + 2] - C[2];
    const dist = Math.hypot(dx, dy, dz);
    if (dist > maxRange || dist < 0.2) continue;
    const facing = -(dx * N[3 * i] + dy * N[3 * i + 1] + dz * N[3 * i + 2]) / dist;   // cos to the normal
    if (facing <= 0) continue;                         // camera behind the surface
    P[0] = X[3 * i]; P[1] = X[3 * i + 1]; P[2] = X[3 * i + 2];
    if (occluded(scene, C, P)) continue;
    seen.push(i);
    if (ok && dist <= goodRange && facing >= scene.cosMax) good.push(i);
  }
  const out = { seen: Int32Array.from(seen), good: Int32Array.from(good) };
  if (scene._posCache.size > 400) scene._posCache.clear();
  scene._posCache.set(key, out);
  return out;
}

/** Which of `idx` fall inside the camera frame of `pose` (in front, within the image minus the margin). */
export function inFrame(scene, pose, idx) {
  const { X } = scene, { fx, fy, cx, cy, image_size: [W, H] } = scene.cam, m = scene.margin;
  const R = pose.R, C = pose.pos;
  const out = [];
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const dx = X[3 * i] - C[0], dy = X[3 * i + 1] - C[1], dz = X[3 * i + 2] - C[2];
    const zc = R[2] * dx + R[5] * dy + R[8] * dz;
    if (zc <= 0.2) continue;
    const u = (fx * (R[0] * dx + R[3] * dy + R[6] * dz)) / zc + cx;
    const v = (fy * (R[1] * dx + R[4] * dy + R[7] * dz)) / zc + cy;
    if (u >= m && u < W - m && v >= m && v < H - m) out.push(i);
  }
  return out;
}

/** What one camera pose sees: {seen, good} index arrays. */
export function viewOf(scene, pose) {
  const v = fromPosition(scene, pose.pos, pose.ok !== false);
  return { seen: inFrame(scene, pose, v.seen), good: inFrame(scene, pose, v.good) };
}

// ---------------------------------------------------------------------------------------------- coverage state
/** Coverage at a frame of the recorded walk, from the measured first_seen / first_good (Infinity = whole video). */
export function stateAtFrame(scene, f = Infinity) {
  const seen = new Uint8Array(scene.n), good = new Uint8Array(scene.n);
  for (let i = 0; i < scene.n; i++) {
    if (scene.firstSeen[i] >= 0 && scene.firstSeen[i] <= f) seen[i] = 1;
    if (scene.firstGood[i] >= 0 && scene.firstGood[i] <= f) good[i] = 1;
  }
  return { seen, good };
}

export function emptyState(scene) {
  return { seen: new Uint8Array(scene.n), good: new Uint8Array(scene.n) };
}

/** Add what a pose sees to a state; returns the number of samples that became inspectable. */
export function film(scene, state, pose) {
  const v = viewOf(scene, pose);
  for (const i of v.seen) state.seen[i] = 1;
  let gained = 0;
  for (const i of v.good) if (!state.good[i]) { state.good[i] = 1; gained++; }
  return gained;
}

const pct = (a, b) => Math.round((1000 * a) / Math.max(1, b)) / 10;

export function stats(scene, state) {
  const seenK = [0, 0, 0], goodK = [0, 0, 0];
  const goodG = new Array(scene.groups.length).fill(0);
  let seen = 0, good = 0;
  for (let i = 0; i < scene.n; i++) {
    if (state.seen[i] || state.good[i]) { seen++; seenK[scene.kind[i]]++; }
    if (state.good[i]) { good++; goodK[scene.kind[i]]++; goodG[scene.group[i]]++; }
  }
  const byKind = {};
  KINDS.forEach((k, j) => {
    byKind[k] = { seen_pct: pct(seenK[j], scene.kindTotal[j]), inspectable_pct: pct(goodK[j], scene.kindTotal[j]),
                  area_m2: round1(scene.kindTotal[j] * scene.step ** 2) };
  });
  return {
    overall: { seen_pct: pct(seen, scene.n), inspectable_pct: pct(good, scene.n), good, n: scene.n },
    byKind,
    byGroup: scene.groups.map((g, j) => ({ ...g, inspectable_pct: pct(goodG[j], scene.groupTotal[j]),
                                           uncovered_m2: round1((scene.groupTotal[j] - goodG[j]) * scene.step ** 2) })),
  };
}

const round1 = (v) => Math.round(v * 10) / 10;

/** What still has to be filmed, like Coverage.gaps(): wall stretches not inspectable at any height (>= 0.4 m),
 *  floors and ceilings under 80% inspectable. Longest / largest first. */
export function gaps(scene, state, minLen = 0.4) {
  const out = [];
  const step = scene.step;
  scene.groups.forEach((g, j) => {
    if (g.kind === "wall") {
      const wall = scene.walls.find((w) => w.id === g.id);
      if (!wall) return;
      const cols = new Map();
      for (let i = 0; i < scene.n; i++) {
        if (scene.group[i] !== j) continue;
        const t = wall.axis === "x" ? scene.X[3 * i + 2] : scene.X[3 * i];
        const k = Math.round(t / step);
        const c = cols.get(k) || [0, 0];
        c[0] += state.good[i]; c[1]++;
        cols.set(k, c);
      }
      const keys = [...cols.keys()].sort((a, b) => a - b);
      let run = [];
      const flush = () => {
        if (run.length * step >= minLen - 1e-9) {
          out.push({ group: j, label: g.label, kind: "wall", from_t: round2(run[0] * step),
                     to_t: round2((run[run.length - 1] + 1) * step), length_m: round2(run.length * step) });
        }
        run = [];
      };
      for (const k of keys) {
        const [gd, tot] = cols.get(k);
        if (gd / tot < 0.5 && (!run.length || k === run[run.length - 1] + 1)) run.push(k);
        else { flush(); if (gd / tot < 0.5) run.push(k); }
      }
      flush();
    } else {
      let gd = 0;
      for (let i = 0; i < scene.n; i++) if (scene.group[i] === j) gd += state.good[i];
      const tot = scene.groupTotal[j];
      if (tot && gd / tot < 0.8) out.push({ group: j, label: g.label, kind: g.kind, uncovered_m2: round1((tot - gd) * step ** 2) });
    }
  });
  return out.sort((a, b) => (b.length_m ?? b.uncovered_m2) - (a.length_m ?? a.uncovered_m2));
}

const round2 = (v) => Math.round(v * 100) / 100;

// ---------------------------------------------------------------------------------------------- walking
export function walkable(scene, x, z) {
  return scene.floorCells.has(cellKey(x, z));
}

/** Can the phone move from a to b (floor under b, no wall in between at waist height)? */
export function canMove(scene, a, b) {
  if (!walkable(scene, b[0], b[2])) return false;
  const lift = (q) => [q[0], 1.0, q[2]];
  const A = lift(a), B = lift(b);
  for (const w of scene.walls) {
    const ax = w.axis === "x";
    const p0 = ax ? A[0] : A[2], p1 = ax ? B[0] : B[2];
    if ((p0 - w.offset) * (p1 - w.offset) > 0 || p0 === p1) continue;
    const s = (w.offset - p0) / (p1 - p0);
    const t = ax ? A[2] + s * (B[2] - A[2]) : A[0] + s * (B[0] - A[0]);
    if (t < w.t0 || t > w.t1) continue;
    if (!w.holes.some((h) => t >= h.t0 - 0.05 && t <= h.t1 + 0.05 && h.y0 <= 1.0 && h.y1 >= 1.0)) return false;
  }
  return true;
}

/** Standing spots on a lattice: on the floor, not inside furniture, 35 cm clear of walls. */
export function standingSpots(scene, spacing = 0.5) {
  if (scene._spots?.spacing === spacing) return scene._spots.list;
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  for (let i = 0; i < scene.n; i++) {
    if (scene.kind[i] !== FLOOR) continue;
    const x = scene.X[3 * i], z = scene.X[3 * i + 2];
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
  }
  const list = [];
  for (let x = Math.ceil(x0 / spacing) * spacing; x <= x1; x += spacing) {
    for (let z = Math.ceil(z0 / spacing) * spacing; z <= z1; z += spacing) {
      const near = [[0, 0], [0.3, 0], [-0.3, 0], [0, 0.3], [0, -0.3]].every(([a, b]) => walkable(scene, x + a, z + b));
      if (!near) continue;
      const clear = scene.walls.every((w) => {
        const d = Math.abs((w.axis === "x" ? x : z) - w.offset);
        const t = w.axis === "x" ? z : x;
        return d >= 0.35 || t < w.t0 - 0.2 || t > w.t1 + 0.2;
      });
      if (clear) list.push([round2(x), scene.height, round2(z)]);
    }
  }
  scene._spots = { spacing, list };
  return list;
}

// ---------------------------------------------------------------------------------------------- guidance
const YAWS = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180];
const PITCHES = [-45, -20, 0, 30, 55, 75];

function newGood(state, idx) {
  let k = 0;
  for (const i of idx) if (!state.good[i]) k++;
  return k;
}

function mainGroup(scene, state, idx) {
  const count = new Map();
  for (const i of idx) if (!state.good[i]) count.set(scene.group[i], (count.get(scene.group[i]) || 0) + 1);
  let best = -1, bc = 0;
  for (const [g, c] of count) if (c > bc) { best = g; bc = c; }
  return best >= 0 ? scene.groups[best] : null;
}

/** Best views from one position: [{yaw, pitch, gain, idx}] sorted by gain. */
function viewsAt(scene, state, pos, yaws, pitches) {
  const v = fromPosition(scene, pos);
  const out = [];
  for (const yaw of yaws) {
    for (const pitch of pitches) {
      const idx = inFrame(scene, poseFromAngles(pos, yaw, pitch), v.good);
      const gain = newGood(state, idx);
      if (gain) out.push({ yaw, pitch, gain, idx });
    }
  }
  return out.sort((a, b) => b.gain - a.gain);
}

const SPOT_YAWS = [0, 30, 60, 90, 120, 150, 180, -150, -120, -90, -60, -30];
const SPOT_PITCHES = [-40, 0, 45, 70];

/** The fixed set of views from every standing spot. Geometry only, so it is computed once per scene; gains
 *  are recounted against the current coverage each time. `warmSpots` fills it in slices for a responsive page. */
function spotViews(scene) {
  if (!scene._spotViews) warmSpots(scene, Infinity);
  return scene._spotViews;
}

export function warmSpots(scene, budget = 50) {
  const spots = standingSpots(scene);
  const out = (scene._spotViews ??= []);
  const t0 = Date.now();
  while (out.length < spots.length && Date.now() - t0 < budget) {
    const s = spots[out.length];
    const v = fromPosition(scene, s);
    const views = [];
    for (const yaw of SPOT_YAWS) for (const pitch of SPOT_PITCHES) {
      const idx = Int32Array.from(inFrame(scene, poseFromAngles(s, yaw, pitch), v.good));
      if (idx.length) views.push({ yaw, pitch, idx });
    }
    out.push({ pos: s, views });
  }
  return out.length / Math.max(1, spots.length);           // 1 = done
}

/** How optimistic the layout-based visibility is for this video: replay the recorded path, compare with the
 *  measured coverage. `precision` = share of predicted-inspectable samples the depth maps also counted; predicted
 *  gains are multiplied by it. Runs in slices: call until it returns {done: true}. */
export function calibrate(scene, budget = 40) {
  const c = (scene._calib ??= { f: 0, pred: emptyState(scene), done: false });
  const frames = scene.frames.t.length;
  const t0 = Date.now();
  while (c.f < frames && Date.now() - t0 < budget) film(scene, c.pred, poseFromFrame(scene, c.f++));
  if (c.f >= frames && !c.done) {
    let both = 0, pred = 0, meas = 0;
    for (let i = 0; i < scene.n; i++) {
      const m = scene.firstGood[i] >= 0;
      pred += c.pred.good[i]; meas += m; both += c.pred.good[i] && m;
    }
    Object.assign(c, { done: true, precision: pred ? both / pred : 1, recall: meas ? both / meas : 1,
                       predicted_pct: pct(pred, scene.n), measured_pct: pct(meas, scene.n) });
  }
  return { ...c, pred: undefined, progress: c.f / Math.max(1, frames) };
}

/** Expected inspectable coverage when `state` grew out of `base` by predicted views worth `factor` each. */
export function expectedPct(scene, state, base, factor = 1) {
  let have = 0, added = 0;
  for (let i = 0; i < scene.n; i++) {
    if (base.good[i]) have++;
    else if (state.good[i]) added++;
  }
  return pct(have + factor * added, scene.n);
}

function turnPhrase(dyaw, dpitch) {
  const parts = [];
  if (Math.abs(dyaw) >= 20) {
    parts.push(Math.abs(dyaw) >= 160 ? "Turn around" : `Turn ${dyaw > 0 ? "left" : "right"} about ${Math.round(Math.abs(dyaw) / 10) * 10}°`);
  }
  if (Math.abs(dpitch) >= 15) parts.push(`tilt ${dpitch > 0 ? "up" : "down"} about ${Math.round(Math.abs(dpitch) / 5) * 5}°`);
  if (!parts.length) return "Hold this view";
  const s = parts.join(", then ");
  return s[0].toUpperCase() + s.slice(1);
}

function tiltPhrase(pitch) {
  if (pitch >= 15) return `tilt up about ${Math.round(pitch / 5) * 5}°`;
  if (pitch <= -15) return `tilt down about ${Math.round(-pitch / 5) * 5}°`;
  return "hold the phone level";
}

function bearingPhrase(rel) {
  const a = Math.abs(rel);
  if (a < 25) return "ahead";
  if (a < 70) return rel > 0 ? "ahead on your left" : "ahead on your right";
  if (a < 115) return rel > 0 ? "to your left" : "to your right";
  return "behind you";
}

function what(group, gainM2) {
  if (!group) return `${gainM2.toFixed(1)} m² not covered yet`;
  return `${group.label}: ${gainM2.toFixed(1)} m² not covered yet`;
}

/**
 * The next instruction for a person holding the phone at `pose`, given what is already covered.
 * Returns {action: keep|turn|move|done|stuck, text, detail, gain_m2, view?: {pos, yaw, pitch}, pct_after}.
 */
export function suggest(scene, state, pose, opts = {}) {
  const minGain = opts.minGain ?? 15;                   // samples (0.15 m² at 10 cm) worth acting on
  const st = stats(scene, state).overall;
  const a2 = scene.step ** 2;
  const here = viewOf(scene, pose);
  const nowGain = newGood(state, here.good);
  const { yaw, pitch } = anglesOf(pose);
  const reached = opts.expectedPct ?? st.inspectable_pct;
  const done = reached >= scene.target;
  if (nowGain >= minGain) {
    const g = mainGroup(scene, state, here.good);
    return { action: "keep", text: "Keep filming this view", detail: `${what(g, nowGain * a2)} in frame`,
             gain_m2: round1(nowGain * a2), view: { pos: pose.pos, yaw, pitch } };
  }
  // 1. turn or tilt where you stand
  const turns = viewsAt(scene, state, pose.pos, YAWS.map((d) => yaw + d), PITCHES);
  // among similar gains, prefer the smaller turn
  const turnScore = (v) => v.gain / (1 + Math.abs(wrap180(v.yaw - yaw)) / 720);
  const bestTurn = turns.sort((a, b) => turnScore(b) - turnScore(a))[0];
  // 2. walk to a nearby spot (cost grows with distance); only spots warmed so far, so this never blocks
  let bestMove = null;
  for (const sp of scene._spotViews ?? []) {
    const d = Math.hypot(sp.pos[0] - pose.pos[0], sp.pos[2] - pose.pos[2]);
    if (d < 0.4 || d > (opts.maxWalk ?? 4)) continue;
    let best = null, bg = 0;
    for (const v of sp.views) { const g = newGood(state, v.idx); if (g > bg) { bg = g; best = v; } }
    if (!best) continue;
    const score = bg / (1 + d / 1.5);
    if (!bestMove || score > bestMove.score) bestMove = { ...best, gain: bg, pos: sp.pos, dist: d, score };
  }
  // walking is harder to follow than turning: only send the person somewhere if it pays clearly more
  const turnOk = bestTurn && bestTurn.gain >= minGain;
  const moveBetter = bestMove && bestMove.gain >= minGain && (!turnOk || bestMove.gain > 1.6 * bestTurn.gain);
  if (turnOk && !moveBetter) {
    const g = mainGroup(scene, state, bestTurn.idx);
    return { action: "turn", text: turnPhrase(wrap180(bestTurn.yaw - yaw), bestTurn.pitch - pitch),
             detail: what(g, bestTurn.gain * a2), gain_m2: round1(bestTurn.gain * a2),
             view: { pos: pose.pos, yaw: bestTurn.yaw, pitch: bestTurn.pitch } };
  }
  if (moveBetter) {
    const g = mainGroup(scene, state, bestMove.idx);
    const toYaw = Math.atan2(-(bestMove.pos[0] - pose.pos[0]), -(bestMove.pos[2] - pose.pos[2])) / DEG;
    const where = bearingPhrase(wrap180(toYaw - yaw));
    const face = turnPhrase(wrap180(bestMove.yaw - toYaw), bestMove.pitch);
    return { action: "move", text: `Walk ${bestMove.dist.toFixed(1)} m ${where}`,
             detail: `${face.replace(/^Hold this view$/, "Look straight ahead")} to film ${what(g, bestMove.gain * a2)}`,
             gain_m2: round1(bestMove.gain * a2), view: { pos: bestMove.pos, yaw: bestMove.yaw, pitch: bestMove.pitch } };
  }
  if (done) {
    return { action: "done", text: `Coverage target reached: about ${reached}% inspectable`,
             detail: "Film any remaining red areas if you can, then stop.", gain_m2: 0 };
  }
  return { action: "stuck", text: "Nothing more to gain nearby",
           detail: "The remaining areas are blocked or out of reach from here. Try the route plan.", gain_m2: 0 };
}

/**
 * Predict a short route of views that brings inspectable coverage to the target: greedy next-best views over
 * standing spots x directions, then ordered as a walk from the start pose. Each spot reports the coverage
 * expected after it. Samples in `opts.base` (the measured part) count fully; predicted ones count
 * `opts.gainFactor` each (from calibrate), so the plan aims past the target by the model's optimism.
 */
export function plan(scene, state, start, opts = {}) {
  const target = opts.targetPct ?? scene.target;
  const maxSpots = opts.maxSpots ?? 12;
  const factor = opts.gainFactor ?? 1;
  const base = opts.base ?? state;
  const good = state.good.slice();
  const views = [];
  for (const sp of spotViews(scene)) for (const v of sp.views) views.push({ ...v, pos: sp.pos });
  const need = (target / 100) * scene.n;
  let have = 0, added = 0;
  for (let i = 0; i < scene.n; i++) { if (base.good[i]) have++; else if (good[i]) added++; }
  const value = () => have + factor * added;
  const startPct = pct(value(), scene.n);
  const picked = [];
  while (value() < need && picked.length < maxSpots) {
    let best = null, bg = 0;
    for (const v of views) {
      const g = newGood({ good }, v.idx);
      if (g > bg) { bg = g; best = v; }
    }
    if (!best || bg < (opts.minGain ?? 10)) break;
    picked.push(best);
    for (const i of best.idx) if (!good[i]) { good[i] = 1; added++; }
  }
  // walk order: nearest next spot from the start position
  const order = [];
  let at = start.pos;
  const left = picked.slice();
  while (left.length) {
    let k = 0, kd = Infinity;
    left.forEach((v, j) => { const d = Math.hypot(v.pos[0] - at[0], v.pos[2] - at[2]); if (d < kd) { kd = d; k = j; } });
    const v = left.splice(k, 1)[0];
    order.push({ ...v, walk_m: round1(kd) });
    at = v.pos;
  }
  const g2 = state.good.slice();
  let a2 = 0;
  for (let i = 0; i < scene.n; i++) if (!base.good[i] && g2[i]) a2++;
  const spots = order.map((v) => {
    const gpre = newGood({ good: g2 }, v.idx);
    const grp = mainGroup(scene, { good: g2 }, v.idx);
    for (const i of v.idx) if (!g2[i]) { g2[i] = 1; a2++; }
    return { pos: v.pos, yaw: v.yaw, pitch: v.pitch, walk_m: v.walk_m, gain_m2: round1(gpre * scene.step ** 2),
             target: grp ? grp.label : "", pct_after: pct(have + factor * a2, scene.n), how: tiltPhrase(v.pitch) };
  });
  return { start_pct: startPct, final_pct: pct(value(), scene.n), target_pct: target, reached: value() >= need,
           gain_factor: factor, spots };
}
