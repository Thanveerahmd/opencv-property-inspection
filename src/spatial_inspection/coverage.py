"""Inspection coverage: which parts of the walls, floor and ceiling the camera has seen, and seen well enough to
inspect (ported from the inspection platform so the numbers match).

Surfaces are sampled every `step` metres on the room layout. For every frame each sample is projected into the
camera with cv2.projectPoints and checked against that frame's depth map, so furniture in front of a wall occludes
it. A sample is inspectable when a frame sees it within `good_range` metres, at most `max_angle_deg` off
straight-on, and the frame is not BAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scene import Walkthrough

SURF_WALL, SURF_FLOOR, SURF_CEIL = 0, 1, 2
_SIDE = {("x", 1): "west", ("x", -1): "east", ("z", 1): "north", ("z", -1): "south"}


@dataclass
class Samples:
    xyz: np.ndarray            # (N, 3)
    normal: np.ndarray         # (N, 3) facing the room
    kind: np.ndarray           # (N,) SURF_*
    group: list[str]           # (N,) wall id, room id (floor) or room id + "_ceiling"
    t: np.ndarray              # (N,) position along the wall (walls) else 0
    step: float
    groups: dict = field(default_factory=dict)   # group id -> metadata (label, kind, length/area)


def _in_footprint(xz: np.ndarray, fp: dict, pad: float = 0.05) -> np.ndarray:
    c = np.asarray(fp["center"], float)
    sx, sz = np.asarray(fp["size"], float) / 2 + pad
    a = np.deg2rad(fp.get("angle_deg", 0.0))
    d = xz - c
    u = d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a)
    v = -d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)
    return (np.abs(u) <= sx) & (np.abs(v) <= sz)


def make_samples(layout: dict, step: float = 0.1) -> Samples:
    ceil = float(layout.get("ceiling_height_m", 2.6))
    P, N, Kd, G, T = [], [], [], [], []
    groups: dict = {}
    for w in layout["walls"]:
        ts = np.arange(w["t0"] + step / 2, w["t1"], step)
        ys = np.arange(step / 2, ceil, step)
        tt, yy = np.meshgrid(ts, ys, indexing="ij")
        tt, yy = tt.ravel(), yy.ravel()
        if w["axis"] == "x":
            p = np.c_[np.full_like(tt, w["offset"]), yy, tt]
            n = np.array([w["sign"], 0, 0], float)
        else:
            p = np.c_[tt, yy, np.full_like(tt, w["offset"])]
            n = np.array([0, 0, w["sign"]], float)
        P.append(p)
        N.append(np.tile(n, (len(p), 1)))
        Kd.append(np.full(len(p), SURF_WALL))
        G += [w["id"]] * len(p)
        T.append(tt)
        # the wall's normal points into the room, so a wall facing +z is the room's north wall (plan has -z up)
        groups[w["id"]] = {"kind": "wall", "length_m": round(w["t1"] - w["t0"], 2),
                           "label": f"{_SIDE.get((w['axis'], int(w['sign'])), '')} wall ({w['id']})".strip()}
    rooms = layout.get("rooms", [])
    parts = []
    for r in rooms:
        x0, z0, x1, z1 = r["bbox_m"]
        xx, zz = np.meshgrid(np.arange(x0 + step / 2, x1, step), np.arange(z0 + step / 2, z1, step), indexing="ij")
        parts.append(np.c_[xx.ravel(), zz.ravel()])
    xz = np.vstack(parts) if parts else np.zeros((0, 2))
    rid = np.full(len(xz), "", object)
    for r in rooms:
        x0, z0, x1, z1 = r["bbox_m"]
        m = (rid == "") & (xz[:, 0] >= x0 - 0.05) & (xz[:, 0] <= x1 + 0.05) & (xz[:, 1] >= z0 - 0.05) & (xz[:, 1] <= z1 + 0.05)
        rid[m] = r["id"]
    names = {r["id"]: r.get("name", r["id"]) for r in rooms}
    objs = [o for o in layout.get("objects", []) if not o.get("ceiling_mounted") and o.get("footprint")]
    under = np.zeros(len(xz), bool)
    for o in objs:
        under |= _in_footprint(xz, o["footprint"])
    for g in [r["id"] for r in rooms]:
        m = rid == g
        if not m.any():
            continue
        fxz, cxz = xz[m & ~under], xz[m]
        P.append(np.c_[fxz[:, 0], np.zeros(len(fxz)), fxz[:, 1]])
        N.append(np.tile([0, 1, 0], (len(fxz), 1)))
        Kd.append(np.full(len(fxz), SURF_FLOOR))
        G += [g] * len(fxz)
        T.append(np.zeros(len(fxz)))
        P.append(np.c_[cxz[:, 0], np.full(len(cxz), ceil), cxz[:, 1]])
        N.append(np.tile([0, -1, 0], (len(cxz), 1)))
        Kd.append(np.full(len(cxz), SURF_CEIL))
        G += [g + "_ceiling"] * len(cxz)
        T.append(np.zeros(len(cxz)))
        groups[g] = {"kind": "floor", "label": f"{names[g]} floor", "area_m2": round(len(fxz) * step * step, 2),
                     "furniture_excluded_m2": round(int((m & under).sum()) * step * step, 2)}
        groups[g + "_ceiling"] = {"kind": "ceiling", "label": f"{names[g]} ceiling",
                                  "area_m2": round(len(cxz) * step * step, 2)}
    return Samples(xyz=np.vstack(P), normal=np.vstack(N).astype(float), kind=np.concatenate(Kd), group=G,
                   t=np.concatenate(T), step=step, groups=groups)


@dataclass
class Coverage:
    samples: Samples
    first_seen: np.ndarray      # (N,) frame index or -1
    first_good: np.ndarray      # (N,) frame index or -1
    best_dist: np.ndarray       # (N,) metres (inf if unseen)
    good_by_frame: list         # per frame: indices of samples inspectable in that frame
    seen_by_frame: list         # per frame: indices of samples seen (any quality) in that frame
    inspected: np.ndarray       # (N,) bool: seen inspectably by an analysed keyframe
    params: dict

    def state(self, upto: int | None = None) -> np.ndarray:
        """0 unseen, 1 seen (far or oblique), 2 inspectable, 3 inspected; `upto` limits to frames so far."""
        s = np.zeros(len(self.first_seen), np.int8)
        lim = np.inf if upto is None else upto
        s[(self.first_seen >= 0) & (self.first_seen <= lim)] = 1
        s[(self.first_good >= 0) & (self.first_good <= lim)] = 2
        if upto is None:
            s[self.inspected] = 3
        return s

    def stats(self, upto: int | None = None) -> dict:
        st = self.state(upto)
        S = self.samples
        out = {"by_kind": {}, "by_group": {}}
        for k, name in ((SURF_WALL, "walls"), (SURF_FLOOR, "floor"), (SURF_CEIL, "ceiling")):
            m = S.kind == k
            tot = max(1, int(m.sum()))
            out["by_kind"][name] = {"seen_pct": _pct(st[m] >= 1, tot), "inspectable_pct": _pct(st[m] >= 2, tot),
                                    "inspected_pct": _pct(st[m] >= 3, tot), "area_m2": round(tot * S.step * S.step, 1)}
        G = np.array(S.group)
        for g, meta in S.groups.items():
            m = G == g
            tot = max(1, int(m.sum()))
            out["by_group"][g] = {**meta, "seen_pct": _pct(st[m] >= 1, tot), "inspectable_pct": _pct(st[m] >= 2, tot),
                                  "inspected_pct": _pct(st[m] >= 3, tot)}
        tot = max(1, len(st))
        out["overall"] = {"seen_pct": _pct(st >= 1, tot), "inspectable_pct": _pct(st >= 2, tot),
                          "inspected_pct": _pct(st >= 3, tot)}
        return out

    def gaps(self, min_len_m: float = 0.4) -> list[dict]:
        """Surfaces never seen inspectably, largest first: what to go back and film."""
        S = self.samples
        st = self.state()
        G = np.array(S.group)
        out = []
        for g, meta in S.groups.items():
            if meta["kind"] != "wall":
                continue
            m = np.where(G == g)[0]
            ts = np.round(S.t[m] / S.step).astype(int)
            cols: dict = {}
            for i, tk in zip(m, ts, strict=True):
                cols.setdefault(tk, []).append(st[i] >= 2)
            keys = sorted(cols)
            bad = {k for k in keys if np.mean(cols[k]) < 0.5}
            run: list[int] = []
            for k in keys + [None]:
                if k is not None and k in bad and (not run or k == run[-1] + 1):
                    run.append(k)
                    continue
                if run and len(run) * S.step >= min_len_m:
                    out.append({"surface": g, "kind": "wall", "label": meta["label"], "from_t": round(run[0] * S.step, 2),
                                "to_t": round((run[-1] + 1) * S.step, 2), "length_m": round(len(run) * S.step, 2)})
                run = [k] if (k is not None and k in bad) else []
        for g, meta in S.groups.items():
            if meta["kind"] in ("floor", "ceiling"):
                m = G == g
                frac = float((st[m] >= 2).mean()) if m.any() else 1
                if frac < 0.8:
                    out.append({"surface": g, "kind": meta["kind"], "label": meta["label"], "length_m": None,
                                "uncovered_m2": round(float((st[m] < 2).sum()) * S.step ** 2, 1),
                                "inspectable_pct": round(100 * frac, 1)})
        return sorted(out, key=lambda x: -(x.get("length_m") or x.get("uncovered_m2") or 0))


def _pct(mask: np.ndarray, tot: int) -> float:
    return round(100 * float(mask.sum()) / tot, 1)


def compute_coverage(w: Walkthrough, samples: Samples, max_range: float = 6.0, good_range: float = 3.0,
                     max_angle_deg: float = 70.0, margin_px: int = 4) -> Coverage:
    X = samples.xyz.astype(np.float64)
    n = len(X)
    first_seen = np.full(n, -1, int)
    first_good = np.full(n, -1, int)
    best = np.full(n, np.inf)
    good_by_frame, seen_by_frame = [], []
    cos_max = np.cos(np.deg2rad(max_angle_deg))
    for fi, f in enumerate(w.frames):
        if f.depth is None or f.c2w is None:
            good_by_frame.append(np.zeros(0, int))
            seen_by_frame.append(np.zeros(0, int))
            continue
        w2c = np.linalg.inv(f.c2w)
        z = X @ w2c[2, :3] + w2c[2, 3]
        front = np.where(z > 0.2)[0]
        if len(front) == 0:      # facing away from every surface sample (OpenCV 5 projectPoints returns None then)
            good_by_frame.append(np.zeros(0, int))
            seen_by_frame.append(np.zeros(0, int))
            continue
        h, wd = f.depth.shape
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        uv, _ = cv2.projectPoints(X[front], rvec, w2c[:3, 3], f.K_depth, None)
        uv = uv.reshape(-1, 2)
        ok = (uv[:, 0] >= margin_px) & (uv[:, 0] < wd - margin_px) & (uv[:, 1] >= margin_px) & (uv[:, 1] < h - margin_px)
        idx = front[ok]
        if len(idx) == 0:
            good_by_frame.append(np.zeros(0, int))
            seen_by_frame.append(np.zeros(0, int))
            continue
        ui, vi = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        d = f.depth[vi, ui]
        zz = z[idx]
        tol = np.maximum(0.12, 0.06 * zz)
        # visible unless the depth map shows something clearly in front; missing depth counts as not visible
        vis = (d > 0) & (d >= zz - tol)
        idx = idx[vis]
        ray = X[idx] - f.c2w[:3, 3]
        dist = np.linalg.norm(ray, axis=1)
        cosang = np.abs(np.sum(-ray / dist[:, None] * samples.normal[idx], axis=1))
        seen = idx[dist <= max_range]
        first_seen[seen[first_seen[seen] < 0]] = fi
        best[idx] = np.minimum(best[idx], dist)
        bad = f.quality.get("verdict", f.quality.get("platform", "")) == "BAD"
        good = idx[(dist <= good_range) & (cosang >= cos_max) & (not bad)]
        first_good[good[first_good[good] < 0]] = fi
        good_by_frame.append(good)
        seen_by_frame.append(seen)
    return Coverage(samples=samples, first_seen=first_seen, first_good=first_good, best_dist=best,
                    good_by_frame=good_by_frame, seen_by_frame=seen_by_frame, inspected=np.zeros(n, bool),
                    params={"max_range_m": max_range, "good_range_m": good_range, "max_angle_deg": max_angle_deg,
                            "sample_step_m": samples.step})


def select_keyframes(cov: Coverage, sharpness: np.ndarray, max_frames: int = 16, min_frames: int = 6,
                     min_gain_frac: float = 0.005, min_gap: int = 3, seen_weight: float = 0.25) -> list[int]:
    """Greedy set cover: each pick maximises newly covered surface times sharpness."""
    weights = {SURF_WALL: 1.0, SURF_FLOOR: 1.0, SURF_CEIL: 0.6}
    wv = np.array([weights.get(int(k), 1.0) for k in cov.samples.kind])
    total = float(wv.sum())
    got = np.zeros(len(wv))
    sharp = np.clip(sharpness / (np.percentile(sharpness, 90) + 1e-6), 0.2, 1.0)
    chosen: list[int] = []
    n = len(cov.good_by_frame)
    while len(chosen) < max_frames:
        best, bf = 0.0, -1
        for f in range(n):
            if any(abs(f - c) < min_gap for c in chosen):
                continue
            g, s = cov.good_by_frame[f], cov.seen_by_frame[f]
            gain = float((wv[g] * (1.0 - got[g])).sum() + (wv[s] * np.clip(seen_weight - got[s], 0, None)).sum())
            gain *= sharp[f]
            if gain > best:
                best, bf = gain, f
        if bf < 0 or best < min_gain_frac * total:
            break
        chosen.append(bf)
        got[cov.seen_by_frame[bf]] = np.maximum(got[cov.seen_by_frame[bf]], seen_weight)
        got[cov.good_by_frame[bf]] = 1.0
    if len(chosen) < min(min_frames, n):
        for sl in np.array_split(np.arange(n), min(min_frames, n)):
            if len(chosen) >= min_frames:
                break
            if any(sl[0] - min_gap < c <= sl[-1] + min_gap for c in chosen):
                continue
            chosen.append(int(sl[np.argmax(sharpness[sl])]))
    return sorted(chosen)


def mark_inspected(cov: Coverage, frames: list[int]) -> None:
    for f in frames:
        cov.inspected[cov.good_by_frame[f]] = True


def sample_state_for_frame(cov: Coverage, frame: int) -> dict:
    """What one frame contributes: how many samples it sees and sees inspectably, by surface kind."""
    S = cov.samples
    out = {}
    for k, name in ((SURF_WALL, "walls"), (SURF_FLOOR, "floor"), (SURF_CEIL, "ceiling")):
        out[name] = {"seen": int((S.kind[cov.seen_by_frame[frame]] == k).sum()),
                     "inspectable": int((S.kind[cov.good_by_frame[frame]] == k).sum())}
    return out
