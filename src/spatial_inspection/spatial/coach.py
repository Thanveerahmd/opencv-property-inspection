"""Coverage coach data (coach.json): what the browser needs to replay a walkthrough with a coverage overlay and to
simulate filming the rest of the property (frontend/coach.html, frontend/coach/engine.js).

The browser re-implements the visibility test of coverage.compute_coverage analytically: walls with their door
openings and furniture boxes from layout.json stand in for the per-frame depth maps. So the samples, the rules
(ranges, angle, frame margin) and the camera conventions written here must stay in step with compute_coverage.
Coordinates are the 3D engine's model frame: x right, y up, z back, floor at y = 0, metres. Camera rotations are
OpenCV camera-to-model (x right, y down, z forward)."""
from __future__ import annotations

import numpy as np

from .coverage import SURF_CEIL, SURF_FLOOR, Coverage
from .slam_data import SlamRun

VERSION = 1
# unit normals by code: samples carry a code instead of three floats
NORMALS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]


def _normal_code(n: np.ndarray) -> np.ndarray:
    table = np.array(NORMALS, float)
    return np.argmax(n @ table.T, axis=1).astype(int)


def _holes(w: dict) -> list[dict]:
    """Door and opening rectangles of a wall as (t range, y range); t is z for x-walls and x for z-walls."""
    out = []
    ti = 1 if w["axis"] == "x" else 0
    for h in (w.get("doors") or []) + (w.get("openings") or []):
        if h.get("xz0") is not None and h.get("xz1") is not None:
            a, b = float(h["xz0"][ti]), float(h["xz1"][ti])
        elif h.get("along_m"):
            a, b = (float(w["t0"]) + float(v) for v in h["along_m"])
        else:
            continue
        out.append({"t0": round(min(a, b), 3), "t1": round(max(a, b), 3),
                    "y0": round(float(h.get("bottom_m", 0.0)), 3), "y1": round(float(h.get("top_m", 2.0)), 3)})
    return out


def coach_data(run: SlamRun, cov: Coverage, keyframes: list[int], *, name: str, target_pct: float,
               video: str = "source_video.mp4", margin_px: int = 4) -> dict:
    S = cov.samples
    L = run.layout
    groups = list(S.groups)
    gidx = {g: i for i, g in enumerate(groups)}
    walls = [{"id": w["id"], "label": S.groups.get(w["id"], {}).get("label", w["id"]), "axis": w["axis"],
              "offset": round(float(w["offset"]), 3), "sign": int(w["sign"]),
              "t0": round(float(w["t0"]), 3), "t1": round(float(w["t1"]), 3), "holes": _holes(w)}
             for w in L["walls"]]
    boxes = []
    for o in L.get("objects", []):
        fp = o.get("footprint")
        if o.get("ceiling_mounted") or not fp:
            continue
        lo, hi = o.get("bbox_min"), o.get("bbox_max")
        boxes.append({"id": o.get("id", ""), "label": o.get("label", ""),
                      "center": [round(float(v), 3) for v in fp["center"]],
                      "size": [round(float(v), 3) for v in fp["size"]], "angle_deg": round(float(fp.get("angle_deg", 0)), 2),
                      "y0": round(float(lo[1]) if lo else float(o.get("bottom_m", 0.0)), 3),
                      "y1": round(float(hi[1]) if hi else float(o.get("height_m", 1.0)), 3)})
    # intrinsics at video resolution; the frame margin of compute_coverage is in depth-map pixels
    W, H = run.image_size
    K = np.median(run.K, axis=0)
    margin = margin_px * W / run.depth.shape[2]
    st = cov.state()
    stats = cov.stats()
    return {
        "version": VERSION, "name": name, "video": video,
        "params": {**cov.params, "target_pct": target_pct, "margin_px": round(float(margin), 2)},
        "ceiling_m": round(run.ceiling, 3),
        "camera": {"image_size": [int(W), int(H)], "fx": round(float(K[0, 0]), 2), "fy": round(float(K[1, 1]), 2),
                   "cx": round(float(K[0, 2]), 2), "cy": round(float(K[1, 2]), 2)},
        "walls": walls, "boxes": boxes,
        "rooms": [{"id": r["id"], "name": r["name"], "bbox_m": [round(float(v), 3) for v in r["bbox_m"]],
                   "area_m2": r.get("area_m2")} for r in L.get("rooms", [])],
        "groups": [{"id": g, **S.groups[g]} for g in groups],
        "normals": NORMALS,
        "samples": {
            "n": int(len(S.xyz)),
            "xyz_cm": np.round(S.xyz * 100).astype(int).ravel().tolist(),
            "normal": _normal_code(S.normal).tolist(),
            "kind": S.kind.astype(int).tolist(),
            "group": [gidx[g] for g in S.group],
            "first_seen": cov.first_seen.astype(int).tolist(),
            "first_good": cov.first_good.astype(int).tolist(),
            "inspected": cov.inspected.astype(int).tolist(),
        },
        "frames": {
            "t": [round(float(t), 3) for t in run.times],
            "pos": np.round(run.c2w[:, :3, 3], 3).tolist(),
            "rot": np.round(run.c2w[:, :3, :3].reshape(run.n, 9), 4).tolist(),
            "ok": [int(q != "BAD") for q in run.frame_quality],
        },
        "keyframes": [int(f) for f in keyframes],
        "measured": {"overall": stats["overall"], "by_kind": stats["by_kind"],
                     "inspectable_samples": int((st >= 2).sum()),
                     "floor_samples": int((S.kind == SURF_FLOOR).sum()),
                     "ceiling_samples": int((S.kind == SURF_CEIL).sum())},
    }
