"""Room layout from the fused point cloud, for new videos with no precomputed layout.

The room is fitted as a box: its heading from the wall normals (folded every 90 degrees), the floor at y = 0, the
ceiling from downward-facing points, and each wall from points facing into the room. Furniture fronts sit further in
and glimpses through a doorway further out, so each wall takes the outermost well-supported 10 cm band unless a band
in front of it spans nearly all of its width and height. The result uses the inspection platform's layout format.
"""

from __future__ import annotations

import numpy as np

from .fusion import Fused
from .scene import Walkthrough


def manhattan_yaw(normals: np.ndarray) -> float:
    """Heading of the room's walls (radians), from horizontal normals folded every 90 degrees."""
    m = np.abs(normals[:, 1]) < 0.3
    phi = np.arctan2(normals[m, 0], normals[m, 2])
    return float(np.arctan2(np.sin(4 * phi).sum(), np.cos(4 * phi).sum()) / 4)


def rotate_walkthrough(w: Walkthrough, yaw: float) -> np.ndarray:
    """Turn the model frame about +y by -yaw so the walls line up with x and z. Returns the 4x4 applied."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    for f in w.frames:
        if f.c2w is not None:
            f.c2w = T @ f.c2w
    return T


def wall_position(v: np.ndarray, c: np.ndarray, y: np.ndarray, outward: int) -> float | None:
    """Wall offset along one axis from points facing into the room (see the module docstring)."""
    if len(v) < 30:
        return None
    k = np.round(v / 0.1).astype(int)
    keys, counts = np.unique(k, return_counts=True)
    top = counts.max()
    bands = []
    for key, cnt in zip(keys, counts, strict=True):
        if cnt < max(20, 0.25 * top):
            continue
        m = k == key
        bands.append((key, np.percentile(c[m], 5), np.percentile(c[m], 95), np.percentile(y[m], 5), np.percentile(y[m], 95)))
    bands.sort(key=lambda b: b[0] * outward)

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    for i in range(len(bands) - 1, -1, -1):
        key, c0, c1, y0, y1 = bands[i]
        hidden = any(overlap(p[1], p[2], c0, c1) >= 0.8 * (c1 - c0) and overlap(p[3], p[4], y0, y1) >= 0.7 * (y1 - y0)
                     for p in bands[:i])
        if not hidden:
            near = np.abs(k - key) <= 1
            return float(np.median(v[near]))
    return None


def layout_from_points(fused: Fused, floor_y: float = 0.0, cameras: np.ndarray | None = None,
                       clearance: float = 0.15) -> dict:
    """A one-room box layout in the platform's format. The model frame must already be wall-aligned.

    `cameras` (N x 3 camera positions) constrains the fit: the room must contain the path it was filmed from, so each
    wall is looked for only beyond the path on its own side. Walls that still cannot be found fall back to the extent
    of the points, and the layout's `confidence` says so."""
    p, n = fused.points, fused.normals
    down = (n[:, 1] < -0.85) & (p[:, 1] > floor_y + 1.6)
    ceil = float(np.percentile(p[down, 1], 85)) if down.sum() > 50 else floor_y + 2.6
    horiz = np.abs(n[:, 1]) < 0.3
    lo, hi = np.percentile(p[:, [0, 2]], 2, axis=0), np.percentile(p[:, [0, 2]], 98, axis=0)
    sets = {
        ("x", 1): horiz & (n[:, 0] > 0.8), ("x", -1): horiz & (n[:, 0] < -0.8),
        ("z", 1): horiz & (n[:, 2] > 0.8), ("z", -1): horiz & (n[:, 2] < -0.8),
    }
    cam_lo = cam_hi = None
    if cameras is not None and len(cameras):
        cam_lo, cam_hi = cameras[:, [0, 2]].min(axis=0), cameras[:, [0, 2]].max(axis=0)
    off, fell_back = {}, []
    for (axis, sign), m in sets.items():
        a, b = (0, 2) if axis == "x" else (2, 0)
        j = 0 if axis == "x" else 1
        if cam_lo is not None:   # a wall facing +axis bounds the room from below, so it lies below the whole path
            m = m & ((p[:, a] < cam_lo[j] - clearance) if sign == 1 else (p[:, a] > cam_hi[j] + clearance))
        v = wall_position(p[m, a], p[m, b], p[m, 1], -sign)
        fallback = lo[j] if sign == 1 else hi[j]
        if cam_lo is not None:
            fallback = min(fallback, cam_lo[j] - clearance) if sign == 1 else max(fallback, cam_hi[j] + clearance)
        if v is None:
            fell_back.append(f"{axis}{'+' if sign == 1 else '-'}")
        off[(axis, sign)] = fallback if v is None else v
    x0, x1, z0, z1 = off[("x", 1)], off[("x", -1)], off[("z", 1)], off[("z", -1)]
    walls = [
        {"id": "wall_00", "axis": "x", "offset": x0, "sign": 1, "t0": z0, "t1": z1},
        {"id": "wall_01", "axis": "x", "offset": x1, "sign": -1, "t0": z0, "t1": z1},
        {"id": "wall_02", "axis": "z", "offset": z0, "sign": 1, "t0": x0, "t1": x1},
        {"id": "wall_03", "axis": "z", "offset": z1, "sign": -1, "t0": x0, "t1": x1},
    ]
    for wl in walls:
        wl["offset"] = round(float(wl["offset"]), 3)
        wl["t0"], wl["t1"] = round(float(min(wl["t0"], wl["t1"])), 3), round(float(max(wl["t0"], wl["t1"])), 3)
        wl["length_m"] = round(wl["t1"] - wl["t0"], 2)
    return {
        "units": "metres, model frame (x right, y up, z back, floor y = 0)",
        "source": "opencv-points",
        "confidence": "high" if not fell_back else "medium" if len(fell_back) == 1 else "low",
        "walls_not_found": fell_back,
        "ceiling_height_m": round(ceil - floor_y, 3),
        "walls": walls,
        "rooms": [{"id": "room_0", "name": "Room", "bbox_m": [round(float(min(x0, x1)), 3), round(float(min(z0, z1)), 3),
                                                              round(float(max(x0, x1)), 3), round(float(max(z0, z1)), 3)]}],
        "objects": [],
    }
