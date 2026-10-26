"""Place an image finding in the room: back-project its pixel through the frame's depth and pose, then snap it to
the nearest wall, floor or ceiling sample so it lands on a surface the report can name."""

from __future__ import annotations

import numpy as np

from .coverage import SURF_CEIL, SURF_FLOOR, Samples
from .scene import Frame

KIND_NAME = {0: "wall", SURF_FLOOR: "floor", SURF_CEIL: "ceiling"}


def pixel_to_world(f: Frame, u: float, v: float) -> np.ndarray | None:
    """Model-frame point under image pixel (u, v), using the frame's depth map (median of a small window)."""
    if f.depth is None or f.c2w is None:
        return None
    h, w = f.depth.shape
    sx, sy = f.K_depth[0, 0] / f.K[0, 0], f.K_depth[1, 1] / f.K[1, 1]
    du = (u - f.K[0, 2]) * sx + f.K_depth[0, 2]
    dv = (v - f.K[1, 2]) * sy + f.K_depth[1, 2]
    x0, x1 = int(max(0, du - 3)), int(min(w, du + 4))
    y0, y1 = int(max(0, dv - 3)), int(min(h, dv + 4))
    win = f.depth[y0:y1, x0:x1]
    win = win[win > 0]
    if win.size == 0:
        return None
    z = float(np.median(win))
    cam = np.array([(du - f.K_depth[0, 2]) / f.K_depth[0, 0] * z, (dv - f.K_depth[1, 2]) / f.K_depth[1, 1] * z, z, 1.0])
    return (f.c2w @ cam)[:3]


def snap(samples: Samples, xyz: np.ndarray, max_dist: float = 0.5) -> dict | None:
    """Nearest surface sample to a 3D point: its position, surface kind, group and normal."""
    d = np.linalg.norm(samples.xyz - xyz, axis=1)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        return None
    return {"xyz": samples.xyz[i].round(3).tolist(), "surface": KIND_NAME.get(int(samples.kind[i]), "wall"),
            "group": samples.group[i], "normal": samples.normal[i].round(3).tolist(), "snap_m": round(float(d[i]), 3),
            "sample": i}


def merge_repeats(findings: list[dict], radius: float = 0.35) -> list[dict]:
    """The same defect seen from several keyframes: keep the best-scored one per kind and place."""
    kept: list[dict] = []
    for f in sorted(findings, key=lambda x: -x["score"]):
        if f.get("xyz") is None:
            kept.append(f)
            continue
        dup = next((k for k in kept if k.get("xyz") is not None and k["kind"] == f["kind"]
                    and np.linalg.norm(np.subtract(k["xyz"], f["xyz"])) < radius), None)
        if dup is None:
            f["views"] = [f["frame"]]
            kept.append(f)
        else:
            dup.setdefault("views", [dup["frame"]]).append(f["frame"])
    return kept
