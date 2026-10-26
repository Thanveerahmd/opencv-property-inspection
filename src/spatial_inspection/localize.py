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


def surface_hit(f: Frame, u: float, v: float, layout: dict) -> dict | None:
    """Where the ray through image pixel (u, v) first meets the room's walls, floor or ceiling.

    Returns the hit point, which surface it is on, and the hit's depth along the camera axis (comparable with the
    depth map). None when the ray leaves the room without meeting a surface the layout knows about."""
    if f.c2w is None or not layout or not layout.get("walls"):
        return None
    d_cam = np.linalg.inv(f.K) @ np.array([u, v, 1.0])
    d = f.c2w[:3, :3] @ d_cam
    o = f.c2w[:3, 3]
    ceil = float(layout.get("ceiling_height_m", 2.6))
    rooms = [r["bbox_m"] for r in layout.get("rooms", [])]

    def in_room(p: np.ndarray, m: float = 0.3) -> bool:
        return not rooms or any(x0 - m <= p[0] <= x1 + m and z0 - m <= p[2] <= z1 + m for x0, z0, x1, z1 in rooms)

    hits = []   # (distance along the ray, surface, wall, plane axis, plane offset)
    for y, name in ((0.0, "floor"), (ceil, "ceiling")):
        if abs(d[1]) > 1e-6:
            t = (y - o[1]) / d[1]
            if t > 0.05 and in_room(o + t * d):
                hits.append((t, name, None, 1, y))
    for w in layout["walls"]:
        a = 0 if w["axis"] == "x" else 2
        if abs(d[a]) < 1e-6:
            continue
        t = (w["offset"] - o[a]) / d[a]
        p = o + t * d
        along = p[2] if a == 0 else p[0]
        if t > 0.05 and w["t0"] - 0.05 <= along <= w["t1"] + 0.05 and -0.05 <= p[1] <= ceil + 0.05:
            hits.append((t, "wall", w, a, w["offset"]))
    if not hits:
        return None
    t, surface, wall, axis, offset = min(hits, key=lambda h: h[0])
    return {"xyz": o + t * d, "surface": surface, "wall": wall, "axis": axis, "offset": offset,
            "z_cam": float(t * d_cam[2])}


def protrusion(f: Frame, mask: np.ndarray, axis: int, offset: float, step: int = 4, min_m: float = 0.1) -> float | None:
    """Share of a candidate's pixels whose depth is at least `min_m` in front of the surface plane they look at.

    A stain or crack lies in the plane; a chair pushed against the wall, a door handle or a sofa sticks out of it."""
    if f.depth is None or f.c2w is None:
        return None
    vs, us = np.nonzero(mask[::step, ::step])
    if len(us) < 8:
        return None
    us, vs = us * step, vs * step
    rays = np.linalg.inv(f.K) @ np.vstack([us, vs, np.ones_like(us)]).astype(float)   # z = 1: t is camera depth
    d = f.c2w[:3, :3] @ rays
    o = f.c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_plane = (offset - o[axis]) / d[axis]
    sx, sy = f.K_depth[0, 0] / f.K[0, 0], f.K_depth[1, 1] / f.K[1, 1]
    du = np.clip(((us - f.K[0, 2]) * sx + f.K_depth[0, 2]).astype(int), 0, f.depth.shape[1] - 1)
    dv = np.clip(((vs - f.K[1, 2]) * sy + f.K_depth[1, 2]).astype(int), 0, f.depth.shape[0] - 1)
    z_obs = f.depth[dv, du].astype(float)
    ok = np.isfinite(z_obs) & (z_obs > 0) & np.isfinite(z_plane) & (z_plane > 0)
    if ok.sum() < 8:
        return None
    return float(np.mean(z_obs[ok] < z_plane[ok] - min_m))


def on_door(hit: dict, layout: dict, margin: float = 0.1) -> dict | None:
    """The door (from the layout) that a wall hit lands on, if any."""
    if hit["surface"] != "wall":
        return None
    p, wid = hit["xyz"], hit["wall"]["id"]
    along = p[2] if hit["axis"] == 0 else p[0]
    for door in layout.get("doors_and_openings", []):
        if door.get("wall") != wid:
            continue
        k = 1 if hit["axis"] == 0 else 0
        lo, hi = sorted((door["xz0"][k], door["xz1"][k]))
        if lo - margin <= along <= hi + margin and door.get("bottom_m", 0) - margin <= p[1] <= door.get("top_m", 2.1) + margin:
            return door
    return None


def observed_depth(f: Frame, bbox) -> float | None:
    """Median depth-map value inside an image box, or None where the depth map has nothing valid."""
    if f.depth is None:
        return None
    x, y, bw, bh = bbox
    sx, sy = f.K_depth[0, 0] / f.K[0, 0], f.K_depth[1, 1] / f.K[1, 1]
    h, w = f.depth.shape
    x0 = int(np.clip((x - f.K[0, 2]) * sx + f.K_depth[0, 2], 0, w - 1))
    x1 = int(np.clip((x + bw - f.K[0, 2]) * sx + f.K_depth[0, 2], x0 + 1, w))
    y0 = int(np.clip((y - f.K[1, 2]) * sy + f.K_depth[1, 2], 0, h - 1))
    y1 = int(np.clip((y + bh - f.K[1, 2]) * sy + f.K_depth[1, 2], y0 + 1, h))
    win = f.depth[y0:y1, x0:x1].astype(np.float32)
    win = win[np.isfinite(win) & (win > 0)]
    return float(np.median(win)) if win.size >= 4 else None


def place_on_surface(f: Frame, bbox, layout: dict, samples: Samples, mask: np.ndarray | None = None,
                     tolerance: float = 0.25, max_protrusion: float = 0.35) -> dict:
    """Decide whether an image candidate lies on the building's own surface, and where.

    The candidate's centre ray is cast onto the room's planes, then checked against the depth map and the layout:
    - something well in front of the plane (a chair, a sofa, a person) is an object, not the surface;
    - something well beyond it was seen through a doorway or window;
    - a hit inside a door the layout found is on the door, not on wall paint.
    None of these is evidence about the wall, floor or ceiling, so the candidate is ruled out with that reason."""
    x, y, bw, bh = bbox

    def out(reason: str | None, place: dict | None) -> dict:
        return {"on_surface": reason is None and place is not None,
                "reason": reason if reason or place else "outside the room model", "place": place}

    hit = surface_hit(f, x + bw / 2, y + bh / 2, layout)
    if hit is None:
        xyz = pixel_to_world(f, x + bw / 2, y + bh / 2)
        return out(None, snap(samples, xyz) if xyz is not None else None)
    z_obs = observed_depth(f, bbox)
    margin = max(tolerance, 0.12 * hit["z_cam"])
    gap = None if z_obs is None else round(hit["z_cam"] - z_obs, 3)
    place = snap(samples, hit["xyz"])
    if place is not None:
        place["depth_gap_m"] = gap
    if gap is not None and gap > margin:
        return out(f"on an object {gap:.2f} m in front of the {hit['surface']}", place)
    if gap is not None and gap < -margin:
        return out(f"seen {-gap:.2f} m beyond the {hit['surface']} (through an opening)", place)
    door = on_door(hit, layout)
    if door is not None:
        return out(f"on the {door.get('type', 'door')} in {door['wall']}, not on wall paint", place)
    if mask is not None:
        share = protrusion(f, mask, hit["axis"], hit["offset"])
        if place is not None:
            place["protrusion"] = None if share is None else round(share, 2)
        if share is not None and share > max_protrusion:
            return out(f"{share:.0%} of it stands out of the {hit['surface']} plane (furniture or fittings)", place)
    return out(None, place)


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
