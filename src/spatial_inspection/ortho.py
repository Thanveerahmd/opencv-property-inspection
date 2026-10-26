"""Flat, metric views of the room: every wall, the floor and the ceiling unrolled into an image built from the
walkthrough frames, and a floor plan drawn with OpenCV.

A wall is a plane, so the mapping from wall pixels to a frame's pixels is a homography, H = K [r1 r2 t] written in wall
coordinates. Every frame that sees the wall well is warped onto it with warpPerspective, pixels hidden behind
furniture (the frame's depth is much nearer than the wall) are dropped, and the views are combined: each wall pixel
takes the sharpest, closest, most head-on view, and a multi-band blender hides the seams between views.

Run on a cached platform output:  python -m spatial_inspection.ortho <slam_dir> <video> <out_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spatial_inspection.scene import Frame, Walkthrough, load_platform_run

UNSEEN_BGR = (128, 128, 128)
UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Opening:
    kind: str       # door, doorway, window/pass-through, ...
    a0: float       # metres along the wall from its left end (seen from inside the room)
    a1: float
    bottom: float   # metres above the floor
    top: float


@dataclass
class Wall:
    """A rectangle on a room surface, seen from inside the room. Walls, the floor and the ceiling all use it.

    Image convention: columns run along u (left to right), rows run against v (top to bottom), so pixel (0, 0) is
    the model point origin + height_m * v. For the floor and ceiling u is +x and v is -z, the floor plan's orientation
    (the ceiling is a reflected ceiling plan: it lines up with the floor plan, as if seen in a mirror on the floor).
    """

    id: str
    origin: np.ndarray      # bottom-left corner, model frame
    u: np.ndarray           # unit, along the wall, left to right
    v: np.ndarray           # unit, up the wall (+y for walls)
    normal: np.ndarray      # unit, into the room
    length_m: float
    height_m: float
    openings: list[Opening] = field(default_factory=list)

    def size_px(self, px_per_m: float) -> tuple[int, int]:
        """(width, height) of the rectified image."""
        return max(1, round(self.length_m * px_per_m)), max(1, round(self.height_m * px_per_m))

    def pixel_to_model(self, px_per_m: float) -> np.ndarray:
        """3x3 matrix taking a wall pixel (col, row, 1) to its model-frame point."""
        top_left = self.origin + self.height_m * self.v
        return np.column_stack([self.u / px_per_m, -self.v / px_per_m, top_left])

    def model_to_pixel(self, xyz, px_per_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Columns, rows and signed distance from the plane (positive inside the room) for model points."""
        d = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) - self.origin
        return d @ self.u * px_per_m, (self.height_m - d @ self.v) * px_per_m, d @ self.normal

    def corners(self) -> np.ndarray:
        """The four corners in the model frame: bottom-left, bottom-right, top-right, top-left."""
        o, a, b = self.origin, self.length_m * self.u, self.height_m * self.v
        return np.array([o, o + a, o + a + b, o + b])


@dataclass
class RectifiedWall:
    image: np.ndarray       # BGR, unseen pixels neutral grey
    weight: np.ndarray      # float32, summed view weight per pixel (0 where no frame saw it)
    coverage: np.ndarray    # bool, True where at least one frame saw the surface
    px_per_m: float
    wall: Wall
    source: np.ndarray      # int32, index of the frame each pixel was taken from, -1 where unseen
    frames_used: list[int] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        return 100.0 * float(np.count_nonzero(self.coverage)) / self.coverage.size


def wall_planes(layout: dict) -> list[Wall]:
    """Walls of the platform's layout.json as planes, each with its doors and openings (none for no layout)."""
    if not layout:
        return []
    height = float(layout.get("ceiling_height_m", 2.5))
    walls = []
    for w in layout.get("walls", []):
        axis = 0 if w["axis"] == "x" else 2     # the wall is the plane model[axis] = offset
        along = 2 if axis == 0 else 0           # t0 / t1 run along the other horizontal axis
        normal = np.zeros(3)
        normal[axis] = 1.0 if w.get("sign", 1) >= 0 else -1.0
        u = np.cross(UP, normal)                # left to right for someone inside the room facing the wall
        t0, t1 = sorted((float(w["t0"]), float(w["t1"])))
        origin = np.zeros(3)
        origin[axis] = float(w["offset"])
        origin[along] = t0 if u[along] > 0 else t1
        wall = Wall(id=str(w.get("id", f"wall_{len(walls):02d}")), origin=origin, u=u, v=UP.copy(), normal=normal,
                    length_m=t1 - t0, height_m=height)
        for o in (w.get("doors") or []) + (w.get("openings") or []):
            wall.openings.append(_opening(o, wall))
        walls.append(wall)
    return walls


def _opening(o: dict, wall: Wall) -> Opening:
    if "xz0" in o and "xz1" in o:
        ends = [(np.array([p[0], 0.0, p[1]]) - wall.origin) @ wall.u for p in (o["xz0"], o["xz1"])]
    else:
        # the platform measures along_m from the wall's right-hand end (seen from inside)
        ends = [wall.length_m - float(a) for a in o["along_m"]]
    a0, a1 = (float(np.clip(a, 0.0, wall.length_m)) for a in sorted(ends))
    return Opening(kind=str(o.get("type", "opening")), a0=a0, a1=a1, bottom=float(o.get("bottom_m", 0.0)),
                   top=float(o.get("top_m", wall.height_m)))


def _plan_bounds(layout: dict | None, margin_m: float = 0.1) -> tuple[float, float, float, float] | None:
    """(xmin, zmin, xmax, zmax) around every wall (or room box), None when the layout has neither."""
    pts = []
    for w in wall_planes(layout):
        for p in (w.origin, w.origin + w.length_m * w.u):
            pts.append((p[0], p[2]))
    if not pts:
        for r in (layout or {}).get("rooms") or []:
            x0, z0, x1, z1 = r["bbox_m"]
            pts += [(x0, z0), (x1, z1)]
    if not pts:
        return None
    pts = np.array(pts)
    (x0, z0), (x1, z1) = pts.min(0) - margin_m, pts.max(0) + margin_m
    return float(x0), float(z0), float(x1), float(z1)


def floor_plane(layout: dict) -> Wall:
    """The floor under every wall, oriented like the floor plan (x right, -z up)."""
    x0, z0, x1, z1 = _plan_bounds(layout)
    return Wall("floor", origin=np.array([x0, 0.0, z1]), u=np.array([1.0, 0, 0]), v=np.array([0, 0, -1.0]),
                normal=UP.copy(), length_m=x1 - x0, height_m=z1 - z0)


def ceiling_plane(layout: dict) -> Wall:
    """The ceiling as a reflected ceiling plan: it lines up with the floor plan."""
    x0, z0, x1, z1 = _plan_bounds(layout)
    h = float(layout.get("ceiling_height_m", 2.5))
    return Wall("ceiling", origin=np.array([x0, h, z1]), u=np.array([1.0, 0, 0]), v=np.array([0, 0, -1.0]),
                normal=-UP, length_m=x1 - x0, height_m=z1 - z0)


def frame_sharpness(frame: Frame) -> float:
    """Variance of the Laplacian of the frame, cached in frame.quality; low means motion blur or defocus."""
    if "sharpness" not in frame.quality:
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        frame.quality["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return frame.quality["sharpness"]


@dataclass
class _View:
    frame: Frame
    x0: int                 # the view covers wall pixels [y0:y0 + h, x0:x0 + w]
    y0: int
    image: np.ndarray
    weight: np.ndarray


def _warp_view(frame: Frame, wall: Wall, A: np.ndarray, size: tuple[int, int], grid: tuple[np.ndarray, np.ndarray],
               max_dist_m: float, max_angle_deg: float, sharp_gain: float) -> _View | None:
    """One frame warped onto the part of the wall it sees, with a per-pixel weight (0 where it does not see it)."""
    w2c = np.linalg.inv(frame.c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    # camera on the wrong side of the wall (outside the room) sees its back
    if (frame.position - wall.origin) @ wall.normal < 0.1:
        return None
    M = R @ A
    M[:, 2] += t                    # wall pixel -> camera coordinates
    H_img = frame.K @ M             # wall pixel -> image pixel, the plane homography K [r1 r2 t]
    w_img, h_img = frame.image.shape[1], frame.image.shape[0]
    cos_min = math.cos(math.radians(max_angle_deg))
    # plain floats keep the per-pixel maths in float32
    m, hm, n_cam = M.tolist(), H_img.tolist(), (R @ wall.normal).tolist()

    def visible(cc, rr):
        X = cc * m[0][0] + rr * m[0][1] + m[0][2]
        Y = cc * m[1][0] + rr * m[1][1] + m[1][2]
        Z = cc * m[2][0] + rr * m[2][1] + m[2][2]
        dist = np.sqrt(X * X + Y * Y + Z * Z)
        cos = -(X * n_cam[0] + Y * n_cam[1] + Z * n_cam[2]) / np.maximum(dist, 1e-6)
        # K's last row is (0, 0, 1), so the homography's third coordinate is the camera depth Z
        zs = np.where(Z > 0.05, Z, 1.0)
        x = (hm[0][0] * cc + hm[0][1] * rr + hm[0][2]) / zs
        y = (hm[1][0] * cc + hm[1][1] * rr + hm[1][2]) / zs
        edge = np.minimum(np.minimum(x, w_img - 1 - x), np.minimum(y, h_img - 1 - y))
        ok = (Z > 0.1) & (edge >= 0) & (dist < max_dist_m) & (cos > cos_min)
        return ok, Z, dist, cos, edge

    # a coarse grid finds the part of the wall in view before anything is warped
    W, H = size
    nc, nr = min(W, 40), min(H, 24)
    gx, gy = np.linspace(0, W - 1, nc), np.linspace(0, H - 1, nr)
    ok = visible(*np.meshgrid(gx, gy))[0]
    if not ok.any():
        return None
    rows, cols = np.nonzero(ok)
    x0 = int(gx[max(cols.min() - 1, 0)])
    x1 = int(math.ceil(gx[min(cols.max() + 1, nc - 1)])) + 1
    y0 = int(gy[max(rows.min() - 1, 0)])
    y1 = int(math.ceil(gy[min(rows.max() + 1, nr - 1)])) + 1
    roi = (x1 - x0, y1 - y0)

    ok, Z, dist, cos, edge = visible(grid[0][y0:y1, x0:x1], grid[1][y0:y1, x0:x1])
    if not ok.any():
        return None
    # source resolution on the wall (image pixels per metre), eased off near the image border and the limits
    weight = float(frame.K[0, 0]) * cos / np.maximum(dist, 0.3)
    weight *= np.clip(edge / 24.0, 0.0, 1.0)
    weight *= np.clip((max_dist_m - dist) / 0.5, 0.0, 1.0)
    weight *= np.clip((cos - cos_min) / 0.15, 0.0, 1.0)
    weight *= sharp_gain

    shift = np.array([[1.0, 0, x0], [0, 1, y0], [0, 0, 1]])     # ROI pixel -> wall pixel
    if frame.depth is not None and frame.K_depth is not None:
        H_depth = frame.K_depth @ M @ shift
        d = cv2.warpPerspective(frame.depth, H_depth, roi, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # something sits in front of the wall where the depth is well short of the plane; depth edges are soft,
        # so the hidden area is grown a little
        hidden = ((d > 0) & (d < Z - (0.15 + 0.1 * Z))).astype(np.uint8)
        hidden = cv2.dilate(hidden, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        ok &= hidden == 0
        weight = np.where(d > 0, weight, 0.35 * weight)    # no depth: cannot rule out furniture
    weight = np.where(ok, weight, 0.0).astype(np.float32)
    if weight.max() <= 0:
        return None
    # replicate the border so the blender's coarse bands do not pull black in from outside the frame
    image = cv2.warpPerspective(frame.image, H_img @ shift, roi, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    return _View(frame, x0, y0, image, weight)


def rectify_wall(walkthrough: Walkthrough, wall: Wall, px_per_m: float = 100.0, frames: list[Frame] | None = None,
                 max_dist_m: float = 4.0, max_angle_deg: float = 70.0, blend: str = "multiband") -> RectifiedWall:
    """Front-on, metric image of a wall (or the floor / ceiling) from every frame that sees it well.

    blend is "multiband" (each pixel from its best view, seams hidden by a multi-band blender) or "average"
    (weighted average of all views, softer but blurrier where poses disagree).
    """
    if frames is None:
        frames = walkthrough.frames
    if any(f.image is None for f in frames):
        walkthrough.load_images()
    frames = [f for f in frames if f.c2w is not None and f.image is not None]
    size = wall.size_px(px_per_m)
    W, H = size
    A = wall.pixel_to_model(px_per_m)
    grid = tuple(g.astype(np.float32) for g in np.meshgrid(np.arange(W), np.arange(H)))
    sharp = np.array([frame_sharpness(f) for f in frames]) if frames else np.ones(0)
    ref = float(np.median(sharp)) if len(sharp) else 1.0
    gains = np.clip(sharp / max(ref, 1e-6), 0.3, 1.5)

    for reach in (max_dist_m, 1.6 * max_dist_m):
        views = []
        for f, g in zip(frames, gains, strict=True):
            v = _warp_view(f, wall, A, size, grid, reach, max_angle_deg, float(g))
            if v is not None:
                views.append(v)
        # a wall the camera never came near is still better shown from afar than not at all
        if views:
            break
    return _combine(views, wall, size, px_per_m, blend)


def rectify_floor(walkthrough: Walkthrough, px_per_m: float = 100.0, **kw) -> RectifiedWall:
    """Top-down, metric image of the floor, oriented like the floor plan."""
    return rectify_wall(walkthrough, floor_plane(walkthrough.layout), px_per_m, **kw)


def rectify_ceiling(walkthrough: Walkthrough, px_per_m: float = 100.0, **kw) -> RectifiedWall:
    """Reflected ceiling plan: the ceiling seen from below, lined up with the floor plan."""
    return rectify_wall(walkthrough, ceiling_plane(walkthrough.layout), px_per_m, **kw)


def _combine(views: list[_View], wall: Wall, size: tuple[int, int], px_per_m: float, blend: str) -> RectifiedWall:
    W, H = size
    total = np.zeros((H, W), np.float32)
    best_w = np.zeros((H, W), np.float32)
    best = np.full((H, W), -1, np.int32)
    for i, v in enumerate(views):
        h, w = v.weight.shape
        sl = np.s_[v.y0:v.y0 + h, v.x0:v.x0 + w]
        total[sl] += v.weight
        better = v.weight > best_w[sl]
        best_w[sl][better] = v.weight[better]
        best[sl][better] = i
    seen = total > 0
    image = np.full((H, W, 3), UNSEEN_BGR, np.uint8)
    if views and blend == "average":
        acc = np.zeros((H, W, 3), np.float32)
        for v in views:
            h, w = v.weight.shape
            acc[v.y0:v.y0 + h, v.x0:v.x0 + w] += v.image.astype(np.float32) * v.weight[..., None]
        image[seen] = np.clip(acc[seen] / total[seen, None] + 0.5, 0, 255).astype(np.uint8)
    elif views:
        bands = int(np.clip(np.log2(max(min(W, H), 2) / 8), 1, 5))
        blender = cv2.detail.MultiBandBlender(0, bands)
        blender.prepare((0, 0, W, H))
        for i in np.unique(best[seen]):
            v = views[int(i)]
            h, w = v.weight.shape
            mask = (best[v.y0:v.y0 + h, v.x0:v.x0 + w] == i).astype(np.uint8) * 255
            blender.feed(v.image.astype(np.int16), mask, (v.x0, v.y0))
        out, _ = blender.blend(None, None)
        image[seen] = np.clip(out[:H, :W][seen], 0, 255).astype(np.uint8)
    # best is -1 where unseen, which picks the trailing -1
    source = np.array([v.frame.index for v in views] + [-1], np.int32)[best]
    return RectifiedWall(image=image, weight=total, coverage=seen, px_per_m=px_per_m,
                         wall=wall, source=source, frames_used=sorted({v.frame.index for v in views}))


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, color=(255, 255, 255),
           bg=(40, 40, 40), thickness: int = 1) -> None:
    """Text on a filled box so it reads on any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base + 1), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _tag(f: dict, k: int) -> str:
    """Short marker text for a finding: the number in its id ("F007" -> "7"), else its position in the list."""
    fid = str(f.get("id", k))
    m = re.search(r"(\d+)$", fid)
    return str(int(m.group(1))) if m else fid[:4]


def _describe(f: dict) -> str:
    text = str(f.get("label") or f.get("kind") or "")
    status = f.get("status")
    return f"{text} ({status})" if status else text


def _placed(findings: list[dict] | None) -> list[tuple[int, dict]]:
    """(list position from 1, finding) for findings that have a model position."""
    return [(k, f) for k, f in enumerate(findings or [], 1) if f.get("xyz") is not None]


def draw_findings(rect: RectifiedWall, findings: list[dict], max_dist_m: float = 0.4) -> np.ndarray:
    """The rectified image with a numbered marker for every finding that lies on this surface.

    A finding is a dict with "xyz" (model frame, or None when not placed) and "label" or "kind"; optional "id"
    (its digits are the number shown), "status" and "radius_m".
    """
    out = rect.image.copy()
    h, w = out.shape[:2]
    for k, f in _placed(findings):
        cols, rows, dist = rect.wall.model_to_pixel(f["xyz"], rect.px_per_m)
        c, r = float(cols[0]), float(rows[0])
        if abs(dist[0]) > max_dist_m or not (0 <= c < w and 0 <= r < h):
            continue
        num = _tag(f, k)
        radius = max(8, round(float(f.get("radius_m", 0.12)) * rect.px_per_m))
        centre = (round(c), round(r))
        cv2.circle(out, centre, radius, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(out, centre, radius, (40, 40, 220), 2, cv2.LINE_AA)
        text = f"{num} {_describe(f)}".strip()
        _label(out, text, (centre[0] + radius + 4, centre[1] + 5), 0.5, bg=(40, 40, 200))
    return out


def draw_openings(rect: RectifiedWall) -> np.ndarray:
    """The rectified image with the layout's doors and openings outlined (for review)."""
    out = rect.image.copy()
    s, Hm = rect.px_per_m, rect.wall.height_m
    for o in rect.wall.openings:
        p0 = (round(o.a0 * s), round((Hm - o.top) * s))
        p1 = (round(o.a1 * s), round((Hm - o.bottom) * s))
        cv2.rectangle(out, p0, p1, (0, 140, 255), 2, cv2.LINE_AA)
        _label(out, o.kind, (p0[0] + 4, p0[1] + 18), 0.45, bg=(0, 110, 210))
    return out


class _Plan:
    """Model (x, z) to floor-plan pixels: x to the right, -z up."""

    def __init__(self, bounds: tuple[float, float, float, float], px_per_m: float, pad: tuple[int, int, int, int]):
        self.x0, self.z0, x1, z1 = bounds
        self.s = px_per_m
        self.left, self.top, right, bottom = pad
        self.size = (round((x1 - self.x0) * px_per_m) + self.left + right,
                     round((z1 - self.z0) * px_per_m) + self.top + bottom)

    def px(self, xz) -> np.ndarray:
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([(xz[:, 0] - self.x0) * self.s + self.left, (xz[:, 1] - self.z0) * self.s + self.top])

    def pt(self, xz) -> tuple[int, int]:
        p = self.px(xz)[0]
        return round(p[0]), round(p[1])


def _segments_without(length: float, openings: list[Opening]) -> list[tuple[float, float]]:
    """Parts of [0, length] not covered by the openings."""
    segs, a = [], 0.0
    for o in sorted(openings, key=lambda o: o.a0):
        if o.a0 > a:
            segs.append((a, o.a0))
        a = max(a, o.a1)
    if a < length:
        segs.append((a, length))
    return [(s0, s1) for s0, s1 in segs if s1 - s0 > 1e-3]


def _xz(p: np.ndarray) -> tuple[float, float]:
    return float(p[0]), float(p[2])


def _passages(layout: dict) -> list[Segment]:
    """Walked passages (gaps in a wall line the camera went through) as (x, z) segments."""
    segs = []
    for p in layout.get("walked_passages", []):
        a, b = np.zeros(3), np.zeros(3)
        axis, along = (0, 2) if p["wall_axis"] == "x" else (2, 0)
        a[axis] = b[axis] = p["offset"]
        a[along], b[along] = p["t0"], p["t1"]
        segs.append((_xz(a), _xz(b)))
    return segs


Segment = tuple[tuple[float, float], tuple[float, float]]


def _missing_corners(walls: list[Wall], max_ext_m: float = 2.5) -> list[Segment]:
    """Wall extensions that close corners the layout left open.

    A wall end that touches no other wall is extended until it meets the line of another free end's wall, when the
    two walls face into the same room (each one's normal points towards the other); collinear free ends facing the
    same way are bridged directly.
    """
    ends = []                       # (point, outward direction, wall)
    for wall in walls:
        p0, p1 = np.array(_xz(wall.origin)), np.array(_xz(wall.origin + wall.length_m * wall.u))
        d = np.array(_xz(wall.u))
        ends += [(p0, -d, wall), (p1, d, wall)]

    def dist_to_wall(p, wall):
        a, b = np.array(_xz(wall.origin)), np.array(_xz(wall.origin + wall.length_m * wall.u))
        t = np.clip((p - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1)
        return np.linalg.norm(a + t * (b - a) - p)

    free = [e for e in ends if all(dist_to_wall(e[0], w) > 0.2 for w in walls if w is not e[2])]
    cands = []
    for i, (pi, di, wi) in enumerate(free):
        ni = np.array(_xz(wi.normal))
        for j in range(i + 1, len(free)):
            pj, dj, wj = free[j]
            if wj is wi:
                continue
            nj = np.array(_xz(wj.normal))
            cross = di[0] * dj[1] - di[1] * dj[0]
            if abs(cross) > 0.5:
                # perpendicular: solve pi + s di = pj + r dj
                s, r = np.linalg.solve(np.column_stack([di, -dj]), pj - pi)
                corner = pi + s * di
                faces = (pi - di - corner) @ nj > 0 and (pj - dj - corner) @ ni > 0
                if -0.2 <= s <= max_ext_m and -0.2 <= r <= max_ext_m and faces:
                    cands.append((s + r, i, j, [(pi, corner), (pj, corner)]))
            elif di @ dj < -0.9 and ni @ nj > 0.9 and abs((pj - pi) @ ni) < 0.15:
                gap = (pj - pi) @ di
                if 0 < gap <= max_ext_m:
                    cands.append((gap, i, j, [(pi, pj)]))
    used, segs = set(), []
    for _, i, j, s in sorted(cands, key=lambda c: c[0]):
        if i in used or j in used:
            continue
        used |= {i, j}
        segs += [(tuple(a), tuple(b)) for a, b in s]
    return segs


def _enclosed(lines: np.ndarray, gap_px: int, min_area_px: float) -> np.ndarray:
    """Regions of at least min_area_px that a flood fill from the image corner cannot reach, after closing gaps
    up to gap_px in the line drawing."""
    k = max(3, gap_px | 1)
    fill = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    cv2.floodFill(fill, np.zeros((fill.shape[0] + 2, fill.shape[1] + 2), np.uint8), (0, 0), 128)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((fill == 0).astype(np.uint8), connectivity=4)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area_px)
    return np.isin(labels, big[big > 0]).astype(np.uint8) * 255


def _room_mask(plan: _Plan, walls: list[Wall], layout: dict,
               inside_xz: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """Floor area enclosed by the walls: everything a flood fill from outside cannot reach.

    Gaps where wall ends do not quite meet are closed with growing structuring elements until the camera path lies
    inside. When the walls never close (part of the room unmapped), the layout's room boxes stand in for the
    missing walls. Returns the mask and whether the walls alone enclosed it.
    """
    w, h = plan.size
    lines = np.zeros((h, w), np.uint8)
    thick = max(2, round(0.08 * plan.s))
    segs = [(_xz(wall.origin), _xz(wall.origin + wall.length_m * wall.u)) for wall in walls]
    segs += _passages(layout) + _missing_corners(walls)
    for a, b in segs:
        cv2.line(lines, plan.pt(a), plan.pt(b), 255, thick)
    path = None
    if inside_xz is not None and len(inside_xz):
        path = np.round(plan.px(inside_xz)).astype(int)
        path = path[(path[:, 0] >= 0) & (path[:, 0] < w) & (path[:, 1] >= 0) & (path[:, 1] < h)]

    def inside_share(mask):
        if path is None or not len(path):
            return 1.0 if mask.any() else 0.0
        return np.count_nonzero(mask[path[:, 1], path[:, 0]]) / len(path)

    min_area = 1.0 * plan.s * plan.s
    best = np.zeros((h, w), np.uint8)
    for gap_m in (0.15, 0.4, 0.8, 1.2):
        keep = _enclosed(lines, round(gap_m * plan.s), min_area)
        if keep.any():
            best = keep
            if inside_share(keep) > 0.5:
                break
    rooms = layout.get("rooms") or []
    walled = inside_share(best) > 0.5
    if not walled and rooms:
        for r in rooms:
            x0, z0, x1, z1 = r["bbox_m"]
            cv2.rectangle(lines, plan.pt((x0, z0)), plan.pt((x1, z1)), 255, thick)
        boxed = _enclosed(lines, round(0.15 * plan.s), min_area)
        if inside_share(boxed) > inside_share(best):
            best, walled = boxed, False
    # the walls themselves are drawn on top; grow the floor under them so it meets their inner faces
    return cv2.dilate(best, cv2.getStructuringElement(cv2.MORPH_RECT, (thick + 2, thick + 2))), walled


def _dashed(img, p0, p1, color, thickness=2, dash=8):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    n = max(1, int(np.linalg.norm(p1 - p0) // dash))
    for i in range(0, n, 2):
        a, b = p0 + (p1 - p0) * i / n, p0 + (p1 - p0) * min(i + 1, n) / n
        cv2.line(img, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, thickness, cv2.LINE_AA)


def _positions(trajectory) -> np.ndarray | None:
    if trajectory is None:
        return None
    if isinstance(trajectory, Walkthrough):
        trajectory = [f.c2w for f in trajectory.frames if f.c2w is not None]
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, :3, 3]
    return arr.reshape(-1, 3) if arr.size else None


def floor_plan(layout: dict, px_per_m: float = 80, findings: list[dict] | None = None, trajectory=None,
               title: str | None = None, objects: bool = True) -> np.ndarray:
    """Floor plan of the layout as a BGR image: walls, doors, the enclosed floor, the camera path and findings.

    trajectory is camera positions (N, 3), camera-to-model matrices (N, 4, 4) or a Walkthrough; findings are dicts
    with "xyz" (model frame; None when not placed, then left off) and "label" or "kind".
    """
    layout = layout or {}
    walls = wall_planes(layout)
    traj = _positions(trajectory)
    placed = _placed(findings)
    b = _plan_bounds(layout, 0.0)
    pts = [b[:2], b[2:]] if b else []
    if traj is not None:
        pts += list(traj[:, [0, 2]])
    pts += [(f["xyz"][0], f["xyz"][2]) for _, f in placed]
    if not pts:
        raise ValueError("nothing to draw: no walls, rooms, camera path or placed findings")
    pts = np.array(pts, dtype=np.float64)
    m = 0.6
    bounds = (*(pts.min(0) - m), *(pts.max(0) + m))
    legend = placed[:14]
    legend_h = 24 * (len(legend) + (len(placed) > len(legend))) + (16 if legend else 0)
    plan = _Plan(bounds, px_per_m, (20, 56, 20, 64 + legend_h))
    w, h = plan.size
    w = max(w, 420)
    plan.size = (w, h)
    img = np.full((h, w, 3), (250, 249, 247), np.uint8)

    # 1 m grid
    gx = np.arange(math.ceil(bounds[0]), bounds[2])
    gz = np.arange(math.ceil(bounds[1]), bounds[3])
    for x in gx:
        cv2.line(img, plan.pt((x, bounds[1])), plan.pt((x, bounds[3])), (238, 236, 233), 1)
    for z in gz:
        cv2.line(img, plan.pt((bounds[0], z)), plan.pt((bounds[2], z)), (238, 236, 233), 1)

    # floor enclosed by the walls, filled lightly
    floor, walled = _room_mask(plan, walls, layout, traj[:, [0, 2]] if traj is not None else None)
    contours, _ = cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [cv2.approxPolyDP(c, 0.03 * px_per_m, True) for c in contours]
    tint = img.copy()
    cv2.fillPoly(tint, polys, (214, 228, 240))
    img = cv2.addWeighted(tint, 0.6, img, 0.4, 0)
    cv2.polylines(img, polys, True, (190, 205, 220), 1, cv2.LINE_AA)

    if objects:
        for o in layout.get("objects", []):
            fp = o.get("footprint")
            if not fp or o.get("ceiling_mounted"):
                continue
            box = cv2.boxPoints(((0, 0), tuple(fp["size"]), float(fp.get("angle_deg", 0.0))))
            box = plan.px(box + np.asarray(fp["center"]))
            cv2.fillPoly(img, [np.round(box).astype(np.int32)], (222, 222, 222), cv2.LINE_AA)
            cv2.polylines(img, [np.round(box).astype(np.int32)], True, (170, 170, 170), 1, cv2.LINE_AA)

    # area of each region the walls enclose (a room box standing in for missing walls is no measurement)
    for c in contours if walled else []:
        area = cv2.contourArea(c) / px_per_m**2
        if area < 1.0:
            continue
        mo = cv2.moments(c)
        cx, cy = int(mo["m10"] / mo["m00"]), int(mo["m01"] / mo["m00"])
        text = f"{area:.1f} m2"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, text, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 110, 100), 1, cv2.LINE_AA)

    # walls: drawn behind their inner face, broken at doors and openings
    thick = max(3, round(0.12 * px_per_m))
    for wall in walls:
        back = -wall.normal * 0.06
        for a0, a1 in _segments_without(wall.length_m, wall.openings):
            p0 = plan.pt(_xz(wall.origin + back + a0 * wall.u))
            p1 = plan.pt(_xz(wall.origin + back + a1 * wall.u))
            cv2.line(img, p0, p1, (45, 40, 40), thick, cv2.LINE_8)
        for o in wall.openings:
            q0 = wall.origin + o.a0 * wall.u
            q1 = wall.origin + o.a1 * wall.u
            if o.kind == "door":
                _door(img, plan, q0, q1, wall.normal)
            elif "window" in o.kind:
                for off in (-0.02, -0.1):
                    p0, p1 = plan.pt(_xz(q0 + off * wall.normal)), plan.pt(_xz(q1 + off * wall.normal))
                    cv2.line(img, p0, p1, (200, 140, 40), 2, cv2.LINE_AA)
            else:
                _dashed(img, plan.pt(_xz(q0 + back)), plan.pt(_xz(q1 + back)), (60, 160, 60), 2)
        # label inside the room, set off the wall so it never sits on the line
        text = f"{wall.id} {wall.length_m:.2f} m"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        c = plan.pt(_xz(wall.origin + 0.5 * wall.length_m * wall.u + 0.12 * wall.normal))
        nx, nz = wall.normal[0], wall.normal[2]
        x = c[0] if nx > 0.5 else c[0] - tw if nx < -0.5 else c[0] - tw // 2
        y = c[1] + th if nz > 0.5 else c[1] if nz < -0.5 else c[1] + th // 2
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 80, 80), 1, cv2.LINE_AA)
    for a, b in _passages(layout):
        _dashed(img, plan.pt(a), plan.pt(b), (60, 160, 60), 2)

    # camera path
    if traj is not None and len(traj) > 1:
        path = np.round(plan.px(traj[:, [0, 2]])).astype(np.int32)
        cv2.polylines(img, [path], False, (200, 120, 30), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in path[0]), 6, (60, 170, 60), -1, cv2.LINE_AA)
        _label(img, "start", (int(path[0][0]) + 9, int(path[0][1]) + 5), 0.4, bg=(60, 150, 60))
        tail = path[max(0, len(path) - 6)]
        if np.linalg.norm(path[-1] - tail) > 2:
            cv2.arrowedLine(img, tuple(int(v) for v in tail), tuple(int(v) for v in path[-1]), (200, 120, 30), 2,
                            cv2.LINE_AA, tipLength=0.5)

    # findings
    for k, f in placed:
        c = plan.pt((f["xyz"][0], f["xyz"][2]))
        num = _tag(f, k)
        cv2.circle(img, c, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, c, 11, (40, 40, 210), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, num, (c[0] - tw // 2, c[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 210), 1,
                    cv2.LINE_AA)
    y = h - legend_h - 8
    for k, f in legend:
        y += 24
        text = f"{_tag(f, k)}  {f.get('id', '')}  {_describe(f)}"
        cv2.circle(img, (30, y - 5), 9, (40, 40, 210), 2, cv2.LINE_AA)
        cv2.putText(img, text, (46, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
    if len(placed) > len(legend):
        cv2.putText(img, f"... and {len(placed) - len(legend)} more", (46, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (50, 50, 50), 1, cv2.LINE_AA)

    # title, scale bar, up arrow
    if title:
        cv2.putText(img, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    x0, yb = 24, h - legend_h - 30
    for i in range(4):
        a, b = x0 + round(i * px_per_m / 4), x0 + round((i + 1) * px_per_m / 4)
        cv2.rectangle(img, (a, yb), (b, yb + 8), (40, 40, 40), -1 if i % 2 == 0 else 1)
    cv2.putText(img, "0", (x0 - 4, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "1 m", (x0 + round(px_per_m) - 10, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1,
                cv2.LINE_AA)
    ax, ay = w - 40, 52
    cv2.arrowedLine(img, (ax, ay), (ax, ay - 34), (40, 40, 40), 2, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, "up = -z", (ax - 70, ay + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def _door(img: np.ndarray, plan: _Plan, hinge: np.ndarray, end: np.ndarray, normal: np.ndarray) -> None:
    """Door leaf opened into the room and its swing arc, hinged at the opening's first end."""
    width = float(np.linalg.norm(end - hinge))
    leaf = hinge + width * normal
    c, e, lf = plan.pt(_xz(hinge)), plan.pt(_xz(end)), plan.pt(_xz(leaf))
    color = (40, 110, 210)
    cv2.line(img, c, lf, color, 2, cv2.LINE_AA)
    a_end = math.degrees(math.atan2(e[1] - c[1], e[0] - c[0]))
    a_leaf = math.degrees(math.atan2(lf[1] - c[1], lf[0] - c[0]))
    start = a_end if (a_leaf - a_end) % 360 < 180 else a_leaf
    r = round(width * plan.s)
    cv2.ellipse(img, c, (r, r), 0, start, start + 90, color, 1, cv2.LINE_AA)


def contact_sheet(tiles: list[tuple[str, np.ndarray]], width: int = 1800, tile_h: int = 260) -> np.ndarray:
    """Captioned thumbnails packed into rows; a caption may hold several lines."""
    gap, line_h = 14, 20
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    placed, x, y, row_h = [], gap, gap, 0
    for caption, im in tiles:
        lines = caption.split("\n")
        s = min(tile_h / im.shape[0], (width - 2 * gap) / im.shape[1])
        thumb = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
        text_w = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines)
        slot_w = max(thumb.shape[1], text_w)
        slot_h = len(lines) * line_h + 6 + thumb.shape[0]
        if x > gap and x + slot_w + gap > width:
            x, y, row_h = gap, y + row_h + gap, 0
        placed.append((lines, thumb, x, y))
        x += slot_w + gap
        row_h = max(row_h, slot_h)
    sheet = np.full((y + row_h + gap, width, 3), 245, np.uint8)
    for lines, thumb, x, y in placed:
        for k, t in enumerate(lines):
            cv2.putText(sheet, t, (x, y + 14 + k * line_h), font, scale, (40, 40, 40), 1, cv2.LINE_AA)
        top = y + len(lines) * line_h + 6
        sheet[top:top + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    return sheet


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Rectified walls, floor and ceiling, and a floor plan for one walkthrough.")
    p.add_argument("slam_dir", type=Path, help="the platform's cached 3D-engine output for the video")
    p.add_argument("video", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--px-per-m", type=float, default=100.0)
    p.add_argument("--blend", choices=["multiband", "average"], default="multiband")
    p.add_argument("--findings", type=Path, help="JSON list of findings, each with xyz (model frame) and label")
    args = p.parse_args(argv)

    w = load_platform_run(args.slam_dir, args.video)
    if w.layout is None:
        raise SystemExit(f"no layout.json under {args.slam_dir}")
    w.load_images()
    findings = json.loads(args.findings.read_text(encoding="utf-8")) if args.findings else []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    surfaces = wall_planes(w.layout) + [floor_plane(w.layout), ceiling_plane(w.layout)]
    tiles, summary = [], []
    for s in surfaces:
        t = time.perf_counter()
        r = rectify_wall(w, s, args.px_per_m, blend=args.blend)
        secs = time.perf_counter() - t
        img = draw_findings(r, findings) if findings else r.image
        cv2.imwrite(str(args.out_dir / f"{s.id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles.append((f"{s.id}  {s.length_m:.2f} x {s.height_m:.2f} m\nseen {r.coverage_pct:.0f}%, "
                      f"{len(r.frames_used)} frames", draw_openings(r)))
        summary.append({"id": s.id, "length_m": round(s.length_m, 3), "height_m": round(s.height_m, 3),
                        "coverage_pct": round(r.coverage_pct, 1), "frames": len(r.frames_used),
                        "seconds": round(secs, 2)})
        print(f"{s.id}: {s.length_m:.2f} x {s.height_m:.2f} m, seen {r.coverage_pct:.0f}% from "
              f"{len(r.frames_used)} frames, {secs:.1f} s")
    plan = floor_plan(w.layout, findings=findings, trajectory=w, title=w.name)
    cv2.imwrite(str(args.out_dir / "floor_plan.png"), plan)
    sheet = contact_sheet([("floor plan", plan)] + tiles)
    cv2.imwrite(str(args.out_dir / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    (args.out_dir / "ortho.json").write_text(json.dumps({"walkthrough": w.name, "px_per_m": args.px_per_m,
                                                         "surfaces": summary}, indent=2), encoding="utf-8")
    print(f"wrote {len(surfaces)} surfaces, floor_plan.png and contact_sheet.jpg to {args.out_dir}")


if __name__ == "__main__":
    main()
