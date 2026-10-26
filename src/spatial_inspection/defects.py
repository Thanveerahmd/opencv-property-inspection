"""Defect candidates found with classical OpenCV image analysis: cracks, stains, mould, rust and peeling paint.

Every detector works on a copy of the image scaled to at most `max_side` pixels, so thresholds do not depend much on
the camera resolution, and returns candidates in the input image's pixel coordinates. Scores are heuristic
confidences in 0..1 for ranking what a reviewer (or a learned model) should look at first; they are not calibrated
probabilities.

How each kind is found:
- crack: CLAHE, then black-hat (dark lines) and top-hat (light lines, e.g. cracks in dark tiles) confirmed by the
  ximgproc ridge filter, hysteresis, removal of regular straight line families (grout and tile joints, found with
  HoughLinesP), thinning to a skeleton and keeping long thin components (length, width from distanceTransform)
- stain: colour deviation in Lab from the local surface colour (large median filter), towards darker, yellower or
  browner only, ranked with spectral-residual saliency and shape
- mould: clusters of small dark speckles (black-hat spots, counted with a box filter)
- rust: orange-brown HSV and Lab range, ranked by saturation and surface roughness
- peeling: dense edges with no dominant orientation (Canny density times 1 - structure tensor coherence)

measure() turns a candidate into metres with a depth map: a plane is fitted to the depth around the candidate and
every pixel is intersected with it, so lengths and areas are not inflated by depth noise.

Overlaps between kinds are settled in detect_all: cracks inside flaking paint or rust, stains inside rust and peeling
inside a mould speckle cluster are scored down, since the larger region explains them.

Known limitations (from the sample images in tests/data, where the expected kind is in the top three for all six):
- detectors are generic and fire on any matching pattern: object outlines (a toilet seat, a skirting board) and
  flake edges can pass as cracks, dark tiles or shadows as stains, a wooden bed frame as rust; treat the output as
  candidates for review
- ceiling_crack_1: the torn, open crack is reported as a stain (a brown gap); its thin side branches are found as
  cracks, but so is fine plaster texture in the shaded corner at a similar score (about 0.55)
- rusty_door_lock_2 and water_damaged_wall_1: rust and the dark streaks it leaves come out as both rust and stain
- hairline cracks that are fainter than the surface texture (heavy stucco) are found only in pieces
- none of the samples shows mould; that detector is checked only on a synthetic speckle cluster
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np

KINDS = ("crack", "stain", "mould", "rust", "peeling")
MIN_SIDE = 24   # smaller crops hold too little surface to judge
COLOURS = {  # BGR
    "crack": (40, 40, 230),
    "stain": (220, 120, 30),
    "mould": (60, 180, 40),
    "rust": (0, 140, 255),
    "peeling": (200, 60, 200),
}


@dataclass
class Candidate:
    kind: str
    score: float                        # 0..1
    bbox: tuple[int, int, int, int]     # x, y, w, h in image pixels
    mask: np.ndarray | None = None      # uint8 0/255, same size as the image
    contour: np.ndarray | None = None   # largest outer contour, (N, 1, 2) int32
    features: dict = field(default_factory=dict)  # "*_px" values are in image pixels, "*_px2" in square pixels

    def as_dict(self) -> dict:
        """JSON-friendly summary without the mask and contour."""
        return {"kind": self.kind, "score": round(self.score, 3), "bbox": [int(v) for v in self.bbox],
                "features": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.features.items()}}


@dataclass
class _Work:
    """The image at working resolution, with the colour spaces the detectors share."""

    bgr: np.ndarray
    scale: float                  # working pixels per input pixel
    shape: tuple[int, int]        # input (h, w)

    @cached_property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    @cached_property
    def lab(self) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(self.bgr, (0, 0), 1.5), cv2.COLOR_BGR2LAB)

    @cached_property
    def hsv(self) -> np.ndarray:
        return cv2.cvtColor(cv2.GaussianBlur(self.bgr, (0, 0), 1.5), cv2.COLOR_BGR2HSV)

    @cached_property
    def roughness(self) -> np.ndarray:
        """Local standard deviation of the grey level, averaged over a small neighbourhood."""
        g = self.gray.astype(np.float32)
        std = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
        return cv2.blur(std, (15, 15))

    @cached_property
    def rust_colour(self) -> np.ndarray:
        """Rough orange-brown pixels: hue, saturation and value range, red and yellow in Lab, and texture."""
        H, S, V = cv2.split(self.hsv)
        a = self.lab[..., 1].astype(np.int16) - 128
        b = self.lab[..., 2].astype(np.int16) - 128
        colour = ((H <= 24) | (H >= 168)) & (S >= 55) & (V >= 25) & (V <= 200) & (a > 4) & (b > 3)
        # smooth brown paint, wood and shadowed gaps are not corroded
        return colour & (self.roughness > 6)

    @cached_property
    def edges(self) -> np.ndarray:
        return cv2.Canny(cv2.GaussianBlur(self.gray, (0, 0), 1.5), 50, 120)

    @property
    def diag(self) -> float:
        return math.hypot(*self.gray.shape)

    @property
    def area(self) -> int:
        return self.gray.shape[0] * self.gray.shape[1]


def _prepare(img: np.ndarray, max_side: int) -> _Work:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    return _Work(bgr=small, scale=s, shape=(h, w))


def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _odd(v: float, lo: int = 3) -> int:
    v = max(lo, int(v))
    return v if v % 2 else v + 1


def _robust_level(values: np.ndarray, k: float) -> float:
    """median + k * MAD, a noise-relative threshold."""
    med = float(np.median(values))
    return med + k * (float(np.median(np.abs(values.astype(np.float32) - med))) + 1.0)


def _keep_seeded(weak: np.ndarray, strong: np.ndarray) -> np.ndarray:
    """Hysteresis: the weak components that contain at least one strong pixel."""
    n, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(weak, dtype=bool)
    seeds = np.unique(labels[strong & weak])
    keep = np.zeros(n, bool)
    keep[seeds] = True
    keep[0] = False
    return keep[labels]


def _blob(kind: str, score: float, mask: np.ndarray, features: dict) -> Candidate:
    """Candidate at working resolution from a region mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    bbox = cv2.boundingRect(cv2.findNonZero(mask))
    return Candidate(kind, float(np.clip(score, 0.0, 1.0)), tuple(int(v) for v in bbox), mask, contour, features)


def _region_stats(mask: np.ndarray) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.countNonZero(mask))
    hull = cv2.contourArea(cv2.convexHull(c))
    _, _, bw, bh = cv2.boundingRect(c)
    return {"area_px2": area, "solidity": area / max(hull, 1.0), "extent": area / max(bw * bh, 1)}


def _regions(mask: np.ndarray, min_area: float, limit: int = 30):
    """Masks of the largest connected regions with at least min_area pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    order = [i for i in np.argsort(-stats[:, cv2.CC_STAT_AREA]) if i > 0 and stats[i, cv2.CC_STAT_AREA] >= min_area]
    for i in order[:limit]:
        yield (labels == i).astype(np.uint8) * 255


# ---------------------------------------------------------------- cracks

def _regular(gaps: np.ndarray) -> bool:
    """Two neighbouring gaps between parallel lines that look like one spacing (allowing a missed line)."""
    r = max(gaps) / max(min(gaps), 1e-6)
    return r <= 1.6 or (round(r) in (2, 3) and abs(r - round(r)) <= 0.2)


def _grid_lines(mask: np.ndarray, diag: float) -> np.ndarray:
    """Pixels on regularly spaced families of long straight lines (grout, tile joints, boards), not cracks."""
    out = np.zeros(mask.shape, np.uint8)
    segs = cv2.HoughLinesP(mask.astype(np.uint8) * 255, 2, np.pi / 180, threshold=50,
                           minLineLength=int(0.05 * diag), maxLineGap=12)
    if segs is None or len(segs) < 3:
        return out.astype(bool)
    segs = segs.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    for i, (x1, y1, x2, y2) in enumerate(segs):
        d = np.abs(ang - ang[i])
        family = np.minimum(d, 180 - d) <= 10
        t = np.radians(ang[i])
        offsets = -mid[family, 0] * np.sin(t) + mid[family, 1] * np.cos(t)
        own = -mid[i, 0] * np.sin(t) + mid[i, 1] * np.cos(t)
        # distinct parallel lines, then whether this one sits in a run of three evenly spaced ones
        lines = []
        for o in np.sort(offsets):
            if lines and o - lines[-1][-1] <= 15:
                lines[-1].append(o)
            else:
                lines.append([o])
        centres = np.array([np.mean(g) for g in lines])
        k = int(np.argmin(np.abs(centres - own)))
        regular = any(_regular(np.diff(centres[j:j + 3])) for j in range(max(0, k - 2), min(k, len(centres) - 3) + 1))
        if regular:
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 1, 9)
    return out.astype(bool)


def _skeleton_length(skel: np.ndarray) -> float:
    """Length of a one-pixel skeleton: each outline walks every branch twice."""
    contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(c, True) for c in contours) / 2


def _crack_candidates(wk: _Work) -> list[Candidate]:
    h, w = wk.gray.shape
    diag = wk.diag
    eq = cv2.GaussianBlur(cv2.createCLAHE(2.0, (8, 8)).apply(wk.gray), (0, 0), 1.0)
    kernel = _ellipse(15)
    ridge = cv2.ximgproc.RidgeDetectionFilter.create(ddepth=cv2.CV_32F, ksize=3, out_dtype=cv2.CV_32F)
    min_len = max(30.0, 0.04 * diag)
    out = []
    for polarity, src in (("dark", eq), ("light", 255 - eq)):
        # black-hat of the (inverted) image: how much darker a pixel is than its closed surroundings
        bh = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
        r = ridge.getRidgeFilteredImage(src)
        strong = (bh > max(18.0, _robust_level(bh, 6))) & (r > 200)
        weak = (bh > max(10.0, _robust_level(bh, 3))) & (r > 80)
        line = _keep_seeded(weak, strong)
        line &= ~_grid_lines(line, diag)
        line = line.astype(np.uint8) * 255
        # a step between two surfaces survives closing, a thin line on one surface does not
        closed = cv2.GaussianBlur(src.astype(np.float32) + bh, (0, 0), 3.0)
        step = cv2.magnitude(cv2.Sobel(closed, cv2.CV_32F, 1, 0), cv2.Sobel(closed, cv2.CV_32F, 0, 1))
        dist = cv2.distanceTransform(line, cv2.DIST_L2, 3)
        # pieces of one crack separated by small gaps are measured together
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(line, _ellipse(7)), connectivity=8)
        for i in range(1, n):
            x, y, bw, bh_, _ = stats[i]
            if max(bw, bh_) < min_len * 0.7:
                continue
            sl = np.s_[y:y + bh_, x:x + bw]
            comp = ((labels[sl] == i) & (line[sl] > 0)).astype(np.uint8) * 255
            skel = cv2.ximgproc.thinning(comp)
            ys, xs = np.nonzero(skel)
            if len(xs) < 5:
                continue
            length = _skeleton_length(skel)
            width = max(1.0, 2 * float(dist[sl][ys, xs].mean()) - 1)
            contrast = float(bh[sl][ys, xs].mean())
            elong = length / width
            if length < min_len or elong < 8 or contrast < 12:
                continue
            pts = np.column_stack([xs, ys]).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            along = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
            offset = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
            edge_ratio = float(step[sl][ys, xs].mean()) * 5 / contrast
            tortuosity = length / math.hypot(bw, bh_)
            s_len = 1 - math.exp(-length / (0.12 * diag))
            s_con = np.clip((contrast - 10) / 30, 0, 1)
            s_thin = np.clip((elong - 8) / 30, 0, 1)
            # an outline between two different surfaces rather than a line within one
            s_surface = float(np.clip(1.3 - edge_ratio, 0.3, 1.0))
            # cracks run somewhere; texture and flake outlines wind back and forth
            s_smooth = float(np.clip(1 - (tortuosity - 1.8) / 1.5, 0.3, 1.0))
            prior = 1.0 if polarity == "dark" else 0.9
            score = (0.45 * s_len + 0.35 * s_con + 0.2 * s_thin) * s_surface * s_smooth * prior
            mask = np.zeros((h, w), np.uint8)
            mask[sl] = comp
            out.append(_blob("crack", score, mask, {
                "polarity": polarity, "length_px": length, "width_px": width, "contrast": contrast,
                "elongation": elong, "straightness_px": float(np.sqrt(np.mean(offset ** 2))),
                "chord_ratio": float(np.ptp(along) / max(length, 1.0)), "edge_ratio": edge_ratio,
                "tortuosity": tortuosity,
                "area_px2": float(cv2.countNonZero(comp))}))
    return out


# ---------------------------------------------------------------- stains

def _surface_colour(lab: np.ndarray) -> np.ndarray:
    """The local surface colour: a large median filter on a small copy, so stains do not pull it towards themselves."""
    h, w = lab.shape[:2]
    s = 160 / max(h, w)
    small = cv2.resize(lab, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)
    k = _odd(min(small.shape[:2]) * 0.3)
    bg = cv2.merge([cv2.medianBlur(np.ascontiguousarray(small[..., c]), k) for c in range(3)])
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def _stain_candidates(wk: _Work) -> list[Candidate]:
    lab = wk.lab.astype(np.float32)
    bg = _surface_colour(wk.lab).astype(np.float32)
    L, a, b = lab[..., 0] * (100 / 255), lab[..., 1] - 128, lab[..., 2] - 128
    dL = L - bg[..., 0] * (100 / 255)
    da, db = a - (bg[..., 1] - 128), b - (bg[..., 2] - 128)
    # water marks are darker, yellower or browner than the paint, never bluer and rarely vivid
    dev = np.sqrt((0.8 * np.maximum(-dL, 0)) ** 2 + np.maximum(db, 0) ** 2 + (0.5 * np.maximum(da, 0)) ** 2)
    gate = (db > -4) & (np.hypot(a, b) < 45)
    weak = (dev > 10) & gate
    mask = _keep_seeded(weak, (dev > 18) & gate).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15))
    ok, sal = cv2.saliency.StaticSaliencySpectralResidual.create().computeSaliency(wk.bgr)
    sal = sal.astype(np.float32) if ok else np.zeros(wk.gray.shape, np.float32)
    sal_mean = float(sal.mean()) + 1e-6
    out = []
    for region in _regions(mask, 0.003 * wk.area):
        m = region > 0
        f = _region_stats(region)
        f.update({"deviation": float(dev[m].mean()), "darker": float(-dL[m].mean()), "yellower": float(db[m].mean()),
                  "area_frac": f["area_px2"] / wk.area, "saliency": float(sal[m].mean() / sal_mean)})
        s_dev = np.clip((f["deviation"] - 8) / 20, 0, 1)
        s_area = math.sqrt(min(1.0, f["area_frac"] / 0.05))
        s_sal = np.clip(f["saliency"] / 2, 0, 1)
        # neat rectangles are doors, frames and furniture rather than stains
        s_shape = 0.5 if f["extent"] > 0.85 and f["solidity"] > 0.95 else 1.0
        # water marks carry a yellow-brown tint; plain dark patches are more often shadows or dark materials
        s_tint = float(np.clip(0.5 + f["yellower"] / 8, 0.5, 1.0))
        score = (0.45 * s_dev + 0.35 * s_area + 0.2 * s_sal) * s_shape * s_tint
        out.append(_blob("stain", score, region, f))
    return out


# ---------------------------------------------------------------- mould

def _mould_candidates(wk: _Work) -> list[Candidate]:
    h, w = wk.gray.shape
    g = cv2.GaussianBlur(wk.gray, (3, 3), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, _ellipse(9))
    spots = (bh > max(22.0, _robust_level(bh, 6))).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(spots, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    bw, bh_ = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    speck = (area >= 3) & (area <= 80) & (np.maximum(bw, bh_) <= 3 * np.minimum(bw, bh_))
    speck[0] = False
    if speck.sum() < 10:
        return []
    pts = np.zeros((h, w), np.float32)
    cx, cy = cents[speck].T
    pts[np.clip(cy.astype(int), 0, h - 1), np.clip(cx.astype(int), 0, w - 1)] = 1
    win = _odd(max(15, 0.1 * min(h, w)))
    dens = cv2.boxFilter(pts, -1, (win, win), normalize=False)
    cluster = cv2.dilate((dens >= 8).astype(np.uint8) * 255, _ellipse(win // 2))
    spot_mask = speck[labels]
    halo = cv2.dilate(spot_mask.astype(np.uint8), _ellipse(5)) > 0
    edges = wk.edges > 0
    out = []
    for region in _regions(cluster, 0.004 * wk.area, limit=15):
        m = region > 0
        n_spots = int(pts[m].sum())
        e = edges & m
        f = _region_stats(region)
        f.update({"spots": n_spots, "spot_cover": float(spot_mask[m].mean()), "area_frac": f["area_px2"] / wk.area,
                  "spots_per_window": float(dens[m].max()),
                  # on mould the edges are the speckles; flaking paint and rust have long outlines between them
                  "spot_edge_share": float((e & halo).sum() / max(e.sum(), 1)),
                  "rust_colour": float(wk.rust_colour[m].mean())})
        s_density = np.clip((f["spots_per_window"] - 8) / 20, 0, 1)
        s_cover = np.clip(f["spot_cover"] / 0.08, 0, 1)
        s_count = np.clip(n_spots / 150, 0, 1)
        s_spotty = np.clip((f["spot_edge_share"] - 0.3) / 0.4, 0.15, 1)
        score = (0.5 * s_density + 0.3 * s_cover + 0.2 * s_count) * s_spotty * (1 - 0.7 * f["rust_colour"])
        out.append(_blob("mould", score, region, f))
    return out


# ---------------------------------------------------------------- rust

def _rust_candidates(wk: _Work) -> list[Candidate]:
    H, S, V = cv2.split(wk.hsv)
    colour = wk.rust_colour
    mask = cv2.morphologyEx(colour.astype(np.uint8) * 255, cv2.MORPH_OPEN, _ellipse(3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(15))
    rough = wk.roughness
    out = []
    for region in _regions(mask, 0.002 * wk.area):
        m = region > 0
        f = _region_stats(region)
        f.update({"area_frac": f["area_px2"] / wk.area, "saturation": float(S[m].mean()), "hue": float(H[m].mean()),
                  "value": float(V[m].mean()), "roughness": float(rough[m].mean()),
                  "colour_fill": float(colour[m].mean())})
        s_col = np.clip((f["saturation"] - 40) / 60, 0, 1) * f["colour_fill"]
        s_area = math.sqrt(min(1.0, f["area_frac"] / 0.03))
        s_rough = np.clip((f["roughness"] - 6) / 8, 0, 1)
        out.append(_blob("rust", (0.4 * s_col + 0.6 * s_area) * (0.3 + 0.7 * s_rough), region, f))
    return out


# ---------------------------------------------------------------- peeling

def _peeling_candidates(wk: _Work) -> list[Candidate]:
    h, w = wk.gray.shape
    g = cv2.GaussianBlur(wk.gray, (0, 0), 1.5)
    win = _odd(max(15, 0.04 * min(h, w)))
    dens = cv2.boxFilter(wk.edges.astype(np.float32) / 255, -1, (win, win))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    j11 = cv2.boxFilter(gx * gx, -1, (win, win))
    j22 = cv2.boxFilter(gy * gy, -1, (win, win))
    j12 = cv2.boxFilter(gx * gy, -1, (win, win))
    # structure tensor coherence: 1 for one dominant edge direction (joints, frames), 0 for flaky, random texture
    coh = np.sqrt((j11 - j22) ** 2 + 4 * j12 ** 2) / (j11 + j22 + 1e-6)
    peel = cv2.boxFilter(dens * (1 - coh), -1, (2 * win + 1, 2 * win + 1))
    mask = (peel > 0.035).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(win // 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(win))
    out = []
    for region in _regions(mask, 0.01 * wk.area, limit=15):
        m = region > 0
        f = _region_stats(region)
        f.update({"area_frac": f["area_px2"] / wk.area, "edge_density": float(dens[m].mean()),
                  "coherence": float(coh[m].mean()), "surround_density": float(dens[~m].mean()) if (~m).any() else 0.0,
                  "rust_colour": float(wk.rust_colour[m].mean())})
        s_dens = np.clip((f["edge_density"] - 0.04) / 0.1, 0, 1)
        s_area = math.sqrt(min(1.0, f["area_frac"] / 0.15))
        s_iso = np.clip((0.6 - f["coherence"]) / 0.4, 0, 1)
        s_contrast = np.clip(f["edge_density"] / (f["surround_density"] + 0.01) / 3, 0, 1)
        score = (0.35 * s_dens + 0.25 * s_area + 0.2 * s_iso + 0.2 * s_contrast) * (1 - 0.6 * f["rust_colour"])
        out.append(_blob("peeling", score, region, f))
    return out


_DETECTORS = {"crack": _crack_candidates, "stain": _stain_candidates, "mould": _mould_candidates,
              "rust": _rust_candidates, "peeling": _peeling_candidates}


# ---------------------------------------------------------------- shared

def _to_input(c: Candidate, wk: _Work) -> Candidate:
    """Scale a working-resolution candidate back to the input image."""
    if wk.scale == 1.0:
        return c
    h, w = wk.shape
    s = 1.0 / wk.scale
    x, y, bw, bh = c.bbox
    x0, y0 = int(x * s), int(y * s)
    x1, y1 = min(w, math.ceil((x + bw) * s)), min(h, math.ceil((y + bh) * s))
    mask = None
    if c.mask is not None:
        mask = np.zeros((h, w), np.uint8)
        mask[y0:y1, x0:x1] = cv2.resize(c.mask[y:y + bh, x:x + bw], (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    contour = None if c.contour is None else np.round(c.contour * s).astype(np.int32)
    features = {}
    for k, v in c.features.items():
        features[k] = v * s * s if k.endswith("_px2") else v * s if k.endswith("_px") else v
    return Candidate(c.kind, c.score, (x0, y0, x1 - x0, y1 - y0), mask, contour, features)


def _iou(a: tuple, b: tuple) -> tuple[float, float]:
    """Intersection over union, and over the smaller box."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min(a[2] * a[3], b[2] * b[3])
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1), inter / max(small, 1)


def nms(candidates: list[Candidate], same_iou: float = 0.3, cross_iou: float = 0.6) -> list[Candidate]:
    """Drop candidates that overlap a better one: loosely within a kind, only near-duplicates across kinds."""
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: -c.score):
        ok = True
        for k in kept:
            iou, inside = _iou(c.bbox, k.bbox)
            if c.kind == k.kind:
                # a thin crack's box is mostly empty, so nesting says nothing about cracks
                ok = iou <= same_iou and (c.kind == "crack" or inside <= 0.85)
            else:
                ok = iou <= cross_iou
            if not ok:
                break
        if ok:
            kept.append(c)
    return kept


def detect(img: np.ndarray, kind: str, max_side: int = 1280, min_score: float = 0.05,
           limit: int = 12) -> list[Candidate]:
    """Candidates of one kind, best first."""
    return detect_all(img, kinds=[kind], max_side=max_side, min_score=min_score, limit=limit)


def detect_all(img: np.ndarray, kinds: list[str] | tuple[str, ...] | None = None, max_side: int = 1280,
               min_score: float = 0.05, limit: int = 12) -> list[Candidate]:
    """Candidates of every requested kind (all by default), overlaps suppressed, best first.

    limit caps the candidates kept per kind.
    """
    kinds = list(kinds or KINDS)
    for kind in kinds:
        if kind not in _DETECTORS:
            raise ValueError(f"unknown defect kind {kind!r}, expected one of {KINDS}")
    if img is None or img.size == 0 or min(img.shape[:2]) < MIN_SIDE:
        return []
    wk = _prepare(img, max_side)
    raw = {kind: _DETECTORS[kind](wk) for kind in kinds}
    # thin dark lines inside flaking paint or rust are flake and pit outlines; dark patches on rust are the rust
    _explain(raw.get("crack", []), raw.get("peeling", []) + raw.get("rust", []), 0.5)
    _explain(raw.get("stain", []), raw.get("rust", []), 0.4)
    _explain(raw.get("peeling", []), raw.get("mould", []), 0.5)
    found = []
    for cands in raw.values():
        found += nms([c for c in cands if c.score >= min_score])[:limit]
    return [_to_input(c, wk) for c in nms(found)]


def _explain(targets: list[Candidate], regions: list[Candidate], strength: float) -> None:
    """Lower candidates that lie mostly inside regions of another kind which explain them better."""
    if not targets or not regions:
        return
    cover = np.zeros(regions[0].mask.shape, np.float32)
    for r in regions:
        cover = np.maximum(cover, (r.mask > 0) * np.float32(r.score))
    for c in targets:
        inside = c.mask > 0
        share = float(np.mean(cover[inside] > 0))
        if share > 0.5:
            c.features["explained_by"] = regions[0].kind if len({r.kind for r in regions}) == 1 else "texture"
            c.features["explained_share"] = share
            c.score *= 1 - strength * share * float(cover[inside].max())


def detect_cracks(img: np.ndarray, **kw) -> list[Candidate]:
    return detect(img, "crack", **kw)


def detect_stains(img: np.ndarray, **kw) -> list[Candidate]:
    return detect(img, "stain", **kw)


def detect_mould(img: np.ndarray, **kw) -> list[Candidate]:
    return detect(img, "mould", **kw)


def detect_rust(img: np.ndarray, **kw) -> list[Candidate]:
    return detect(img, "rust", **kw)


def detect_peeling(img: np.ndarray, **kw) -> list[Candidate]:
    return detect(img, "peeling", **kw)


# ---------------------------------------------------------------- measurement

def _fit_plane(P: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Plane n . X = d through 3D points, refitted once without outliers; n faces the camera."""
    if len(P) < 20:
        return None
    for _ in range(2):
        c = P.mean(axis=0)
        _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
        n = vt[2]
        if sv[1] < 1e-9:
            return None
        res = np.abs((P - c) @ n)
        keep = res <= 2.5 * np.median(res) + 0.005
        if keep.all() or keep.sum() < 20:
            break
        P = P[keep]
    if n[2] > 0:
        n = -n
    return n, float(n @ c)


def measure(c: Candidate, depth: np.ndarray | None = None, K_depth: np.ndarray | None = None,
            image_size: tuple[int, int] | None = None) -> dict:
    """Size of a candidate: always in pixels, and in metres when a depth map and its intrinsics are given.

    depth holds metres along the camera z axis and may have a lower resolution than the image; K_depth is for the
    depth map and is scaled to image_size (width, height), which defaults to the mask size. Returns length_px,
    width_px, area_px and, with depth, length_m, width_mm, area_m2, depth_m and tilt_deg (surface normal against the
    viewing direction). Cracks are measured along their skeleton; other kinds use their minimum-area rectangle.
    """
    mask = c.mask
    if mask is None:
        if image_size is None or c.contour is None:
            raise ValueError("measure needs a candidate mask, or a contour and image_size")
        mask = np.zeros((image_size[1], image_size[0]), np.uint8)
        cv2.drawContours(mask, [c.contour], -1, 255, cv2.FILLED)
    mh, mw = mask.shape
    W, H = image_size or (mw, mh)
    x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(mask))
    roi = (mask[y:y + bh, x:x + bw] > 0).astype(np.uint8) * 255
    roi = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ox, oy = x - 1, y - 1
    area_px = float(cv2.countNonZero(roi))
    if c.kind == "crack":
        skel = cv2.ximgproc.thinning(roi)
        ys, xs = np.nonzero(skel)
        width_px = max(1.0, 2 * float(cv2.distanceTransform(roi, cv2.DIST_L2, 3)[ys, xs].mean()) - 1)
        # an end pixel covers half a pixel beyond its centre
        nb = cv2.filter2D((skel > 0).astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
        ends = int(np.count_nonzero((skel > 0) & (nb == 2)))
        paths = [p.reshape(-1, 2).astype(np.float64) + (ox, oy) for p in
                 cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
        length_px = sum(_path_length(p) for p in paths) / 2 + 0.5 * ends
        lines = None
    else:
        contour = max(cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        box = cv2.boxPoints(cv2.minAreaRect(contour)) + (ox, oy)
        e1, e2 = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
        mids = [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, (box[1] + box[2]) / 2, (box[3] + box[0]) / 2]
        # long axis joins the midpoints of the short sides
        lines = (mids[2:], mids[:2]) if e1 >= e2 else (mids[:2], mids[2:])
        length_px, width_px = max(e1, e2), min(e1, e2)
        ys, xs = np.nonzero(skel := roi)
    out = {"kind": c.kind, "length_px": float(length_px), "width_px": float(width_px), "area_px": area_px,
           "length_m": None, "width_mm": None, "area_m2": None, "depth_m": None, "tilt_deg": None}
    if depth is None or K_depth is None:
        return out

    # everything below is in image pixels; masks of another size are mapped onto the image
    sx_m, sy_m = W / mw, H / mh
    dh, dw = depth.shape[:2]
    K = np.diag([W / dw, H / dh, 1.0]) @ np.asarray(K_depth, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def rays(u, v):
        u = (np.asarray(u, np.float64) + 0.5) * sx_m - 0.5
        v = (np.asarray(v, np.float64) + 0.5) * sy_m - 0.5
        return np.column_stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)])

    # surface plane from the depth in and around the candidate (cracks themselves often have bad depth)
    pad = int(0.25 * max(bw, bh)) + 4
    gx0, gy0, gx1, gy1 = max(0, x - pad), max(0, y - pad), min(mw, x + bw + pad), min(mh, y + bh + pad)
    step = max(1, int(math.sqrt((gx1 - gx0) * (gy1 - gy0) / 3000)))
    gv, gu = np.mgrid[gy0:gy1:step, gx0:gx1:step]
    gu, gv = gu.ravel(), gv.ravel()
    r = rays(gu, gv)
    du = np.clip(((r[:, 0] * fx + cx + 0.5) * dw / W).astype(int), 0, dw - 1)
    dv = np.clip(((r[:, 1] * fy + cy + 0.5) * dh / H).astype(int), 0, dh - 1)
    z = depth[dv, du].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < 100)
    if ok.sum() < 3:
        return out
    plane = _fit_plane(r[ok] * z[ok, None])
    if plane is None:
        n, d = np.array([0.0, 0.0, -1.0]), -float(np.median(z[ok]))
    else:
        n, d = plane

    def on_plane(u, v):
        rr = rays(u, v)
        nr = rr @ n
        nr = np.where(np.abs(nr) < 1e-6, -1e-6, nr)
        return rr * (d / nr)[:, None], rr

    P, rr = on_plane(xs + ox, ys + oy)
    zc = float(np.median(P[:, 2]))
    foot = zc / math.sqrt(fx * fy) * math.sqrt(sx_m * sy_m)   # metres per mask pixel, facing the camera
    if c.kind == "crack":
        length_m = sum(_path_length(on_plane(p[:, 0], p[:, 1])[0]) for p in paths) / 2 + 0.5 * ends * foot
    else:
        length_m = float(np.linalg.norm(np.subtract(*on_plane(*np.array(lines[0]).T)[0])))
        out["width_m"] = float(np.linalg.norm(np.subtract(*on_plane(*np.array(lines[1]).T)[0])))
    # each pixel's patch on a tilted plane: z^2 / (fx fy |n . r|), with r the ray scaled to z = 1
    av, au = np.nonzero(roi)
    Pa, ra = on_plane(au + ox, av + oy)
    cos = np.maximum(np.abs(ra @ n), 0.2 * np.linalg.norm(ra, axis=1))
    area_m2 = float(np.sum(Pa[:, 2] ** 2 / (fx * fy * cos)) * sx_m * sy_m)
    out.update({"length_m": float(length_m), "width_mm": float(width_px * foot * 1000), "area_m2": area_m2,
                "depth_m": zc, "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[2])))))})
    return out


def _path_length(P: np.ndarray) -> float:
    """Length of a closed path of 2D or 3D points."""
    return float(np.linalg.norm(np.diff(P, axis=0, append=P[:1]), axis=1).sum())


# ---------------------------------------------------------------- drawing

def draw(img: np.ndarray, candidates: list[Candidate], alpha: float = 0.35) -> np.ndarray:
    """Annotated copy: tinted masks, outlines, boxes and "kind score" labels, coloured per kind."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    t = max(1, round(max(h, w) / 900))
    tint = out.copy()
    for c in candidates:
        if c.mask is not None:
            m = cv2.dilate(c.mask, _ellipse(2 * t + 1)) if c.kind == "crack" else c.mask
            tint[m > 0] = COLOURS.get(c.kind, (255, 255, 255))
    out = cv2.addWeighted(tint, alpha, out, 1 - alpha, 0)
    for c in candidates:
        col = COLOURS.get(c.kind, (255, 255, 255))
        if c.mask is not None and c.kind == "crack":
            out[cv2.dilate(c.mask, _ellipse(t)) > 0] = col
        elif c.contour is not None:
            cv2.drawContours(out, [c.contour], -1, col, t)
        x, y, bw, bh = c.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), col, t)
        label = f"{c.kind} {c.score:.2f}"
        fs = 0.5 * t
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        ty = y - 4 if y - th - base - 4 > 0 else y + th + base + 4
        cv2.rectangle(out, (x, ty - th - base), (x + tw + 4, ty + base // 2), col, cv2.FILLED)
        cv2.putText(out, label, (x + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t, cv2.LINE_AA)
    return out
