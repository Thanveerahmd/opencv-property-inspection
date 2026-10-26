"""Camera trajectory and metric depth for a whole walkthrough from a single phone camera, with OpenCV 5.

1. Corners are followed from frame to frame with pyramidal Lucas-Kanade flow (kept only when tracking back lands
   where it started), and an essential matrix fitted with MAGSAC between consecutive frames rejects tracks that do
   not move with the scene.
2. Start: the first frames are searched for a pair with enough parallax; its essential-matrix pose and triangulated
   points fix the offset of the first frame's network depth (see depth.py), which then gives every tracked corner a
   3D point.
3. Each new frame is placed with solvePnPRansac (refined with Levenberg-Marquardt) against the 3D points of the
   corners it still sees. The frame's network depth is then aligned to those points (inverse depth = a * disparity
   + b), so the depth and the trajectory share one scale, and new corners get 3D points from that depth.
4. Once a corner has been seen from two poses at least 5 degrees apart, its point is re-triangulated with
   cv2.triangulatePoints, so the map is held by geometry rather than only by the network.
5. When a frame cannot be placed (too few points), the essential-matrix rotation is applied and the camera is kept
   where it was.
6. Gravity and scale: the floor is the lowest large upward-facing plane in the aligned depth of many frames (RANSAC
   on points whose normals point roughly opposite the average camera "down"). The model is rotated so the floor
   normal is +y, moved so the floor is y = 0, and scaled so the median camera height is 1.45 m (a phone at chest
   height).

7. Scale drift: chaining depth from frame to frame lets the scale wander (about 15% over 8 s on the sample walk).
   floor_scale_correction reads the floor's distance below the camera in every frame's own depth and rescales the
   depth and the trajectory steps so the phone stays at a steady height (on by default; steady_height=False skips it).
8. Field of view: when no K is given it is self-calibrated from essential-matrix inlier counts (estimate_hfov).

One camera cannot measure distance, so the metric scale comes only from the camera-height assumption; see
process_video's meta for the numbers of each run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from spatial_inspection.depth import (
    DEFAULT_SHIFT,
    DepthEstimator,
    DepthFit,
    backproject,
    fit_inverse_depth,
    sample_map,
    to_metric,
)
from spatial_inspection.scene import Frame, Walkthrough
from spatial_inspection.tracking import LK_PARAMS, intrinsics_guess, relative_pose, track_points
from spatial_inspection.video import sample_frames, video_info

CAMERA_HEIGHT = 1.45      # metres, phone held at chest height
TRIANGULATE_DEG = 5.0     # parallax needed before a corner's point is re-triangulated
MAX_DEPTH_FACTOR = 6.0    # depth beyond this multiple of the frame's median is dropped
FLOOR_PEAK_FRACTION = 0.3  # the floor is the lowest height bin of upward-facing points with this share of the peak


def _lk(prev_gray: np.ndarray, gray: np.ndarray, pts: np.ndarray, max_backtrack_px: float = 1.0):
    """Where `pts` moved to, and which of them were followed reliably (forward-backward check)."""
    if len(pts) == 0:
        return pts, np.zeros(0, bool)
    p = pts.reshape(-1, 1, 2).astype(np.float32)
    nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p, None, **LK_PARAMS)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
    err = np.linalg.norm(p - back, axis=2).ravel()
    h, w = gray.shape
    q = nxt.reshape(-1, 2)
    inside = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] < h - 1)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (err < max_backtrack_px) & inside
    return q, ok


def _rays(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    r = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _angles_deg(r0: np.ndarray, r1: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((r0 * r1).sum(1), -1, 1)))


def _project(K: np.ndarray, w2c: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def _w2c(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.ravel(tvec)
    return T


def bootstrap_depth(grays: list[np.ndarray], K: np.ndarray, disp0: np.ndarray, depth_scale: np.ndarray,
                    max_frames: int = 40, min_parallax_deg: float = 2.0) -> tuple[DepthFit | None, int]:
    """Offset for the first frame's network depth, from the first later frame with enough parallax.

    Returns the fit (in the arbitrary unit-baseline scale) and the frame it used, or (None, -1).
    """
    p0 = cv2.goodFeaturesToTrack(grays[0], maxCorners=1500, qualityLevel=0.01, minDistance=7)
    if p0 is None:
        return None, -1
    p0 = p0.reshape(-1, 2)
    cur = p0.copy()
    for j in range(1, min(max_frames, len(grays))):
        cur, ok = _lk(grays[j - 1], grays[j], cur)
        p0, cur = p0[ok], cur[ok]
        if len(p0) < 50:
            break
        pose = relative_pose(p0, cur, K)
        if pose is None:
            continue
        r0, r1 = _rays(K, p0), _rays(K, cur)
        if np.median(_angles_deg(r0 @ pose.R.T, r1)) < min_parallax_deg:
            continue
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([pose.R, pose.t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P0, P1, p0.T.astype(np.float64), cur.T.astype(np.float64))
        X = (Xh[:3] / Xh[3]).T
        T1 = np.eye(4)
        T1[:3, :3], T1[:3, 3] = pose.R, pose.t
        uv0, z0 = _project(K, np.eye(4), X)
        uv1, z1 = _project(K, T1, X)
        par = _angles_deg(r0, _rays(K, cur) @ pose.R)  # angle between the two viewing rays, in frame 0
        good = ((z0 > 0) & (z1 > 0) & (np.linalg.norm(uv0 - p0, axis=1) < 2) & (np.linalg.norm(uv1 - cur, axis=1) < 2)
                & (par > 1.0))
        if good.sum() < 30:
            continue
        fit = fit_inverse_depth(sample_map(disp0, p0[good] * depth_scale), z0[good], threshold=0.15)
        if fit is not None and fit.inliers >= 30:
            return fit, j
    return None, -1


@dataclass
class _Tracks:
    """Corners followed from frame to frame, each with the frame it was first seen in and maybe a 3D point."""

    uv: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    next_id: int = 0
    first: dict = field(default_factory=dict)   # id -> (walkthrough index, uv)
    X: dict = field(default_factory=dict)       # id -> world point
    tri: set = field(default_factory=set)       # ids whose point was triangulated

    def keep(self, mask: np.ndarray) -> None:
        for i in self.ids[~mask]:
            self.first.pop(int(i), None)
            self.X.pop(int(i), None)
            self.tri.discard(int(i))
        self.uv, self.ids = self.uv[mask], self.ids[mask]

    def add(self, uv: np.ndarray, k: int) -> None:
        ids = np.arange(self.next_id, self.next_id + len(uv))
        self.next_id += len(uv)
        for i, p in zip(ids, uv, strict=True):
            self.first[int(i)] = (k, p.copy())
        self.uv = np.vstack([self.uv, uv.astype(np.float32)])
        self.ids = np.concatenate([self.ids, ids])

    def with_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices into the current tracks that have a 3D point, and those points."""
        idx = np.array([i for i, t in enumerate(self.ids) if int(t) in self.X], dtype=np.int64)
        X = np.array([self.X[int(self.ids[i])] for i in idx]).reshape(-1, 3)
        return idx, X


@dataclass
class Trajectory:
    """Output of track_walkthrough, before levelling: poses and depth in one arbitrary scale."""

    c2w: list[np.ndarray]
    depth: list[np.ndarray]
    fits: list[DepthFit | None]
    states: list[str]
    K_depth: np.ndarray
    stats: dict


def track_walkthrough(frames: list[Frame], estimator: DepthEstimator, max_tracks: int = 1200,
                      min_tracks: int = 700, progress: Callable[[int, int], None] | None = None) -> Trajectory:
    """Pose and aligned depth for every frame, in the first camera's frame and an arbitrary scale."""
    K = frames[0].K
    h, w = frames[0].image.shape[:2]
    dsize = estimator.depth_size((w, h))
    Kd = estimator.depth_K(K, (w, h))
    to_d = np.array([dsize[0] / w, dsize[1] / h])
    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in frames]
    t_depth = t_track = 0.0

    t0 = time.perf_counter()
    disp0 = estimator.predict(frames[0].image)
    t_depth += time.perf_counter() - t0
    t0 = time.perf_counter()
    boot, boot_frame = bootstrap_depth(grays, K, disp0, to_d)
    t_track += time.perf_counter() - t0

    tracks = _Tracks()
    c2ws: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    fits: list[DepthFit | None] = []
    states: list[str] = []
    w2c = np.eye(4)
    shift = boot.shift if boot else DEFAULT_SHIFT
    last_fit = boot or DepthFit(a=1.0, b=DEFAULT_SHIFT, inliers=0, points=0, rel_err=float("nan"))
    pnp_inliers = []

    for k, f in enumerate(frames):
        t0 = time.perf_counter()
        disp = disp0 if k == 0 else estimator.predict(f.image)
        t_depth += time.perf_counter() - t0
        t0 = time.perf_counter()
        gray = grays[k]
        state = "START"
        if k > 0:
            prev_uv = tracks.uv
            uv, ok = _lk(grays[k - 1], gray, prev_uv)
            tracks.uv = uv
            R_rel = None
            if ok.sum() >= 8:
                E, emask = cv2.findEssentialMat(prev_uv[ok], uv[ok], K, method=cv2.USAC_MAGSAC, prob=0.999,
                                                threshold=1.0)
                if E is not None and E.shape[0] >= 3:
                    _, R_rel, _, _ = cv2.recoverPose(E[:3], prev_uv[ok], uv[ok], K, mask=emask.copy())
                    ok[np.flatnonzero(ok)[emask.ravel() == 0]] = False
            tracks.keep(ok)

            idx, X = tracks.with_points()
            placed = False
            if len(idx) >= 12:
                rvec = cv2.Rodrigues(w2c[:3, :3])[0]
                tvec = w2c[:3, 3].reshape(3, 1).copy()
                found, rvec, tvec, inl = cv2.solvePnPRansac(
                    X.astype(np.float64), tracks.uv[idx].astype(np.float64), K, None, rvec, tvec,
                    useExtrinsicGuess=True, iterationsCount=200, reprojectionError=2.5, confidence=0.999,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if found and inl is not None and len(inl) >= max(12, 0.25 * len(idx)):
                    inl = inl.ravel()
                    rvec, tvec = cv2.solvePnPRefineLM(X[inl].astype(np.float64),
                                                      tracks.uv[idx][inl].astype(np.float64), K, None, rvec, tvec)
                    w2c = _w2c(rvec, tvec)
                    placed = True
                    state = "PNP"
                    pnp_inliers.append(len(inl))
                    outlier = np.ones(len(idx), bool)
                    outlier[inl] = False
                    keep = np.ones(len(tracks.ids), bool)
                    keep[idx[outlier]] = False
                    tracks.keep(keep)
            if not placed:
                C = -w2c[:3, :3].T @ w2c[:3, 3]
                if R_rel is not None:
                    w2c = w2c.copy()
                    w2c[:3, :3] = R_rel @ w2c[:3, :3]
                    w2c[:3, 3] = -w2c[:3, :3] @ C
                    state = "ROTATION_ONLY"
                else:
                    state = "LOST"

        # align this frame's network depth to the points it sees
        idx, X = tracks.with_points()
        fit = None
        if k == 0:
            fit = last_fit
        elif len(idx) >= 12:
            _, z = _project(K, w2c, X)
            d = sample_map(disp, tracks.uv[idx] * to_d)
            front = z > 0
            d, z = d[front], z[front]
            spread = np.percentile(z, 90) / max(np.percentile(z, 10), 1e-9) if len(z) else 1.0
            low = np.percentile(disp, 2)
            if spread > 1.5:
                fit = fit_inverse_depth(d, z)
                if fit is not None and (fit.inliers < 20 or fit.a * low + fit.b <= 0):
                    fit = None
            if fit is None:
                fit = fit_inverse_depth(d, z, shift=shift)
        if fit is None:
            fit = DepthFit(a=last_fit.a, b=last_fit.b, inliers=0, points=0, rel_err=float("nan"))
            state = state if state != "PNP" else "PNP_NO_DEPTH_FIT"
        else:
            shift = fit.shift
            last_fit = fit
        depth = to_metric(disp, fit.a, fit.b)
        valid = depth[depth > 0]
        if len(valid):
            depth[depth > MAX_DEPTH_FACTOR * float(np.median(valid))] = 0  # the far tail is noise, and unbounded
        c2w = np.linalg.inv(w2c)

        # re-triangulate corners seen from far enough apart
        if k > 0 and len(tracks.ids):
            r_now = _rays(K, tracks.uv) @ c2w[:3, :3].T
            groups: dict[int, list[int]] = {}
            for i, t in enumerate(tracks.ids):
                t = int(t)
                if t in tracks.tri:
                    continue
                fk = tracks.first[t][0]
                if fk < k:
                    groups.setdefault(fk, []).append(i)
            for fk, members in groups.items():
                members = np.array(members)
                uv0 = np.array([tracks.first[int(tracks.ids[i])][1] for i in members], np.float64)
                c2w0 = c2ws[fk]
                par = _angles_deg(_rays(K, uv0) @ c2w0[:3, :3].T, r_now[members])
                sel = members[par > TRIANGULATE_DEG]
                if len(sel) == 0:
                    continue
                uv0 = uv0[par > TRIANGULATE_DEG]
                w2c0 = np.linalg.inv(c2w0)
                Xh = cv2.triangulatePoints(K @ w2c0[:3], K @ w2c[:3], uv0.T, tracks.uv[sel].T.astype(np.float64))
                Xt = (Xh[:3] / Xh[3]).T
                e0, z0 = _project(K, w2c0, Xt)
                e1, z1 = _project(K, w2c, Xt)
                good = ((z0 > 0) & (z1 > 0) & (np.linalg.norm(e0 - uv0, axis=1) < 2.0)
                        & (np.linalg.norm(e1 - tracks.uv[sel], axis=1) < 2.0))
                for i, X1 in zip(sel[good], Xt[good], strict=True):
                    tracks.X[int(tracks.ids[i])] = X1
                    tracks.tri.add(int(tracks.ids[i]))

        # top up the corners and give the ones without a point a 3D point from this frame's depth
        if len(tracks.ids) < min_tracks:
            mask = np.full(gray.shape, 255, np.uint8)
            for x, y in tracks.uv.astype(int):
                cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
            new = cv2.goodFeaturesToTrack(gray, maxCorners=max_tracks - len(tracks.ids), qualityLevel=0.01,
                                          minDistance=8, mask=mask)
            if new is not None:
                tracks.add(new.reshape(-1, 2), k)
        missing = np.array([i for i, t in enumerate(tracks.ids) if int(t) not in tracks.X], dtype=np.int64)
        if len(missing):
            uvd = tracks.uv[missing] * to_d
            ij = np.clip(np.round(uvd).astype(int), 0, [dsize[0] - 1, dsize[1] - 1])
            z = depth[ij[:, 1], ij[:, 0]]
            ok = z > 0
            if ok.any():
                Xc = backproject(depth, Kd, uvd[ok])
                Xc *= (z[ok] / np.maximum(Xc[:, 2], 1e-9))[:, None]  # use the unblurred depth at the pixel
                Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
                for i, X1 in zip(missing[ok], Xw, strict=True):
                    tracks.X[int(tracks.ids[i])] = X1

        c2ws.append(c2w)
        depths.append(depth)
        fits.append(fit)
        states.append(state)
        t_track += time.perf_counter() - t0
        if progress:
            progress(k + 1, len(frames))

    stats = {
        "bootstrap_frame": boot_frame,
        "bootstrap_inliers": boot.inliers if boot else 0,
        "states": {s: states.count(s) for s in sorted(set(states))},
        "pnp_inliers_median": float(np.median(pnp_inliers)) if pnp_inliers else 0.0,
        "depth_fit_rel_err_median": float(np.nanmedian([f.rel_err for f in fits if f is not None] or [np.nan])),
        "seconds_depth_network": round(t_depth, 2),
        "seconds_tracking": round(t_track, 2),
    }
    return Trajectory(c2w=c2ws, depth=depths, fits=fits, states=states, K_depth=Kd, stats=stats)


def _normals(P: np.ndarray) -> np.ndarray:
    """Per-pixel unit normals of an organised (H, W, 3) point grid, facing the camera."""
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    flip = (n * P).sum(2) > 0
    n[flip] *= -1
    return n


def _fit_plane(P: np.ndarray, threshold: float, iterations: int = 300, seed: int = 0):
    """RANSAC plane (unit normal n, offset d with n.x + d = 0) and its inlier mask."""
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(iterations):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int((np.abs(P @ n - n @ a) < threshold).sum())
        if count > best_count:
            best, best_count = (n, -n @ a), count
    n, d = best
    inl = np.abs(P @ n + d) < threshold
    c = P[inl].mean(0)
    n = np.linalg.eigh(np.cov((P[inl] - c).T))[1][:, 0]  # direction of least spread
    d = -n @ c
    return n, d, np.abs(P @ n + d) < threshold


def find_floor(c2w: list[np.ndarray], depths: list[np.ndarray], K_depth: np.ndarray, max_frames: int = 60,
               stride: int = 4) -> tuple[np.ndarray, float, dict]:
    """Floor plane (unit normal pointing up, offset) in the trajectory's frame, from the aligned depth maps."""
    C = np.array([T[:3, 3] for T in c2w])
    up = -np.mean([T[:3, 1] for T in c2w], axis=0)  # opposite the average camera y (OpenCV y points down)
    up /= np.linalg.norm(up)
    pts, nrm = [], []
    step = max(1, len(c2w) // max_frames)
    for T, depth in zip(c2w[::step], depths[::step], strict=True):
        Pc = backproject(depth, K_depth)[::stride, ::stride]
        valid = Pc[:, :, 2] > 0
        n = _normals(Pc)
        valid &= np.isfinite(n).all(2)
        # a normal is unreliable where a neighbour has no depth
        valid &= cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        pts.append(Pc[valid] @ T[:3, :3].T + T[:3, 3])
        nrm.append(n[valid] @ T[:3, :3].T)
    P = np.vstack(pts)
    N = np.vstack(nrm)
    scale = float(np.median(np.linalg.norm(P - C.mean(0), axis=1)))
    info = {}
    for _ in range(2):
        facing_up = N @ up > np.cos(np.radians(25))
        heights = P[facing_up] @ up
        cam_h = float(np.median(C @ up))
        below = heights < cam_h
        if below.sum() < 100:
            raise RuntimeError("no floor seen")
        hb = heights[below]
        bin_w = 0.02 * scale
        lo, hi = np.percentile(hb, [0.5, 99.5])
        hist, edges = np.histogram(hb, bins=int(np.clip((hi - lo) / bin_w, 10, 1000)), range=(lo, hi))
        big = np.flatnonzero(hist >= FLOOR_PEAK_FRACTION * hist.max())
        level = 0.5 * (edges[big[0]] + edges[big[0] + 1])  # the lowest of the large flat levels
        cand = P[facing_up][below][np.abs(hb - level) < 3 * bin_w]
        if len(cand) > 20000:
            cand = cand[np.random.default_rng(0).choice(len(cand), 20000, replace=False)]
        n, d, inl = _fit_plane(cand, threshold=bin_w)
        if n @ up < 0:
            n, d = -n, -d
        up = n
        info = {"floor_points": int(inl.sum()), "upward_points": int(facing_up.sum())}
    return up, float(d), info


def level_and_scale(c2w: list[np.ndarray], depths: list[np.ndarray], K_depth: np.ndarray,
                    camera_height: float = CAMERA_HEIGHT) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Rotate, move and scale the trajectory and depth into the model frame (metres, y up, floor at y = 0)."""
    n, d, info = find_floor(c2w, depths, K_depth)
    C = np.array([T[:3, 3] for T in c2w])
    heights = C @ n + d
    s = camera_height / float(np.median(heights))
    fwd = c2w[0][:3, 2] - (c2w[0][:3, 2] @ n) * n
    if np.linalg.norm(fwd) < 1e-6:
        fwd = c2w[0][:3, 0] - (c2w[0][:3, 0] @ n) * n
    z_axis = -fwd / np.linalg.norm(fwd)   # the model's z points back, so the first view looks along -z
    x_axis = np.cross(n, z_axis)
    Ra = np.vstack([x_axis, n, z_axis])
    origin = C[0] - heights[0] * n       # the floor point under the first camera
    out = []
    for T in c2w:
        M = np.eye(4)
        M[:3, :3] = Ra @ T[:3, :3]
        M[:3, 3] = s * Ra @ (T[:3, 3] - origin)
        out.append(M)
    info.update(scale=s, camera_height_spread=float(np.percentile(heights * s, 90) - np.percentile(heights * s, 10)))
    return out, [dm * s for dm in depths], info


def floor_scale_correction(c2w: list[np.ndarray], depths: list[np.ndarray], K_depth: np.ndarray,
                           camera_height: float = CAMERA_HEIGHT, window: int = 9, stride: int = 4):
    """Undo slow scale drift by holding the camera at a steady height above the floor, frame by frame.

    Works on a levelled trajectory (y up). In each frame the floor's distance below the camera is read from that
    frame's own depth (the most common height among upward-facing points); its ratio to camera_height is the local
    scale error. Smoothed over `window` frames and interpolated where the floor is out of view, it rescales each
    frame's depth and each step of the trajectory. Returns new poses, depths and the per-frame scale factors.
    """
    n = len(c2w)
    s = np.full(n, np.nan)
    bin_w = 0.03 * camera_height
    for k, (T, depth) in enumerate(zip(c2w, depths, strict=True)):
        Pc = backproject(depth, K_depth)[::stride, ::stride]
        nrm = _normals(Pc)
        valid = (Pc[:, :, 2] > 0) & np.isfinite(nrm).all(2)
        valid &= cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        up = nrm[valid] @ T[1, :3]  # y component of the world normal
        h = -(Pc[valid] @ T[1, :3])  # how far below the camera each point is
        h = h[(up > np.cos(np.radians(25))) & (h > 0.5 * camera_height) & (h < 2.0 * camera_height)]
        if len(h) < 200:
            continue
        hist, edges = np.histogram(h, bins=int(1.5 * camera_height / bin_w), range=(0.5 * camera_height, 2.0 * camera_height))
        i = int(np.argmax(hist))
        peak = 0.5 * (edges[i] + edges[i + 1])
        s[k] = camera_height / float(np.median(h[np.abs(h - peak) < 2 * bin_w]))
    known = np.flatnonzero(np.isfinite(s))
    if len(known) < 3:
        return c2w, depths, np.ones(n)
    log_s = np.interp(np.arange(n), known, np.log(s[known]))
    half = window // 2
    log_s = np.array([np.median(log_s[max(0, k - half):k + half + 1]) for k in range(n)])
    s = np.exp(log_s)
    out = [c2w[0].copy()]
    for k in range(1, n):
        T = c2w[k].copy()
        step = c2w[k][:3, 3] - c2w[k - 1][:3, 3]
        T[:3, 3] = out[-1][:3, 3] + np.sqrt(s[k] * s[k - 1]) * step
        out.append(T)
    return out, [d * f for d, f in zip(depths, s, strict=True)], s


def estimate_hfov(frames: list[Frame], pairs: int = 20, gap: int = 2, default: float = 66.0) -> tuple[float, dict]:
    """Horizontal field of view that best explains the motion between frame pairs.

    For each candidate, essential matrices are fitted with MAGSAC on corners tracked between frames `gap` apart; a
    wrong focal length bends the epipolar geometry and fewer corners fit. The mean inlier fraction peaks near the
    true value (96 degrees for the 0.5x camera of the sample walkthroughs). Falls back to `default` when the peak is
    too flat to trust.
    """
    grays = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in frames]
    step = max(1, (len(grays) - gap) // pairs)
    matches = []
    for k in range(0, len(grays) - gap, step):
        p0, p1 = track_points(grays[k], grays[k + gap], max_corners=800)
        if len(p0) >= 60:
            matches.append((p0, p1))
    candidates = np.arange(50.0, 116.0, 5.0)
    if len(matches) < 4:
        return default, {"hfov_deg": default, "hfov_source": "default (too few frame pairs)"}
    score = []
    for hfov in candidates:
        K = intrinsics_guess(grays[0].shape, hfov)
        frac = []
        for p0, p1 in matches:
            E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.0)
            frac.append(0.0 if mask is None else float(mask.mean()))
        score.append(np.mean(frac))
    score = np.array(score)
    i = int(np.argmax(score))
    best = float(candidates[i])
    if 0 < i < len(score) - 1:  # parabola through the peak and its neighbours
        den = score[i - 1] - 2 * score[i] + score[i + 1]
        if den < 0:
            best += 5.0 * 0.5 * (score[i - 1] - score[i + 1]) / den
    if score.max() - score.min() < 0.005:
        return default, {"hfov_deg": default, "hfov_source": "default (no clear peak)"}
    return round(best, 1), {"hfov_deg": round(best, 1), "hfov_source": "self-calibrated",
                            "hfov_scores": {int(c): round(float(v), 4) for c, v in zip(candidates, score, strict=True)}}


def process_video(path: str | Path, fps: float = 10.0, max_frames: int | None = None,
                  progress: Callable[[str, float], None] | None = None, K: np.ndarray | None = None,
                  width: int = 848, model: str | Path | None = None, camera_height: float = CAMERA_HEIGHT,
                  keep_images: bool = False, steady_height: bool = True) -> Walkthrough:
    """A walkthrough with a pose and metric depth for every sampled frame, from the video alone (CPU only).

    K is for frames resized to `width`; without it the field of view is self-calibrated (estimate_hfov).
    steady_height corrects scale drift by assuming the phone stays at about the same height (floor_scale_correction).
    progress, when given, is called as progress(stage, fraction done) with stage "frames", "tracking" or "levelling".
    """
    path = Path(path)
    timings = {}
    t0 = time.perf_counter()
    frames = sample_frames(path, fps=fps, width=width, max_frames=max_frames, K=K,
                           progress=(lambda i, n: progress("frames", i / max(n, 1))) if progress else None)
    if len(frames) < 2:
        raise ValueError(f"{path}: fewer than two frames")
    timings["sample_s"] = round(time.perf_counter() - t0, 2)
    calib = {"hfov_source": "given"}
    if K is None:
        t0 = time.perf_counter()
        hfov, calib = estimate_hfov(frames)
        K = intrinsics_guess(frames[0].image.shape, hfov)
        for f in frames:
            f.K = K.copy()
        timings["calibrate_s"] = round(time.perf_counter() - t0, 2)
    estimator = DepthEstimator(model)
    t0 = time.perf_counter()
    traj = track_walkthrough(frames, estimator,
                             progress=(lambda i, n: progress("tracking", i / n)) if progress else None)
    timings["track_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    c2w, depth, level = level_and_scale(traj.c2w, traj.depth, traj.K_depth, camera_height)
    if steady_height:
        c2w, depth, factors = floor_scale_correction(c2w, depth, traj.K_depth, camera_height)
        c2w, depth, level = level_and_scale(c2w, depth, traj.K_depth, camera_height)
        level["drift_correction_range"] = [round(float(factors.min()), 3), round(float(factors.max()), 3)]
    else:
        factors = np.ones(len(frames))
    timings["level_s"] = round(time.perf_counter() - t0, 2)
    if progress:
        progress("levelling", 1.0)
    size = (frames[0].image.shape[1], frames[0].image.shape[0])
    for f, T, dm, fit, state, factor in zip(frames, c2w, depth, traj.fits, traj.states, factors, strict=True):
        f.c2w = T
        f.depth = dm.astype(np.float32)
        f.K_depth = traj.K_depth.copy()
        f.quality["pose"] = state
        f.quality["scale_correction"] = round(float(factor), 3)
        f.quality["depth_fit"] = None if fit is None else {"a": fit.a, "b": fit.b, "inliers": fit.inliers,
                                                           "rel_err": fit.rel_err}
        if not keep_images:
            f.image = None
    timings["total_s"] = round(sum(timings.values()), 2)
    timings["per_frame_s"] = round(timings["total_s"] / len(frames), 3)
    meta = {"source": "opencv-odometry", "sample_fps": fps, "camera_height_m": camera_height, "timings": timings,
            "tracking": traj.stats, "levelling": level, "calibration": calib, "K": frames[0].K.tolist()}
    return Walkthrough(name=path.stem, video=path, fps=video_info(path)[0], size=size, frames=frames, meta=meta)
