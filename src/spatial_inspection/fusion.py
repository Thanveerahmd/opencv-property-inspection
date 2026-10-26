"""Fuse the walkthrough's depth maps into one 3D model with OpenCV 5's TSDF volume (cv2.Volume, hash-TSDF).

The volume gives a clean surface point cloud with normals. Colours come from projecting each point into the frame
that saw it most squarely, with cv2.projectPoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .scene import Walkthrough


@dataclass
class Fused:
    points: np.ndarray    # (N, 3) model frame, metres
    normals: np.ndarray   # (N, 3)
    colors: np.ndarray    # (N, 3) BGR uint8
    voxel: float
    frames_used: int


def fuse(w: Walkthrough, voxel: float = 0.03, max_depth: float = 5.0, every: int = 1) -> Fused:
    """Integrate every frame that has a pose and depth into a hash-TSDF volume and extract the surface."""
    frames = [f for f in w.frames[::every] if f.depth is not None and f.c2w is not None]
    if not frames:
        raise ValueError("no frames with depth and pose to fuse")
    h, wd = frames[0].depth.shape
    K = frames[0].K_depth.astype(np.float32)
    s = cv2.VolumeSettings(cv2.VolumeType_HashTSDF)
    s.setIntegrateWidth(wd)
    s.setIntegrateHeight(h)
    s.setRaycastWidth(wd)
    s.setRaycastHeight(h)
    s.setCameraIntegrateIntrinsics(K)
    s.setCameraRaycastIntrinsics(K)
    s.setVoxelSize(voxel)
    s.setTsdfTruncateDistance(4 * voxel)
    s.setDepthFactor(1.0)            # depth already in metres, float32
    s.setMaxDepth(max_depth)
    volume = cv2.Volume(cv2.VolumeType_HashTSDF, s)
    for f in frames:
        depth = np.ascontiguousarray(f.depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0
        volume.integrate(depth, f.c2w.astype(np.float32))
    pts, nrm = volume.fetchPointsNormals()
    # the volume returns 4-component vectors (x, y, z, padding)
    pts = np.asarray(pts, np.float32)
    nrm = np.asarray(nrm, np.float32)
    pts = pts.reshape(-1, pts.shape[-1])[:, :3]
    nrm = nrm.reshape(-1, nrm.shape[-1])[:, :3]
    keep = np.isfinite(pts).all(1) & np.isfinite(nrm).all(1)
    pts, nrm = pts[keep], nrm[keep]
    return Fused(points=pts, normals=nrm, colors=colorize(w, pts, nrm), voxel=voxel, frames_used=len(frames))


def colorize(w: Walkthrough, pts: np.ndarray, nrm: np.ndarray, max_frames: int = 40) -> np.ndarray:
    """Colour each point from the frame that sees it most squarely and closely."""
    cols = np.full((len(pts), 3), 128, np.uint8)
    best = np.full(len(pts), -np.inf)
    frames = [f for f in w.frames if f.image is not None and f.c2w is not None]
    if not frames or len(pts) == 0:      # OpenCV 5 projectPoints returns None for no points
        return cols
    step = max(1, len(frames) // max_frames)
    for f in frames[::step]:
        w2c = np.linalg.inv(f.c2w)
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        uv, _ = cv2.projectPoints(pts.astype(np.float64), rvec, w2c[:3, 3], f.K, None)
        uv = uv.reshape(-1, 2)
        h, wd = f.image.shape[:2]
        ok = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < wd - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        ray = pts - f.c2w[:3, 3]
        dist = np.linalg.norm(ray, axis=1) + 1e-6
        score = np.abs(np.sum(ray / dist[:, None] * nrm, axis=1)) / dist
        better = ok & (score > best)
        if not better.any():
            continue
        u = uv[better, 0].astype(int)
        v = uv[better, 1].astype(int)
        cols[better] = f.image[v, u]
        best[better] = score[better]
    return cols


def save_ply(path, fused: Fused) -> None:
    """Write the coloured point cloud with normals (cv2.savePointCloud writes PLY)."""
    # savePointCloud wants (N, 1, 3) vectors
    rgb = (fused.colors[:, ::-1].astype(np.float32) / 255.0).reshape(-1, 1, 3)
    cv2.savePointCloud(str(path), fused.points.reshape(-1, 1, 3).astype(np.float32),
                       fused.normals.reshape(-1, 1, 3).astype(np.float32), rgb)


def export_web_points(path, fused: Fused, max_points: int = 300_000) -> int:
    """Compact binary for the web viewer: uint32 count, then float32 xyz, then uint8 rgb."""
    idx = np.arange(len(fused.points))
    if len(idx) > max_points:
        idx = np.random.default_rng(0).choice(idx, max_points, replace=False)
    with open(path, "wb") as fh:
        fh.write(np.uint32(len(idx)).tobytes())
        fh.write(fused.points[idx].astype(np.float32).tobytes())
        fh.write(fused.colors[idx, ::-1].astype(np.uint8).tobytes())
    return len(idx)
