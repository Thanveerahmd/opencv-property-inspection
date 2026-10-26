"""Per-frame depth from Depth Anything V2 (small) running in OpenCV's DNN module, made metric by alignment.

The network sees one image and returns relative inverse depth: larger means nearer, but the scale and offset are
unknown and change from frame to frame. Sparse points with known depth (triangulated by the odometry, or carried over
from earlier frames) pin them down: inverse depth = a * disparity + b is fitted robustly per frame, then
depth = 1 / (a * disparity + b).

Depth maps are kept at a modest resolution (504 x 280 for a 16:9 frame), with the camera matrix scaled to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spatial_inspection import models

MODEL_NAME = "depth_anything_v2_small"  # in models.REGISTRY; the file is looked up with models.path
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14  # the ViT works on 14 px patches, so input sides must be multiples of 14
# b / a measured for this model against metric depth on indoor walkthroughs; used only when a frame has too few
# points of its own to tell the offset
DEFAULT_SHIFT = 1.0


def grid_size(image_size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """(width, height) with the longer side `long_side` and both sides multiples of 14, keeping the aspect ratio."""
    w, h = image_size
    s = long_side / max(w, h)
    return max(PATCH, round(w * s / PATCH) * PATCH), max(PATCH, round(h * s / PATCH) * PATCH)


def scale_intrinsics(K: np.ndarray, image_size: tuple[int, int], new_size: tuple[int, int]) -> np.ndarray:
    """Camera matrix for the image resized from image_size to new_size (both (width, height))."""
    sx, sy = new_size[0] / image_size[0], new_size[1] / image_size[1]
    return np.diag([sx, sy, 1.0]) @ K


class DepthEstimator:
    """Depth Anything V2 small (ONNX) on the CPU through cv2.dnn."""

    def __init__(self, model: str | Path | None = None, input_long: int = 518, depth_long: int = 504):
        model = Path(model) if model else models.path(MODEL_NAME)
        if not model.exists():
            raise FileNotFoundError(f"depth model not found: {model}")
        self.net = cv2.dnn.readNet(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_long = input_long
        self.depth_long = depth_long
        self._std = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)

    def depth_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """(width, height) of the depth maps for images of this (width, height)."""
        return grid_size(image_size, self.depth_long)

    def depth_K(self, K: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        return scale_intrinsics(K, image_size, self.depth_size(image_size))

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Relative inverse depth (larger = nearer) at the depth resolution, float32."""
        h, w = img.shape[:2]
        size = grid_size((w, h), self.input_long)
        mean = tuple(255.0 * m for m in IMAGENET_MEAN)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, size, mean=mean, swapRB=True, crop=False)
        blob /= self._std
        self.net.setInput(blob)
        out = self.net.forward()
        disp = out.reshape(out.shape[-2:]).astype(np.float32)
        return cv2.resize(disp, self.depth_size((w, h)), interpolation=cv2.INTER_LINEAR)


@dataclass
class DepthFit:
    """inverse depth = a * disparity + b, from `inliers` of `points` sparse points; rel_err is the median relative
    depth error on the inliers."""

    a: float
    b: float
    inliers: int
    points: int
    rel_err: float

    @property
    def shift(self) -> float:
        return self.b / self.a


def sample_map(m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear samples of a single-channel map at (N, 2) pixel positions."""
    if len(uv) == 0:
        return np.empty(0, np.float32)
    xy = np.ascontiguousarray(uv, np.float32).reshape(-1, 1, 2)
    return cv2.remap(m, xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def _weighted_fit(d: np.ndarray, z: np.ndarray, w: np.ndarray, shift: float | None) -> tuple[float, float]:
    # each point asks for (a * d + b) * z = 1, which weights residuals by relative depth error
    if shift is not None:
        g = (d + shift) * z
        a = float((w * g).sum() / max((w * g * g).sum(), 1e-12))
        return a, a * shift
    A = np.column_stack([d * z, z]) * np.sqrt(w)[:, None]
    y = np.sqrt(w)[:, None]
    ok, sol = cv2.solve(A.astype(np.float64), y.astype(np.float64), flags=cv2.DECOMP_SVD)
    return float(sol[0, 0]), float(sol[1, 0])


def fit_inverse_depth(disp: np.ndarray, z: np.ndarray, shift: float | None = None, threshold: float = 0.08,
                      iterations: int = 200, min_points: int = 8, seed: int = 0) -> DepthFit | None:
    """Robust fit of 1 / z = a * disp + b from sparse points (RANSAC on pairs, then Huber reweighting).

    With `shift` given, b is tied to a (b = a * shift) and only the scale is fitted, which is all a frame whose
    points span a narrow depth range can support. threshold is the inlier bound on relative depth error.
    """
    disp = np.asarray(disp, np.float64)
    z = np.asarray(z, np.float64)
    ok = np.isfinite(disp) & np.isfinite(z) & (z > 0)
    disp, z = disp[ok], z[ok]
    n = len(z)
    if n < min_points:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iterations):
        if shift is None:
            i, j = rng.choice(n, 2, replace=False)
            if abs(disp[i] - disp[j]) < 1e-6:
                continue
            a = (1 / z[i] - 1 / z[j]) / (disp[i] - disp[j])
            b = 1 / z[i] - a * disp[i]
        else:
            i = rng.integers(n)
            a = 1 / (z[i] * (disp[i] + shift))
            b = a * shift
        if a <= 0:
            continue
        count = int((np.abs((a * disp + b) * z - 1) < threshold).sum())
        if count > best_count:
            best, best_count = (a, b), count
    if best is None or best_count < min_points:
        return None
    a, b = best
    for _ in range(5):
        r = np.abs((a * disp + b) * z - 1)
        inl = r < 2 * threshold
        if inl.sum() < min_points:
            break
        w = np.where(r[inl] < threshold / 2, 1.0, (threshold / 2) / np.maximum(r[inl], 1e-9))  # Huber weights
        a, b = _weighted_fit(disp[inl], z[inl], w, shift)
        if a <= 0:
            a, b = best
            break
    pred = a * disp + b
    rel = np.abs(z * pred - 1)
    inl = rel < threshold
    if inl.sum() < min_points:
        return None
    return DepthFit(a=a, b=b, inliers=int(inl.sum()), points=n, rel_err=float(np.median(rel[inl])))


def depth_edges(disp: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Pixels on depth discontinuities, where back-projected points would float between surfaces."""
    log_d = np.log(np.maximum(disp, 1e-6))
    gx = cv2.Sobel(log_d, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(log_d, cv2.CV_32F, 0, 1, ksize=3) / 8
    edges = (np.abs(gx) + np.abs(gy)) > threshold
    return cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def to_metric(disp: np.ndarray, a: float, b: float, max_depth: float | None = None,
              drop_edges: bool = True) -> np.ndarray:
    """Depth along the camera z axis from relative inverse depth and a fit; 0 where it cannot be trusted."""
    inv = a * disp + b
    depth = np.zeros_like(disp, dtype=np.float32)
    ok = inv > (1.0 / max_depth if max_depth else 1e-6)
    depth[ok] = 1.0 / inv[ok]
    if drop_edges:
        depth[depth_edges(disp)] = 0
    return depth


def backproject(depth: np.ndarray, K: np.ndarray, uv: np.ndarray | None = None) -> np.ndarray:
    """Camera-frame 3D points: for the given (N, 2) pixels, or for every pixel as an (H, W, 3) array."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if uv is None:
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return np.dstack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])
    z = sample_map(depth, uv)
    return np.column_stack([(uv[:, 0] - cx) / fx * z, (uv[:, 1] - cy) / fy * z, z])
