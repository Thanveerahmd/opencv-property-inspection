"""Camera motion between two frames with OpenCV 5.

Corners are tracked with pyramidal Lucas-Kanade optical flow and kept only when tracking back lands where they
started. An essential matrix fitted with MAGSAC rejects the remaining outliers, and recoverPose gives the rotation and
the direction of travel. Two views cannot tell the distance travelled, so the translation is a unit vector.
"""

from dataclasses import dataclass

import cv2
import numpy as np

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


@dataclass
class RelativePose:
    """Maps points from the first camera's frame to the second's: x2 = R @ x1 + t (t has unit length)."""

    R: np.ndarray
    t: np.ndarray
    inliers: int
    tracked: int

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0])))


def intrinsics_guess(shape: tuple[int, ...], hfov_deg: float = 66.0) -> np.ndarray:
    """Camera matrix for an image of this shape, assuming a typical phone main camera field of view."""
    h, w = shape[:2]
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def track_points(prev_gray: np.ndarray, gray: np.ndarray, max_corners: int = 1500,
                 max_backtrack_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Corners in the first image and where they moved to in the second, as two (N, 2) arrays."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    if pts is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK_PARAMS)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK_PARAMS)
    backtrack = np.linalg.norm(pts - back, axis=2).ravel()
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (backtrack < max_backtrack_px)
    return pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]


def relative_pose(p0: np.ndarray, p1: np.ndarray, K: np.ndarray, threshold_px: float = 1.0) -> RelativePose | None:
    """Rotation and direction of travel from matched points, or None when there is too little to go on."""
    if len(p0) < 8:
        return None
    E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=threshold_px)
    if E is None or E.shape[0] < 3:
        return None
    # degenerate configurations can return several stacked solutions; the first is the best scored
    E = E[:3]
    inliers, R, t, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    t = t.ravel()
    norm = np.linalg.norm(t)
    if inliers < 8 or norm == 0:
        return None
    return RelativePose(R=R, t=t / norm, inliers=int(inliers), tracked=len(p0))


def frame_pose(prev_bgr: np.ndarray, bgr: np.ndarray, K: np.ndarray) -> RelativePose | None:
    """Camera motion between two colour frames of a walkthrough."""
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p0, p1 = track_points(prev_gray, gray)
    return relative_pose(p0, p1, K)
