"""Frames from a walkthrough video, each with a quality verdict.

Frames are sampled at a target rate and resized to a working width. Each sampled frame is scored on three things
that decide whether it is worth tracking and inspecting:

- sharpness: variance of the Laplacian, measured on a copy resized to 640 px wide so the number does not depend on
  the video's resolution. Motion blur and defocus pull it down. A plain white wall scores low as well, and is just
  as useless for tracking, so a low score reads "blurred or featureless"; the verdict also compares each frame with
  the walkthrough's own median.
- exposure: mean brightness and the fractions of pixels clipped to black (< 16) or white (> 245).
- motion: median DIS optical flow magnitude to the previous sampled frame, in pixels at the working width. Large
  jumps break tracking between samples.

Thresholds (applied in `verdict`):

| metric                      | POOR below / above | BAD below / above |
| ---                         | ---                | ---               |
| sharpness                   | < 40               | < 8               |
| sharpness / video median    | < 0.35             | < 0.15            |
| mean brightness             | < 45 or > 215      | < 20 or > 240     |
| clipped black fraction      | > 0.35             | > 0.70            |
| clipped white fraction      | > 0.20             | > 0.50            |
| motion, fraction of width   | > 0.06             | > 0.15            |

Dark frames (mean below 70) can be brightened with CLAHE on the lightness channel before anything else sees them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from spatial_inspection.scene import Frame
from spatial_inspection.tracking import intrinsics_guess

ANALYSIS_WIDTH = 640    # sharpness is measured at this width
FLOW_WIDTH = 320        # motion is measured at this width, then scaled back to the working width
DARK_MEAN = 70          # frames darker than this get CLAHE when enhancement is on

POOR = dict(sharpness=40.0, sharpness_rel=0.35, mean_low=45.0, mean_high=215.0, black=0.35, white=0.20, motion=0.06)
BAD = dict(sharpness=8.0, sharpness_rel=0.15, mean_low=20.0, mean_high=240.0, black=0.70, white=0.50, motion=0.15)
VERDICTS = ("GOOD", "POOR", "BAD")


def _gray(img: np.ndarray, width: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    if w == width:
        return gray
    return cv2.resize(gray, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def enhance_dark(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Brighten a dark frame with CLAHE on the L channel, leaving colour alone."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class MotionMeter:
    """Median DIS optical flow between consecutive frames, in pixels at the frames' own width."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev: np.ndarray | None = None

    def __call__(self, img: np.ndarray) -> float | None:
        small = _gray(img, FLOW_WIDTH)
        prev, self.prev = self.prev, small
        if prev is None or prev.shape != small.shape:
            return None
        flow = self.dis.calc(prev, small, None)
        mag = np.linalg.norm(flow.reshape(-1, 2), axis=1)
        return float(np.median(mag)) * img.shape[1] / FLOW_WIDTH


def measure(img: np.ndarray, motion_px: float | None = None) -> dict:
    """Sharpness, exposure and (when given) motion for one frame. The verdict needs the whole video; see `verdict`."""
    gray = _gray(img, ANALYSIS_WIDTH)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel() / gray.size
    q = {
        "sharpness": round(sharp, 2),
        "brightness": round(float(gray.mean()), 2),
        "black": round(float(hist[:16].sum()), 4),
        "white": round(float(hist[246:].sum()), 4),
        "motion_px": None if motion_px is None else round(motion_px, 2),
        "width": img.shape[1],
    }
    return q


def verdict(q: dict, median_sharpness: float | None = None) -> tuple[str, list[str]]:
    """GOOD / POOR / BAD for a measured frame and the reasons for anything below GOOD."""
    rel = None if not median_sharpness else q["sharpness"] / median_sharpness
    motion = None if q.get("motion_px") is None else q["motion_px"] / q["width"]
    for level, th in (("BAD", BAD), ("POOR", POOR)):
        reasons = []
        if q["sharpness"] < th["sharpness"]:
            reasons.append("blurred or featureless")
        if rel is not None and rel < th["sharpness_rel"]:
            reasons.append("blurred for this video")
        if q["brightness"] < th["mean_low"] or q["black"] > th["black"]:
            reasons.append("dark")
        if q["brightness"] > th["mean_high"] or q["white"] > th["white"]:
            reasons.append("overexposed")
        if motion is not None and motion > th["motion"]:
            reasons.append("fast motion")
        if reasons:
            return level, reasons
    return "GOOD", []


def assess(img: np.ndarray, prev: np.ndarray | None = None, median_sharpness: float | None = None) -> dict:
    """Measure and judge a single frame, optionally against the frame before it."""
    motion = None
    if prev is not None:
        meter = MotionMeter()
        meter(prev)
        motion = meter(img)
    q = measure(img, motion)
    q["verdict"], q["reasons"] = verdict(q, median_sharpness)
    return q


def video_info(path: str | Path) -> tuple[float, int, tuple[int, int]]:
    """Frame rate, frame count and (width, height) of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return fps, n, size


def sample_frames(path: str | Path, fps: float = 10.0, width: int = 848, max_frames: int | None = None,
                  K: np.ndarray | None = None, enhance: bool = True, start_s: float = 0.0,
                  progress: Callable[[int, int], None] | None = None) -> list[Frame]:
    """Every n-th frame (n = round(video fps / fps)), resized to `width`, with quality measured and judged.

    K is for the resized image; when not given it is guessed from the image size (see tracking.intrinsics_guess).
    """
    src_fps, n_total, _ = video_info(path)
    step = max(1, round(src_fps / fps))  # a whole number of source frames, so samples are evenly spaced
    cap = cv2.VideoCapture(str(path))
    first = round(start_s * src_fps)
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[Frame] = []
    meter = MotionMeter()
    want = first
    n = first
    expected = max(1, int((n_total - first) / step)) if n_total > 0 else 0
    if max_frames:
        expected = min(expected, max_frames)
    while max_frames is None or len(frames) < max_frames:
        if not cap.grab():
            break
        if n == want:
            ok, img = cap.retrieve()
            if not ok:
                break
            h, w = img.shape[:2]
            if w != width:
                img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            q = measure(img)
            if enhance and q["brightness"] < DARK_MEAN:
                img = enhance_dark(img)
                q["clahe"] = True
            q["motion_px"] = None if (m := meter(img)) is None else round(m, 2)
            frames.append(Frame(index=len(frames), video_frame=n, t=n / src_fps,
                                K=K if K is not None else intrinsics_guess(img.shape), image=img, quality=q))
            want = first + len(frames) * step
            if progress:
                progress(len(frames), expected)
        n += 1
    cap.release()
    judge(frames)
    return frames


def judge(frames: list[Frame]) -> None:
    """Set each measured frame's verdict, comparing its sharpness with the walkthrough's median."""
    if not frames:
        return
    median = float(np.median([f.quality["sharpness"] for f in frames]))
    for f in frames:
        f.quality["verdict"], f.quality["reasons"] = verdict(f.quality, median)


def assess_frames(frames: list[Frame]) -> None:
    """Measure and judge frames that already have images (for example after Walkthrough.load_images).

    Other entries in each frame's quality dict (pose state, depth fit) are kept.
    """
    meter = MotionMeter()
    measured = []
    for f in frames:
        if f.image is None:
            meter.prev = None
            continue
        motion = meter(f.image)
        q = measure(f.image, motion)
        q["clahe"] = f.quality.get("clahe", False)
        f.quality.update(q)
        measured.append(f)
    judge(measured)
