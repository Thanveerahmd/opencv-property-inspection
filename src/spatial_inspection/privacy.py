"""Redact people and private text before an image leaves the device or the server.

Faces are found with YuNet (cv2.FaceDetectorYN) and text regions (letters and documents, screens, photo captions,
name tags) with the PP-OCRv3 DB text detector. Each box is padded, then blurred, pixelated or filled until it cannot
be read. Detection runs on a copy no larger than max_side; boxes are mapped back to full resolution.
"""

from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np

from spatial_inspection import models
from spatial_inspection.ocr import detect_text

FACE_SCORE = 0.6     # lower than the Zoo demo's 0.9: missing a face costs more than blurring a poster
FACE_NMS = 0.3
FACE_PAD = 0.25      # fraction of the box added on every side (hair, ears)
TEXT_PAD = 0.15
METHODS = ("blur", "pixelate", "fill")


@lru_cache(maxsize=8)
def face_detector(size: tuple[int, int], score_threshold: float = FACE_SCORE) -> cv2.FaceDetectorYN:
    """YuNet for one input (width, height), cached."""
    return cv2.FaceDetectorYN.create(str(models.ensure("face_detection_yunet")), "", size, score_threshold, FACE_NMS,
                                     5000)


def detect_faces(img: np.ndarray, max_side: int = 1280, score_threshold: float = FACE_SCORE) -> list[dict]:
    """Faces as dict(box=[x, y, w, h], score, landmarks=5x[x, y]) in image pixels."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    small = img if size == (w, h) else cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    _, faces = face_detector(size, score_threshold).detect(np.ascontiguousarray(small[:, :, :3]))
    if faces is None:
        return []
    sx, sy = w / size[0], h / size[1]
    out = []
    for f in faces:
        box = [f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy]
        marks = [[f[4 + 2 * i] * sx, f[5 + 2 * i] * sy] for i in range(5)]
        out.append({"box": [round(float(v), 1) for v in box], "score": float(f[14]),
                    "landmarks": [[round(float(v), 1) for v in p] for p in marks]})
    return out


def pad_box(box, pad: float, shape: tuple[int, ...]) -> list[int]:
    """[x, y, w, h] grown by `pad` of its smaller side on every side, clipped to the image."""
    x, y, w, h = box
    m = pad * min(w, h)
    H, W = shape[:2]
    x0, y0 = max(0, int(np.floor(x - m))), max(0, int(np.floor(y - m)))
    x1, y1 = min(W, int(np.ceil(x + w + m))), min(H, int(np.ceil(y + h + m)))
    return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]


def obscure(img: np.ndarray, box, method: str = "blur") -> None:
    """Make one [x, y, w, h] region unreadable, in place."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return
    roi = img[y:y + h, x:x + w]
    if method == "blur":
        # sigma near half the smaller side wipes out strokes and facial features; blur twice to flatten residue
        sigma = max(4.0, 0.5 * min(w, h))
        roi[:] = cv2.GaussianBlur(cv2.GaussianBlur(roi, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE), (0, 0),
                                  sigma, borderType=cv2.BORDER_REPLICATE)
    elif method == "pixelate":
        block = max(4, min(w, h) // 4)
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
        roi[:] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif method == "fill":
        roi[:] = cv2.mean(roi)[: roi.shape[2] if roi.ndim == 3 else 1]
    else:
        raise ValueError(f"method must be one of {METHODS}")


def redact(img: np.ndarray, faces: bool = True, text: bool = True, method: str = "blur",
           max_side: int = 960) -> tuple[np.ndarray, dict]:
    """A copy with faces and text regions made unreadable, and a report: counts, padded [x, y, w, h] boxes, timing."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    t0 = time.perf_counter()
    out = img.copy()
    face_boxes, text_boxes = [], []
    if faces:
        face_boxes = [pad_box(f["box"], FACE_PAD, img.shape) for f in detect_faces(img, max(max_side, 1280))]
    if text:
        for quad, _ in detect_text(img, max_side):
            x, y, w, h = cv2.boundingRect(np.round(quad).astype(np.int32))
            text_boxes.append(pad_box([x, y, w, h], TEXT_PAD, img.shape))
    for box in face_boxes + text_boxes:
        obscure(out, box, method)
    report = {"faces": len(face_boxes), "text": len(text_boxes), "face_boxes": face_boxes, "text_boxes": text_boxes,
              "method": method, "ms": round((time.perf_counter() - t0) * 1000, 1)}
    return out, report
