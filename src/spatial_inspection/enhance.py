"""Image helpers for inspection crops: exposure statistics, denoising, local contrast and detail.

enhance_for_review() is meant for the crops a person (or a vision model) looks at, not for the detectors: it denoises
first so that contrast enhancement does not amplify sensor noise, then lifts local contrast in the lightness channel
only (colours stay true, which matters for rust and stains), then brings out fine detail such as hairline cracks.
"""

from __future__ import annotations

import cv2
import numpy as np


def exposure_stats(img: np.ndarray, dark: int = 5, bright: int = 250) -> dict:
    """Mean and spread of the grey level, percentiles, and the fractions of clipped dark and bright pixels."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    n = max(float(hist.sum()), 1.0)
    p = hist / n
    levels = np.arange(256, dtype=np.float64)
    mean = float(p @ levels)
    cdf = np.cumsum(p)
    return {
        "mean": mean,
        "std": float(np.sqrt(p @ (levels - mean) ** 2)),
        "p5": int(np.searchsorted(cdf, 0.05)),
        "p95": int(np.searchsorted(cdf, 0.95)),
        "dark_clipped": float(p[:dark + 1].sum()),
        "bright_clipped": float(p[bright:].sum()),
    }


def clahe_lab(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Local contrast enhancement (CLAHE) on the Lab lightness channel; colour is left alone."""
    if img.ndim == 2:
        return cv2.createCLAHE(clip, (tiles, tiles)).apply(img)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    L = cv2.createCLAHE(clip, (tiles, tiles)).apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, strength: float = 5.0) -> np.ndarray:
    """Non-local means denoising; strength around 3 to 10 for phone video frames."""
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def detail(img: np.ndarray, sigma_s: float = 10.0, sigma_r: float = 0.15) -> np.ndarray:
    """Edge-preserving detail enhancement (cv2.detailEnhance): fine texture and cracks stand out."""
    if img.ndim == 2:
        return cv2.cvtColor(cv2.detailEnhance(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, sigma_s, sigma_r),
                            cv2.COLOR_BGR2GRAY)
    return cv2.detailEnhance(img, None, sigma_s, sigma_r)


def sharpen(img: np.ndarray, amount: float = 0.6, sigma: float = 1.5) -> np.ndarray:
    """Unsharp mask: adds back `amount` of the difference from a Gaussian blur."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def enhance_for_review(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """A clearer copy of an inspection crop: denoised, local contrast lifted, fine detail brought out.

    The amount of each step follows the crop's own statistics: dark or flat crops get more contrast, and very large
    crops skip non-local means (it is slow) in favour of a light bilateral filter.
    """
    if img is None or img.size == 0 or min(img.shape[:2]) < 8:
        return img
    stats = exposure_stats(img)
    big = img.shape[0] * img.shape[1] > max_side * max_side
    out = cv2.bilateralFilter(img, 5, 25, 5) if big else denoise(img, 4.0 if stats["std"] > 20 else 6.0)
    # flat or dark crops need a stronger push to show surface detail
    clip = 3.0 if stats["std"] < 25 or stats["mean"] < 70 else 2.0
    tiles = max(2, min(8, min(img.shape[:2]) // 64))
    out = clahe_lab(out, clip, tiles)
    if img.ndim == 3:
        out = detail(out, 8.0, 0.12)
    else:
        out = sharpen(out, 0.5)
    return out
