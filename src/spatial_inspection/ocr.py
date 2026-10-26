"""Read labels: appliance rating plates, electrical panel labels, water heater stickers.

The PP-OCRv3 DB detector (cv2.dnn.TextDetectionModel_DB) finds text lines as quadrilaterals. Each line is straightened
with a perspective warp to 32 px high at its own aspect ratio (squashing a whole line into the CRNN's nominal 100x32
garbles it), split into words at wide gaps in its ink profile, and every word is read by the CRNN recogniser
(cv2.dnn.TextRecognitionModel) with greedy CTC decoding. The recogniser knows digits and lowercase letters only: no
spaces, points or symbols, so text comes back lowercase and "3.0kW" reads "30kw".

find_years and find_ratings pull facts out of the lines, for checks such as a water heater's age.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

import cv2
import numpy as np

from spatial_inspection import models

# PP-OCRv3 DB settings from the OpenCV Zoo demo
DB_MEAN = (122.67891434, 116.66876762, 104.00698793)
DB_BINARY_THRESHOLD = 0.3
DB_POLYGON_THRESHOLD = 0.5
DB_MAX_CANDIDATES = 200
DB_UNCLIP_RATIO = 2.0
# CRNN EN settings from the OpenCV Zoo demo; class 0 is the CTC blank
VOCABULARY = "0123456789abcdefghijklmnopqrstuvwxyz"
REC_HEIGHT = 32
REC_MIN_WIDTH = 100


def detection_size(shape: tuple[int, ...], max_side: int = 960) -> tuple[int, int]:
    """DB input (width, height) for an image: no larger than max_side, each side a multiple of 32."""
    h, w = shape[:2]
    s = min(1.0, max_side / max(h, w))
    return max(32, int(round(w * s / 32)) * 32), max(32, int(round(h * s / 32)) * 32)


@lru_cache(maxsize=8)
def text_detector(size: tuple[int, int]) -> cv2.dnn.TextDetectionModel_DB:
    """DB text detector for one input (width, height), cached."""
    det = cv2.dnn.TextDetectionModel_DB(str(models.ensure("text_detection_ppocr")))
    det.setBinaryThreshold(DB_BINARY_THRESHOLD).setPolygonThreshold(DB_POLYGON_THRESHOLD)
    det.setMaxCandidates(DB_MAX_CANDIDATES).setUnclipRatio(DB_UNCLIP_RATIO)
    det.setInputParams(1.0 / 255.0, size, DB_MEAN)
    return det


@lru_cache(maxsize=1)
def recogniser() -> cv2.dnn.TextRecognitionModel:
    """CRNN EN recogniser, cached. Its input width is set per word."""
    rec = cv2.dnn.TextRecognitionModel(str(models.ensure("text_recognition_crnn")))
    rec.setDecodeType("CTC-greedy")
    rec.setVocabulary(list(VOCABULARY))
    rec.setInputParams(1.0 / 127.5, (REC_MIN_WIDTH, REC_HEIGHT), (127.5, 127.5, 127.5))
    return rec


def _bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[:, :, :3]


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def detect_text(img: np.ndarray, max_side: int = 960) -> list[tuple[np.ndarray, float]]:
    """Text lines as (quad, score); quad is (4, 2) float32 in image pixels: bottom-left, top-left, top-right,
    bottom-right."""
    quads, scores = text_detector(detection_size(img.shape, max_side)).detect(_bgr(img))
    return [(np.asarray(q, np.float32).reshape(4, 2), float(s)) for q, s in zip(quads, scores, strict=True)]


def straighten(gray: np.ndarray, quad: np.ndarray, height: int = REC_HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Warp a text quad to an upright strip `height` px high at its own aspect ratio; returns (strip, 3x3 warp)."""
    q = np.asarray(quad, np.float32)
    text_h = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    text_w = (np.linalg.norm(q[2] - q[1]) + np.linalg.norm(q[3] - q[0])) / 2
    w = max(8, int(round(height * text_w / max(text_h, 1.0))))
    dst = np.float32([[0, height - 1], [0, 0], [w - 1, 0], [w - 1, height - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    flags = cv2.INTER_AREA if text_h > height else cv2.INTER_LINEAR
    return cv2.warpPerspective(gray, M, (w, height), flags=flags, borderMode=cv2.BORDER_REPLICATE), M


def split_words(strip: np.ndarray) -> list[tuple[int, int]]:
    """Column ranges [x0, x1) of the words in a straightened line, cut at gaps much wider than the letter spacing."""
    h, w = strip.shape
    _, ink = cv2.threshold(strip, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ink.mean() > 0.5:  # ink is the minority class, whichever polarity the label prints in
        ink = 1 - ink
    cols = ink[h // 8: h - h // 8].sum(axis=0) > 0
    xs = np.flatnonzero(cols)
    if len(xs) == 0:
        return [(0, w)]
    # runs of empty columns between the first and last inked column
    edges = np.flatnonzero(np.diff(cols[xs[0]: xs[-1] + 1].astype(np.int8))) + xs[0] + 1
    gaps = list(zip(edges[0::2], edges[1::2], strict=False))  # (start, end) of each empty run
    if not gaps:
        return [(0, w)]
    lengths = np.array([e - s for s, e in gaps])
    cut = max(0.2 * h, 2.0 * float(np.median(lengths)))
    words, start = [], 0
    for (s, e), n in zip(gaps, lengths, strict=True):
        if n >= cut:
            words.append((start, int(s + min(h // 4, n // 2))))
            start = int(e - min(h // 4, n // 2))
    words.append((start, w))
    return words


def _ctc(strip: np.ndarray) -> list[tuple[str, float, float]]:
    """Greedy CTC over one upright strip: (character, probability, column in the strip) for each emitted character."""
    h, w = strip.shape[:2]
    if h != REC_HEIGHT:
        w = max(1, int(round(w * REC_HEIGHT / h)))
        strip = cv2.resize(strip, (w, REC_HEIGHT), interpolation=cv2.INTER_AREA)
    # the CNN downsamples width by 4; pad short strips so there are enough CTC steps
    target = max(32, -(-w // 4) * 4)
    left = (target - w) // 2
    if target > w:
        strip = cv2.copyMakeBorder(strip, 0, 0, left, target - w - left, cv2.BORDER_REPLICATE)
    rec = recogniser()
    rec.setInputSize((target, REC_HEIGHT))
    out = rec.predict(np.ascontiguousarray(strip))
    logits = np.asarray(out[0] if isinstance(out, (list, tuple)) else out, np.float32).reshape(-1, len(VOCABULARY) + 1)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    best = p.argmax(axis=1)
    # collapse repeats, drop blanks (class 0)
    emit = np.flatnonzero((best != 0) & np.r_[True, best[1:] != best[:-1]])
    step = target / len(best)
    return [(VOCABULARY[best[t] - 1], float(p[t, best[t]]), (t + 0.5) * step - left) for t in emit]


def recognise(strip: np.ndarray) -> tuple[str, float]:
    """Text and confidence (mean probability of the emitted characters) for one upright strip of text."""
    chars = _ctc(strip)
    return "".join(c for c, _, _ in chars), float(np.mean([p for _, p, _ in chars])) if chars else 0.0


def read_text(img: np.ndarray, max_side: int = 960, min_confidence: float = 0.0) -> list[dict]:
    """Every text line in an image, top to bottom: dict(text, confidence, quad, words). Words are separated by
    spaces; confidence is the mean probability of the characters; quad is a (4, 2) list in image pixels
    (bottom-left, top-left, top-right, bottom-right)."""
    gray = _gray(img)
    lines = []
    for quad, _ in detect_text(img, max_side):
        strip, M = straighten(gray, quad)
        chars = _ctc(strip)
        if not chars:
            continue
        spans = split_words(strip)
        cuts = [(a[1] + b[0]) / 2 for a, b in zip(spans[:-1], spans[1:], strict=True)]
        per_word: list[list[tuple[str, float]]] = [[] for _ in spans]
        for c, prob, x in chars:
            per_word[int(np.searchsorted(cuts, x))].append((c, prob))
        Minv = np.linalg.inv(M)
        words = []
        for (x0, x1), wc in zip(spans, per_word, strict=True):
            if not wc:
                continue
            corners = np.float32([[x0, REC_HEIGHT - 1], [x0, 0], [x1 - 1, 0], [x1 - 1, REC_HEIGHT - 1]])
            wq = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Minv).reshape(4, 2)
            words.append({"text": "".join(c for c, _ in wc), "confidence": float(np.mean([p for _, p in wc])),
                          "quad": wq.round(1).tolist()})
        conf = float(np.mean([p for _, p, _ in chars]))
        if conf < min_confidence:
            continue
        lines.append({"text": " ".join(wd["text"] for wd in words), "confidence": conf,
                      "quad": quad.round(1).tolist(), "words": words})
    lines.sort(key=lambda d: (min(p[1] for p in d["quad"]), min(p[0] for p in d["quad"])))
    return lines


# ---------------------------------------------------------------------------------------------------------------
# facts from recognised lines

# letters the recogniser confuses with digits, fixed only between digits ("2o14" -> "2014") or next to one for "o"
_BETWEEN_DIGITS = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "z": "2", "b": "8"})


def normalise(text: str) -> str:
    """Lowercase, with digit look-alikes inside numbers corrected ("2o14" -> "2014", "23ov" -> "230v")."""
    t = text.lower()
    for _ in range(3):
        fixed = re.sub(r"(?<=\d)[oilszb](?=\d)", lambda m: m.group().translate(_BETWEEN_DIGITS), t)
        fixed = re.sub(r"(?<=\d)o|o(?=\d)", "0", fixed)
        if fixed == t:
            break
        t = fixed
    return t


def _texts(lines: Iterable[str | dict] | str) -> list[str]:
    if isinstance(lines, str):
        lines = [lines]
    return [normalise(x["text"] if isinstance(x, dict) else str(x)) for x in lines]


_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d|2030)(?!\d)")


def find_years(lines: Iterable[str | dict] | str) -> list[int]:
    """Four-digit years from 1950 to 2030 standing on their own (not inside a longer number), sorted, unique."""
    return sorted({int(m) for t in _texts(lines) for m in _YEAR.findall(t)})


_NUM = r"(?<![a-z\d.,])(\d+(?:[.,]\d+)?)"
_RATINGS = [
    ("voltage", "V", re.compile(_NUM + r"(?:\s*[-~/]\s*(\d+(?:[.,]\d+)?))?\s*(?:v|vac|volts?)(?![a-z])")),
    ("frequency", "Hz", re.compile(_NUM + r"(?:\s*/\s*(\d+))?\s*hz(?![a-z])")),
    ("power", "kW", re.compile(_NUM + r"()\s*(?:kw)(?![a-z])")),
    ("power", "W", re.compile(_NUM + r"()\s*(?:w|watts?)(?![a-z])")),
    ("current", "A", re.compile(_NUM + r"()\s*(?:a|amps?)(?![a-z])")),
    ("volume", "L", re.compile(_NUM + r"()\s*(?:l|ltrs?|litres?|liters?)(?![a-z])")),
    ("pressure", "bar", re.compile(_NUM + r"()\s*bar(?![a-z])")),
    ("pressure", "kPa", re.compile(_NUM + r"()\s*kpa(?![a-z])")),
]


def find_ratings(lines: Iterable[str | dict] | str) -> list[dict]:
    """Electrical and capacity ratings, e.g. "230V", "220-240v", "50Hz", "3kW", "16A", "80L", "6 bar", as
    dict(kind, value, unit, text); ranges also carry `upper`."""
    found = []
    for t in _texts(lines):
        for kind, unit, rx in _RATINGS:
            for m in rx.finditer(t):
                item = {"kind": kind, "value": float(m.group(1).replace(",", ".")), "unit": unit, "text": m.group()}
                if m.group(2):
                    item["upper"] = float(m.group(2).replace(",", "."))
                found.append(item)
    return found
