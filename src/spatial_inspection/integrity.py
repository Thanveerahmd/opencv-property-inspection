"""Evidence integrity: is this walkthrough one honest, continuous recording of the property? All checks are OpenCV.

- Reused footage: a perceptual hash (cv2.img_hash.PHash) of sampled frames, compared with every earlier walkthrough.
  The same footage submitted for a different property is the classic claim or valuation fraud (high).
- One continuous take: a hard cut, where the colour histogram (cv2.compareHist) changes at once between two
  neighbouring frames, means the video was edited together from separate clips (medium).
- Frozen or looped footage: runs of identical frames, or a stretch that repeats an earlier stretch (medium).
- A recording of a screen: moire from the screen's pixel grid shows as sharp peaks in the frequency domain
  (cv2.dft), and a phone or monitor bezel as a dark border around a bright quadrilateral (high when both show).
- Evidence quality: when most frames are blurred or dark, little can be judged from the video (low).

Signals are advisory: they route the case to a person (REFER) and never reject it on their own.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .scene import Walkthrough

_lock = threading.Lock()
_hasher = cv2.img_hash.PHash_create()


def phash(img: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash (8 bytes) of a BGR image."""
    return _hasher.compute(img)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(cv2.norm(a, b, cv2.NORM_HAMMING))


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def hard_cuts(images: list[np.ndarray], times: list[float], threshold: float = 0.3) -> list[dict]:
    """Points where two neighbouring frames share almost no colour distribution: the video jumps to another scene."""
    cuts = []
    prev = None
    for img, t in zip(images, times, strict=True):
        h = _hist(img)
        if prev is not None:
            corr = float(cv2.compareHist(prev, h, cv2.HISTCMP_CORREL))
            if corr < threshold:
                cuts.append({"t": round(t, 2), "correlation": round(corr, 3)})
        prev = h
    return cuts


def frozen_and_looped(hashes: list[np.ndarray], times: list[float], min_gap_s: float = 2.0, run: int = 5) -> dict:
    """Seconds of identical consecutive frames, and stretches that repeat an earlier stretch frame for frame."""
    frozen = sum(times[i] - times[i - 1] for i in range(1, len(hashes)) if hamming(hashes[i], hashes[i - 1]) == 0)
    n = len(hashes)
    loops = []
    if n > run:
        D = np.array([[hamming(a, b) for b in hashes] for a in hashes])
        for i in range(n - run):
            for j in range(i + 1, n - run):
                if times[j] - times[i] < min_gap_s:
                    continue
                if all(D[i + k, j + k] <= 2 for k in range(run)) and not any(lp["from_t"] == round(times[i - 1], 2)
                                                                            for lp in loops if i > 0):
                    loops.append({"from_t": round(times[i], 2), "repeats_at_t": round(times[j], 2)})
                    break
    return {"frozen_s": round(frozen, 2), "loops": loops[:5]}


def moire_score(img: np.ndarray, size: int = 256) -> float:
    """Prominence of isolated high-frequency peaks in the image spectrum (a screen's pixel grid); about 1 for a scene."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    s = min(size, h, w)
    c = g[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    c = (c - c.mean()) * cv2.createHanningWindow((s, s), cv2.CV_32F)
    spec = cv2.dft(c, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(spec[..., 0], spec[..., 1])
    mag = np.fft.fftshift(np.log1p(mag))
    yy, xx = np.mgrid[:s, :s] - s // 2
    r = np.hypot(yy, xx)
    band = (r > s * 0.12) & (r < s * 0.48)
    vals = mag[band]
    local = cv2.blur(mag, (9, 9))
    excess = (mag - local)[band]
    return float(excess.max() / (vals.std() + 1e-6))


def bezel_score(img: np.ndarray) -> float:
    """Share of the frame's outer border that is near-black and flat, around a brighter quadrilateral."""
    g = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    ring = np.ones_like(g, bool)
    ring[10:-10, 10:-10] = False
    dark = float(np.mean(g[ring] < 30))
    if dark < 0.5:
        return 0.0
    _, bright = cv2.threshold(g, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    quad = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
    frac = cv2.contourArea(c) / g.size
    return dark if len(quad) == 4 and 0.25 < frac < 0.92 else 0.0


def _index_path(root: Path) -> Path:
    return Path(root) / "_index" / "footage.json"


def _load_index(root: Path) -> list[dict]:
    p = _index_path(root)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    except (OSError, ValueError):
        return []


def reused_footage(name: str, prop: str | None, fp: list[str], index: list[dict], max_bits: int = 6,
                   min_share: float = 0.4) -> list[dict]:
    """Earlier walkthroughs whose sampled frames match this one's."""
    mine = [np.frombuffer(bytes.fromhex(h), np.uint8)[None] for h in fp]
    hits = []
    for e in index:
        if e["name"] == name:
            continue
        theirs = [np.frombuffer(bytes.fromhex(h), np.uint8)[None] for h in e["hashes"]]
        if not theirs:
            continue
        share = float(np.mean([min(hamming(a, b) for b in theirs) <= max_bits for a in mine]))
        if share >= min_share:
            hits.append({"name": e["name"], "property": e.get("property"), "share": round(share, 2)})
    return sorted(hits, key=lambda h: -h["share"])


def check(w: Walkthrough, root: Path | None = None, prop: str | None = None, samples: int = 32,
          register: bool = True) -> dict:
    """All integrity checks for a walkthrough whose images are loaded. `root` holds the cross-walkthrough index."""
    frames = [f for f in w.frames if f.image is not None]
    images, times = [f.image for f in frames], [f.t for f in frames]
    hashes = [phash(im) for im in images]
    pick = np.linspace(0, len(frames) - 1, min(samples, len(frames))).round().astype(int) if frames else []
    fp = [hashes[i].tobytes().hex() for i in pick]

    flags: list[dict] = []
    reused: list[dict] = []
    if root is not None:
        with _lock:
            index = _load_index(root)
            reused = reused_footage(w.name, prop, fp, index)
            if register:
                index = [e for e in index if e["name"] != w.name]
                index.append({"name": w.name, "property": prop, "hashes": fp,
                              "at": datetime.now().isoformat(timespec="seconds")})
                p = _index_path(root)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(index[-2000:]), encoding="utf-8")
    for r in reused:
        other = r.get("property")
        if prop and other and other != prop:
            flags.append({"code": "footage_other_property", "level": "high",
                          "message": f"{r['share']:.0%} of the frames match walkthrough {r['name']}, submitted for "
                                     f"a different property ({other})"})
        else:
            flags.append({"code": "footage_resubmitted", "level": "medium",
                          "message": f"{r['share']:.0%} of the frames match an earlier walkthrough ({r['name']})"})

    cuts = hard_cuts(images, times)
    if cuts:
        flags.append({"code": "edited_cuts", "level": "medium",
                      "message": f"{len(cuts)} hard cut(s), the first at {cuts[0]['t']} s: not one continuous take"})
    fl = frozen_and_looped(hashes, times)
    if fl["frozen_s"] >= 1.0:
        flags.append({"code": "frozen_frames", "level": "medium",
                      "message": f"{fl['frozen_s']} s of identical frames (paused or padded video)"})
    if fl["loops"]:
        lp = fl["loops"][0]
        flags.append({"code": "looped_footage", "level": "medium",
                      "message": f"footage from {lp['from_t']} s repeats at {lp['repeats_at_t']} s"})

    probe = [images[i] for i in pick[:: max(1, len(pick) // 12)]] if len(pick) else []
    moire = [moire_score(im) for im in probe]
    bezel = [bezel_score(im) for im in probe]
    moire_share = float(np.mean([m > 3.0 for m in moire])) if moire else 0.0
    bezel_share = float(np.mean([b > 0.5 for b in bezel])) if bezel else 0.0
    if moire_share >= 0.5 and bezel_share >= 0.3:
        flags.append({"code": "screen_recording", "level": "high",
                      "message": "looks like a recording of a screen (pixel-grid moire and a bezel), not the room"})
    elif moire_share >= 0.5 or bezel_share >= 0.5:
        flags.append({"code": "screen_recording_possible", "level": "medium",
                      "message": "parts look like a screen being filmed (" +
                                 ("pixel-grid moire" if moire_share >= 0.5 else "a dark bezel") + ")"})

    verdicts = [f.quality.get("verdict") for f in frames]
    bad = float(np.mean([v == "BAD" for v in verdicts])) if verdicts else 0.0
    if bad >= 0.5:
        flags.append({"code": "poor_quality", "level": "low",
                      "message": f"{bad:.0%} of the frames are blurred, dark or shaking"})

    level = "high" if any(f["level"] == "high" for f in flags) else \
            "medium" if any(f["level"] == "medium" for f in flags) else "low"
    return {"level": level, "flags": flags, "property": prop, "fingerprint": fp,
            "checks": {"frames": len(frames), "reused": reused, "hard_cuts": cuts[:10], "frozen_s": fl["frozen_s"],
                       "loops": fl["loops"], "moire_median": round(float(np.median(moire)), 2) if moire else None,
                       "moire_share": round(moire_share, 2), "bezel_share": round(bezel_share, 2),
                       "bad_frames_share": round(bad, 2)},
            "notes": "Container metadata (capture date, device) is not read: OpenCV does not expose it."}
