"""Open-vocabulary boxes with Grounding DINO (IDEA-Research/grounding-dino-tiny, Apache-2.0).

Two jobs: naming the furniture the 3D engine separated from the room structure (engine3d/v1prod/labels.py), and a
second, semantic opinion on the defect candidates OpenCV finds on the walls (walkthrough.py). The model runs on the
GPU when there is one; without torch/transformers every call returns None and callers fall back to OpenCV only.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_ID = os.environ.get("SI_GDINO_MODEL", "IDEA-Research/grounding-dino-tiny")
_model = None


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("SI_GDINO", "on").lower() not in ("0", "off", "no")


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        net = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(dev).eval()
        _model = (proc, net, dev)
    return _model


def release() -> None:
    """Free the GPU memory before the next heavy step."""
    global _model
    if _model is not None:
        import torch
        _model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def detect(rgb: np.ndarray, labels: list[str], box_threshold: float = 0.3, text_threshold: float = 0.25,
           max_side: int = 800) -> list[dict] | None:
    """Boxes for the given phrases on one RGB image: [{label, score, box: [x0, y0, x1, y1]}] in input pixels."""
    if not available():
        return None
    import torch
    from PIL import Image
    proc, net, dev = _load()
    H, W = rgb.shape[:2]
    s = min(1.0, max_side / max(H, W))
    img = Image.fromarray(rgb)
    if s < 1:
        img = img.resize((int(W * s), int(H * s)), Image.BILINEAR)
    text = ". ".join(labels) + "."
    inputs = proc(images=img, text=text, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = net(**inputs)
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=box_threshold,
                                                      text_threshold=text_threshold, target_sizes=[(H, W)])[0]
    names = res.get("text_labels") or res.get("labels")
    dets = []
    for box, score, name in zip(res["boxes"].tolist(), res["scores"].tolist(), names, strict=False):
        name = name if isinstance(name, str) else labels[int(name)] if int(name) < len(labels) else ""
        if not name:
            continue
        x0, y0, x1, y1 = box
        dets.append({"label": name.strip(), "score": round(float(score), 3),
                     "box": [max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)]})
    return dets
