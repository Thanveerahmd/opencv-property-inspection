"""Is this candidate an object rather than damage? MobileNetV2 (OpenCV Zoo, ImageNet) through cv2.dnn.

The defect detectors look for colour, texture and line patterns, and household objects have those too: a brown
plastic chair reads as rust, a wall clock's rim as a stain, a cable as a crack. A small ImageNet classifier knows
those objects, so a candidate whose crop it recognises as furniture, a fitting or an appliance is ruled out, with the
object's name as the reason. Damage itself is not an ImageNet class, so the classifier is only ever used to say no.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models

# ImageNet classes that are things found in or on a room, not part of its walls, floor or ceiling
HOUSEHOLD = {
    409: "analog clock", 412: "bin", 421: "banister", 423: "chair", 435: "bathtub", 446: "binder", 453: "bookcase",
    487: "phone", 492: "chest", 493: "chest of drawers", 495: "cabinet", 508: "keyboard", 516: "cradle", 520: "cot",
    526: "desk", 527: "computer", 528: "telephone", 530: "digital clock", 532: "dining table", 534: "dishwasher",
    539: "doormat", 545: "electric fan", 548: "entertainment unit", 553: "file cabinet", 556: "fire screen",
    559: "folding chair", 564: "bed", 598: "home theatre", 600: "hook", 611: "jigsaw puzzle", 619: "lampshade",
    620: "laptop", 632: "loudspeaker", 648: "medicine cabinet", 651: "microwave", 664: "monitor", 669: "mosquito net",
    703: "bench", 708: "pedestal", 721: "pillow", 728: "plastic bag", 729: "plate rack", 741: "rug", 750: "quilt",
    753: "radiator", 760: "refrigerator", 761: "remote control", 765: "rocking chair", 782: "screen", 789: "shoji screen",
    794: "curtain", 799: "sliding door", 811: "heater", 818: "spotlight", 827: "stove", 831: "sofa", 844: "switch",
    846: "table lamp", 851: "television", 854: "curtain", 857: "chair", 859: "toaster", 861: "toilet seat",
    868: "tray", 876: "tub", 882: "vacuum cleaner", 883: "vase", 892: "wall clock", 894: "wardrobe", 896: "washbasin",
    897: "washing machine", 904: "window screen", 905: "window blind", 917: "picture", 921: "picture",
}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_net: cv2.dnn.Net | None = None


def available() -> bool:
    return models.available("image_classification_mobilenet")


def _model() -> cv2.dnn.Net:
    global _net
    if _net is None:
        _net = cv2.dnn.readNet(str(models.ensure("image_classification_mobilenet")))
    return _net


def classify(crop: np.ndarray) -> np.ndarray:
    """ImageNet class probabilities (1000) for a BGR crop: resize to 256, centre 224, ImageNet normalisation."""
    img = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)[16:240, 16:240]
    img = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    net = _model()
    net.setInput(cv2.dnn.blobFromImage(img))
    logits = net.forward().ravel().astype(np.float64)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def household_object(crop: np.ndarray, min_top: float = 0.2, min_mass: float = 0.35) -> dict | None:
    """The household object a crop shows, if the classifier is fairly sure it shows one; else None.

    Sure means the most likely class is a household object with at least `min_top` probability, or the household
    classes together hold at least `min_mass`."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    p = classify(crop)
    idx = np.fromiter(HOUSEHOLD, int)
    mass = float(p[idx].sum())
    top = int(np.argmax(p))
    best = int(idx[np.argmax(p[idx])])
    if (top in HOUSEHOLD and p[top] >= min_top) or mass >= min_mass:
        return {"object": HOUSEHOLD[best], "p": round(float(p[best]), 3), "household_mass": round(mass, 3)}
    return None
