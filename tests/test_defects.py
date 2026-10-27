import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial_inspection import defects
from spatial_inspection.defects import Candidate, detect_all, draw, measure
from spatial_inspection.enhance import clahe_lab, enhance_for_review, exposure_stats

DATA = Path(__file__).parent / "data"
DEBUG = Path(__file__).parents[1] / "data" / "debug" / "defects"
# the sample defect photos are not published with the code; the tests that need them skip without them
photos = pytest.mark.skipif(not (DATA / "cracked_wall_1.jpg").exists(), reason="sample defect photos not present")

# sample image -> the kind a reviewer would expect among the three best candidates
SAMPLES = {
    "ceiling_crack_1": "crack",
    "cracked_floor_tile_2": "crack",
    "cracked_wall_1": "crack",
    "peeling_paint_ceiling_2": "peeling",
    "rusty_door_lock_2": "rust",
    "water_damaged_wall_1": "stain",
}


def plain_wall(h=480, w=848, colour=(196, 204, 210), noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), colour, np.float32) + rng.normal(0, noise, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


@photos
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sample_finds_expected_kind(name):
    img = cv2.imread(str(DATA / f"{name}.jpg"))
    assert img is not None

    found = detect_all(img)

    assert found, name
    assert [c.score for c in found] == sorted((c.score for c in found), reverse=True)
    assert SAMPLES[name] in [c.kind for c in found[:3]], [(c.kind, round(c.score, 2)) for c in found[:5]]
    for c in found:
        assert 0.0 <= c.score <= 1.0
        assert c.mask.shape == img.shape[:2]
        json.dumps(c.features)
        json.dumps(c.as_dict())
    if DEBUG.parent.exists():
        DEBUG.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(DEBUG / f"{name}.jpg"), draw(img, found[:8]))


@pytest.mark.parametrize("colour", [(196, 204, 210), (150, 180, 200), (90, 95, 100)])
def test_plain_wall_has_no_confident_candidate(colour):
    found = detect_all(plain_wall(colour=colour))
    assert all(c.score <= 0.5 for c in found), [(c.kind, round(c.score, 2)) for c in found]


def test_plain_wall_with_uneven_light_has_no_confident_candidate():
    img = plain_wall().astype(np.float32)
    ramp = np.linspace(0.75, 1.05, img.shape[1], dtype=np.float32)
    img = np.clip(img * ramp[None, :, None], 0, 255).astype(np.uint8)
    found = detect_all(img)
    assert all(c.score <= 0.5 for c in found), [(c.kind, round(c.score, 2)) for c in found]


def test_dark_hairline_on_a_wall_is_a_crack():
    img = plain_wall()
    pts = np.array([[200, 100], [260, 180], [300, 260], [330, 330], [390, 420]], np.int32)
    cv2.polylines(img, [pts], False, (110, 112, 115), 2, cv2.LINE_AA)

    found = detect_all(img)

    assert found[0].kind == "crack"
    assert found[0].score > 0.5
    length = sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    assert abs(found[0].features["length_px"] - length) < 0.15 * length


def test_speckle_cluster_is_mould():
    img = plain_wall()
    rng = np.random.default_rng(3)
    for _ in range(250):
        x, y = rng.normal(500, 45), rng.normal(240, 35)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), (60, 70, 62), -1, cv2.LINE_AA)

    found = detect_all(img)

    mould = [c for c in found if c.kind == "mould"]
    assert mould and mould[0].score > 0.5
    x, y, w, h = mould[0].bbox
    assert x < 500 < x + w and y < 240 < y + h


def line_candidate(shape, p0, p1, thickness=1):
    mask = np.zeros(shape, np.uint8)
    cv2.line(mask, p0, p1, 255, thickness)
    return Candidate("crack", 0.9, cv2.boundingRect(cv2.findNonZero(mask)), mask)


def test_measure_line_on_a_fronto_parallel_wall():
    # 200 px at 2 m with fx = 500 is 0.8 m; the depth map is a quarter of the image size
    h, w = 480, 640
    c = line_candidate((h, w), (200, 240), (399, 240))
    depth = np.full((h // 4, w // 4), 2.0, np.float32)
    K_depth = np.array([[125.0, 0, 80], [0, 125.0, 60], [0, 0, 1]])

    m = measure(c, depth, K_depth, (w, h))

    assert m["length_px"] == pytest.approx(200, abs=1)
    assert m["length_m"] == pytest.approx(0.8, rel=0.01)
    assert m["width_mm"] == pytest.approx(4.0, rel=0.05)
    assert m["depth_m"] == pytest.approx(2.0, rel=1e-3)
    assert m["tilt_deg"] < 1


def test_measure_line_on_a_tilted_wall():
    # wall turned 40 degrees about the vertical axis, 2.5 m away at the image centre
    h, w = 480, 640
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    t = np.radians(40)
    n = np.array([np.sin(t), 0, -np.cos(t)])
    d = n @ np.array([0, 0, 2.5])
    v, u = np.mgrid[0:h, 0:w]
    rays = np.dstack([(u - 320) / 500, (v - 240) / 500, np.ones((h, w))])
    depth = (d / (rays @ n)).astype(np.float32)
    c = line_candidate((h, w), (220, 300), (420, 300))

    m = measure(c, depth, K, (w, h))

    P = [np.array([(x - 320) / 500, (300 - 240) / 500, 1]) * depth[300, x] for x in (220, 420)]
    assert m["length_m"] == pytest.approx(np.linalg.norm(P[1] - P[0]), rel=0.02)
    assert m["tilt_deg"] == pytest.approx(40, abs=1)


def test_measure_area_of_a_patch():
    h, w = 480, 640
    mask = np.zeros((h, w), np.uint8)
    mask[200:250, 300:400] = 255
    c = Candidate("stain", 0.7, (300, 200, 100, 50), mask)
    depth = np.full((h, w), 2.0, np.float32)
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])

    m = measure(c, depth, K, (w, h))

    assert m["area_m2"] == pytest.approx(5000 * (2 / 500) ** 2, rel=0.01)
    assert m["length_m"] == pytest.approx(0.4, rel=0.02)


def test_measure_without_depth_gives_pixels_only():
    c = line_candidate((240, 320), (10, 20), (110, 20))
    m = measure(c, None, None, (320, 240))
    assert m["length_px"] == pytest.approx(100, abs=1)
    assert m["length_m"] is None and m["area_m2"] is None


@photos
def test_detect_all_handles_crops_and_odd_inputs():
    img = cv2.imread(str(DATA / "cracked_wall_1.jpg"))
    for crop in (img[1100:1500, 1150:1450], img[:30, :30], img[:5, :5], cv2.cvtColor(img[:300, :300], cv2.COLOR_BGR2GRAY)):
        found = detect_all(crop)
        for c in found:
            assert c.mask.shape == crop.shape[:2]
            json.dumps(c.features)
    assert detect_all(np.zeros((0, 0, 3), np.uint8)) == []
    with pytest.raises(ValueError):
        detect_all(img[:100, :100], kinds=["dust"])


@photos
def test_crop_of_a_found_crack_confirms_it():
    img = cv2.imread(str(DATA / "cracked_wall_1.jpg"))
    best = next(c for c in detect_all(img) if c.kind == "crack")
    x, y, w, h = best.bbox
    pad = int(0.25 * max(w, h)) + 12
    crop = img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
    assert "crack" in [c.kind for c in detect_all(crop)[:3]]


def test_draw_marks_the_image():
    img = plain_wall(240, 320)
    c = line_candidate((240, 320), (20, 50), (200, 60), 3)
    out = draw(img, [c])
    assert out.shape == img.shape and np.any(out != img)


def test_exposure_stats_counts_clipped_pixels():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:, 50:] = 255
    s = exposure_stats(img)
    assert s["dark_clipped"] == pytest.approx(0.5)
    assert s["bright_clipped"] == pytest.approx(0.5)
    assert s["mean"] == pytest.approx(127.5)


@photos
def test_enhance_for_review_lifts_contrast_and_keeps_shape():
    img = cv2.imread(str(DATA / "cracked_wall_1.jpg"))[1200:1500, 1150:1450]
    flat = cv2.addWeighted(img, 0.4, np.full_like(img, 140), 0.6, 0)
    out = enhance_for_review(flat)
    assert out.shape == flat.shape and out.dtype == np.uint8
    assert exposure_stats(out)["std"] > exposure_stats(flat)["std"]
    assert clahe_lab(flat).shape == flat.shape
    assert enhance_for_review(img[:4, :4]).shape == (4, 4, 3)


def test_kinds_are_known():
    assert set(defects.KINDS) == {"crack", "stain", "mould", "rust", "peeling"}
