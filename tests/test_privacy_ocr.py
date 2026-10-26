import cv2
import numpy as np
import pytest

from spatial_inspection import models, ocr, privacy

OCR_MODELS = ("text_detection_ppocr", "text_recognition_crnn")


def need(*names):
    missing = [n for n in names if not models.available(n)]
    if missing:
        pytest.skip(f"model files not downloaded: {', '.join(missing)} (python -m spatial_inspection.models download)")


@pytest.fixture(scope="module")
def plate():
    """A synthetic water heater rating plate: large dark text on a light label."""
    img = np.full((480, 848, 3), 235, np.uint8)
    cv2.putText(img, "MODEL WH50 2014 230V", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(img, "50Hz 80L", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4, cv2.LINE_AA)
    return img


def test_model_paths_resolve_and_checksums_match(monkeypatch, tmp_path):
    monkeypatch.setenv(models.ENV_VAR, str(tmp_path))
    assert models.path("face_detection_yunet") == tmp_path / "opencv" / "face_detection_yunet_2023mar.onnx"
    monkeypatch.delenv(models.ENV_VAR)
    present = [n for n in models.REGISTRY if models.available(n)]
    if not present:
        pytest.skip("no model files downloaded")
    for name in present:
        assert models.path(name).is_relative_to(models.models_dir())
        assert models.sha256(models.path(name)) == models.REGISTRY[name].sha256
        assert models.ensure(name) == models.path(name)


def test_ocr_reads_the_year_and_ratings_on_a_label(plate):
    need(*OCR_MODELS)
    lines = ocr.read_text(plate)

    assert len(lines) == 2
    assert all({"text", "confidence", "quad"} <= set(ln) for ln in lines)
    assert any("2014" in ocr.normalise(ln["text"]) for ln in lines)
    assert 2014 in ocr.find_years(lines)
    kinds = {(r["kind"], r["value"]) for r in ocr.find_ratings(lines)}
    assert ("voltage", 230.0) in kinds
    assert ("frequency", 50.0) in kinds
    # the first line's box sits around the text drawn at y = 200
    ys = [p[1] for p in lines[0]["quad"]]
    assert min(ys) < 170 < 200 < max(ys)


def test_redact_blurs_the_text(plate):
    need("text_detection_ppocr", "face_detection_yunet")
    out, report = privacy.redact(plate)

    assert report["text"] == 2 and report["faces"] == 0
    assert out.shape == plate.shape and not np.shares_memory(out, plate)
    x, y, w, h = 40, 165, 550, 40  # where "MODEL WH50 2014 230V" was drawn
    before = plate[y:y + h, x:x + w].astype(np.float32).var()
    after = out[y:y + h, x:x + w].astype(np.float32).var()
    assert after < 0.1 * before
    if all(models.available(n) for n in OCR_MODELS):
        assert not ocr.find_years(ocr.read_text(out))


def test_redact_leaves_a_blank_image_alone():
    need("text_detection_ppocr", "face_detection_yunet")
    blank = np.full((480, 848, 3), 200, np.uint8)
    out, report = privacy.redact(blank, method="pixelate")

    assert report["faces"] == 0 and report["text"] == 0
    assert np.array_equal(out, blank)


def test_years_and_ratings_from_noisy_text():
    lines = ["mfd 2o14", {"text": "220-240v 50hz 16a"}, "3kw 120l 6 bar", "serial 77312014889", "1949 2031"]
    assert ocr.find_years(lines) == [2014]
    got = {(r["kind"], r["unit"], r["value"], r.get("upper")) for r in ocr.find_ratings(lines)}
    assert got == {("voltage", "V", 220.0, 240.0), ("frequency", "Hz", 50.0, None), ("current", "A", 16.0, None),
                   ("power", "kW", 3.0, None), ("volume", "L", 120.0, None), ("pressure", "bar", 6.0, None)}
    assert ocr.normalise("23ov 8ol") == "230v 80l"
