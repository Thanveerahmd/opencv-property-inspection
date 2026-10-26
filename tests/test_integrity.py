import cv2
import numpy as np

from spatial_inspection import integrity
from spatial_inspection.scene import Frame, Walkthrough

K = np.array([[300.0, 0, 160], [0, 300.0, 90], [0, 0, 1]])


def scene(seed: int) -> np.ndarray:
    """A wide textured 'room' to pan across."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (180, 1600, 3), dtype=np.uint8), (0, 0), 6)
    return cv2.normalize(img, None, 40, 220, cv2.NORM_MINMAX)


def walk(name: str, images: list[np.ndarray]) -> Walkthrough:
    frames = [Frame(i, i, i / 10, K, image=im, quality={"verdict": "GOOD"}) for i, im in enumerate(images)]
    return Walkthrough(name=name, video=None, fps=10.0, size=(320, 180), frames=frames, layout=None, meta={})


def pan(room: np.ndarray, n: int = 60, step: int = 20) -> list[np.ndarray]:
    return [room[:, i * step: i * step + 320].copy() for i in range(n)]


def test_honest_continuous_pan_has_no_flags():
    r = integrity.check(walk("a", pan(scene(1))), root=None)
    assert r["flags"] == [] and r["level"] == "low"


def test_same_footage_for_another_property_is_high(tmp_path):
    frames = pan(scene(2))
    integrity.check(walk("first", frames), root=tmp_path, prop="12 Lake Rd")
    r = integrity.check(walk("second", frames), root=tmp_path, prop="7 Hill St")
    assert r["level"] == "high" and r["flags"][0]["code"] == "footage_other_property"
    again = integrity.check(walk("first", frames), root=tmp_path, prop="12 Lake Rd")   # re-analysing is not reuse
    assert all(f["code"] != "footage_resubmitted" or "second" in f["message"] for f in again["flags"])


def test_spliced_video_shows_a_hard_cut():
    a = pan(scene(3), 30)
    b = [cv2.applyColorMap(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_OCEAN) for im in pan(scene(4), 30)]
    r = integrity.check(walk("cut", a + b), root=None)
    assert any(f["code"] == "edited_cuts" for f in r["flags"])
    assert abs(r["checks"]["hard_cuts"][0]["t"] - 3.0) < 0.11


def test_looped_and_frozen_footage():
    frames = pan(scene(5), 70)
    frames[40:50] = [f.copy() for f in frames[10:20]]
    frames[55:70] = [frames[54].copy()] * 15     # 1.5 s paused
    r = integrity.check(walk("loop", frames), root=None)
    codes = {f["code"] for f in r["flags"]}
    assert {"looped_footage", "frozen_frames"} <= codes
