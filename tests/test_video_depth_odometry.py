from pathlib import Path

import cv2
import numpy as np
import pytest

import spatial_inspection  # noqa: F401  (thread cap and low priority, before any OpenCV work)
from spatial_inspection import models
from spatial_inspection.depth import fit_inverse_depth, to_metric
from spatial_inspection.odometry import level_and_scale, process_video, track_walkthrough
from spatial_inspection.scene import Frame
from spatial_inspection.tracking import intrinsics_guess
from spatial_inspection.video import assess, assess_frames

VIDEO = Path(__file__).resolve().parents[1] / "data" / "samples" / "demo_property_lite.mp4"
RANK = {"GOOD": 0, "POOR": 1, "BAD": 2}


def textured(h=480, w=848, seed=0):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur((rng.random((h, w)) * 255).astype(np.float32), (0, 0), 1.5)
    noise = cv2.normalize(noise, None, 30, 225, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def test_quality_gate_flags_a_blurred_frame():
    sharp = textured()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)

    q_sharp, q_blur = assess(sharp), assess(blurred)

    assert q_blur["sharpness"] < 0.2 * q_sharp["sharpness"]
    assert q_sharp["verdict"] == "GOOD"
    assert RANK[q_blur["verdict"]] > RANK[q_sharp["verdict"]]


def test_assess_frames_keeps_other_quality_entries_and_measures_motion():
    img = textured()
    moved = np.roll(img, 6, axis=1)
    K = intrinsics_guess(img.shape)
    frames = [Frame(index=i, video_frame=i, t=0.1 * i, K=K, image=im, quality={"pose": "PNP"})
              for i, im in enumerate([img, moved])]

    assess_frames(frames)

    assert all(f.quality["verdict"] in RANK and f.quality["pose"] == "PNP" for f in frames)
    assert frames[0].quality["motion_px"] is None
    assert abs(frames[1].quality["motion_px"] - 6) < 1.5


def test_depth_alignment_recovers_a_known_affine():
    rng = np.random.default_rng(3)
    disp = rng.uniform(0.5, 6.0, 400)
    a, b = 0.18, 0.21
    z = 1 / (a * disp + b) * (1 + rng.normal(0, 0.01, 400))
    z[:80] *= rng.uniform(1.5, 3.0, 80)  # a fifth of the points are wrong

    fit = fit_inverse_depth(disp, z)
    tied = fit_inverse_depth(disp, z, shift=b / a)

    assert fit is not None and abs(fit.a - a) / a < 0.03 and abs(fit.b - b) / b < 0.05
    assert fit.inliers >= 300
    assert tied is not None and abs(tied.a - a) / a < 0.02
    depth = to_metric(disp.reshape(20, 20).astype(np.float32), fit.a, fit.b, drop_edges=False)
    assert np.allclose(depth.ravel()[80:], 1 / (a * disp[80:] + b), rtol=0.05)


# a small ray-cast room: floor 1.4 m below the first camera (OpenCV y points down) and a wall 6 m ahead
FLOOR_Y, WALL_Z = 1.4, 6.0


def render(K, c2w, size, tex):
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    rays = np.dstack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)])  # camera z = 1
    rw = rays @ c2w[:3, :3].T
    C = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(rw[..., 1] > 1e-6, (FLOOR_Y - C[1]) / rw[..., 1], np.inf)
        t_wall = np.where(rw[..., 2] > 1e-6, (WALL_Z - C[2]) / rw[..., 2], np.inf)
    on_floor = t_floor < t_wall
    t = np.where(on_floor, t_floor, t_wall)  # camera z-depth, since the rays have z = 1
    X = C + rw * t[..., None]
    s = tex.shape[0] / 16.0  # texture pixels per metre
    mx = np.where(on_floor, X[..., 0], X[..., 0] + 8.0) * s + tex.shape[1] / 2
    my = np.where(on_floor, X[..., 2], X[..., 1] + 4.0) * s + tex.shape[0] / 4
    img = cv2.remap(tex, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), t.astype(np.float32)


class FakeDepthNet:
    """Stands in for the network: true inverse depth through an affine the odometry does not know."""

    def __init__(self, depths, a=2.0, b=0.3):
        self.queue = [a / d + b for d in depths]

    def depth_size(self, image_size):
        return image_size

    def depth_K(self, K, image_size):
        return K.copy()

    def predict(self, img):
        return self.queue.pop(0).astype(np.float32)


def test_odometry_follows_a_synthetic_walk_and_levels_it():
    rng = np.random.default_rng(5)
    tex = cv2.GaussianBlur((rng.random((1600, 1600)) * 255).astype(np.float32), (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 20, 235, cv2.NORM_MINMAX).astype(np.uint8)
    size = (424, 240)
    K = intrinsics_guess((size[1], size[0]), 80.0)
    step = np.array([0.06, 0.0, 0.03])
    frames, depths = [], []
    for i in range(16):
        c2w = np.eye(4)
        c2w[:3, :3] = cv2.Rodrigues(np.array([0.0, np.radians(0.4 * i), 0.0]))[0]
        c2w[:3, 3] = step * i
        img, depth = render(K, c2w, size, tex)
        frames.append(Frame(index=i, video_frame=i, t=0.1 * i, K=K, image=img))
        depths.append(depth)

    traj = track_walkthrough(frames, FakeDepthNet(depths))

    moved = traj.c2w[-1][:3, 3] - traj.c2w[0][:3, 3]
    assert traj.states.count("PNP") >= 13
    assert moved @ step / (np.linalg.norm(moved) * np.linalg.norm(step)) > 0.95

    c2w, metric, info = level_and_scale(traj.c2w, traj.depth, traj.K_depth, camera_height=1.45)
    heights = np.array([T[1, 3] for T in c2w])
    assert np.allclose(heights, 1.45, atol=0.08)
    assert c2w[0][:3, 1] @ np.array([0, -1.0, 0]) > 0.99  # camera y (down) points down the model's y axis
    # the levelled scale makes the depth metric: the true floor is 1.4 m below, so the true wall is 6 m * 1.45 / 1.4
    wall = metric[0][: size[1] // 4, size[0] // 2 - 20: size[0] // 2 + 20]
    assert abs(np.median(wall[wall > 0]) - WALL_Z * 1.45 / FLOOR_Y) < 0.5


@pytest.mark.skipif(not VIDEO.exists() or not models.available("depth_anything_v2_small"),
                    reason="needs the sample video and the depth model")
def test_process_video_on_the_first_two_seconds():
    seen = []

    w = process_video(VIDEO, fps=10, max_frames=20, progress=lambda stage, frac: seen.append((stage, frac)))

    assert len(w.frames) == 20
    for f in w.frames:
        assert f.c2w.shape == (4, 4) and np.isfinite(f.c2w).all()
        assert f.depth.dtype == np.float32 and (f.depth > 0).mean() > 0.3
        assert f.K_depth.shape == (3, 3)
        assert f.quality["verdict"] in RANK
    heights = np.array([f.c2w[1, 3] for f in w.frames])
    assert abs(np.median(heights) - 1.45) < 0.1
    assert 0.3 < np.median(w.frames[0].depth[w.frames[0].depth > 0]) < 8
    assert {s for s, _ in seen} >= {"frames", "tracking"} and all(0 <= p <= 1 for _, p in seen)
    assert w.meta["timings"]["per_frame_s"] > 0
