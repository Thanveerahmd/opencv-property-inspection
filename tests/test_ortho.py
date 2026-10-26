from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial_inspection.ortho import Wall, draw_findings, floor_plan, rectify_wall, wall_planes
from spatial_inspection.scene import Frame, Walkthrough

SAMPLE_LAYOUT = Path("D:/quntbit/property-inspection-platform/data/slam/demo_property_lite/03_floor_plan/layout.json")
K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
SIZE = (640, 480)


def texture(w, h, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.circle(t, (w // 3, h // 2), h // 5, (20, 200, 240), -1)
    cv2.rectangle(t, (w // 2, h // 4), (w - 20, h // 2), (200, 30, 30), -1)
    return t


def look_at(position, target):
    """Camera-to-model 4x4 for an OpenCV camera (x right, y down, z forward) with the model's +y up."""
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([x, y, z])
    c2w[:3, 3] = position
    return c2w


def wall_2x1_5():
    return Wall("w", origin=np.zeros(3), u=np.array([1.0, 0, 0]), v=np.array([0, 1.0, 0]),
                normal=np.array([0, 0, 1.0]), length_m=2.0, height_m=1.5)


def render(wall, tex, c2w, px_per_m=100.0):
    """The frame a camera at c2w would see of the textured wall: the plane homography applied to the texture."""
    w2c = np.linalg.inv(c2w)
    M = w2c[:3, :3] @ wall.pixel_to_model(px_per_m)
    M[:, 2] += w2c[:3, 3]
    return cv2.warpPerspective(tex, K @ M, SIZE, flags=cv2.INTER_LINEAR, borderValue=(90, 90, 90))


def walkthrough(frames):
    return Walkthrough(name="synthetic", video=Path("synthetic.mp4"), fps=30.0, size=SIZE, frames=frames)


def ncc(a, b, mask):
    a = a[mask].astype(np.float64).ravel()
    b = b[mask].astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def test_rectify_wall_recovers_texture_from_an_oblique_view():
    wall = wall_2x1_5()
    tex = texture(200, 150)
    c2w = look_at(np.array([1.4, 0.9, 2.2]), [0.9, 0.7, 0.0])     # off to the side, looking down a little
    frame = Frame(index=0, video_frame=0, t=0.0, K=K, c2w=c2w, image=render(wall, tex, c2w))

    rect = rectify_wall(walkthrough([frame]), wall, px_per_m=100.0)

    assert rect.image.shape == tex.shape
    seen = cv2.erode(rect.coverage.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert seen.mean() > 0.6
    assert ncc(rect.image, tex, seen) > 0.8
    assert set(np.unique(rect.source[seen])) == {0}


def test_depth_in_front_of_the_wall_hides_it_and_a_second_view_fills_in():
    wall = wall_2x1_5()
    tex = texture(200, 150, seed=1)
    c2w_a = look_at(np.array([1.0, 0.75, 2.0]), [1.0, 0.75, 0.0])
    c2w_b = look_at(np.array([0.3, 0.75, 1.8]), [1.2, 0.75, 0.0])
    depth = np.zeros((SIZE[1], SIZE[0]), np.float32)        # unknown everywhere ...
    depth[200:280, 280:360] = 0.8                           # ... except an object 0.8 m away, in front of the wall
    a = Frame(index=0, video_frame=0, t=0.0, K=K, c2w=c2w_a, image=render(wall, tex, c2w_a), depth=depth, K_depth=K)

    alone = rectify_wall(walkthrough([a]), wall)
    # the object's patch on the wall: frame A looks straight at the wall's centre from 2 m
    assert not alone.coverage[70:80, 95:105].any()
    assert alone.coverage[10:30, 10:60].all()

    b = Frame(index=1, video_frame=3, t=0.1, K=K, c2w=c2w_b, image=render(wall, tex, c2w_b))
    both = rectify_wall(walkthrough([a, b]), wall)
    assert both.coverage[70:80, 95:105].all()
    assert set(np.unique(both.source[70:80, 95:105])) == {1}
    seen = cv2.erode(both.coverage.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert ncc(both.image, tex, seen) > 0.8


def test_camera_behind_the_wall_sees_nothing():
    wall = wall_2x1_5()
    c2w = look_at(np.array([1.0, 0.75, -2.0]), [1.0, 0.75, 0.0])
    frame = Frame(index=0, video_frame=0, t=0.0, K=K, c2w=c2w, image=np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    rect = rectify_wall(walkthrough([frame]), wall)
    assert not rect.coverage.any()
    assert (rect.image == 128).all()


def fitted_layout():
    """The shape the pipeline's own layout fit produces: four walls, one room, no doors."""
    walls = [
        {"id": "wall_00", "axis": "x", "offset": 0.0, "sign": 1, "t0": 0.0, "t1": 4.0},
        {"id": "wall_01", "axis": "x", "offset": 3.0, "sign": -1, "t0": 0.0, "t1": 4.0},
        {"id": "wall_02", "axis": "z", "offset": 0.0, "sign": 1, "t0": 0.0, "t1": 3.0},
        {"id": "wall_03", "axis": "z", "offset": 4.0, "sign": -1, "t0": 0.0, "t1": 3.0},
    ]
    return {"ceiling_height_m": 2.5, "walls": walls, "rooms": [{"id": "room_0", "bbox_m": [0, 0, 3, 4]}]}


def check_walls(walls):
    assert walls
    for w in walls:
        for a in (w.u, w.v, w.normal):
            assert np.linalg.norm(a) == pytest.approx(1.0)
        assert abs(w.u @ w.v) < 1e-9 and abs(w.u @ w.normal) < 1e-9 and abs(w.v @ w.normal) < 1e-9
        # u, v and the normal are right-handed, so the image is not mirrored when seen from inside
        assert np.allclose(np.cross(w.u, w.v), w.normal)
        assert w.length_m > 0 and w.height_m > 0
        for o in w.openings:
            assert 0 <= o.a0 <= o.a1 <= w.length_m + 1e-9


def test_wall_planes_on_a_fitted_layout_face_into_the_room():
    walls = wall_planes(fitted_layout())
    check_walls(walls)
    centre = np.array([1.5, 1.2, 2.0])
    for w in walls:
        assert (centre - w.origin) @ w.normal > 0
        assert w.length_m == pytest.approx(4.0 if w.id in ("wall_00", "wall_01") else 3.0)
    assert wall_planes(None) == []


@pytest.mark.skipif(not SAMPLE_LAYOUT.exists(), reason="sample layout not available")
def test_wall_planes_on_the_sample_layout():
    import json
    layout = json.loads(SAMPLE_LAYOUT.read_text(encoding="utf-8"))
    walls = wall_planes(layout)
    check_walls(walls)
    assert len(walls) == len(layout["walls"])
    door = next(o for w in walls for o in w.openings if w.id == "wall_00")
    assert door.a1 - door.a0 == pytest.approx(1.0, abs=0.01)


def test_findings_are_drawn_on_the_wall_and_the_plan():
    wall = wall_2x1_5()
    tex = texture(200, 150)
    c2w = look_at(np.array([1.0, 0.75, 2.0]), [1.0, 0.75, 0.0])
    frame = Frame(index=0, video_frame=0, t=0.0, K=K, c2w=c2w, image=render(wall, tex, c2w))
    rect = rectify_wall(walkthrough([frame]), wall)
    findings = [{"id": "F001", "kind": "crack", "status": "candidate", "xyz": [0.5, 1.0, 0.02]},
                {"id": "F002", "kind": "stain", "status": "candidate", "xyz": None},
                {"id": "F003", "kind": "damp", "xyz": [1.5, 0.5, 1.5]}]     # 1.5 m off the wall: not drawn
    out = draw_findings(rect, findings)
    diff = np.abs(out.astype(int) - rect.image.astype(int)).sum(axis=2) > 0
    assert diff[50, 30:70].any()                # marker ring around (50, 50)
    assert not diff[110:, 140:].any()

    plan = floor_plan(fitted_layout(), findings=findings, trajectory=[np.array([1.0, 1.5, 1.0]),
                                                                      np.array([2.0, 1.5, 3.0])])
    assert plan.ndim == 3 and plan.shape[2] == 3 and plan.shape[0] > 200 and plan.shape[1] > 200
