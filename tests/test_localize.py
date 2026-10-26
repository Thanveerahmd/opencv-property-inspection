import numpy as np

from spatial_inspection.coverage import make_samples
from spatial_inspection.localize import place_on_surface, surface_hit
from spatial_inspection.scene import Frame

# one room, 4 m x 4 m; the camera stands in the middle at 1.4 m and looks at the north wall (z = -2)
LAYOUT = {
    "ceiling_height_m": 2.6,
    "walls": [{"id": "wall_n", "axis": "z", "offset": -2.0, "sign": 1, "t0": -2.0, "t1": 2.0},
              {"id": "wall_s", "axis": "z", "offset": 2.0, "sign": -1, "t0": -2.0, "t1": 2.0},
              {"id": "wall_w", "axis": "x", "offset": -2.0, "sign": 1, "t0": -2.0, "t1": 2.0},
              {"id": "wall_e", "axis": "x", "offset": 2.0, "sign": -1, "t0": -2.0, "t1": 2.0}],
    "rooms": [{"id": "room_1", "bbox_m": [-2.0, -2.0, 2.0, 2.0]}],
    "doors_and_openings": [{"type": "door", "wall": "wall_n", "xz0": [1.0, -2.0], "xz1": [1.9, -2.0],
                            "bottom_m": 0.0, "top_m": 2.1}],
}
K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])


def frame(depth_value: float = 2.0) -> Frame:
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])   # OpenCV camera (z forward, y down) in the model frame (y up, z back)
    c2w[:3, 3] = [0.0, 1.4, 0.0]
    return Frame(0, 0, 0.0, K, c2w, np.full((240, 320), depth_value, np.float32), K / np.array([[2], [2], [1]]))


def test_ray_hits_the_wall_in_front():
    hit = surface_hit(frame(), 320, 240, LAYOUT)
    assert hit["surface"] == "wall" and hit["wall"]["id"] == "wall_n"
    assert abs(hit["z_cam"] - 2.0) < 1e-6 and np.allclose(hit["xyz"], [0.0, 1.4, -2.0])


def test_candidate_on_the_wall_is_kept_and_placed():
    samples = make_samples(LAYOUT, 0.1)
    r = place_on_surface(frame(2.0), (300, 220, 40, 40), LAYOUT, samples)
    assert r["on_surface"] and r["place"]["group"] == "wall_n"


def test_object_in_front_of_the_wall_is_ruled_out():
    samples = make_samples(LAYOUT, 0.1)
    r = place_on_surface(frame(1.4), (300, 220, 40, 40), LAYOUT, samples)
    assert not r["on_surface"] and "in front of the wall" in r["reason"]


def test_candidate_on_a_door_is_ruled_out():
    samples = make_samples(LAYOUT, 0.1)
    u = 320 + 400 * 1.4 / 2.0   # x = 1.4 m on the north wall, inside the door
    r = place_on_surface(frame(2.0), (u - 10, 250, 20, 20), LAYOUT, samples)
    assert not r["on_surface"] and "door" in r["reason"]


def test_protruding_object_is_ruled_out_by_its_mask():
    samples = make_samples(LAYOUT, 0.1)
    f = frame(2.0)
    f.depth[100:140, 140:180] = 1.6         # a chair back 0.4 m proud of the wall, under the candidate's mask
    mask = np.zeros((480, 640), np.uint8)
    mask[200:280, 280:360] = 255
    box = (230, 150, 180, 180)              # the box itself is mostly wall, so its median depth matches the wall
    assert place_on_surface(f, box, LAYOUT, samples)["on_surface"]
    r = place_on_surface(f, box, LAYOUT, samples, mask=mask)
    assert not r["on_surface"] and "stands out" in r["reason"]
