import cv2
import numpy as np

from spatial_inspection.tracking import frame_pose, intrinsics_guess, relative_pose, track_points


def rotation(deg_xyz):
    return cv2.Rodrigues(np.radians(np.asarray(deg_xyz, dtype=np.float64)))[0]


def project(K, X):
    x = (K @ X.T).T
    return (x[:, :2] / x[:, 2:]).astype(np.float64)


def test_relative_pose_recovers_rotation_and_direction():
    rng = np.random.default_rng(0)
    K = intrinsics_guess((720, 1280))
    # points spread through a room 2 to 6 m in front of the first camera
    X1 = np.column_stack([rng.uniform(-2, 2, 400), rng.uniform(-1.2, 1.2, 400), rng.uniform(2, 6, 400)])
    R_true = rotation([2.0, -8.0, 1.0])
    t_true = np.array([0.3, 0.02, 0.1])
    X2 = (R_true @ X1.T).T + t_true
    p0 = project(K, X1) + rng.normal(0, 0.3, (400, 2))
    p1 = project(K, X2) + rng.normal(0, 0.3, (400, 2))

    pose = relative_pose(p0, p1, K)

    assert pose is not None
    rot_err = np.degrees(np.linalg.norm(cv2.Rodrigues(pose.R @ R_true.T)[0]))
    dir_err = np.degrees(np.arccos(np.clip(pose.t @ (t_true / np.linalg.norm(t_true)), -1, 1)))
    assert rot_err < 0.5
    assert dir_err < 3.0
    assert pose.inliers > 300


def test_relative_pose_needs_enough_points():
    K = intrinsics_guess((480, 640))
    assert relative_pose(np.zeros((5, 2)), np.zeros((5, 2)), K) is None


def test_optical_flow_follows_a_shifted_image():
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur((rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 2.0)
    shift = np.float32([[1, 0, 4.0], [0, 1, -2.5]])
    shifted = cv2.warpAffine(texture, shift, (640, 480), borderMode=cv2.BORDER_REFLECT)

    p0, p1 = track_points(texture, shifted)

    assert len(p0) > 200
    dx, dy = np.median(p1 - p0, axis=0)
    assert abs(dx - 4.0) < 0.1
    assert abs(dy + 2.5) < 0.1


def test_frame_pose_handles_a_frame_with_nothing_to_track():
    blank = np.zeros((480, 640, 3), np.uint8)
    assert frame_pose(blank, blank, intrinsics_guess(blank.shape)) is None
