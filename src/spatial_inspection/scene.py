"""The data every stage shares: a walkthrough is a list of frames, each with an image, a camera and (once known)
a pose and a depth map.

Conventions (the same as the inspection platform's 3D engine, so its cached outputs load directly):
- camera: OpenCV (x right, y down, z forward); K is the 3x3 intrinsic matrix for the frame's image size
- model frame: metres, x right, y up, z back, floor at y = 0
- c2w: 4x4 camera-to-model transform
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int                      # position in the walkthrough
    video_frame: int                # frame number in the source video
    t: float                        # seconds from the start
    K: np.ndarray                   # 3x3 intrinsics for `image`
    c2w: np.ndarray | None = None   # 4x4 camera-to-model, None until tracked
    depth: np.ndarray | None = None  # metres along the camera z axis, float32, may be lower resolution than the image
    K_depth: np.ndarray | None = None
    image: np.ndarray | None = None  # BGR, loaded on demand
    quality: dict = field(default_factory=dict)   # sharpness, exposure, motion, verdict ("GOOD" / "POOR" / "BAD")

    @property
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)


@dataclass
class Walkthrough:
    name: str
    video: Path
    fps: float
    size: tuple[int, int]          # image (width, height)
    frames: list[Frame]
    layout: dict | None = None     # room layout (walls, floor, ceiling) once known
    meta: dict = field(default_factory=dict)

    def load_images(self) -> None:
        """Read every frame's image from the video, in one pass."""
        want = {f.video_frame: f for f in self.frames if f.image is None}
        if not want:
            return
        cap = cv2.VideoCapture(str(self.video))
        last = max(want)
        n = 0
        while n <= last:
            ok, img = cap.read()
            if not ok:
                break
            f = want.get(n)
            if f is not None:
                w, h = self.size
                f.image = img if img.shape[1] == w and img.shape[0] == h else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            n += 1
        cap.release()


def load_platform_run(slam_dir: Path, video: Path, name: str | None = None) -> Walkthrough:
    """Load the inspection platform's cached 3D-engine output (cameras, depth maps, layout) for a video."""
    slam_dir = Path(slam_dir)
    cams = json.loads((slam_dir / "05_trajectory" / "cameras.json").read_text(encoding="utf-8"))
    w, h = cams["image_size"]
    z = np.load(slam_dir / "06_depth" / "depth_maps.npz")
    depth = z["depth_filtered_m"]
    K_depth = z["K_processed"]
    # depth maps are keyed by video frame number
    depth_index = {int(i): k for k, i in enumerate(z["frame_index"])}
    quality = {}
    csv = slam_dir / "05_trajectory" / "poses_per_frame.csv"
    if csv.exists():
        rows = csv.read_text(encoding="utf-8").splitlines()
        head = rows[0].split(",")
        for line in rows[1:]:
            r = dict(zip(head, line.split(","), strict=False))
            quality[int(r["video_frame"])] = r.get("frame_quality", "")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    frames = []
    for c in cams["frames"]:
        if c.get("pose_state", "VALID") not in ("VALID", "REFINED", "UNCHECKED"):
            continue
        vf = int(c["video_frame"])
        k = depth_index.get(vf)
        frames.append(Frame(
            index=len(frames),
            video_frame=vf,
            t=vf / fps,
            K=np.asarray(c["K"], dtype=np.float64),
            c2w=np.asarray(c["c2w"], dtype=np.float64),
            depth=None if k is None else depth[k].astype(np.float32),
            K_depth=None if k is None else np.asarray(K_depth[k], dtype=np.float64),
            quality={"platform": quality.get(vf, ""), "pose_state": c.get("pose_state", "")},
        ))
    layout_file = slam_dir / "03_floor_plan" / "layout.json"
    layout = json.loads(layout_file.read_text(encoding="utf-8")) if layout_file.exists() else None
    return Walkthrough(name=name or Path(video).stem, video=Path(video), fps=fps, size=(w, h), frames=frames, layout=layout,
                       meta={"source": "platform-cache", "slam_dir": str(slam_dir)})
