"""Render a room's Gaussian splats (10_splats/room.splat) from the engine's cameras, next to the video frames.

    .venv-splat/Scripts/python.exe engine3d/render_splats.py --engine data/engine/<name> --out frames/ [--every 2]

Each output frame is the video frame on the left and the splat render from the same camera on the right, with the
splats' colours matched to that frame's exposure by one least-squares gain (as in train_splats.py's evaluation).
Most of these cameras are training views; train_splats.py scores the held-out ones.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SI_THREADS", "2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

cv2.setNumThreads(int(os.environ.get("SI_THREADS", "2")))
torch.set_num_threads(int(os.environ.get("SI_THREADS", "2")))


def load_splat(path: Path, dev: str):
    rec = np.frombuffer(path.read_bytes(), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
    quats = F.normalize(t(rec["r"].astype(np.float32) - 128) / 128, dim=-1)
    return (t(rec["p"]), quats, t(rec["s"]), t(rec["c"][:, 3].astype(np.float32) / 255),
            t(rec["c"][:, :3].astype(np.float32) / 255))


def main():
    from gsplat import rasterization

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--video", help="default: the video in the engine's run_config.json")
    ap.add_argument("--out", required=True, help="folder for the side-by-side JPEG frames")
    ap.add_argument("--every", type=int, default=2, help="use every n-th camera")
    a = ap.parse_args()
    engine, out = Path(a.engine), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = a.video or json.loads((engine / "run_config.json").read_text(encoding="utf-8"))["video"]
    cams = json.loads((engine / "05_trajectory" / "cameras.json").read_text(encoding="utf-8"))
    W, H = cams["image_size"]
    frames = cams["frames"][:: a.every]
    want = {int(f["video_frame"]): f for f in frames}
    dev = "cuda"
    means, quats, scales, opac, rgb = load_splat(engine / "10_splats" / "room.splat", dev)
    cap = cv2.VideoCapture(video)
    i = n = 0
    while want:
        ok, img = cap.read()
        if not ok:
            break
        f = want.pop(i, None)
        i += 1
        if f is None:
            continue
        if (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        w2c = torch.from_numpy(np.linalg.inv(np.asarray(f["c2w"], np.float64)).astype(np.float32)).to(dev)[None]
        K = torch.tensor(f["K"], dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            r, _, _ = rasterization(means, quats, scales, opac, rgb, w2c, K, W, H, packed=False)
        r = r[0].clamp(0, 1)
        gt = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(dev).float() / 255
        gain = (r * gt).sum((0, 1)) / (r * r).sum((0, 1)).clamp_min(1e-6)
        render = cv2.cvtColor(((r * gain).clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{n:05d}.jpg"), np.hstack([img, render]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    print(f"{n} frames -> {out}")


if __name__ == "__main__":
    main()
