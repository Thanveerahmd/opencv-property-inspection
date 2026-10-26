"""Compare the CPU odometry (odometry.process_video) with the inspection platform's cached GPU run for the same video.

Camera positions at matching video frames are aligned with a 7 DoF similarity (Umeyama) and the absolute trajectory
error is reported, then depth maps are compared pixel by pixel where both have depth.

    .venv/Scripts/python scripts/eval_odometry.py
    .venv/Scripts/python scripts/eval_odometry.py --video data/samples/tom_home.mp4 \
        --slam-dir D:/quntbit/property-inspection-platform/data/slam/tom_home
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import spatial_inspection  # noqa: E402, F401  (thread cap and low priority)
from spatial_inspection.odometry import process_video  # noqa: E402
from spatial_inspection.scene import Walkthrough, load_platform_run  # noqa: E402
from spatial_inspection.tracking import intrinsics_guess  # noqa: E402

PLATFORM = Path("D:/quntbit/property-inspection-platform/data/slam")


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale s, rotation R and translation t minimising |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / a.var(0).sum())
    return s, R, mu_d - s * R @ mu_s


def evaluate(ours: Walkthrough, ref: Walkthrough) -> dict:
    by_vf = {f.video_frame: f for f in ref.frames}
    pairs = [(f, by_vf[f.video_frame]) for f in ours.frames if f.video_frame in by_vf and f.c2w is not None]
    if len(pairs) < 3:
        raise SystemExit("fewer than three frames in common")
    P = np.array([a.position for a, _ in pairs])
    Q = np.array([b.position for _, b in pairs])
    s, R, t = umeyama(P, Q)
    err = np.linalg.norm((s * P @ R.T + t) - Q, axis=1)
    rot = []
    for a, b in pairs:
        dR = b.c2w[:3, :3].T @ R @ a.c2w[:3, :3]
        rot.append(np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0])))
    path_len = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())

    abs_err, rel_err, rel_err_aligned, rel_err_framescale = [], [], [], []
    for a, b in pairs:
        if a.depth is None or b.depth is None:
            continue
        da = a.depth if a.depth.shape == b.depth.shape else cv2.resize(a.depth, b.depth.shape[::-1],
                                                                        interpolation=cv2.INTER_NEAREST)
        m = (da > 0) & (b.depth > 0)
        if m.sum() < 100:
            continue
        x, y = da[m].astype(np.float64), b.depth[m].astype(np.float64)
        abs_err.append(np.abs(x - y))
        rel_err.append(np.abs(x - y) / y)
        rel_err_aligned.append(np.abs(s * x - y) / y)
        rel_err_framescale.append(np.abs(x * np.median(y / x) - y) / y)
    cat = np.concatenate
    return {
        "frames_compared": len(pairs),
        "reference_path_m": round(path_len, 2),
        "ate_rmse_m": round(float(np.sqrt((err ** 2).mean())), 3),
        "ate_median_m": round(float(np.median(err)), 3),
        "sim3_scale": round(s, 3),
        "rotation_err_median_deg": round(float(np.median(rot)), 2),
        "depth_frames": len(abs_err),
        "depth_abs_err_median_m": round(float(np.median(cat(abs_err))), 3) if abs_err else None,
        "depth_absrel": round(float(np.mean(cat(rel_err))), 3) if rel_err else None,
        "depth_rel_err_median": round(float(np.median(cat(rel_err))), 3) if rel_err else None,
        "depth_rel_err_median_sim3_scaled": round(float(np.median(cat(rel_err_aligned))), 3) if abs_err else None,
        "depth_rel_err_median_per_frame_scaled": round(float(np.median(cat(rel_err_framescale))), 3)
        if abs_err else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, default=ROOT / "data" / "samples" / "demo_property_lite.mp4")
    ap.add_argument("--slam-dir", type=Path, default=None, help="cached platform run (default: by video name)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--hfov", type=float, default=None, help="horizontal field of view in degrees for K")
    ap.add_argument("--ref-k", action="store_true", help="use the reference run's intrinsics")
    args = ap.parse_args()

    slam_dir = args.slam_dir or PLATFORM / args.video.stem
    ref = load_platform_run(slam_dir, args.video)
    K = None
    if args.ref_k:
        K = ref.frames[0].K.copy()
    elif args.hfov:
        K = intrinsics_guess((ref.size[1], ref.size[0]), args.hfov)

    shown = {}

    def progress(stage, frac):
        if frac >= shown.get(stage, 0) + 0.25 or frac >= 1 and shown.get(stage, 0) < 1:
            shown[stage] = frac
            print(f"  {stage} {100 * frac:.0f}%", flush=True)

    ours = process_video(args.video, fps=args.fps, max_frames=args.max_frames, progress=progress, K=K)
    meta = ours.meta
    print(json.dumps({"frames": len(ours.frames), "timings_s": meta["timings"], "tracking": meta["tracking"],
                      "calibration": {k: v for k, v in meta["calibration"].items() if k != "hfov_scores"},
                      "levelling": meta["levelling"], "eval": evaluate(ours, ref)}, indent=1, default=str))
    sc = np.array([f.quality.get("scale_correction", 1.0) for f in ours.frames])
    print("drift correction per 10 frames:", sc[::10].round(2).tolist())


if __name__ == "__main__":
    main()
