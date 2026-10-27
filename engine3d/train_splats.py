"""Photoreal room: 3D Gaussian splats trained on the 3D engine's cameras.

    .venv-splat/Scripts/python.exe engine3d/train_splats.py --engine data/engine/<name> [--iters 7000]

The engine has already solved the hard part: a camera pose and intrinsics for every frame (Depth Anything 3 refined by
OpenCV 5 tracking and ICP) and a dense, metric surface (TSDF mesh + point cloud) in the room's model frame (x right,
y up, z back, metres, floor y = 0). Gaussians start on that surface and are optimised with gsplat (Apache-2.0) so that
rendering them from each camera reproduces its video frame. Every 10th frame is held out to measure how realistic the
result is from views it was not trained on.

  - per-frame exposure (RGB gain + offset) absorbs the phone's auto-exposure, so the colours are the room's, not the
    frame's; the exported splats use the neutral exposure
  - densify / prune with gsplat's DefaultStrategy; L1 + 0.2 D-SSIM loss
  - laptop guard: CPU thread pools capped, and training pauses while the GPU is above 85 C until it is below 75 C

Outputs in <engine>/10_splats/: room.splat (web viewer format: position, scale, RGBA, rotation; 32 bytes a splat),
metrics.json (PSNR / SSIM on training and held-out frames), compare.jpg (held-out frame | splat render).

Runs in its own environment (.venv-splat: Python 3.10, PyTorch 2.4 + CUDA 12.4, the prebuilt gsplat wheel); see
requirements-splat.txt.
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

THREADS = os.environ.get("SI_THREADS", "2")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, THREADS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from plyfile import PlyData  # noqa: E402

cv2.setNumThreads(int(THREADS))
torch.set_num_threads(int(THREADS))
C0 = 0.28209479177387814          # SH degree-0 basis constant


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data
def load_frames(engine: Path, video: str):
    cams = json.loads((engine / "05_trajectory" / "cameras.json").read_text(encoding="utf-8"))
    W, H = cams["image_size"]
    want = {int(f["video_frame"]): f for f in cams["frames"]}
    cap = cv2.VideoCapture(video)
    imgs, Ks, w2c, vf = [], [], [], []
    i = 0
    while want:
        ok, im = cap.read()
        if not ok:
            break
        f = want.pop(i, None)
        i += 1
        if f is None:
            continue
        if (im.shape[1], im.shape[0]) != (W, H):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        Ks.append(np.asarray(f["K"], np.float32))
        w2c.append(np.linalg.inv(np.asarray(f["c2w"], np.float64)).astype(np.float32))
        vf.append(int(f["video_frame"]))
    cap.release()
    return np.stack(imgs), np.stack(Ks), np.stack(w2c), vf, (W, H)


def load_points(engine: Path, voxel: float, max_points: int):
    """Starting Gaussians: TSDF mesh vertices + the dense point cloud, one per `voxel` cell."""
    xyz, rgb = [], []
    for rel in ("04_geometry/mesh_fused_regularised.ply", "04_geometry/point_cloud_dense.ply"):
        p = engine / rel
        if not p.exists():
            continue
        v = PlyData.read(str(p))["vertex"]
        xyz.append(np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32))
        names = v.data.dtype.names
        rgb.append(np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255 if "red" in names
                   else np.full((len(v.data), 3), 0.5, np.float32))
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    xyz, rgb = xyz[first], rgb[first]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


# ------------------------------------------------------------------------------------------------ helpers
def ssim(x, y, win=11, sigma=1.5):
    """Mean SSIM of two (1, 3, H, W) images in [0, 1]."""
    g = torch.exp(-((torch.arange(win, device=x.device) - win // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    w = g.expand(3, 1, win, win)
    mu_x, mu_y = F.conv2d(x, w, groups=3, padding=win // 2), F.conv2d(y, w, groups=3, padding=win // 2)
    sxx = F.conv2d(x * x, w, groups=3, padding=win // 2) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=3, padding=win // 2) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=3, padding=win // 2) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def psnr(x, y):
    return float(10 * torch.log10(1.0 / F.mse_loss(x, y).clamp_min(1e-10)))


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cool_down(hot=85, cool=75):
    t = gpu_temp()
    if t is None or t < hot:
        return
    log(f"GPU at {t} C: pausing until it is below {cool} C")
    while (t := gpu_temp()) is not None and t > cool:
        time.sleep(5)


def export_splat(path: Path, params, min_opacity=0.04):
    """antimatter15 .splat: float32 xyz, float32 scale xyz, uint8 RGBA, uint8 quaternion (w, x, y, z); big first."""
    with torch.no_grad():
        means = params["means"].detach().cpu().numpy()
        scales = torch.exp(params["scales"]).detach().cpu().numpy()
        q = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
        alpha = torch.sigmoid(params["opacities"]).detach().cpu().numpy()
        rgb = np.clip(params["sh0"][:, 0, :].detach().cpu().numpy() * C0 + 0.5, 0, 1)
    keep = alpha >= min_opacity
    means, scales, q, alpha, rgb = means[keep], scales[keep], q[keep], alpha[keep], rgb[keep]
    order = np.argsort(-(scales.prod(1) * alpha))
    rec = np.zeros(len(order), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"] = means[order], scales[order]
    rec["c"] = np.concatenate([rgb[order], alpha[order, None]], 1) * 255
    rec["r"] = np.clip(q[order] * 128 + 128, 0, 255)
    path.write_bytes(rec.tobytes())
    return int(keep.sum())


# ------------------------------------------------------------------------------------------------ training
def train(engine: Path, video: str, iters: int, voxel: float, max_points: int, hold_every: int):
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    dev = "cuda"
    out = engine / "10_splats"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    imgs, Ks, w2c, vf, (W, H) = load_frames(engine, video)
    n = len(imgs)
    test = [i for i in range(n) if i % hold_every == hold_every // 2]
    train_ids = [i for i in range(n) if i not in set(test)]
    log(f"{n} frames {W}x{H}: {len(train_ids)} for training, {len(test)} held out")
    xyz, rgb = load_points(engine, voxel, max_points)
    log(f"{len(xyz):,} starting Gaussians on the engine's surface ({voxel * 100:.1f} cm cells)")

    images = torch.from_numpy(imgs).to(dev)                         # uint8 (n, H, W, 3)
    K_t, V_t = torch.from_numpy(Ks).to(dev), torch.from_numpy(w2c).to(dev)
    centres = np.linalg.inv(w2c)[:, :3, 3]
    scene_scale = float(np.linalg.norm(centres - centres.mean(0), axis=1).max() * 1.1)

    N = len(xyz)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz).to(dev)),
        "scales": torch.nn.Parameter(torch.full((N, 3), math.log(voxel), device=dev)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.5, device=dev))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(rgb).to(dev) - 0.5) / C0)[:, None, :]),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    means_decay = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / iters))
    exposure = torch.nn.Parameter(torch.zeros(n, 6, device=dev))   # RGB gain - 1, RGB offset per frame
    exp_opt = torch.optim.Adam([exposure], lr=1e-3)

    strategy = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(iters * 0.6), reset_every=3000,
                               refine_every=100, prune_opa=0.005, verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(i):
        r, a, info = rasterization(params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                                   torch.sigmoid(params["opacities"]), params["sh0"], V_t[i:i + 1], K_t[i:i + 1],
                                   W, H, sh_degree=0, packed=False)
        return r[0].clamp(0, 1), info

    rng = np.random.default_rng(0)
    for step in range(iters):
        if step % 250 == 0:
            cool_down()
        i = int(rng.choice(train_ids))
        img, info = render(i)
        g = exposure[i]
        img = img * (1 + g[:3]) + g[3:]
        gt = images[i].float() / 255
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = 0.8 * F.l1_loss(img, gt) + 0.2 * (1 - ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        exp_opt.step()
        exp_opt.zero_grad(set_to_none=True)
        means_decay.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 500 == 0 or step == iters - 1:
            log(f"step {step:5d}/{iters}  loss {float(loss):.4f}  gaussians {len(params['means']):,}")

    # ---------------------------------------------------------------- evaluation
    def scores(ids, matched):
        p, s = [], []
        with torch.no_grad():
            for i in ids:
                img, _ = render(i)
                gt = images[i].float() / 255
                if matched:      # held-out frames have no learned exposure: match it by a least-squares gain
                    gain = (img * gt).sum((0, 1)) / (img * img).sum((0, 1)).clamp_min(1e-6)
                    img = (img * gain).clamp(0, 1)
                else:
                    g = exposure[i]
                    img = (img * (1 + g[:3]) + g[3:]).clamp(0, 1)
                p.append(psnr(img, gt))
                s.append(float(ssim(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])))
        return round(float(np.mean(p)), 2), round(float(np.mean(s)), 3)

    tr = scores(train_ids[:: max(1, len(train_ids) // 30)], matched=False)
    te = scores(test, matched=True)
    log(f"PSNR / SSIM  training {tr[0]} dB / {tr[1]}  held out {te[0]} dB / {te[1]}")
    with torch.no_grad():
        rows = []
        for i in test[:: max(1, len(test) // 4)][:4]:
            img, _ = render(i)
            gt = images[i].float() / 255
            gain = (img * gt).sum((0, 1)) / (img * img).sum((0, 1)).clamp_min(1e-6)
            pair = torch.cat([gt, (img * gain).clamp(0, 1)], 1).cpu().numpy()
            rows.append((pair * 255).astype(np.uint8))
        cv2.imwrite(str(out / "compare.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    kept = export_splat(out / "room.splat", params)
    m = {"frames": n, "train_frames": len(train_ids), "held_out_frames": len(test), "iterations": iters,
         "start_gaussians": N, "gaussians": len(params["means"]), "exported": kept,
         "psnr_train": tr[0], "ssim_train": tr[1], "psnr_held_out": te[0], "ssim_held_out": te[1],
         "minutes": round((time.time() - t0) / 60, 1), "size_mb": round((out / "room.splat").stat().st_size / 2**20, 1),
         "gsplat": __import__("gsplat").__version__, "torch": torch.__version__}
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    log(f"{kept:,} splats -> {out / 'room.splat'} ({m['size_mb']} MB) in {m['minutes']} min")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="the 3D engine's output folder (data/engine/<name>)")
    ap.add_argument("--video", help="default: the video in the engine's run_config.json")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--voxel", type=float, default=0.012, help="starting Gaussian spacing in metres")
    ap.add_argument("--max-points", type=int, default=600_000)
    ap.add_argument("--hold-every", type=int, default=10, help="hold out 1 frame in this many for evaluation")
    a = ap.parse_args()
    engine = Path(a.engine)
    video = a.video or json.loads((engine / "run_config.json").read_text(encoding="utf-8"))["video"]
    train(engine, video, a.iters, a.voxel, a.max_points, a.hold_every)


if __name__ == "__main__":
    main()
