"""Walkthrough inspection: 3D engine -> OpenCV analysis -> agent -> the viewer's files.

    python -m spatial_inspection.walkthrough --video room.mp4 --name room --property "12 Lake Rd"

1. 3D engine (engine3d/run_production.py, a subprocess on the GPU): Depth Anything 3 depth, pose and intrinsics;
   OpenCV 5 LK optical flow, SIFT + MAGSAC pose verification and trajectory refinement; persistent landmarks and loop
   candidates; frame-to-model point-to-plane ICP; 1.5 cm TSDF with free-space carving; gravity + Manhattan model frame;
   clean room shell with walls, floor and ceiling textured from the video; objects named by Grounding DINO.
   Then, in its own environment (.venv-splat), photoreal 3D Gaussian splats trained on the engine's cameras
   (engine3d/train_splats.py); optional, the build goes on without them.
2. OpenCV analysis on the engine's cameras and depth (pipeline.run): quality gate, evidence integrity, privacy
   redaction, cv2.Volume TSDF, surface coverage, keyframes, defect candidates placed on the room planes (furniture,
   doors and openings ruled out by depth, Grounding DINO and MobileNetV2), rating-plate text, rectified walls.
3. Agent (agent.run_agent): looks again from better views, asks for gaps to be re-filmed, grades the risk and
   proposes a decision for a person to approve.
4. Viewer files in results/<name>/ (the Property Inspection Platform's format): walkthrough.json, index.html (3D
   viewer), inspection_3d*.glb with numbered finding pins, live_map.mp4, coverage_map.png, coach.json, keyframes,
   crops, annotated keyframes, the engine's models / maps / videos, report.html, summary.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine3d"
ENGINE_OUT = Path(os.environ.get("SI_ENGINE_DIR", ROOT / "data" / "engine"))
WEB = Path(__file__).parent / "web"
# advanced: 1.5 cm TSDF, two ICP alignment passes, preview videos; lite: 3 cm, one pass, no videos
LITE_ARGS = ["--voxel", "0.03", "--align-iters", "1", "--no-video"]
ENGINE_STEPS = 14          # the engine prints [1] .. [14] as it goes
POOLS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SPLAT_ENV = Path(os.environ.get("SI_SPLAT_ENV", ROOT / ".venv-splat"))

DEFECT_LABEL = {"crack": "Crack", "stain": "Stain / water mark", "water_damage": "Water damage", "mould": "Mould",
                "mold": "Mould", "peeling": "Peeling paint", "peeling_paint": "Peeling paint", "rust": "Rust",
                "hole": "Hole", "spalling": "Spalling", "broken_glass": "Broken glass"}
STATE = {"confirmed": "DETECTED", "needs_review": "NEEDS_REVIEW", "candidate": "NEEDS_REVIEW", "dismissed": "DISMISSED"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------- 1. 3D engine
def engine_ready(d: Path) -> bool:
    return (d / "05_trajectory" / "cameras.json").exists() and (d / "03_floor_plan" / "layout.json").exists()


def run_engine(video: Path, out: Path, mode: str = "advanced", progress=None, log_file: Path | None = None) -> Path:
    """Run the 3D engine; `progress(fraction, line)` follows its numbered steps."""
    args = [sys.executable, "run_production.py", "--video", str(video.resolve()), "--output-dir", str(out.resolve()),
            "--chunk-size", os.environ.get("SI_DA3_CHUNK", "200")]
    args += (LITE_ARGS if mode == "lite" else []) + os.environ.get("SI_ENGINE_ARGS", "").split()
    threads = os.environ.get("SI_THREADS", "2")      # every thread pool capped: a laptop must stay usable
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "SI_THREADS": threads,
           **{k: os.environ.get(k, threads) for k in POOLS}}
    log(f"3D engine ({mode}): Depth Anything 3 + OpenCV 5 SLAM on {video.name}")
    fh = open(log_file, "a", encoding="utf-8") if log_file else None
    p = subprocess.Popen(args, cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        if fh:
            fh.write(line)
            fh.flush()
        m = re.match(r"\s*\[(\d+)\]\s+(.*)", line)
        if m and progress:
            progress(min(1.0, (int(m.group(1)) - 1) / ENGINE_STEPS), m.group(2).strip())
    p.wait()
    if fh:
        fh.close()
    if p.returncode != 0 or not engine_ready(out):
        raise RuntimeError(f"3D engine failed (exit {p.returncode}); see {log_file or out}")
    return out


def splat_python() -> Path | None:
    for p in (SPLAT_ENV / "Scripts" / "python.exe", SPLAT_ENV / "bin" / "python"):
        if p.exists():
            return p
    return None


def run_splats(slam_dir: Path, video: Path, progress=None, log_file: Path | None = None) -> bool:
    """Photoreal Gaussian splats on the engine's cameras (its own environment; skipped when it is not installed).
    A failure is logged and the walkthrough is built without splats."""
    py = splat_python()
    if py is None or os.environ.get("SI_SPLATS", "on").lower() in ("0", "off", "no"):
        return False
    threads = os.environ.get("SI_THREADS", "2")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "SI_THREADS": threads, **{k: os.environ.get(k, threads) for k in POOLS}}
    args = [str(py), str(ENGINE / "train_splats.py"), "--engine", str(slam_dir.resolve()), "--video", str(video.resolve()),
            "--iters", os.environ.get("SI_SPLAT_ITERS", "7000")]
    log("Gaussian splats: training on the engine's cameras")
    fh = open(log_file, "a", encoding="utf-8") if log_file else None
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        if fh:
            fh.write(line)
            fh.flush()
        m = re.search(r"step\s+(\d+)/(\d+)", line)
        if m and progress:
            progress(int(m.group(1)) / int(m.group(2)), "Photoreal Gaussian splats")
    p.wait()
    if fh:
        fh.close()
    ok = p.returncode == 0 and (slam_dir / "10_splats" / "room.splat").exists()
    if not ok:
        log(f"Gaussian splats failed (exit {p.returncode}); building without them")
    return ok


# ---------------------------------------------------------------------------------------------------- 4. viewer files
def _h264(src: Path, dst: Path) -> bool:
    """The engine writes MPEG-4 part 2 (not playable in browsers) -> H.264."""
    import imageio_ffmpeg
    r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
                        "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "24", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-movflags", "+faststart", str(dst)])
    return r.returncode == 0 and dst.exists()


def _frames_to_h264(frames_dir: Path, fps: float, dst: Path) -> None:
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", f"{fps:.3f}", "-i",
                    str(frames_dir / "%05d.jpg"), "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "23", "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(dst)], check=True)


ENGINE_MODELS = [("slam/room_slam.glb", "02_slam_model/room_slam_open_top.glb", "SLAM model (point layers + mesh)"),
                 ("slam/room_clean.glb", "01_clean_model/room_clean.glb", "Clean room model (walls + furniture)"),
                 ("slam/objects.glb", "01_clean_model/objects_only.glb", "Furniture / objects only")]
ENGINE_MAPS = [("slam/floor_plan.svg", "03_floor_plan/floor_plan.svg", "2D floor plan"),
               ("slam/occupancy_map.png", "03_floor_plan/occupancy_map.png", "Occupancy map"),
               ("slam/navigation_map.png", "03_floor_plan/navigation_map.png", "Navigation map (A* path)"),
               ("slam/slam_model_top.jpg", "07_previews/slam_model_top.jpg", "SLAM model — top view"),
               ("slam/clean_model_top.jpg", "07_previews/clean_model_top.jpg", "Clean model — top view"),
               ("slam/structure_segmentation.jpg", "07_previews/structure_segmentation.jpg", "Structure segmentation"),
               ("slam/objects_gallery.jpg", "07_previews/objects_gallery.jpg", "Detected objects"),
               ("slam/model_vs_video.jpg", "07_previews/model_vs_video.jpg", "Model vs video"),
               ("slam/depth_examples.jpg", "06_depth/depth_examples.jpg", "Depth examples")]
ENGINE_VIDEOS = [("slam/walkthrough_video_vs_model.mp4", "07_previews/walkthrough_video_vs_model.mp4", "Video vs 3D model"),
                 ("slam/depth_preview.mp4", "06_depth/depth_preview.mp4", "Depth (Depth Anything 3)"),
                 ("slam/slam_diagnostic.mp4", "07_previews/slam_diagnostic.mp4", "SLAM tracking diagnostic"),
                 ("slam/orbit_clean_model.mp4", "07_previews/orbit_clean_model.mp4", "Orbit of the clean model")]


def collect_extras(out: Path, slam_dir: Path, video: Path, keyframes: list[dict]) -> dict:
    from .spatial import glb3d
    (out / "slam").mkdir(exist_ok=True)
    ex = {"models": [], "maps": [], "videos": [], "keyframes": []}
    for kind, items in (("models", ENGINE_MODELS), ("maps", ENGINE_MAPS)):
        for dst, src, label in items:
            if (slam_dir / src).exists():
                shutil.copy(slam_dir / src, out / dst)
                if dst == "slam/room_clean.glb":
                    glb3d.clean_glb_file(out / dst)
                if dst.endswith(".glb"):
                    glb3d.make_unlit(out / dst)
                ex[kind].append({"file": dst, "label": label})
    if video.exists() and _h264(video, out / "source_video.mp4"):
        ex["videos"].append({"file": "source_video.mp4", "label": "Original walkthrough video"})
    for dst, src, label in ENGINE_VIDEOS:
        if (slam_dir / src).exists() and _h264(slam_dir / src, out / dst):
            ex["videos"].append({"file": dst, "label": label})
    ex["keyframes"] = [{k: kf[k] for k in ("inspection_id", "time_s", "image", "status", "room", "n_detected", "n_review",
                                          "error", "annotated", "report")} for kf in keyframes]
    return ex


def place_pin_tips(findings: list[dict], base: float = 0.35, min_gap: float = 0.28) -> None:
    """Pin head of every finding: out of the surface along its normal, fanned out when another head is too close."""
    placed: list[np.ndarray] = []
    for fd in findings:
        if fd.get("xyz") is None:
            fd["tip"] = None
            continue
        p = np.asarray(fd["xyz"], float)
        n = np.asarray(fd.get("normal") or [0, 1, 0], float)
        n = n / max(np.linalg.norm(n), 1e-6)
        side = np.cross(n, [0, 1, 0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([1.0, 0, 0])
        tip = p + n * base
        for k in range(1, 9):
            if all(np.linalg.norm(tip - q) >= min_gap for q in placed):
                break
            sgn = 1 if k % 2 else -1
            step = (k + 1) // 2
            tip = p + n * (base + 0.12 * step) + side * sgn * min_gap * step + np.array([0, 0.12 * step, 0])
        placed.append(tip)
        fd["tip"] = [round(float(v), 3) for v in tip]
        fd["normal"] = [round(float(v), 4) for v in n]


def _size_text(size: dict) -> str:
    if not size or size.get("error"):
        return ""
    if size.get("length_m"):
        return f"about {size['length_m'] * 100:.0f} cm long" + (f", {size['width_mm']:.1f} mm wide" if size.get("width_mm") else "")
    if size.get("area_m2"):
        return f"about {size['area_m2'] * 1e4:.0f} cm²"
    return ""


def to_viewer_findings(ag_findings: list[dict], result: dict, run, samples, w, kf_ids: dict, out: Path) -> list[dict]:
    """The OpenCV findings in the platform's schema: placed on the engine's surfaces, numbered, with pin tips."""
    from . import risk
    from .spatial.localize import snap_to_surface
    (out / "crops").mkdir(exist_ok=True)
    vf_to_run = {int(v): i for i, v in enumerate(run.video_frames)}
    found = []
    for f in ag_findings:
        fr = w.frames[f["frame"]]
        ri = vf_to_run.get(int(fr.video_frame), int(np.argmin(np.abs(run.video_frames - fr.video_frame))))
        xyz = None if f.get("xyz") is None else np.asarray(f["xyz"], float)
        surf, snapped, normal = "", xyz, None
        if xyz is not None:
            surf, snapped, _ = snap_to_surface(xyz, samples)
            if surf:
                normal = samples.normal[int(np.argmin(np.linalg.norm(samples.xyz - snapped, axis=1)))].tolist()
            else:
                v = run.c2w[ri][:3, 3] - xyz
                normal = (v / max(np.linalg.norm(v), 1e-6)).round(4).tolist()
                snapped = xyz
        kind = risk.defect_of(f)
        label = DEFECT_LABEL.get((f.get("defect") or f.get("kind") or "").lower(), DEFECT_LABEL.get(kind, "Surface damage"))
        comp = (f.get("surface") or "wall").split("_")[0]
        comp = comp if comp in ("wall", "floor", "ceiling") else "wall"
        group = samples.groups.get(surf, {}) if surf else {}
        size = f.get("size") or {}
        dist = None
        if xyz is not None:
            dist = round(float(np.linalg.norm(run.c2w[ri][:3, 3] - xyz)), 2)
        x, y, bw, bh = f["bbox"]
        size_m = None
        if dist:
            fx = float(run.K[ri][0, 0])
            size_m = [round(bw * dist / fx, 2), round(bh * dist / fx, 2)]
        crop = ""
        if f.get("crop") and (out / f["crop"]).exists():
            shutil.copy(out / f["crop"], out / "crops" / f"{f['id']}.jpg")
            crop = f"crops/{f['id']}.jpg"
        reasons = []
        if f.get("status") != "confirmed":
            reasons.append(f.get("reason") or "found by OpenCV detectors; needs a person to confirm")
        if f.get("grounding"):
            g = f["grounding"]
            reasons.append(f"Grounding DINO also sees '{g['label']}' here ({g['score']:.2f})")
        state = STATE.get(f.get("status", "candidate"), "NEEDS_REVIEW")
        where = f"{label} on the {group.get('label', comp)}" + (f", {_size_text(size)}" if _size_text(size) else "") + "."
        desc = " ".join(s for s in [where, f.get("evidence") or ""] if s).strip()
        found.append({
            "n": 0, "issue_id": f["id"], "inspection_id": kf_ids.get(f["frame"], f"KF-{f['frame']:04d}"), "frame": ri,
            "time_s": round(float(run.times[ri]), 2), "component": comp.capitalize(), "defect": label,
            "component_key": comp, "defect_type": kind, "state": state, "severity": risk.severity_of(f),
            "description": desc, "review_reasons": reasons, "professional_review": state != "DETECTED",
            "xyz": None if snapped is None else [round(float(v), 3) for v in snapped],
            "camera_distance_m": dist, "surface": surf, "surface_label": group.get("label", surf), "normal": normal,
            "crop": crop, "size_m": size_m, "size": size, "score": f.get("score"), "kind": f.get("kind"),
            "bbox": f["bbox"], "review": f.get("review"),
            "validation": {"opencv_score": f.get("score"), "views": f.get("views"), "depth_gap_m": f.get("depth_gap_m"),
                           "grounding_dino": f.get("grounding")},
            "models": {"detector": "OpenCV 5 (ridge filter, Lab colour, morphology, contours)",
                       "placement": "Depth Anything 3 depth + OpenCV SLAM pose, snapped to the room planes",
                       "semantic": "Grounding DINO + MobileNetV2 (cv2.dnn)", "decision": "inspection agent + human review"}})
    rank = {"DETECTED": 0, "NEEDS_REVIEW": 1, "DISMISSED": 2}
    found.sort(key=lambda x: (rank.get(x["state"], 3), x["frame"]))
    for i, fd in enumerate(found, 1):
        fd["n"] = i
    place_pin_tips(found)
    return found


def annotate(img: np.ndarray, items: list[dict]) -> np.ndarray:
    out = img.copy()
    col = {"DETECTED": (38, 38, 220), "NEEDS_REVIEW": (12, 150, 234), "DISMISSED": (150, 150, 150)}
    for fd in items:
        x, y, bw, bh = (int(v) for v in fd["bbox"])
        c = col.get(fd["state"], (235, 99, 37))
        cv2.rectangle(out, (x, y), (x + bw, y + bh), c, 3)
        tag = f"#{fd['n']} {fd['defect']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(th + 8, y)
        cv2.rectangle(out, (x, y0 - th - 8), (x + tw + 8, y0), c, -1)
        cv2.putText(out, tag, (x + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def summary_text(name: str, risk: dict, stats: dict, findings: list[dict], gaps: list[dict], integrity: dict | None) -> str:
    """The property-level summary the viewer shows (written from the record, no language model)."""
    det = [f for f in findings if f["state"] == "DETECTED"]
    rev = [f for f in findings if f["state"] == "NEEDS_REVIEW"]
    ov = stats["overall"]
    lines = ["**Overall condition**  ",
             f"Grade {risk['grade']} ({risk['grade_label']}), condition score {risk['condition_score']}, "
             f"risk level {risk['risk_level']}. Decision: {risk['decision']}.", "",
             "**Evidence**  ", f"The video saw {ov['seen_pct']}% of the mapped walls, floor and ceiling; {ov['inspectable_pct']}% "
             f"was filmed close and square enough to inspect and {ov['inspected_pct']}% was analysed in keyframes."]
    if integrity:
        flags = "; ".join(x["message"] for x in integrity.get("flags", [])[:3])
        lines.append(f"Evidence integrity: {integrity.get('level', 'low')} risk" + (f" ({flags})" if flags else "") + ".")
    lines += ["", "**Findings**  "]
    if det or rev:
        lines.append(f"{len(det)} confirmed and {len(rev)} waiting for review.")
        for f in (det + rev)[:8]:
            state = f["state"].replace("_", " ").lower()
            where = f["surface_label"] or f["component"].lower()
            lines.append(f"- #{f['n']} {f['defect']} on the {where} ({f['severity']}, {state})")
    else:
        lines.append("No defects were found on the inspected surfaces.")
    if gaps:
        lines += ["", "**Film again**  "]
        for g in gaps[:5]:
            lines.append(f"- {g['label']}: " + (f"{g['length_m']} m" if g.get("length_m") else
                                                f"{g.get('uncovered_m2', 0)} m² not inspectable"))
    return "\n".join(lines)


def build_viewer(name: str, video: Path, slam_dir: Path, out: Path, w, result: dict, agent: dict, prop: str | None,
                 mode: str = "advanced", live_video: bool = True, t_start: float | None = None) -> dict:
    import trimesh

    from . import risk as risk_mod
    from .spatial import glb3d, map2d
    from .spatial.coach import coach_data
    from .spatial.coverage import compute_coverage, make_samples, mark_inspected
    from .spatial.slam_data import load_slam_run

    t0 = t_start or time.perf_counter()
    run = load_slam_run(slam_dir)
    samples = make_samples(run.layout, 0.1)
    cov = compute_coverage(run, samples)
    vf_to_run = {int(v): i for i, v in enumerate(run.video_frames)}
    kf_frames = [k["frame"] for k in result["keyframes"]]
    kfs = sorted({vf_to_run.get(int(w.frames[i].video_frame), 0) for i in kf_frames})
    mark_inspected(cov, kfs)
    stats, gaps = cov.stats(), cov.gaps()
    log(f"coverage on the engine's room model: {stats['overall']}")

    kf_ids = {i: f"KF-{vf_to_run.get(int(w.frames[i].video_frame), 0):04d}" for i in kf_frames}
    findings = to_viewer_findings(agent.get("findings") or result.get("findings", []), result, run, samples, w, kf_ids, out)
    shown = [f for f in findings if f["state"] != "DISMISSED"]

    # keyframes: image, annotated copy, counts
    (out / "annotated").mkdir(exist_ok=True)
    keyframes = []
    for i in kf_frames:
        kid = kf_ids[i]
        ri = vf_to_run.get(int(w.frames[i].video_frame), 0)
        img_rel = f"keyframes/kf_{i:04d}.jpg"
        mine = [f for f in shown if f["inspection_id"] == kid]
        img = cv2.imread(str(out / img_rel))
        ann = ""
        if img is not None:
            cv2.imwrite(str(out / "annotated" / f"{kid}.jpg"), annotate(img, mine), [cv2.IMWRITE_JPEG_QUALITY, 90])
            ann = f"annotated/{kid}.jpg"
        keyframes.append({"frame": ri, "time_s": round(float(run.times[ri]), 2), "inspection_id": kid, "image": img_rel,
                          "position": [round(float(v), 3) for v in run.c2w[ri][:3, 3]], "status": "done", "room": "",
                          "n_detected": sum(f["state"] == "DETECTED" for f in mine),
                          "n_review": sum(f["state"] == "NEEDS_REVIEW" for f in mine), "error": "", "annotated": ann,
                          "report": ann})

    # 2D maps and the coverage coach
    pm = map2d.PlanMap(run, cov)
    final = pm.render(None, kfs, shown)
    lines = [f"Overall: seen {stats['overall']['seen_pct']}%, inspectable {stats['overall']['inspectable_pct']}%,"
             f" inspected {stats['overall']['inspected_pct']}%", f"Findings: {len(shown)}  |  keyframes: {len(kfs)}"]
    lines += ["Not covered (re-film):"] + [f"  - {g['label']}: " + (f"{g['length_m']} m" if g.get("length_m") else
                                                                     f"{g['uncovered_m2']} m2") for g in gaps[:6]]
    panel = map2d.stats_panel(final.shape[1], 170 + 21 * len(lines), stats, float(run.times[-1]), lines)
    cv2.imwrite(str(out / "coverage_map.png"), np.vstack([final, map2d.legend(final.shape[1]), panel]))
    target = float(risk_mod.RULES.get("min_inspected_pct", 60))
    (out / "coach.json").write_text(json.dumps(coach_data(run, cov, kfs, name=name, target_pct=target), separators=(",", ":")),
                                    encoding="utf-8")
    series = []
    for f in range(run.n):
        s = cov.stats(f)
        series.append({"t": round(float(run.times[f]), 2), "overall_seen": s["overall"]["seen_pct"],
                       "overall_inspectable": s["overall"]["inspectable_pct"],
                       **{f"{k}_seen": v["seen_pct"] for k, v in s["by_kind"].items()},
                       **{f"{k}_inspectable": v["inspectable_pct"] for k, v in s["by_kind"].items()}})

    if live_video:
        log("rendering the live map video ...")
        cap = cv2.VideoCapture(str(video))
        tmp = out / "_frames"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        vw, hold = 640, None
        kf_by_run = {k["frame"]: k for k in keyframes}
        for f in range(run.n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(run.video_frames[f]))
            ok, fr = cap.read()
            m = pm.render(f, kfs, shown)
            if f in kf_by_run and kf_by_run[f]["annotated"]:
                hold = (cv2.imread(str(out / kf_by_run[f]["annotated"])), f, kf_by_run[f]["inspection_id"])
            src = fr if ok else np.zeros((480, 848, 3), np.uint8)
            vid = cv2.resize(src, (vw, int(vw * src.shape[0] / src.shape[1])))
            if hold is not None and hold[0] is not None and f - hold[1] <= 6:
                vid = cv2.resize(hold[0], (vw, int(vw * hold[0].shape[0] / hold[0].shape[1])))
                map2d._put(vid, f"OpenCV analysis {hold[2]}", (10, 26), 0.7, (180, 0, 180), 2)
            s = cov.stats(f)
            nf = sum(1 for x in shown if x["frame"] <= f)
            left_h = max(m.shape[0], vid.shape[0] + 220)
            p2 = map2d.stats_panel(vw, left_h - vid.shape[0], s, float(run.times[f]),
                                   [f"Overall seen {s['overall']['seen_pct']}%  inspectable {s['overall']['inspectable_pct']}%",
                                    f"Findings so far: {nf}", f"Pose: {run.pose_state[f]}  frame quality: {run.frame_quality[f]}"])
            left = np.vstack([vid, p2])
            right = np.vstack([m, np.full((left_h - m.shape[0], m.shape[1], 3), 255, np.uint8)]) if left_h > m.shape[0] else m
            if right.shape[0] > left.shape[0]:
                left = np.vstack([left, np.full((right.shape[0] - left.shape[0], left.shape[1], 3), 255, np.uint8)])
            cv2.imwrite(str(tmp / f"{f:05d}.jpg"), np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cap.release()
        fps = run.n / max(1e-6, float(run.times[-1] - run.times[0])) if run.n > 1 else 10
        _frames_to_h264(tmp, fps, out / "live_map.mp4")
        shutil.rmtree(tmp)

    log("building the 3D models with finding pins ...")
    glb3d.build_glb(run, cov, shown, kfs, out / "inspection_3d.glb", open_top=False, with_gaps=False,
                    drop=("slam_landmarks",), clean=True)
    glb3d.build_glb(run, cov, shown, kfs, out / "inspection_3d_gaps.glb", open_top=True, with_gaps=True)
    clean_glb = slam_dir / "01_clean_model" / "room_clean.glb"
    if clean_glb.exists():
        glb3d.build_glb(run, cov, shown, kfs, out / "inspection_3d_inside.glb", with_gaps=False, base=clean_glb,
                        with_keyframes=False, clean=True)
        glb3d.clean_presentation(trimesh.load(str(clean_glb), force="scene"))

    # risk on the engine's coverage (the evidence the viewer shows)
    res_for_risk = {**result, "coverage": stats}
    risk = risk_mod.assess_result([{**f, "status": f.get("status")} for f in (agent.get("findings") or result.get("findings", []))],
                                  res_for_risk)
    warnings = []
    if run.ceiling > 3.3:
        warnings.append(f"Reconstructed ceiling height is {run.ceiling:.2f} m. If the real ceiling is lower, the metric scale "
                        f"is too large (it assumes the camera was held {run.extras['config'].get('camera_height', 1.5)} m above "
                        "the floor); distances are inflated and 'inspectable' (<= 3 m) is under-counted.")
    if stats["overall"]["seen_pct"] < 50:
        warnings.append("Less than half of the mapped surfaces were seen - the walkthrough did not cover the whole area.")
    rej = result.get("rejected") or []
    by_reason: dict[str, int] = {}
    for r in rej:
        why = r.get("reason", "")
        if any(s in why for s in ("in front of", "stands out", "looks like", "Grounding DINO")):
            key = "furniture or an object in front of the wall"
        elif "door" in why or "window" in why:
            key = "a door or window"
        elif "beyond" in why:
            key = "seen through an opening"
        else:
            key = "outside the room model" if "outside" in why else why
        by_reason[key] = by_reason.get(key, 0) + 1
    dismissed = [f for f in findings if f["state"] == "DISMISSED"]
    if dismissed:
        by_reason["dismissed by the agent after a second look"] = len(dismissed)

    fy = float(np.median(run.K[:, 1, 1]))
    W = {"name": name, "mode": mode, "built": time.strftime("%Y%m%d%H%M%S"), "warnings": warnings, "video": video.name,
         "slam_dir": slam_dir.name, "duration_s": round(float(run.times[-1]), 2), "slam_frames": run.n,
         "image_size": list(run.image_size), "ceiling_m": run.ceiling,
         "rooms": [{"id": r["id"], "name": r["name"], "area_m2": r["area_m2"], "size_m": r["size_m"]}
                   for r in run.layout.get("rooms", [])],
         "coverage": stats, "coverage_params": cov.params, "gaps": gaps, "series": series,
         "keyframes": [{k: kf[k] for k in ("frame", "time_s", "inspection_id", "image", "position")} for kf in keyframes],
         "findings": findings, "runtime_s": round(time.perf_counter() - t0, 1),
         "files": {"live_map": "live_map.mp4" if live_video else None, "coverage_map": "coverage_map.png",
                   "glb": "inspection_3d.glb", "glb_gaps": "inspection_3d_gaps.glb",
                   "glb_inside": "inspection_3d_inside.glb" if (out / "inspection_3d_inside.glb").exists() else None,
                   "report": "report.html"},
         "camera": {"fov_v_deg": round(float(np.degrees(2 * np.arctan(run.image_size[1] / 2 / fy))), 2),
                    "path": [{"t": round(float(run.times[f]), 3), "p": [round(float(v), 3) for v in run.c2w[f][:3, 3]],
                              "f": [round(float(v), 4) for v in run.c2w[f][:3, 2]]} for f in range(run.n)]}}
    W["extras"] = collect_extras(out, slam_dir, video, keyframes)
    splat = slam_dir / "10_splats" / "room.splat"
    if splat.exists():          # photoreal mode in the viewer
        shutil.copy(splat, out / "slam" / "room.splat")
        W["files"]["splat"] = "slam/room.splat"
        m = slam_dir / "10_splats" / "metrics.json"
        W["splats"] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else {}
        if (slam_dir / "10_splats" / "compare.jpg").exists():
            shutil.copy(slam_dir / "10_splats" / "compare.jpg", out / "slam" / "splats_vs_video.jpg")
            W["extras"]["maps"].append({"file": "slam/splats_vs_video.jpg", "label": "Video frame vs Gaussian splats (held out)"})
    W["property_id"] = prop or name
    W["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    W["risk"] = risk
    # one set of numbers everywhere: the report and the dashboard read agent.json
    agent["risk"] = risk
    (out / "agent.json").write_text(json.dumps(agent, indent=1, default=str), encoding="utf-8")
    W["integrity"] = result.get("integrity")
    W["rejected"] = {"total": sum(by_reason.values()), "by_reason": by_reason}
    W["agent"] = {"engine": agent.get("engine"), "trace": agent.get("trace", []), "refilm": agent.get("refilm", []),
                  "proposal": agent.get("decision"), "approval": agent.get("approval")}
    W["engine"] = engine_summary(slam_dir)
    W["opencv"] = {"version": cv2.__version__, "timings_s": result.get("timings_s"), "labels": result.get("labels"),
                   "walls": (result.get("visuals") or {}).get("walls"), "privacy": result.get("privacy")}
    W["ai_summary"] = summary_text(name, risk, stats, findings, gaps, result.get("integrity"))
    (out / "walkthrough.json").write_text(json.dumps(finite(W), indent=1, allow_nan=False), encoding="utf-8")
    shutil.copy(WEB / "walkthrough.html", out / "index.html")
    try:
        from . import report
        (out / "report.html").write_text(report.to_html(out), encoding="utf-8")
    except Exception as e:  # noqa: BLE001  the report is a convenience; the viewer stands without it
        log(f"report skipped: {e}")
    md = [f"# Walkthrough inspection — {name}", "", f"- Video: {video.name} ({W['duration_s']} s, {run.n} SLAM frames)",
          "- Rooms: " + ", ".join(f"{r['name']} {r['area_m2']} m2" for r in W["rooms"]),
          f"- Coverage: seen {stats['overall']['seen_pct']}%, inspectable {stats['overall']['inspectable_pct']}%, "
          f"inspected {stats['overall']['inspected_pct']}%", ""] + [f"> **Warning:** {x}" for x in warnings] + ["", W["ai_summary"]]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    log(f"viewer files written in {time.perf_counter() - t0:.0f}s -> {out}")
    return W


def finite(o):
    """JSON the browser accepts: numpy values as plain numbers, NaN and infinity as null."""
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return finite(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def engine_summary(slam_dir: Path) -> dict:
    """What the 3D engine did, for the viewer and the report (from its evaluation file when present)."""
    s = {"depth": "Depth Anything 3 (DA3-BASE, Apache-2.0) on the GPU",
         "tracking": "OpenCV 5: LK optical flow, SIFT + MAGSAC pose verification, trajectory refinement, loop candidates",
         "fusion": "point-to-plane ICP frame-to-model alignment, TSDF mesh, free-space carving",
         "room": "gravity + Manhattan frame, plane regularisation, textured clean shell",
         "objects": "Grounding DINO multi-view object names"}
    ev = slam_dir / "08_evaluation" / "evaluation.json"
    if ev.exists():
        try:
            e = json.loads(ev.read_text(encoding="utf-8"))
        except ValueError:
            e = {}
        sl = e.get("slam", {})
        s["metrics"] = {
            "frames": sl.get("frames"), "keyframes": sl.get("keyframes"), "pair_checks": sl.get("pair_checks"),
            "pose_states": sl.get("pose_states"), "icp": sl.get("alignment"), "landmarks": sl.get("landmarks"),
            "reprojection_px": sl.get("reprojection"), "loops": sl.get("loops"),
            "opencv_agreement": {"da3_raw": sl.get("opencv_agreement_raw"), "refined": sl.get("opencv_agreement_refined")},
            "smoothness": {"da3_raw": sl.get("smoothness_raw"), "refined": sl.get("smoothness_refined")},
            "walls": [{k: w.get(k) for k in ("id", "length_m", "rms_cm", "p90_abs_cm", "seen_pct")} for w in e.get("walls", [])],
            "photometric": e.get("photometric"), "fidelity": e.get("fidelity"), "objects": e.get("objects"),
            "layout": {k: v for k, v in (e.get("layout") or {}).items() if k != "rooms"}, "minutes": e.get("timing_min")}
    return s


# ---------------------------------------------------------------------------------------------------- all of it
def build(video: Path, name: str, out: Path, prop: str | None = None, mode: str = "advanced", progress=None,
          slam_dir: Path | None = None, live_video: bool = True) -> dict:
    """`progress(step, fraction, detail)` with step in engine / splats / analysis / agent / viewer."""
    from .agent import run_agent
    from .pipeline import STAGES, run
    from .scene import load_platform_run

    t0 = time.perf_counter()
    video = Path(video).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prog = progress or (lambda *a: None)
    slam_dir = Path(slam_dir) if slam_dir else ENGINE_OUT / name
    if not engine_ready(slam_dir):
        run_engine(video, slam_dir, mode=mode, progress=lambda fr, line: prog("engine", fr, line),
                   log_file=out / "engine.log")
    prog("engine", 1.0, "3D model ready")
    if not (slam_dir / "10_splats" / "room.splat").exists():
        run_splats(slam_dir, video, progress=lambda fr, d: prog("splats", fr, d), log_file=out / "engine.log")

    w = load_platform_run(slam_dir, video, name)
    prog("analysis", 0.0, STAGES[0])

    def on_stage(done: str, frac: float) -> None:
        i = STAGES.index(done) + 1
        prog("analysis", frac, STAGES[i] if i < len(STAGES) else "agent")

    result = run(w, out, prop=prop, progress=on_stage)
    prog("agent", 0.0, "agent")
    # local first: the rule-based policy unless SI_AGENT=auto/bedrock asks for the Bedrock model
    agent = run_agent(result, out, w, prefer=os.environ.get("SI_AGENT", "policy"))
    prog("viewer", 0.0, "3D viewer, live map and GLBs")
    W = build_viewer(name, video, slam_dir, out, w, result, agent, prop, mode=mode, live_video=live_video, t_start=t0)
    prog("viewer", 1.0, "done")
    return W


class Status:
    """results/<name>/status.json as the web app writes it, so a run started here shows on the dashboard with its
    progress (shares of the whole job: engine 0-62%, splats 62-70%, analysis 70-88%, agent 88-90%, viewer 90-100%)."""

    SHARE = {"engine": (0.0, 0.62), "splats": (0.62, 0.70), "analysis": (0.70, 0.88), "agent": (0.88, 0.90),
             "viewer": (0.90, 1.0)}
    LABEL = {"quality": "Frame quality gate", "integrity": "Evidence-integrity checks",
             "privacy": "Blurring faces and private text", "fusion": "Fusing depth into 3D (TSDF volume)",
             "layout": "Fitting walls, floor and ceiling", "coverage": "Measuring surface coverage",
             "defects": "Finding and placing defects", "labels": "Reading labels",
             "visuals": "Rectifying walls and drawing the floor plan", "agent": "Agent reviewing the evidence"}

    def __init__(self, path: Path, **info) -> None:
        self.path, self.started, self.last = path, time.time(), 0.0
        self.write(state="processing", stage="Starting", step="engine", progress=0.0, started=self.started, **info)

    def write(self, **kw) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cur = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        cur.update(kw, updated=time.time(), elapsed_s=round(time.time() - self.started))
        self.path.write_text(json.dumps(cur), encoding="utf-8")

    def update(self, step: str, frac: float, detail: str) -> None:
        if time.time() - self.last < 0.7 and frac < 1.0:
            return
        self.last = time.time()
        lo, hi = self.SHARE.get(step, (0.0, 1.0))
        done = lo + (hi - lo) * max(0.0, min(1.0, frac))
        elapsed = time.time() - self.started
        self.write(step=step, stage=self.LABEL.get(detail, detail or step), progress=round(done, 3),
                   eta_s=round(elapsed / done - elapsed) if done > 0.03 else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--name")
    ap.add_argument("--property")
    ap.add_argument("--out", help="default: results/<name>")
    ap.add_argument("--slam-dir", help="existing 3D engine output (default: data/engine/<name>, run when missing)")
    ap.add_argument("--mode", choices=["advanced", "lite"], default="advanced")
    ap.add_argument("--no-video", action="store_true", help="skip live_map.mp4")
    a = ap.parse_args()
    name = a.name or Path(a.video).stem
    out = Path(a.out) if a.out else ROOT / "results" / name
    video = Path(a.video).resolve()
    data = ROOT / "data"
    status = Status(out / "status.json", video=video.name, property=a.property, mode=a.mode,
                    video_path=video.relative_to(data).as_posix() if video.is_relative_to(data) else None)

    def progress(step: str, frac: float, detail: str) -> None:
        log(f"{step} {frac:.0%} {detail}")
        status.update(step, frac, detail)

    try:
        W = build(Path(a.video), name, out, prop=a.property, mode=a.mode, slam_dir=Path(a.slam_dir) if a.slam_dir else None,
                  live_video=not a.no_video, progress=progress)
    except BaseException as e:
        status.write(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    status.write(state="done", stage="Done", step="done", progress=1.0, eta_s=0)
    print(json.dumps({"name": W["name"], "coverage": W["coverage"]["overall"], "findings": len(W["findings"]),
                      "grade": W["risk"]["grade"], "decision": W["risk"]["decision"]}, indent=1))


if __name__ == "__main__":
    main()
