"""One walkthrough, end to end. Every stage is OpenCV 5 work; the agent (agent.py) runs afterwards on the results.

    quality gate -> privacy redaction -> TSDF fusion -> room layout -> coverage -> keyframes -> defect evidence
    -> 3D placement and size -> label reading -> wall orthomosaics and floor plan -> result.json

    python -m spatial_inspection.pipeline --video data/samples/demo_property_lite.mp4 \
        --platform-cache D:/quntbit/property-inspection-platform/data/slam/demo_property_lite --out results/demo
    python -m spatial_inspection.pipeline --video my_walkthrough.mp4 --out results/mine     # OpenCV-only path
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import coverage as cov_mod
from .fusion import export_web_points, fuse
from .layout import layout_from_points, manhattan_yaw, rotate_walkthrough
from .localize import merge_repeats, place_on_surface
from .scene import Walkthrough, load_platform_run

STAGES = ["quality", "integrity", "privacy", "fusion", "layout", "coverage", "defects", "labels", "visuals"]


class Timer:
    def __init__(self, progress=None):
        self.stages: dict[str, float] = {}
        self._t = time.perf_counter()
        self.progress = progress     # progress(stage_done, fraction_of_pipeline_done)

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = round(now - self._t, 2)
        if self.progress is not None and name in STAGES:
            self.progress(name, (STAGES.index(name) + 1) / len(STAGES))
        self._t = now


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def quality_gate(w: Walkthrough) -> None:
    """Sharpness, exposure and motion per frame, unless the video stage already did it."""
    try:
        from .video import assess_frames  # the OpenCV-only video path's gate, when available
        assess_frames(w.frames)
        return
    except (ImportError, AttributeError, TypeError):
        pass
    prev = None
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    for f in w.frames:
        g = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2))
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        mean = float(g.mean())
        motion = 0.0
        if prev is not None:
            flow = dis.calc(prev, small, None)
            motion = float(np.median(np.linalg.norm(flow, axis=2))) * 2
        prev = small
        verdict = "BAD" if sharp < 40 or mean < 25 or motion > 25 else "POOR" if sharp < 90 or mean < 50 else "GOOD"
        f.quality.update({"sharpness": round(sharp, 1), "brightness": round(mean, 1), "motion_px": round(motion, 2),
                          "verdict": f.quality.get("verdict", verdict)})


def run(w: Walkthrough, out: Path, max_keyframes: int = 16, prop: str | None = None, progress=None) -> dict:
    """`prop` names the property the walkthrough was submitted for (used by the reused-footage check);
    `progress(stage, fraction)` is called as each stage finishes."""
    out = Path(out)
    (out / "keyframes").mkdir(parents=True, exist_ok=True)
    (out / "findings").mkdir(exist_ok=True)
    tm = Timer(progress)
    w.load_images()
    quality_gate(w)
    tm.lap("quality")

    # integrity on the frames as filmed: one continuous take, not reused, not a screen being filmed
    from . import integrity
    evidence = integrity.check(w, root=out.parent, prop=prop)
    tm.lap("integrity")

    # privacy first: nothing leaves this stage with a face or private text in it
    privacy = {"faces": 0, "text_regions": 0}
    try:
        from .privacy import redact
        for f in w.frames:
            f.image, rep = redact(f.image)
            privacy["faces"] += int(rep.get("faces", 0))
            privacy["text_regions"] += int(rep.get("text", rep.get("text_regions", 0)))
    except ImportError:
        privacy["skipped"] = "privacy module not available"
    tm.lap("privacy")

    fused = fuse(w, voxel=0.04)
    tm.lap("fusion")
    if w.layout is None:
        yaw = manhattan_yaw(fused.normals)
        T = rotate_walkthrough(w, yaw)
        fused.points = fused.points @ T[:3, :3].T
        fused.normals = fused.normals @ T[:3, :3].T
        cams = np.array([f.c2w[:3, 3] for f in w.frames if f.c2w is not None])
        w.layout = layout_from_points(fused, cameras=cams)
    n_points = export_web_points(out / "points.bin", fused)
    tm.lap("layout")

    samples = cov_mod.make_samples(w.layout, 0.1)
    cov = cov_mod.compute_coverage(w, samples)
    sharp = np.array([f.quality.get("sharpness") or sharpness(f.image) for f in w.frames])
    keyframes = cov_mod.select_keyframes(cov, sharp, max_frames=max_keyframes)
    cov_mod.mark_inspected(cov, keyframes)
    tm.lap("coverage")

    findings, rejected = detect_findings(w, keyframes, samples, out)
    tm.lap("defects")

    labels = read_labels(w, keyframes)
    tm.lap("labels")

    for i in keyframes:
        cv2.imwrite(str(out / "keyframes" / f"kf_{i:04d}.jpg"), w.frames[i].image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    visuals = render_visuals(w, findings, out)
    tm.lap("visuals")

    stats = cov.stats()
    result = {
        "name": w.name,
        "source": w.meta.get("source", "opencv"),
        "opencv": cv2.__version__,
        "frames": len(w.frames),
        "fps": w.fps,
        "image_size": list(w.size),
        "property": prop,
        "quality": summarise_quality(w),
        "integrity": evidence,
        "privacy": privacy,
        "fusion": {"points": int(len(fused.points)), "web_points": n_points, "voxel_m": fused.voxel,
                   "frames": fused.frames_used},
        "layout": w.layout,
        "coverage": stats,
        "gaps": cov.gaps(),
        "keyframes": [{"frame": i, "t": round(w.frames[i].t, 2), "image": f"keyframes/kf_{i:04d}.jpg",
                       "contributes": cov_mod.sample_state_for_frame(cov, i)} for i in keyframes],
        "findings": findings,
        "rejected": rejected,
        "labels": labels,
        "visuals": visuals,
        "trajectory": [w.frames[i].position.round(3).tolist() for i in range(len(w.frames)) if w.frames[i].c2w is not None],
        "timings_s": tm.stages,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, default=_json), encoding="utf-8")
    np.savez_compressed(out / "coverage.npz", xyz=samples.xyz.astype(np.float32), kind=samples.kind,
                        state=cov.state(), group=np.array(samples.group))
    return result


def detect_findings(w: Walkthrough, keyframes: list[int], samples, out: Path) -> tuple[list[dict], list[dict]]:
    """Defect candidates on the keyframes that lie on the building's surfaces, and the ones ruled out (with why)."""
    try:
        from . import defects
    except ImportError:
        return [], []
    from . import grounding
    from . import objects as _objects
    objects = _objects if _objects.available() else None
    semantic = grounding.available()
    found, rejected = [], []
    for i in keyframes:
        f = w.frames[i]
        things, damage = semantic_boxes(f.image) if semantic else ([], [])
        for c in defects.detect_all(f.image)[:6]:
            if c.score < 0.35:
                continue
            x, y, bw, bh = (int(v) for v in c.bbox)
            where = place_on_surface(f, (x, y, bw, bh), w.layout, samples, mask=c.mask)
            if not where["on_surface"]:
                rejected.append({"kind": c.kind, "score": round(float(c.score), 3), "frame": i, "bbox": [x, y, bw, bh],
                                 "reason": where["reason"]})
                continue
            on = next((t for t in things if _inside((x, y, bw, bh), t["box"]) >= 0.6), None)
            if on is not None:
                rejected.append({"kind": c.kind, "score": round(float(c.score), 3), "frame": i, "bbox": [x, y, bw, bh],
                                 "reason": f"on a {on['label']} (Grounding DINO {on['score']:.2f}), not the wall"})
                continue
            pad = int(0.25 * max(bw, bh)) + 12
            crop = f.image[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad]
            obj = objects.household_object(crop) if objects is not None else None
            if obj is not None:
                rejected.append({"kind": c.kind, "score": round(float(c.score), 3), "frame": i, "bbox": [x, y, bw, bh],
                                 "reason": f"looks like a {obj['object']} (MobileNetV2 {obj['p']:.2f}), not damage"})
                continue
            place = where["place"]
            size = {}
            try:
                size = defects.measure(c, f.depth, f.K_depth, (f.image.shape[1], f.image.shape[0]))
                size = {k: round(v, 3) if isinstance(v, float) else v for k, v in size.items()}
            except Exception as e:   # measurement is best effort; the finding still stands
                size = {"error": str(e)}
            fid = f"F{len(found) + 1:03d}"
            try:
                from .enhance import enhance_for_review
                crop = enhance_for_review(crop)
            except ImportError:
                pass
            cv2.imwrite(str(out / "findings" / f"{fid}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            found.append({
                "id": fid, "kind": c.kind, "score": round(float(c.score), 3), "frame": i, "t": round(f.t, 2),
                "bbox": [x, y, bw, bh], "crop": f"findings/{fid}.jpg",
                "features": {k: _json(v) for k, v in (c.features or {}).items()},
                "size": size, "status": "candidate", "xyz": place["xyz"], "surface": place["surface"],
                "group": place["group"], "depth_gap_m": place.get("depth_gap_m"),
                "grounding": next(({"label": d["label"], "score": d["score"]} for d in damage
                                   if _inside((x, y, bw, bh), d["box"]) >= 0.3
                                   or _inside(_xywh(d["box"]), [x, y, x + bw, y + bh]) >= 0.5), None),
            })
    if semantic:
        grounding.release()
    return merge_repeats(found), rejected


# Grounding DINO phrases: things a defect detector mistakes for damage, and the damage itself
THINGS = ["chair", "sofa", "table", "cabinet", "shelf", "bed", "door", "picture frame", "painting", "clock", "fan",
          "lamp", "curtain", "television", "plant", "mirror", "window", "air conditioner", "box", "bag", "pillow",
          "light switch", "power socket", "wire", "pipe"]
DAMAGE = ["crack", "water stain", "mold", "peeling paint", "rust", "hole"]


def semantic_boxes(img: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Grounding DINO boxes on one keyframe: objects (big ones dropped: a box over most of the frame is not one
    object in front of the wall) and visible damage."""
    from . import grounding
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    area = img.shape[0] * img.shape[1]
    things = []
    for k in range(0, len(THINGS), 13):
        things += grounding.detect(rgb, THINGS[k:k + 13], box_threshold=0.35) or []
    things = [t for t in things if (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) < 0.5 * area]
    damage = grounding.detect(rgb, DAMAGE, box_threshold=0.3) or []
    return things, damage


def _xywh(box) -> tuple[float, float, float, float]:
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


def _inside(b, box) -> float:
    """Share of the x, y, w, h box `b` that lies inside the x0, y0, x1, y1 box."""
    x, y, w, h = b
    ix = max(0.0, min(x + w, box[2]) - max(x, box[0])) * max(0.0, min(y + h, box[3]) - max(y, box[1]))
    return ix / max(w * h, 1e-6)


def read_labels(w: Walkthrough, keyframes: list[int]) -> list[dict]:
    try:
        from . import ocr
    except ImportError:
        return []
    out = []
    for i in keyframes:
        try:
            lines = ocr.read_text(w.frames[i].image)
        except Exception:
            continue
        lines = [ln for ln in lines if ln.get("confidence", 0) > 0.5 and len(ln.get("text", "")) >= 3]
        if lines:
            out.append({"frame": i, "lines": [{"text": ln["text"], "confidence": round(float(ln["confidence"]), 2)}
                                              for ln in lines[:12]]})
    return out


def render_visuals(w: Walkthrough, findings: list[dict], out: Path) -> dict:
    vis: dict = {}
    try:
        from . import ortho
    except ImportError:
        return vis
    (out / "walls").mkdir(exist_ok=True)
    traj = [f.position for f in w.frames if f.c2w is not None]
    try:
        plan = ortho.floor_plan(w.layout, findings=findings, trajectory=traj)
        cv2.imwrite(str(out / "floor_plan.png"), plan)
        vis["floor_plan"] = "floor_plan.png"
    except Exception as e:
        vis["floor_plan_error"] = str(e)
    walls = []
    for wall in ortho.wall_planes(w.layout):
        try:
            rect = ortho.rectify_wall(w, wall)
            img = ortho.draw_findings(rect, [f for f in findings if f.get("xyz") is not None])
            p = f"walls/{wall.id}.jpg"
            cv2.imwrite(str(out / p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            seen = float(rect.coverage.mean()) if getattr(rect, "coverage", None) is not None else None
            walls.append({"id": wall.id, "image": p, "length_m": round(float(wall.length_m), 2),
                          "seen_pct": None if seen is None else round(100 * seen, 1)})
        except Exception as e:
            walls.append({"id": getattr(wall, "id", "?"), "error": str(e)})
    vis["walls"] = walls
    return vis


def summarise_quality(w: Walkthrough) -> dict:
    v = [f.quality.get("verdict", "GOOD") for f in w.frames]
    return {k: v.count(k) for k in ("GOOD", "POOR", "BAD")}


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, dict):
        return {k: _json(x) for k, x in v.items()}
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform-cache", help="the inspection platform's 3D-engine output folder for this video")
    ap.add_argument("--name")
    ap.add_argument("--property", help="the property this walkthrough was submitted for (address or reference)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--agent", action="store_true", help="run the inspection agent on the result")
    a = ap.parse_args()
    if a.platform_cache:
        w = load_platform_run(Path(a.platform_cache), Path(a.video), a.name)
    else:
        from .odometry import process_video
        w = process_video(a.video, fps=a.fps)
        if a.name:
            w.name = a.name
    r = run(w, Path(a.out), prop=a.property)
    print(json.dumps({"name": r["name"], "coverage": r["coverage"]["overall"], "findings": len(r["findings"]),
                      "timings_s": r["timings_s"]}, indent=1))
    if a.agent:
        from .agent import run_agent
        ag = run_agent(r, Path(a.out), w)
        print(json.dumps({"engine": ag["engine"], "decision": ag["decision"], "steps": len(ag["trace"])}, indent=1))


if __name__ == "__main__":
    main()
