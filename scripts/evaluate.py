"""Evaluation for the technical report. Runs one heavy job at a time.

    python scripts/evaluate.py --platform D:/quntbit/property-inspection-platform --out results/eval

1. Coverage: this OpenCV 5 pipeline against the inspection platform's GPU engine on the same videos (EXP-000).
2. Room layout: walls fitted from OpenCV-fused points against the platform's layout.
3. Defect evidence: hit rate on labelled sample photos, false alarms on plain synthetic walls.
4. Runtime per stage.
5. Agent: decision, tool calls, re-film requests, better-view searches, and fallback behaviour.
Writes eval.json and evaluation.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import spatial_inspection  # noqa: F401  (thread cap and low priority)
from spatial_inspection.agent import run_agent
from spatial_inspection.fusion import fuse
from spatial_inspection.layout import layout_from_points
from spatial_inspection.pipeline import run
from spatial_inspection.scene import load_platform_run

SAMPLE_LABELS = {  # sample photo -> the defect kind it shows
    "ceiling_crack_1.jpg": "crack", "cracked_wall_1.jpg": "crack", "cracked_floor_tile_2.jpg": "crack",
    "water_damaged_wall_1.jpg": "stain", "peeling_paint_ceiling_2.jpg": "peeling", "rusty_door_lock_2.jpg": "rust",
}
BASELINE = {  # the platform's EXP-000 (Depth Anything 3 + GPU pipeline), same videos
    "demo_property_lite": {"inspectable_pct": 50.6, "inspected_pct": 42.5},
    "tom_home": {"inspectable_pct": 32.3, "inspected_pct": 20.0},
}


def layout_error(platform_layout: dict, fitted: dict) -> dict:
    """For each fitted wall, the distance to the nearest platform wall on the same axis and side."""
    errs = []
    for w in fitted["walls"]:
        same = [p for p in platform_layout["walls"] if p["axis"] == w["axis"] and int(p["sign"]) == int(w["sign"])]
        if same:
            errs.append(min(abs(p["offset"] - w["offset"]) for p in same))
    return {"walls": len(errs), "median_offset_error_m": round(float(np.median(errs)), 3) if errs else None,
            "max_offset_error_m": round(float(np.max(errs)), 3) if errs else None,
            "ceiling_error_m": round(abs(fitted["ceiling_height_m"] - platform_layout["ceiling_height_m"]), 3)}


def defects_eval(data_dir: Path) -> dict:
    from spatial_inspection import defects
    rows, hits = [], 0
    for name, kind in SAMPLE_LABELS.items():
        img = cv2.imread(str(data_dir / name))
        if img is None:
            continue
        t = time.perf_counter()
        cands = defects.detect_all(img)
        ms = (time.perf_counter() - t) * 1000
        top = [c.kind for c in cands[:3]]
        hit = kind in top
        hits += hit
        rows.append({"image": name, "expected": kind, "top3": top, "hit": hit, "ms": round(ms, 1)})
    rng = np.random.default_rng(0)
    false_alarms = 0
    for _ in range(20):
        base = np.full((480, 640, 3), rng.integers(120, 230, 3), np.uint8)
        noisy = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        false_alarms += sum(c.score >= 0.5 for c in defects.detect_all(noisy))
    return {"samples": rows, "top3_hit_rate": round(hits / max(1, len(rows)), 2),
            "false_alarms_on_20_plain_walls": int(false_alarms)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="D:/quntbit/property-inspection-platform")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--samples", default="demo_property_lite,tom_home")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"opencv": cv2.__version__, "threads": cv2.getNumThreads(), "walkthroughs": {}}
    for name in a.samples.split(","):
        slam = Path(a.platform) / "data" / "slam" / name
        video = Path("data/samples") / f"{name}.mp4"
        if not slam.exists() or not video.exists():
            continue
        w = load_platform_run(slam, video, name)
        platform_layout = w.layout
        w.load_images()
        cams = np.array([f.c2w[:3, 3] for f in w.frames if f.c2w is not None])
        fitted = layout_from_points(fuse(w, voxel=0.05), cameras=cams)
        r = run(w, Path("results") / name)
        ag = run_agent(r, Path("results") / name, w)
        tools = [s.get("tool") for s in ag["trace"] if s["kind"] == "tool"]
        report["walkthroughs"][name] = {
            "frames": r["frames"],
            "coverage": r["coverage"]["overall"],
            "baseline": BASELINE.get(name),
            "layout_vs_platform": layout_error(platform_layout, fitted),
            "findings": len(r["findings"]),
            "timings_s": r["timings_s"],
            "agent": {"engine": ag["engine"], "decision": ag["decision"], "tool_calls": len(tools),
                      "better_views": tools.count("find_better_view"), "refilm": len(ag["refilm"]),
                      "confirmed": sum(f.get("status") == "confirmed" for f in ag["findings"]),
                      "dismissed": sum(f.get("status") == "dismissed" for f in ag["findings"]),
                      "rule_respected": (ag["decision"] or {}).get("proposed") == "REFER"
                      if r["coverage"]["overall"]["inspected_pct"] < 60 else True},
        }
        print(name, json.dumps(report["walkthroughs"][name]["coverage"]), flush=True)
    try:
        report["defects"] = defects_eval(Path("tests/data"))
    except ImportError as e:
        report["defects"] = {"error": str(e)}
    (out / "eval.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
