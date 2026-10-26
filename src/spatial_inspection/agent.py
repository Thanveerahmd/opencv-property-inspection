"""The inspection agent: a perception -> decision -> action loop over OpenCV 5 tools.

The agent reads what the OpenCV pipeline measured (coverage, gaps, defect candidates), looks at candidates, and
decides what to do next: get a better view from another frame, measure, read a label, confirm or dismiss a finding,
ask for part of the room to be filmed again, and finally ask a person to approve the decision. Each tool runs OpenCV
on the walkthrough, so what it sees changes the next call.

The decision-maker is a model on Amazon Bedrock (Converse API with tool use). Without AWS credentials the same tools
run under a fixed rule-based policy, so the loop can be demonstrated and tested offline.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import risk
from .scene import Walkthrough

MIN_INSPECTED_PCT = 60.0   # the inspection platform's rule: below this, refer for more evidence

SYSTEM = """You are a property inspection agent. A phone walkthrough video of a room has been analysed with OpenCV 5:
the room was reconstructed in 3D, surface coverage measured (a surface counts as inspectable when filmed within 3 m,
at most 70 degrees off straight-on, in a sharp frame) and defect candidates proposed by classical computer vision.

Candidates on furniture, doors, fittings or seen through openings were already ruled out with 3D geometry.

Work through it like a careful inspector, using the tools:
1. Check coverage first. The rule: below 60% of the surface inspected, the decision must be REFER, and you must
   request re-filming of the largest gaps with a clear, specific instruction for the person filming.
2. Check evidence integrity (reused footage, edits, loops, a screen being filmed). A high flag means REFER.
3. Look at every candidate before deciding on it. If a crop is unclear, too far or blurred, call find_better_view
   and judge the new view. Confirm only defects you can clearly see; dismiss false positives (shadows, grout lines,
   wallpaper patterns, reflections, objects) with a reason; flag_for_review what you cannot settle.
4. Measure confirmed defects and rate severity: minor, moderate or major.
5. If a label or rating plate is visible, read it.
6. Call assess_risk for the condition grade, risk level, repair cost and the rules-based underwriting decision.
7. Finish with request_human_approval: decision ACCEPT (no significant defects, enough coverage), REFER (not enough
   or not trustworthy evidence) or REPAIR (confirmed defects need work), with a short summary. A person makes the
   final call.
Be concise. Never invent measurements; use what the tools return."""

TOOLS = [
    {"name": "coverage_report", "description": "Coverage of walls, floor and ceiling (seen / inspectable / inspected %) "
     "and the largest uncovered surfaces.", "input": {}},
    {"name": "list_candidates", "description": "Defect candidates found by OpenCV: id, kind, score, surface, position, "
     "size, the frames that saw them, and their status.", "input": {}},
    {"name": "look_at", "description": "Show the image crop of a candidate, enhanced (CLAHE, denoise) for review, with "
     "its OpenCV image-quality measurements.", "input": {"finding_id": "string"}},
    {"name": "find_better_view", "description": "Search the other frames of the video for a closer, more square-on, "
     "sharper view of the candidate's 3D position, re-run OpenCV defect detection on it, and show it.",
     "input": {"finding_id": "string"}},
    {"name": "measure", "description": "Metric size of a candidate (length, width, area) from depth and camera "
     "geometry.", "input": {"finding_id": "string"}},
    {"name": "read_label", "description": "Read printed text (rating plates, labels) in a keyframe with OpenCV text "
     "detection and recognition.", "input": {"frame": "integer"}},
    {"name": "confirm_finding", "description": "Confirm a candidate as a real defect.",
     "input": {"finding_id": "string", "defect": "string", "severity": "string", "evidence": "string"}},
    {"name": "dismiss_finding", "description": "Reject a candidate as not a defect.",
     "input": {"finding_id": "string", "reason": "string"}},
    {"name": "flag_for_review", "description": "Leave a candidate for a person to judge, saying what is unclear.",
     "input": {"finding_id": "string", "reason": "string"}},
    {"name": "check_integrity", "description": "OpenCV evidence-integrity checks: footage reused from another "
     "walkthrough (perceptual hashes), hard cuts, frozen or looped frames, a screen being filmed (moire, bezel).",
     "input": {}},
    {"name": "assess_risk", "description": "Rules-based condition score and grade (A-E), risk level, insurance perils, "
     "repair-cost range, valuation impact and underwriting decision from the findings as they stand now.", "input": {}},
    {"name": "request_refilm", "description": "Ask the person filming to capture part of the room again. Sent to the "
     "capture app as an instruction.", "input": {"surface": "string", "instruction": "string", "reason": "string"}},
    {"name": "request_human_approval", "description": "Finish: propose the decision (ACCEPT, REFER or REPAIR) with a "
     "summary for a person to approve.", "input": {"decision": "string", "summary": "string"}},
]


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def add(self, kind: str, **data) -> None:
        self.steps.append({"step": len(self.steps) + 1, "t": round(time.time() - self.started, 2), "kind": kind, **data})


class Tools:
    """OpenCV-backed tools over one analysed walkthrough. Each call returns (text_json, optional_jpeg_bytes)."""

    def __init__(self, result: dict, out: Path, w: Walkthrough | None = None):
        self.r = result
        self.out = Path(out)
        self.w = w
        self.findings = {f["id"]: f for f in result.get("findings", [])}
        self.refilm: list[dict] = []
        self.decision: dict | None = None

    # perception
    def coverage_report(self) -> tuple[dict, bytes | None]:
        c = self.r["coverage"]
        gaps = [{k: g[k] for k in g if k in ("surface", "kind", "label", "length_m", "uncovered_m2", "inspectable_pct")}
                for g in self.r.get("gaps", [])[:6]]
        return {"overall": c["overall"], "by_kind": c["by_kind"], "largest_gaps": gaps,
                "rule": f"REFER if inspected below {MIN_INSPECTED_PCT}%"}, None

    def list_candidates(self) -> tuple[dict, bytes | None]:
        out = []
        for f in self.findings.values():
            out.append({k: f.get(k) for k in ("id", "kind", "score", "surface", "group", "xyz", "size", "frame",
                                              "views", "status")})
        return {"candidates": out}, None

    def look_at(self, finding_id: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no candidate {finding_id}"}, None
        img = cv2.imread(str(self.out / f["crop"]))
        if img is None:
            return {"error": "crop missing"}, None
        return {"id": finding_id, "kind": f["kind"], "score": f["score"], **image_quality(img)}, jpeg(img)

    def find_better_view(self, finding_id: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None or f.get("xyz") is None or self.w is None:
            return {"error": "no 3D position or video for this candidate"}, None
        best = best_view(self.w, np.asarray(f["xyz"]), exclude=f.get("views", [f["frame"]]))
        if best is None:
            return {"result": "no other frame sees this spot better"}, None
        i, u, v, dist, cos = best
        frame = self.w.frames[i]
        half = int(np.clip(160 / max(dist, 0.5), 60, 220))
        x0, y0 = max(0, int(u) - half), max(0, int(v) - half)
        crop = frame.image[y0:int(v) + half, x0:int(u) + half]
        redetect = []
        try:
            from . import defects
            redetect = [{"kind": c.kind, "score": round(float(c.score), 3)} for c in defects.detect_all(crop)[:3]]
        except ImportError:
            pass
        try:
            from .enhance import enhance_for_review
            crop = enhance_for_review(crop)
        except ImportError:
            pass
        name = f"findings/{finding_id}_view{i}.jpg"
        cv2.imwrite(str(self.out / name), crop)
        f.setdefault("extra_views", []).append(name)
        return {"frame": i, "distance_m": round(dist, 2), "angle_deg": round(float(np.degrees(np.arccos(cos))), 1),
                "redetected": redetect, **image_quality(crop), "image": name}, jpeg(crop)

    def measure(self, finding_id: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no candidate {finding_id}"}, None
        return {"id": finding_id, "size": f.get("size") or {}, "surface": f.get("surface")}, None

    def read_label(self, frame: int) -> tuple[dict, bytes | None]:
        if self.w is None:
            hits = [lab for lab in self.r.get("labels", []) if lab["frame"] == frame]
            return {"lines": hits[0]["lines"] if hits else []}, None
        try:
            from . import ocr
            lines = ocr.read_text(self.w.frames[int(frame)].image)
            years = ocr.find_years([ln["text"] for ln in lines]) if hasattr(ocr, "find_years") else []
            return {"lines": [{"text": ln["text"], "confidence": round(float(ln["confidence"]), 2)} for ln in lines[:12]],
                    "years": years}, None
        except (ImportError, IndexError) as e:
            return {"error": str(e)}, None

    # actions
    def confirm_finding(self, finding_id: str, defect: str, severity: str, evidence: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no candidate {finding_id}"}, None
        f.update(status="confirmed", defect=defect, severity=severity.lower(), evidence=evidence)
        return {"ok": True}, None

    def dismiss_finding(self, finding_id: str, reason: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no candidate {finding_id}"}, None
        f.update(status="dismissed", reason=reason)
        return {"ok": True}, None

    def flag_for_review(self, finding_id: str, reason: str) -> tuple[dict, bytes | None]:
        f = self.findings.get(finding_id)
        if f is None:
            return {"error": f"no candidate {finding_id}"}, None
        f.update(status="needs_review", reason=reason)
        return {"ok": True}, None

    def check_integrity(self) -> tuple[dict, bytes | None]:
        ev = self.r.get("integrity") or {}
        return {"level": ev.get("level", "unknown"), "flags": ev.get("flags", []),
                "checks": {k: v for k, v in (ev.get("checks") or {}).items() if k != "reused"}}, None

    def assess_risk(self) -> tuple[dict, bytes | None]:
        a = risk.assess_result(list(self.findings.values()), self.r)
        return {k: a[k] for k in ("condition_score", "grade", "grade_label", "risk_level", "decision", "action",
                                  "reasons", "repair_cost")}, None

    def request_refilm(self, surface: str, instruction: str, reason: str) -> tuple[dict, bytes | None]:
        self.refilm.append({"surface": surface, "instruction": instruction, "reason": reason, "state": "sent"})
        return {"ok": True, "queued": len(self.refilm)}, None

    def request_human_approval(self, decision: str, summary: str) -> tuple[dict, bytes | None]:
        self.decision = {"proposed": decision.upper(), "summary": summary, "state": "awaiting approval"}
        return {"ok": True, "state": "awaiting approval"}, None

    def call(self, name: str, args: dict) -> tuple[dict, bytes | None]:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "call":
            return {"error": f"unknown tool {name}"}, None
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}, None


def image_quality(img: np.ndarray) -> dict:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {"sharpness": round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1), "brightness": round(float(g.mean()), 1),
            "size_px": [int(img.shape[1]), int(img.shape[0])]}


def jpeg(img: np.ndarray, max_side: int = 640) -> bytes:
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def best_view(w: Walkthrough, xyz: np.ndarray, exclude: list[int], max_dist: float = 3.5):
    """The frame that sees a 3D point closest, most squarely and sharply, judged by projection and depth."""
    best, best_score = None, 0.0
    for i, f in enumerate(w.frames):
        if i in exclude or f.c2w is None or f.depth is None or f.image is None:
            continue
        w2c = np.linalg.inv(f.c2w)
        p = w2c[:3, :3] @ xyz + w2c[:3, 3]
        if p[2] < 0.3:
            continue
        u = f.K[0, 0] * p[0] / p[2] + f.K[0, 2]
        v = f.K[1, 1] * p[1] / p[2] + f.K[1, 2]
        h, wd = f.image.shape[:2]
        if not (0.15 * wd < u < 0.85 * wd and 0.15 * h < v < 0.85 * h):
            continue
        dist = float(np.linalg.norm(p))
        if dist > max_dist:
            continue
        du = (u - f.K[0, 2]) * f.K_depth[0, 0] / f.K[0, 0] + f.K_depth[0, 2]
        dv = (v - f.K[1, 2]) * f.K_depth[1, 1] / f.K[1, 1] + f.K_depth[1, 2]
        d = f.depth[int(np.clip(dv, 0, f.depth.shape[0] - 1)), int(np.clip(du, 0, f.depth.shape[1] - 1))]
        if d <= 0 or d < p[2] - max(0.15, 0.08 * p[2]):
            continue    # something stands in front of it in this frame
        cos = float(abs(p[2]) / dist)
        sharp = float(f.quality.get("sharpness") or 100.0)
        score = cos / dist * min(1.0, sharp / 150)
        if score > best_score:
            best, best_score = (i, u, v, dist, cos), score
    return best


# ---------------------------------------------------------------- decision makers

def run_bedrock(tools: Tools, trace: Trace, model_id: str, region: str, max_turns: int = 24) -> None:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)
    spec = [{"toolSpec": {"name": t["name"], "description": t["description"], "inputSchema": {"json": {
        "type": "object", "properties": {k: {"type": v} for k, v in t["input"].items()},
        "required": list(t["input"])}}}} for t in TOOLS]
    messages = [{"role": "user", "content": [{"text": "Inspect this walkthrough and reach a decision. Start with coverage."}]}]
    for _ in range(max_turns):
        resp = client.converse(modelId=model_id, system=[{"text": SYSTEM}], messages=messages,
                               toolConfig={"tools": spec}, inferenceConfig={"maxTokens": 1500, "temperature": 0.2})
        msg = resp["output"]["message"]
        messages.append(msg)
        for block in msg["content"]:
            if "text" in block and block["text"].strip():
                trace.add("thought", text=block["text"].strip())
        uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
        if not uses:
            break
        results = []
        for u in uses:
            out, img = tools.call(u["name"], u.get("input") or {})
            trace.add("tool", tool=u["name"], input=u.get("input") or {}, output=_brief(out), image=out.get("image"))
            content = [{"json": out}]
            if img is not None:
                content.append({"image": {"format": "jpeg", "source": {"bytes": img}}})
            results.append({"toolResult": {"toolUseId": u["toolUseId"], "content": content}})
        messages.append({"role": "user", "content": results})
        if tools.decision is not None:
            break


def run_policy(tools: Tools, trace: Trace) -> None:
    """Rule-based stand-in for the model: the same tools in a fixed order, deciding from their outputs."""
    cov, _ = tools.coverage_report()
    trace.add("tool", tool="coverage_report", input={}, output=_brief(cov))
    inspected = cov["overall"]["inspected_pct"]
    if inspected < MIN_INSPECTED_PCT:
        for g in cov["largest_gaps"][:3]:
            instr = (f"Film the {g['label']} again from about 2 m, phone level, moving slowly"
                     if g["kind"] == "wall" else f"Tilt the phone {'up' if g['kind'] == 'ceiling' else 'down'} and "
                     f"sweep the whole {g['label']} slowly")
            args = {"surface": g["surface"], "instruction": instr, "reason": f"{g['label']} not inspectable"}
            out, _ = tools.request_refilm(**args)
            trace.add("tool", tool="request_refilm", input=args, output=out)
    integ, _ = tools.check_integrity()
    trace.add("tool", tool="check_integrity", input={}, output=_brief(integ))
    cands, _ = tools.list_candidates()
    trace.add("tool", tool="list_candidates", input={}, output={"count": len(cands["candidates"])})
    confirmed = review = 0
    for c in sorted(cands["candidates"], key=lambda x: -x["score"])[:8]:
        look, _ = tools.look_at(c["id"])
        trace.add("tool", tool="look_at", input={"finding_id": c["id"]}, output=_brief(look))
        score, corroborated = c["score"], False
        better, _ = tools.find_better_view(c["id"])
        trace.add("tool", tool="find_better_view", input={"finding_id": c["id"]}, output=_brief(better),
                  image=better.get("image"))
        same = [r["score"] for r in better.get("redetected", []) if r["kind"] == c["kind"]]
        if same:
            score, corroborated = max(score, max(same)), True
        elif "frame" in better:
            score *= 0.6     # a closer view that no longer shows it lowers confidence
        # the policy cannot look at images, and classical detectors cannot tell every object from damage, so it never
        # confirms on its own: evidence worth a look goes to a person (or to the model on Bedrock), weak evidence is
        # dismissed
        if score >= 0.45:
            why = (f"seen again in frame {better.get('frame')} (score {score:.2f})" if corroborated
                   else "not seen again from another view")
            args = {"finding_id": c["id"], "reason": f"possible {c['kind']}: {why}"}
            tool, (out, _) = "flag_for_review", tools.flag_for_review(**args)
            review += 1
        else:
            args = {"finding_id": c["id"], "reason": f"weak evidence (score {score:.2f})"}
            tool, (out, _) = "dismiss_finding", tools.dismiss_finding(**args)
        trace.add("tool", tool=tool, input=args, output=out)
    assessed, _ = tools.assess_risk()
    trace.add("tool", tool="assess_risk", input={}, output=_brief(assessed))
    decision = assessed["action"]
    if inspected < MIN_INSPECTED_PCT or integ.get("level") == "high":
        decision = "REFER"
    summary = (f"{inspected}% inspected; grade {assessed['grade']} ({assessed['condition_score']}); "
               f"{confirmed} defect(s) confirmed, {review} for review; {len(tools.refilm)} re-film request(s); "
               f"integrity {integ.get('level', 'unknown')}. {assessed['decision']}.")
    out, _ = tools.request_human_approval(decision, summary)
    trace.add("tool", tool="request_human_approval", input={"decision": decision, "summary": summary}, output=out)


def run_agent(result: dict, out: Path, w: Walkthrough | None = None, prefer: str = "auto") -> dict:
    """Run the agent on an analysed walkthrough and write agent.json next to result.json."""
    tools = Tools(result, out, w)
    trace = Trace()
    model_id = os.environ.get("BEDROCK_MODEL_ID", "apac.anthropic.claude-sonnet-4-5-20250929-v1:0")
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1"))
    engine = "policy"
    if prefer in ("auto", "bedrock"):
        try:
            run_bedrock(tools, trace, model_id, region)
            engine = f"bedrock:{model_id}"
        except Exception as e:   # no credentials, no model access or no network: fall back, and say so
            trace.add("fallback", error=f"{type(e).__name__}: {str(e)[:200]}")
            if prefer == "bedrock":
                raise
    if tools.decision is None:
        if engine != "policy":
            trace.add("fallback", error="model stopped without a decision")
        run_policy(tools, trace)
        engine = engine if engine != "policy" else "policy"
    findings = list(tools.findings.values())
    assessment = risk.assess_result(findings, result)
    agent = {"engine": engine, "decision": tools.decision, "refilm": tools.refilm, "risk": assessment,
             "findings": findings, "trace": trace.steps, "seconds": round(time.time() - trace.started, 1)}
    (Path(out) / "agent.json").write_text(json.dumps(agent, indent=1, default=str), encoding="utf-8")
    return agent


def main() -> None:
    """Re-run the agent on an analysed walkthrough folder (find_better_view needs the video, so it is skipped)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("folder")
    ap.add_argument("--engine", choices=["auto", "bedrock", "policy"], default="auto")
    a = ap.parse_args()
    out = Path(a.folder)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    ag = run_agent(result, out, None, prefer=a.engine)
    print(json.dumps({"engine": ag["engine"], "decision": ag["decision"], "steps": len(ag["trace"])}, indent=1))


def _brief(d: dict, limit: int = 600) -> dict:
    s = json.dumps(d, default=str)
    return d if len(s) <= limit else {"summary": s[:limit] + "…"}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


if __name__ == "__main__":
    main()
