"""The web endpoint: upload a walkthrough, watch it being processed, review the agent's decision and approve it.

    uvicorn spatial_inspection.server:app --host 0.0.0.0 --port 8000

Results live in RESULTS_DIR (default ./results), one folder per walkthrough. With S3_BUCKET set, uploads and finished
results are also copied to Amazon S3.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

RESULTS = Path(os.environ.get("RESULTS_DIR", "results")).resolve()
UPLOADS = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
WEB = Path(__file__).parent / "web"
RESULTS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_walkthroughs: dict = {}          # name -> Walkthrough kept in memory so the agent can look at other frames

app = FastAPI(title="Spatially Aware Property Inspection", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/files", StaticFiles(directory=RESULTS), name="files")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


@app.middleware("http")
async def no_stale_results(request: Request, call_next):
    """Results change when a walkthrough is re-analysed or approved: never let the browser reuse an old copy."""
    response = await call_next(request)
    if request.url.path.startswith(("/api/", "/files/")):
        response.headers["Cache-Control"] = "no-store"
    return response


def _status(name: str, **kw) -> None:
    p = RESULTS / name / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=time.time())
    p.write_text(json.dumps(cur))


def _s3_put_dir(folder: Path, prefix: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    import boto3
    s3 = boto3.client("s3")
    for f in folder.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"{prefix}/{f.relative_to(folder).as_posix()}")


# share of the whole job each step takes (measured on a laptop CPU), so progress and the time left are honest
STEP_SHARE = {"frames": (0.0, 0.04), "tracking": (0.04, 0.66), "levelling": (0.66, 0.68), "analysis": (0.68, 0.94),
              "agent": (0.94, 1.0)}
STEP_LABEL = {"frames": "Reading frames and checking quality", "tracking": "Depth (cv2.dnn) and camera tracking",
              "levelling": "Levelling the floor and fixing scale", "quality": "Frame quality gate",
              "integrity": "Evidence-integrity checks", "privacy": "Blurring faces and private text",
              "fusion": "Fusing depth into 3D (TSDF volume)", "layout": "Fitting walls, floor and ceiling",
              "coverage": "Measuring surface coverage", "defects": "Finding and placing defects",
              "labels": "Reading labels", "visuals": "Rectifying walls and drawing the floor plan",
              "agent": "Agent reviewing the evidence"}


def _progress(name: str, step: str, frac: float, detail: str | None = None) -> None:
    lo, hi = STEP_SHARE[step]
    p = RESULTS / name / "status.json"
    started = json.loads(p.read_text()).get("started", time.time()) if p.exists() else time.time()
    done = lo + (hi - lo) * max(0.0, min(1.0, frac))
    elapsed = time.time() - started
    eta = round(elapsed / done - elapsed) if done > 0.03 else None
    _status(name, step=step, stage=STEP_LABEL.get(detail or step, detail or step), progress=round(done, 3), eta_s=eta,
            elapsed_s=round(elapsed))


def process(name: str, video: Path, prop: str | None = None) -> None:
    from .agent import run_agent
    from .odometry import process_video
    from .pipeline import STAGES, run

    out = RESULTS / name
    try:
        _status(name, state="processing", stage="Starting", started=time.time(), progress=0.0, property=prop)
        last = [0.0]

        def on_video(step: str, frac: float) -> None:
            if time.time() - last[0] > 0.7 or frac >= 1.0:     # keep status writes light
                last[0] = time.time()
                _progress(name, step, frac)

        w = process_video(str(video), fps=float(os.environ.get("SAMPLE_FPS", "6")), progress=on_video)
        w.name = name
        _progress(name, "analysis", 0.0, STAGES[0])

        def on_stage(done: str, frac: float) -> None:     # the pipeline reports a finished stage: show the next one
            i = STAGES.index(done) + 1
            _progress(name, "analysis", frac, STAGES[i] if i < len(STAGES) else "agent")

        result = run(w, out, prop=prop, progress=on_stage)
        _walkthroughs[name] = w
        _progress(name, "agent", 0.0)
        run_agent(result, out, w)
        _status(name, state="done", stage="Done", step="done", progress=1.0, eta_s=0)
        _s3_put_dir(out, f"results/{name}")
    except Exception as e:
        _status(name, state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    import cv2
    return {"ok": True, "opencv": cv2.__version__, "s3": bool(os.environ.get("S3_BUCKET")),
            "bedrock_model": os.environ.get("BEDROCK_MODEL_ID")}


def _summary(d: Path) -> dict:
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    res, ag = d / "result.json", d / "agent.json"
    item = {"name": d.name, "state": st.get("state", "done" if res.exists() else "unknown"), "stage": st.get("stage"),
            "progress": st.get("progress"), "eta_s": st.get("eta_s"), "property": st.get("property"),
            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))}
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        item.update(inspected_pct=r["coverage"]["overall"]["inspected_pct"], findings=len(r.get("findings", [])),
                    property=r.get("property") or item["property"],
                    integrity=(r.get("integrity") or {}).get("level"))
    if ag.exists():
        a = json.loads(ag.read_text(encoding="utf-8"))
        item["decision"] = a.get("approval") or a.get("decision") or {}
        rk = a.get("risk") or {}
        item.update(grade=rk.get("grade"), risk_level=rk.get("risk_level"), underwriting=rk.get("decision"),
                    repair_cost=rk.get("repair_cost"), condition_score=rk.get("condition_score"),
                    confirmed=sum(f.get("status") == "confirmed" for f in a.get("findings", [])),
                    review=sum(f.get("status") == "needs_review" for f in a.get("findings", [])))
    return item


def _folders() -> list[Path]:
    return [d for d in sorted(RESULTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if d.is_dir() and not d.name.startswith(("_", "."))]


@app.get("/api/walkthroughs")
def list_walkthroughs():
    return [_summary(d) for d in _folders()]


@app.get("/api/dashboard")
def dashboard():
    """Figures across every walkthrough: grades, decisions, risk, repair exposure, coverage and integrity."""
    items = [_summary(d) for d in _folders()]
    done = [i for i in items if i.get("grade")]

    def count(key: str) -> dict:
        c: dict = {}
        for i in done:
            k = i.get(key) or "–"
            c[k] = c.get(k, 0) + 1
        return c

    decisions: dict = {}
    for i in done:
        k = (i.get("decision") or {}).get("decision") or (i.get("decision") or {}).get("proposed") or "–"
        decisions[k] = decisions.get(k, 0) + 1
    cov = [i["inspected_pct"] for i in done if i.get("inspected_pct") is not None]
    return {"walkthroughs": len(items), "analysed": len(done),
            "processing": sum(i["state"] in ("queued", "processing") for i in items),
            "awaiting_approval": sum((i.get("decision") or {}).get("state") == "awaiting approval" for i in done),
            "grades": {g: count("grade").get(g, 0) for g in "ABCDE"}, "risk_levels": count("risk_level"),
            "decisions": decisions, "integrity": count("integrity"),
            "repair_exposure": {"low": sum((i.get("repair_cost") or {}).get("low", 0) for i in done),
                                "high": sum((i.get("repair_cost") or {}).get("high", 0) for i in done),
                                "currency": next(((i.get("repair_cost") or {}).get("currency") for i in done), "USD")},
            "mean_inspected_pct": round(sum(cov) / len(cov), 1) if cov else None,
            "confirmed": sum(i.get("confirmed", 0) for i in done), "review": sum(i.get("review", 0) for i in done),
            "recent": items[:12]}


@app.get("/api/walkthroughs/{name}/report.{fmt}")
def report_file(name: str, fmt: str):
    from . import report
    d = RESULTS / name
    if not (d / "result.json").exists():
        raise HTTPException(404, "unknown walkthrough")
    if fmt == "html":
        return HTMLResponse(report.to_html(d))
    if fmt == "md":
        return PlainTextResponse(report.to_markdown(d), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{name}_report.md"'})
    if fmt == "json":
        return JSONResponse(report.bundle(d), headers={"Content-Disposition": f'attachment; filename="{name}_report.json"'})
    raise HTTPException(404, "report formats: html, md, json")


@app.get("/api/walkthroughs/{name}")
def get_walkthrough(name: str):
    d = RESULTS / name
    if not (d / "result.json").exists():
        st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None
        if st is None:
            raise HTTPException(404, "unknown walkthrough")
        return {"status": st}
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    a = json.loads((d / "agent.json").read_text(encoding="utf-8")) if (d / "agent.json").exists() else None
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"state": "done"}
    return {"result": r, "agent": a, "status": st}


@app.post("/api/walkthroughs")
async def upload(background: BackgroundTasks, video: UploadFile = File(...), property: str = Form("")):  # noqa: A002
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(video.filename or "video").stem)[:40]
    name = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = UPLOADS / f"{name}{Path(video.filename or '.mp4').suffix or '.mp4'}"
    with open(path, "wb") as fh:
        while chunk := await video.read(1 << 20):
            fh.write(chunk)
    prop = property.strip()[:120] or None
    _status(name, state="queued", stage="Queued", video=path.name, property=prop, progress=0.0)
    if os.environ.get("S3_BUCKET"):
        background.add_task(_s3_upload_video, path, name)
    threading.Thread(target=_queued, args=(name, path, prop), daemon=True).start()
    return {"name": name}


_one_at_a_time = threading.Lock()


def _queued(name: str, path: Path, prop: str | None) -> None:
    """Walkthroughs are processed one after another, never side by side, so the machine is not overloaded."""
    with _one_at_a_time:
        process(name, path, prop)


def _s3_upload_video(path: Path, name: str) -> None:
    import boto3
    boto3.client("s3").upload_file(str(path), os.environ["S3_BUCKET"], f"uploads/{name}/{path.name}")


class Approval(BaseModel):
    decision: str
    note: str = ""
    by: str = "inspector"


@app.post("/api/walkthroughs/{name}/approve")
def approve(name: str, body: Approval):
    p = RESULTS / name / "agent.json"
    if not p.exists():
        raise HTTPException(404, "no agent decision to approve")
    with _lock:
        a = json.loads(p.read_text(encoding="utf-8"))
        proposed = (a.get("decision") or {}).get("proposed")
        a["approval"] = {"decision": body.decision.upper(), "proposed": proposed, "agreed": body.decision.upper() == proposed,
                         "note": body.note, "by": body.by, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        a.setdefault("trace", []).append({"step": len(a.get("trace", [])) + 1, "kind": "human", "tool": "approval",
                                          "input": a["approval"], "output": {"ok": True}})
        p.write_text(json.dumps(a, indent=1), encoding="utf-8")
    return a["approval"]


class Review(BaseModel):
    status: str                 # confirmed | dismissed | needs_review
    severity: str = "moderate"  # minor | moderate | major, for confirmed findings
    note: str = ""
    by: str = "inspector"


@app.post("/api/walkthroughs/{name}/findings/{fid}")
def review_finding(name: str, fid: str, body: Review):
    """A person settles a finding; the grade, risk and rules-based decision are recomputed from it."""
    from . import risk
    d = RESULTS / name
    p = d / "agent.json"
    if not p.exists() or body.status not in ("confirmed", "dismissed", "needs_review"):
        raise HTTPException(400, "unknown walkthrough or status")
    with _lock:
        a = json.loads(p.read_text(encoding="utf-8"))
        f = next((x for x in a.get("findings", []) if x["id"] == fid), None)
        if f is None:
            raise HTTPException(404, "unknown finding")
        f.update(status=body.status, reviewed_by=body.by, review_note=body.note)
        if body.status == "confirmed":
            f["severity"] = body.severity.lower()
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        a["risk"] = risk.assess_result(a["findings"], r)
        a.setdefault("trace", []).append({"step": len(a.get("trace", [])) + 1, "kind": "human", "tool": "review_finding",
                                          "input": {"finding_id": fid, **body.model_dump()}, "output": {"ok": True}})
        p.write_text(json.dumps(a, indent=1), encoding="utf-8")
    return {"finding": f, "risk": a["risk"]}


@app.post("/api/walkthroughs/{name}/agent")
def rerun_agent(name: str):
    from .agent import run_agent
    d = RESULTS / name
    if not (d / "result.json").exists():
        raise HTTPException(404, "unknown walkthrough")
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    a = run_agent(r, d, _walkthroughs.get(name))
    return JSONResponse({"engine": a["engine"], "decision": a["decision"]})
