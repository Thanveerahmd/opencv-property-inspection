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

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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


def process(name: str, video: Path) -> None:
    from .agent import run_agent
    from .odometry import process_video
    from .pipeline import run

    out = RESULTS / name
    try:
        _status(name, state="processing", stage="tracking and depth (OpenCV 5)", started=time.time())
        w = process_video(str(video), fps=float(os.environ.get("SAMPLE_FPS", "6")))
        w.name = name
        _status(name, stage="fusion, coverage, defects (OpenCV 5)")
        result = run(w, out)
        _walkthroughs[name] = w
        _status(name, stage="agent")
        run_agent(result, out, w)
        _status(name, state="done", stage="done")
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


@app.get("/api/walkthroughs")
def list_walkthroughs():
    out = []
    for d in sorted(RESULTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
        res = d / "result.json"
        ag = d / "agent.json"
        item = {"name": d.name, "state": st.get("state", "done" if res.exists() else "unknown"), "stage": st.get("stage")}
        if res.exists():
            r = json.loads(res.read_text(encoding="utf-8"))
            item["inspected_pct"] = r["coverage"]["overall"]["inspected_pct"]
            item["findings"] = len(r.get("findings", []))
        if ag.exists():
            a = json.loads(ag.read_text(encoding="utf-8"))
            item["decision"] = (a.get("approval") or a.get("decision") or {})
        out.append(item)
    return out


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
async def upload(background: BackgroundTasks, video: UploadFile = File(...)):
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(video.filename or "video").stem)[:40]
    name = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = UPLOADS / f"{name}{Path(video.filename or '.mp4').suffix or '.mp4'}"
    with open(path, "wb") as fh:
        while chunk := await video.read(1 << 20):
            fh.write(chunk)
    _status(name, state="queued", stage="queued", video=path.name)
    if os.environ.get("S3_BUCKET"):
        background.add_task(_s3_upload_video, path, name)
    threading.Thread(target=process, args=(name, path), daemon=True).start()
    return {"name": name}


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


@app.post("/api/walkthroughs/{name}/agent")
def rerun_agent(name: str):
    from .agent import run_agent
    d = RESULTS / name
    if not (d / "result.json").exists():
        raise HTTPException(404, "unknown walkthrough")
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    a = run_agent(r, d, _walkthroughs.get(name))
    return JSONResponse({"engine": a["engine"], "decision": a["decision"]})
