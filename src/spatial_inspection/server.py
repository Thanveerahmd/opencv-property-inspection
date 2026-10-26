"""The web endpoint: upload a walkthrough, watch it being processed, review the agent's decision and approve it.

    uvicorn spatial_inspection.server:app --host 0.0.0.0 --port 8000

Results live in RESULTS_DIR (default ./results), one folder per walkthrough. With S3_BUCKET set, uploads and finished
results are also copied to Amazon S3.

The console (web/index.html, app.js, styles.css) and the 3D viewer of one walkthrough (web/walkthrough.html) are the
Property Inspection Platform's UI. A finished walkthrough is the folder results/<name>/ (walkthrough.json, GLBs, live
map, coverage map, keyframes, ...), served at /walkthroughs/<name>/.

  GET  /api/v1/health                          OpenCV, GPU, S3 / Bedrock and the job queue
  GET  /api/v1/walkthroughs                    finished walkthroughs (walkthrough.json) and jobs that are not done
  POST /api/v1/walkthrough                     upload a video: queued, processed one at a time
  GET  /api/v1/walkthrough/{name}/log          job status, log tail, progress and place in the queue
  GET  /api/v1/walkthrough/{name}/video        the stored upload (preview while it is processed)
  POST /api/v1/walkthrough/{name}/retry        process a failed or interrupted upload again
  POST /api/v1/walkthrough/{name}/ask          short rule-based answer from the walkthrough record
  GET  /api/v1/portfolio                       grades, decisions and repair exposure across walkthroughs
  POST /api/walkthroughs/{name}/approve        a person approves or overrides the agent's proposal
  POST /api/walkthroughs/{name}/findings/{id}  a person confirms, dismisses or reopens a finding
  POST /api/walkthroughs/{name}/agent          run the agent again
  GET  /api/walkthroughs/{name}/report.{html,md,json}
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
ROOT_DIR = Path(__file__).resolve().parents[2]      # the repository (sample videos under data/)
RESULTS.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()
_walkthroughs: dict = {}          # name -> Walkthrough kept in memory so the agent can look at other frames
NAME_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
# agent finding status -> state of the finding in walkthrough.json (the viewer's vocabulary)
STATE = {"confirmed": "DETECTED", "needs_review": "NEEDS_REVIEW", "candidate": "NEEDS_REVIEW", "dismissed": "DISMISSED"}
QUEUED_LINE = "[queued] waiting for the GPU (one walkthrough at a time)"

app = FastAPI(title="Property Inspection AI · OpenCV 5 edition", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/files", StaticFiles(directory=RESULTS), name="files")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")


@app.middleware("http")
async def no_stale_results(request: Request, call_next):
    """Results change when a walkthrough is re-analysed or approved, and the UI changes too: never reuse an old copy."""
    response = await call_next(request)
    p = request.url.path
    if p.startswith(("/api/", "/files/")) or p.endswith("/walkthrough.json"):
        response.headers["Cache-Control"] = "no-store"
    elif p == "/" or p.startswith("/static/") or p.endswith((".html", "/")):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _status(name: str, **kw) -> None:
    p = RESULTS / name / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=time.time())
    p.write_text(json.dumps(cur))


def _log(name: str, line: str) -> None:
    """One line in results/<name>/run.log (the console shows its tail while the walkthrough is processed)."""
    try:
        with open(RESULTS / name / "run.log", "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {line}\n")
    except OSError:
        pass


def _s3_put_dir(folder: Path, prefix: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        return
    import boto3
    s3 = boto3.client("s3")
    for f in folder.rglob("*"):
        if f.is_file():
            s3.upload_file(str(f), bucket, f"{prefix}/{f.relative_to(folder).as_posix()}")


# share of the whole job each step of walkthrough.build takes, so progress and the time left are honest
STEP_SHARE = {"engine": (0.0, 0.70), "analysis": (0.70, 0.88), "agent": (0.88, 0.90), "viewer": (0.90, 1.0)}
STEP_LABEL = {"engine": "3D engine: Depth Anything 3 + OpenCV 5 SLAM", "quality": "Frame quality gate",
              "integrity": "Evidence-integrity checks", "privacy": "Blurring faces and private text",
              "fusion": "Fusing depth into 3D (TSDF volume)", "layout": "Fitting walls, floor and ceiling",
              "coverage": "Measuring surface coverage", "defects": "Finding and placing defects",
              "labels": "Reading labels", "visuals": "Rectifying walls and drawing the floor plan",
              "agent": "Agent reviewing the evidence", "viewer": "3D viewer, live map and GLBs"}


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
    """3D engine -> OpenCV analysis -> agent -> viewer files, all in walkthrough.build; status.json follows it."""
    from .walkthrough import build

    out = RESULTS / name
    try:
        st = _read_json(out / "status.json") or {}
        mode = st.get("mode") or os.environ.get("SI_MODE", "advanced")      # the upload form's Lite / Advanced
        started = time.time()
        _status(name, state="processing", stage="Starting", step="engine", started=started, progress=0.0, property=prop)
        _log(name, f"processing {Path(video).name} ({mode} mode)")
        last = {"t": 0.0, "line": None}

        def on_progress(step: str, frac: float, detail: str | None = None) -> None:
            line = STEP_LABEL.get(detail or step, detail or step)
            new = line != last["line"]
            if new:
                last["line"] = line
                _log(name, f"{step}: {line}")
            if new or frac >= 1.0 or time.time() - last["t"] > 0.7:     # keep status writes light
                last["t"] = time.time()
                _progress(name, step if step in STEP_SHARE else "viewer", frac, detail)

        build(Path(video), name, out, prop=prop, mode=mode, progress=on_progress)
        _status(name, state="done", stage="Done", step="done", progress=1.0, eta_s=0, elapsed_s=round(time.time() - started))
        _log(name, "done")
        _s3_put_dir(out, f"results/{name}")
    except Exception as e:
        _status(name, state="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        _log(name, f"failed: {type(e).__name__}: {e}")


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(p: Path, data: dict, indent: int = 1) -> None:
    text = json.dumps(data, indent=indent, default=str)
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)
    except OSError:          # Windows: the file is being read (served) right now, so write it in place
        p.write_text(text, encoding="utf-8")
        tmp.unlink(missing_ok=True)


def _dir(name: str) -> Path:
    if not NAME_RX.fullmatch(name):
        raise HTTPException(404, "unknown walkthrough")
    return RESULTS / name


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


# ---------------------------------------------------------------- earlier OpenCV console routes (result.json / agent.json)

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
    """The inspection report from result.json + agent.json (so it shows the latest reviews and approval); a walkthrough
    that only has walkthrough.json gets its own report.html, a Markdown summary or the record itself."""
    from . import report
    d = _dir(name)
    if fmt not in ("html", "md", "json"):
        raise HTTPException(404, "report formats: html, md, json")
    if (d / "result.json").exists():
        if fmt == "html":
            return HTMLResponse(report.to_html(d))
        if fmt == "md":
            return PlainTextResponse(report.to_markdown(d), media_type="text/markdown",
                                     headers={"Content-Disposition": f'attachment; filename="{name}_report.md"'})
        return JSONResponse(report.bundle(d), headers={"Content-Disposition": f'attachment; filename="{name}_report.json"'})
    w = _read_json(d / "walkthrough.json")
    if w is None:
        raise HTTPException(404, "unknown walkthrough")
    if fmt == "html":
        rel = (w.get("files") or {}).get("report") or "report.html"
        f = (d / rel).resolve()
        if not f.is_relative_to(d.resolve()) or not f.is_file():
            raise HTTPException(404, "no report for this walkthrough")
        return FileResponse(f, media_type="text/html")
    if fmt == "md":
        return PlainTextResponse(_markdown(w), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{name}_report.md"'})
    return JSONResponse(w, headers={"Content-Disposition": f'attachment; filename="{name}_report.json"'})


@app.get("/api/walkthroughs/{name}")
def get_walkthrough(name: str):
    d = _dir(name)
    if not (d / "result.json").exists():
        st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else None
        if st is None:
            raise HTTPException(404, "unknown walkthrough")
        return {"status": st}
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    a = json.loads((d / "agent.json").read_text(encoding="utf-8")) if (d / "agent.json").exists() else None
    st = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {"state": "done"}
    return {"result": r, "agent": a, "status": st}


async def _store_upload(video: UploadFile) -> tuple[str, Path]:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(video.filename or "video").stem)[:40] or "video"
    base = f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    name, n = base, 1
    while (RESULTS / name).exists() or name in _jobs:
        n += 1
        name = f"{base}-{n}"
    path = UPLOADS / f"{name}{Path(video.filename or '.mp4').suffix or '.mp4'}"
    with open(path, "wb") as fh:
        while chunk := await video.read(1 << 20):
            fh.write(chunk)
    return name, path


@app.post("/api/walkthroughs")
async def upload(background: BackgroundTasks, video: UploadFile = File(...), property: str = Form("")):  # noqa: A002
    name, path = await _store_upload(video)
    prop = property.strip()[:120] or None
    _status(name, state="queued", stage="Queued", video=path.name, property=prop, progress=0.0, created=time.time())
    _log(name, QUEUED_LINE)
    if os.environ.get("S3_BUCKET"):
        background.add_task(_s3_upload_video, path, name)
    _enqueue(name, path, prop)
    return {"name": name}


# ---------------------------------------------------------------- the job queue: one walkthrough at a time

_one_at_a_time = threading.Lock()
_jobs: dict[str, dict] = {}       # name -> {"created", "state": queued | running} for jobs queued in THIS process
STALE_S = 300                     # a job no process has touched for this long died with an earlier server


def _enqueue(name: str, path: Path, prop: str | None) -> None:
    with _lock:
        _jobs[name] = {"created": time.time(), "state": "queued"}
    threading.Thread(target=_queued, args=(name, path, prop), daemon=True).start()


def _queued(name: str, path: Path, prop: str | None) -> None:
    """Walkthroughs are processed one after another, never side by side, so the machine is not overloaded."""
    try:
        with _one_at_a_time:
            with _lock:
                _jobs[name].update(state="running", started=time.time())
            process(name, path, prop)
    finally:
        with _lock:
            _jobs.pop(name, None)


def _s3_upload_video(path: Path, name: str) -> None:
    import boto3
    boto3.client("s3").upload_file(str(path), os.environ["S3_BUCKET"], f"uploads/{name}/{path.name}")


def _job_status(name: str, st: dict | None) -> str:
    """status.json in the platform's words: queued / running / done / failed / interrupted (or unknown)."""
    st = st or {}
    state = {"processing": "running"}.get(st.get("state"), st.get("state"))
    if state in ("done", "failed", "interrupted"):
        return state
    with _lock:
        job = _jobs.get(name)
    if job:
        return job["state"]
    if state in ("queued", "running"):
        # not queued in this server: another process may be running it, else it died with an earlier server
        touched = [st.get("updated") or 0] + [p.stat().st_mtime for p in (RESULTS / name / "run.log", RESULTS / name / "engine.log")
                                              if p.exists()]
        return state if time.time() - max(touched) < STALE_S else "interrupted"
    return "done" if (RESULTS / name / "walkthrough.json").exists() else "unknown"


def _eta(st: dict, now: float) -> int | None:
    eta = st.get("eta_s")
    return None if eta is None else max(0, round(eta - (now - (st.get("updated") or now))))


def _queue_info(name: str) -> dict:
    """Where a queued job stands: how many walkthroughs are ahead of it and how far the running one is."""
    now = time.time()
    with _lock:
        jobs = {k: dict(v) for k, v in _jobs.items()}
    me = jobs.get(name) or {"created": now}
    ahead = [n for n, j in jobs.items() if j["state"] == "queued" and (j["created"], n) < (me["created"], name)]
    run = next((n for n, j in jobs.items() if j["state"] == "running"), None)
    info = {"ahead": len(ahead) + (run is not None), "waiting_s": round(now - me["created"])}
    if run:
        st = _read_json(RESULTS / run / "status.json") or {}
        started = st.get("started") or jobs[run].get("started") or now
        info.update(running=run, running_started=started, running_elapsed_s=round(now - started), running_eta_s=_eta(st, now))
    return info


def _progress_of(st: dict, state: str) -> dict | None:
    """status.json as the console's progress block: percent, phase (stage text), key (step), elapsed and time left."""
    if not st or state in ("queued", "unknown"):
        return None
    now = time.time()
    started = st.get("started")
    elapsed = (now - started) if started and state == "running" else st.get("elapsed_s")
    if elapsed is None and started:
        elapsed = (st.get("updated") or now) - started
    return {"percent": 100.0 if state == "done" else round(100 * float(st.get("progress") or 0), 1),
            "phase": st.get("stage") or state, "key": st.get("step"), "elapsed_s": round(elapsed or 0),
            "eta_s": 0 if state == "done" else _eta(st, now)}


def _tail_lines(p: Path, n: int) -> list[str]:
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            return [x for x in fh.read().decode("utf-8", "replace").splitlines() if x.strip()][-n:]
    except OSError:
        return []


def _tail(name: str, st: dict, state: str) -> list[str]:
    d = RESULTS / name
    lines = _tail_lines(d / "run.log", 40)
    if state == "running" and st.get("step") == "engine":       # the 3D engine's own output while it runs
        lines += _tail_lines(d / "engine.log", 6)
    if not lines:
        lines = [QUEUED_LINE if state == "queued" else f"[{state}] {st.get('stage') or ''}".rstrip()]
    if state == "failed" and st.get("error"):
        lines.append(f"error: {st['error']}")
    return lines


def _accepted(name: str) -> dict:
    st = _read_json(RESULTS / name / "status.json") or {}
    state = _job_status(name, st)
    out = {"name": name, "mode": st.get("mode"), "status": state, "log": f"/api/v1/walkthrough/{name}/log",
           "view": f"/walkthroughs/{name}/"}
    if state == "queued":
        out["queue"] = _queue_info(name)
    return out


# ---------------------------------------------------------------- the console's routes (/api/v1, the platform's API)

_gpu_cache: dict = {"t": 0.0, "info": {"cuda": False}}


def _gpu() -> dict:
    """GPU name and memory from nvidia-smi (cached): the server never opens a CUDA context itself."""
    if time.time() - _gpu_cache["t"] < 10:
        return _gpu_cache["info"]
    info: dict = {"cuda": False}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        gpu, total, used = (x.strip() for x in out.splitlines()[0].split(","))
        info = {"cuda": True, "gpu": gpu, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used))}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    _gpu_cache.update(t=time.time(), info=info)
    return info


@app.get("/api/v1/health")
def health_v1():
    import cv2
    with _lock:
        jobs = dict(_jobs)
    return {"status": "ok", "opencv": cv2.__version__, **_gpu(), "s3": bool(os.environ.get("S3_BUCKET")),
            "bedrock_model": os.environ.get("BEDROCK_MODEL_ID"), "mode": os.environ.get("SI_MODE", "advanced"),
            "auth": False, "queue": sum(j["state"] == "queued" for j in jobs.values()),
            "running": [n for n, j in jobs.items() if j["state"] == "running"]}


@app.get("/api/v1/walkthroughs")
def list_v1():
    """Finished walkthroughs (newest first), then jobs that are queued, running, failed or interrupted."""
    done, jobs = [], []
    for d in _folders():
        w = _read_json(d / "walkthrough.json") if (d / "walkthrough.json").exists() else None
        if w:
            r = w.get("risk") or {}
            done.append({"name": d.name, "url": f"/walkthroughs/{d.name}/", "duration_s": w.get("duration_s"),
                         "findings": sum(f.get("state") != "DISMISSED" for f in w.get("findings") or []),
                         "coverage": (w.get("coverage") or {}).get("overall"),
                         "rooms": [x.get("name") for x in w.get("rooms") or []], "grade": r.get("grade"),
                         "risk_level": r.get("risk_level"), "decision": r.get("decision"), "mode": w.get("mode"),
                         "property_id": w.get("property_id"),
                         "approval": ((w.get("agent") or {}).get("approval") or {}).get("decision"),
                         "_t": w.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(d.stat().st_mtime))})
        st = _read_json(d / "status.json")
        state = _job_status(d.name, st) if st else "done"
        if state != "done":
            jobs.append({"name": d.name, "status": state, "mode": st.get("mode"), "property_id": st.get("property"),
                         "created": st.get("created") or st.get("started") or st.get("updated"),
                         "log": f"/api/v1/walkthrough/{d.name}/log"})
    done.sort(key=lambda x: x.pop("_t"), reverse=True)
    jobs.sort(key=lambda j: j["created"] or 0)
    return done + jobs


@app.post("/api/v1/walkthrough")
async def create_walkthrough(background: BackgroundTasks, video: UploadFile = File(...), property_id: str = Form(""),
                             property: str = Form(""), mode: str = Form("")):  # noqa: A002
    """Upload a walkthrough video. It is queued and processed in the background, one walkthrough at a time; poll
    /api/v1/walkthrough/{name}/log. mode=lite builds a lighter 3D model (3 cm, one alignment pass, no preview videos)."""
    mode = mode.strip().lower()
    mode = mode if mode in ("lite", "advanced") else os.environ.get("SI_MODE", "advanced")
    name, path = await _store_upload(video)
    prop = (property_id or property).strip()[:120] or None
    _status(name, state="queued", stage="Queued", video=path.name, property=prop, progress=0.0, created=time.time(), mode=mode)
    _log(name, QUEUED_LINE)
    if os.environ.get("S3_BUCKET"):
        background.add_task(_s3_upload_video, path, name)
    _enqueue(name, path, prop)
    return _accepted(name)


@app.get("/api/v1/walkthrough/{name}/log")
def walkthrough_log(name: str):
    d = _dir(name)
    st = _read_json(d / "status.json")
    if st is None and not (d / "walkthrough.json").exists():
        raise HTTPException(404, "no log")
    st = st or {}
    state = _job_status(name, st)
    out = {"status": state, "tail": _tail(name, st, state), "progress": _progress_of(st, state), "mode": st.get("mode"),
           "created": st.get("created"), "started": st.get("started"),
           "finished": st.get("updated") if state in ("done", "failed") else None}
    if state == "queued":
        out["queue"] = _queue_info(name)
    return out


@app.get("/api/v1/walkthrough/{name}/video")
def walkthrough_video(name: str):
    """The stored upload, so the progress view can play it when the browser has no local copy of the video."""
    d = _dir(name)
    st = _read_json(d / "status.json") or {}
    # an upload, or for a run started from the command line its video under the repository's data/ folder
    allowed = [(UPLOADS, st.get("video")), (ROOT_DIR / "data", st.get("video_path"))]
    for base, rel in allowed:
        p = (base / rel).resolve() if rel else None
        if p is not None and p.is_relative_to(base.resolve()) and p.is_file():
            return FileResponse(p)
    if (d / "source_video.mp4").is_file():
        return FileResponse(d / "source_video.mp4")
    raise HTTPException(404, "no stored video")


@app.post("/api/v1/walkthrough/{name}/retry")
def retry_walkthrough(name: str):
    """Process a failed or interrupted walkthrough again from its stored upload (no re-upload)."""
    d = _dir(name)
    st = _read_json(d / "status.json")
    if st is None:
        raise HTTPException(404, "unknown walkthrough")
    state = _job_status(name, st)
    if state in ("queued", "running"):
        raise HTTPException(409, "this walkthrough is already queued or running")
    if state == "done":
        raise HTTPException(409, "this walkthrough already finished")
    path = UPLOADS / (st.get("video") or "")
    if not st.get("video") or not path.is_file():
        raise HTTPException(410, "the uploaded video is no longer stored - upload it again")
    _status(name, state="queued", stage="Queued", step=None, progress=0.0, eta_s=None, error=None, trace=None,
            created=time.time())
    _log(name, "[retry] " + QUEUED_LINE)
    _enqueue(name, path, st.get("property"))
    return _accepted(name)


class Ask(BaseModel):
    question: str


@app.post("/api/v1/walkthrough/{name}/ask")
def ask_walkthrough(name: str, body: Ask):
    """A short answer from the walkthrough record, by rules (no language model): coverage, findings, cost, decision,
    valuation, integrity."""
    w = _read_json(_dir(name) / "walkthrough.json")
    if w is None:
        raise HTTPException(404, "walkthrough not found")
    q = body.question.strip()[:500]
    if not q:
        raise HTTPException(422, "empty question")
    return {"answer": _answer(w, q)}


@app.get("/api/v1/portfolio")
def portfolio():
    """Risk overview across every finished walkthrough (the dashboard)."""
    rows, cur = [], None
    for d in _folders():
        w = _read_json(d / "walkthrough.json") if (d / "walkthrough.json").exists() else None
        if not w or not w.get("risk"):
            continue
        r = w["risk"]
        rc = r.get("repair_cost") or {}
        cur = cur or rc.get("currency")
        rows.append({"id": d.name, "kind": "walkthrough", "property_id": w.get("property_id") or "", "label": d.name,
                     "created_at": w.get("created_at"), "grade": r.get("grade"), "condition_score": r.get("condition_score"),
                     "risk_level": r.get("risk_level"), "decision": r.get("decision"), "repair_low": rc.get("low"),
                     "repair_high": rc.get("high"), "valuation_pct": (r.get("valuation") or {}).get("impact_pct"),
                     "integrity_level": (w.get("integrity") or {}).get("level", "low"),
                     "coverage_pct": (w.get("coverage") or {}).get("overall"),
                     "approval": ((w.get("agent") or {}).get("approval") or {}).get("decision")})

    def by(k: str) -> dict:
        return {v: sum(1 for x in rows if x.get(k) == v) for v in sorted({x.get(k) for x in rows if x.get(k)})}

    return {"n": len(rows), "by_grade": by("grade"), "by_risk_level": by("risk_level"), "by_decision": by("decision"),
            "repair_exposure": {"low": sum(x["repair_low"] or 0 for x in rows), "high": sum(x["repair_high"] or 0 for x in rows),
                                "currency": cur or "USD"},
            "integrity_flagged": sum(1 for x in rows if x["integrity_level"] in ("medium", "high")),
            "rows": sorted(rows, key=lambda x: str(x.get("created_at") or ""), reverse=True)}


# ---------------------------------------------------------------- the person in the loop: approval and finding review

class Approval(BaseModel):
    decision: str
    note: str = ""
    by: str = "inspector"


def _agent_block(a: dict) -> dict:
    """walkthrough.json["agent"] from agent.json (as walkthrough.build writes it)."""
    return {"engine": a.get("engine"), "trace": a.get("trace", []), "refilm": a.get("refilm", []),
            "proposal": a.get("decision"), "approval": a.get("approval")}


def _human_step(trace: list, tool: str, data: dict) -> None:
    trace.append({"step": len(trace) + 1, "kind": "human", "tool": tool, "input": data, "output": {"ok": True}})


def _wt_risk(d: Path, a: dict, w: dict) -> dict | None:
    """The walkthrough's risk, recomputed from the agent's findings on the coverage the viewer shows (as the builder
    does); agent.json's own risk when the pipeline result is missing."""
    from . import risk
    r = _read_json(d / "result.json")
    if r is None or not w.get("coverage"):
        return a.get("risk")
    return risk.assess_result([{**f, "status": f.get("status")} for f in a.get("findings", [])], {**r, "coverage": w["coverage"]})


@app.post("/api/walkthroughs/{name}/approve")
def approve(name: str, body: Approval):
    d = _dir(name)
    decision = body.decision.strip().upper()
    if decision not in ("ACCEPT", "REFER", "REPAIR"):
        raise HTTPException(400, "decision must be ACCEPT, REFER or REPAIR")
    p, wp = d / "agent.json", d / "walkthrough.json"
    with _lock:
        a = _read_json(p) if p.exists() else None
        w = _read_json(wp) if wp.exists() else None
        if a is None and not (w and w.get("agent")):
            raise HTTPException(404, "no agent decision to approve")
        prop = (a.get("decision") if a is not None else w["agent"].get("proposal")) or {}
        proposed = prop.get("proposed") or prop.get("decision")
        approval = {"decision": decision, "proposed": proposed, "agreed": decision == proposed, "note": body.note,
                    "by": body.by, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if a is not None:
            a["approval"] = approval
            _human_step(a.setdefault("trace", []), "approval", approval)
            _write_json(p, a)
        if w is not None:
            if not w.get("agent"):
                w["agent"] = _agent_block(a)
            else:
                w["agent"]["approval"] = approval
                _human_step(w["agent"].setdefault("trace", []), "approval", approval)
            _write_json(wp, w)
    return approval


class Review(BaseModel):
    status: str                 # confirmed | dismissed | needs_review
    severity: str = "moderate"  # minor | moderate | major, for confirmed findings
    note: str = ""
    by: str = "inspector"


@app.post("/api/walkthroughs/{name}/findings/{fid}")
def review_finding(name: str, fid: str, body: Review):
    """A person settles a finding; the grade, risk and rules-based decision are recomputed from it."""
    from . import risk
    d = _dir(name)
    p, wp = d / "agent.json", d / "walkthrough.json"
    if body.status not in ("confirmed", "dismissed", "needs_review") or not (p.exists() or wp.exists()):
        raise HTTPException(400, "unknown walkthrough or status")
    review = {"status": body.status, "severity": body.severity.lower(), "note": body.note, "by": body.by,
              "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    data = {"finding_id": fid, **body.model_dump()}
    with _lock:
        a = _read_json(p) if p.exists() else None
        w = _read_json(wp) if wp.exists() else None
        f = wf = None
        if a is not None:
            f = next((x for x in a.get("findings", []) if x["id"] == fid), None)
            if f is None:
                raise HTTPException(404, "unknown finding")
            f.update(status=body.status, reviewed_by=body.by, review_note=body.note, review=review)
            if body.status == "confirmed":
                f["severity"] = body.severity.lower()
            if (d / "result.json").exists():
                a["risk"] = risk.assess_result(a["findings"], json.loads((d / "result.json").read_text(encoding="utf-8")))
            _human_step(a.setdefault("trace", []), "review_finding", data)
            _write_json(p, a)
        if w is not None:
            wf = next((x for x in w.get("findings", []) if x.get("issue_id") == fid), None)
            if wf is None and a is None:
                raise HTTPException(404, "unknown finding")
            if wf is not None:
                wf.update(state=STATE[body.status], review=review)
                if body.status == "confirmed":
                    wf["severity"] = body.severity.lower()
            if a is not None:
                w["risk"] = _wt_risk(d, a, w) or w.get("risk")
            if w.get("agent") is not None:
                _human_step(w["agent"].setdefault("trace", []), "review_finding", data)
            _write_json(wp, w)
    return {"finding": f or wf, "risk": (w or a).get("risk")}


@app.post("/api/walkthroughs/{name}/agent")
def rerun_agent(name: str):
    from .agent import run_agent
    d = _dir(name)
    if not (d / "result.json").exists():
        raise HTTPException(404, "unknown walkthrough")
    r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    a = run_agent(r, d, _walkthroughs.get(name))
    wp = d / "walkthrough.json"
    with _lock:
        w = _read_json(wp) if wp.exists() else None
        if w is not None:            # the viewer shows the new trace, proposal, finding states and risk
            by_id = {f["id"]: f for f in a.get("findings", [])}
            for wf in w.get("findings", []):
                f = by_id.get(wf.get("issue_id"))
                if f is not None:
                    wf.update(state=STATE.get(f.get("status"), wf.get("state")), review=f.get("review"))
            w["agent"] = _agent_block(a)
            w["risk"] = _wt_risk(d, a, w) or w.get("risk")
            _write_json(wp, w)
    return JSONResponse({"engine": a["engine"], "decision": a["decision"]})


# ---------------------------------------------------------------- rule-based answers and the Markdown summary

STATE_TXT = {"DETECTED": "confirmed", "NEEDS_REVIEW": "needs review", "DISMISSED": "dismissed"}
SEV_RANK = {"severe": 0, "major": 0, "moderate": 1, "minor": 2}
DEFECT_WORDS = {"crack": ("crack",), "mould": ("mould", "mold"), "mold": ("mould", "mold"), "stain": ("stain", "water mark"),
                "water": ("water", "damp", "stain", "mould", "mold", "leak"), "damp": ("damp", "water", "mould", "mold"),
                "leak": ("water", "leak", "stain"), "rust": ("rust",), "peel": ("peel",), "paint": ("peel", "paint"),
                "hole": ("hole",), "spall": ("spall",), "glass": ("glass",), "door": ("door",), "window": ("window",)}


def _money(lo, hi, cur: str) -> str:
    if lo is None and hi is None:
        return "not estimated"
    if hi is None or lo == hi:
        return f"{cur} {lo or hi:,.0f}"
    return f"{cur} {lo:,.0f}–{hi:,.0f}"


def _fline(f: dict) -> str:
    where = f.get("surface_label") or f.get("room") or ""
    return (f"- **{f.get('issue_id')}** {f.get('component', '')} · {f.get('defect', '')} ({f.get('severity') or '?'}, "
            f"{STATE_TXT.get(f.get('state'), f.get('state'))}){' — ' + where if where else ''}")


def _answer(w: dict, q: str) -> str:
    ql = q.lower()

    def has(*words: str) -> bool:
        return any(x in ql for x in words)

    r = w.get("risk") or {}
    rc = r.get("repair_cost") or {}
    cur = rc.get("currency", "USD")
    findings = w.get("findings") or []
    live = [f for f in findings if f.get("state") != "DISMISSED"]
    cov = (w.get("coverage") or {}).get("overall") or {}
    ag = w.get("agent") or {}
    prop = ag.get("proposal") or {}
    out: list[str] = []

    grade = (f"Grade **{r.get('grade', '?')}** ({r.get('grade_label', '')}), "
             f"condition score {r.get('condition_score', '?')} / 100, {r.get('risk_level', '?')} risk."
             if r else "This walkthrough has no risk assessment.")
    if has("grade", "condition", "score", "risk", "overall", "summary", "state of"):
        out += [grade]
    if has("cover", "gap", "film", "miss", "seen", "room", "ceiling", "floor", "wall", "complete", "inspect"):
        out += [f"**Coverage**: {cov.get('inspectable_pct', '?')}% of the surfaces were filmed close and square enough to inspect "
                f"(seen {cov.get('seen_pct', '?')}%, analysed in keyframes {cov.get('inspected_pct', '?')}%)."]
        rooms = w.get("rooms") or []
        if rooms:
            out.append("Rooms mapped: " + ", ".join(f"{x.get('name')} ({x.get('area_m2', '?')} m²)" for x in rooms) + ".")
        gaps = w.get("gaps") or []
        if gaps:
            out.append("Not covered well enough — film these again:")
            out += [f"- {g.get('label')}: " + (f"{g['length_m']} m of wall" if g.get("length_m") else
                                               f"{g.get('uncovered_m2', '?')} m² not inspectable") for g in gaps[:6]]
        for x in (ag.get("refilm") or [])[:4]:
            out.append(f"- Agent re-film request: {x.get('instruction')}")
    if has("cost", "repair", "price", "expens", "money", "budget", "fix", "exposure"):
        out.append(f"**Repair cost** (indicative): {_money(rc.get('low'), rc.get('high'), cur)}.")
        perils = sorted((r.get("insurance") or {}).get("perils") or [], key=lambda p: -((p.get("repair_cost") or [0, 0])[-1] or 0))
        out += [f"- {p.get('label')}: {p.get('findings')} finding(s), {_money(*(p.get('repair_cost') or [None, None]), cur)}"
                for p in perils[:5]]
    if has("decision", "refer", "accept", "proceed", "approv", "why", "underwrit", "agent", "next", "propos"):
        out.append(f"**Underwriting rules**: {r.get('decision', 'no decision')}.")
        out += [f"- {x}" for x in (r.get("reasons") or [])[:6]]
        if prop:
            out.append(f"**Agent proposal**: {prop.get('proposed') or prop.get('decision')}"
                       + (f" — {prop['summary']}" if prop.get("summary") else ""))
        ap = ag.get("approval")
        if ap:
            out.append(f"**Approved** by {ap.get('by')} on {ap.get('at')}: {ap.get('decision')}"
                       + (f" ({ap['note']})" if ap.get("note") else ""))
        elif ag:
            out.append("Not yet approved by a person.")
    if has("value", "valuation", "premium", "insur", "mortgage", "collateral", "lend"):
        val, ins, mort = r.get("valuation") or {}, r.get("insurance") or {}, r.get("mortgage") or {}
        imp = val.get("impact_pct")
        imp_txt = f"{imp[0]}% to {imp[1]}%" if isinstance(imp, list) else "n/a" if imp is None else f"{imp}%"
        basis = f" ({val['basis']})" if val.get("basis") else ""
        out.append(f"**Valuation impact**: {imp_txt}{basis}; premium loading {ins.get('premium_loading_pct', 'n/a')}%; "
                   f"collateral risk {mort.get('collateral_risk', 'n/a')}.")
    if has("integrity", "fraud", "fake", "tamper", "authentic", "screen", "evidence", "genuine"):
        ig = w.get("integrity")
        if ig:
            out.append(f"**Evidence integrity**: {ig.get('level', 'low')} risk.")
            out += [f"- {x.get('message')} ({x.get('level')})" for x in (ig.get("flags") or [])[:5]] or \
                ["- One continuous take, no reuse of earlier footage, no sign of a screen being filmed."]
        ev = r.get("evidence") or {}
        if ev:
            out.append(f"Evidence coverage {ev.get('coverage_pct', '?')}%: {'complete' if ev.get('complete') else 'incomplete'}.")
    kinds = sorted({k for word, ks in DEFECT_WORDS.items() if word in ql for k in ks})
    if kinds or has("finding", "defect", "damage", "problem", "issue", "worst", "serious", "check", "survey", "review", "confirm"):
        pick = [f for f in live if any(k in f"{f.get('defect', '')} {f.get('component', '')} {f.get('description', '')}".lower()
                                       for k in kinds)] if kinds else live
        pick.sort(key=lambda f: (f.get("state") != "DETECTED", SEV_RANK.get(str(f.get("severity")).lower(), 3)))
        n_det = sum(f.get("state") == "DETECTED" for f in pick)
        if kinds and not pick:
            out.append(f"No {' / '.join(kinds)} finding in this walkthrough's record.")
        else:
            matching = f" matching {' / '.join(kinds)}" if kinds else ""
            out.append(f"**{len(pick)} finding(s)**{matching}: {n_det} confirmed, {len(pick) - n_det} need a person's review.")
            out += [_fline(f) for f in pick[:8]]
            if any(f.get("state") == "NEEDS_REVIEW" for f in pick):
                out.append("Findings that need review are unconfirmed: a person should check them on site or in the crops.")
    out = [x for x in out if x]
    if not out:
        out = [grade, f"Decision: {r.get('decision', 'none')}.",
               f"Coverage: {cov.get('inspectable_pct', '?')}% inspectable, {cov.get('inspected_pct', '?')}% analysed.",
               f"{len(live)} finding(s), {sum(f.get('state') == 'DETECTED' for f in live)} confirmed.",
               "Ask about coverage gaps, findings (e.g. mould or cracks), repair cost, the decision, valuation "
               "or evidence integrity."]
    return "\n".join(out)


def _markdown(w: dict) -> str:
    """A Markdown report from walkthrough.json alone (when the pipeline's result.json is not there)."""
    r = w.get("risk") or {}
    rc = r.get("repair_cost") or {}
    cov = (w.get("coverage") or {}).get("overall") or {}
    ap = (w.get("agent") or {}).get("approval")
    lines = [f"# Inspection report — {w.get('name')}", "", f"- Property: {w.get('property_id') or '—'}",
             f"- Created: {w.get('created_at') or '—'}", f"- Video: {w.get('duration_s', '?')} s",
             f"- Grade {r.get('grade', '?')} ({r.get('grade_label', '')}), score {r.get('condition_score', '?')}, "
             f"{r.get('risk_level', '?')} risk", f"- Decision: {r.get('decision', '—')}",
             f"- Repair cost: {_money(rc.get('low'), rc.get('high'), rc.get('currency', 'USD'))}",
             f"- Coverage: seen {cov.get('seen_pct', '?')}%, inspectable {cov.get('inspectable_pct', '?')}%, "
             f"analysed {cov.get('inspected_pct', '?')}%"]
    if ap:
        lines.append(f"- Approved: {ap.get('decision')} by {ap.get('by')} on {ap.get('at')}")
    lines += ["", "## Why this decision", ""] + [f"- {x}" for x in r.get("reasons") or []]
    lines += ["", "## Findings", ""] + [_fline(f) for f in w.get("findings") or []]
    lines += ["", "## Film again", ""] + [f"- {g.get('label')}" for g in w.get("gaps") or []]
    if r.get("disclaimer"):
        lines += ["", f"_{r['disclaimer']}_"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the walkthroughs themselves: results/<name>/

@app.get("/walkthroughs/{name}/", include_in_schema=False)
def walkthrough_page(name: str):
    """The 3D viewer of one walkthrough: always the current web/walkthrough.html (the copy in the folder is for viewing
    the folder on its own), so a walkthrough built before a UI update gets the new viewer too."""
    d = _dir(name)
    if (d / "walkthrough.json").exists():
        return FileResponse(WEB / "walkthrough.html", media_type="text/html")
    if (d / "index.html").exists():
        return FileResponse(d / "index.html", media_type="text/html")
    raise HTTPException(404, "unknown walkthrough")


class _WalkthroughFiles(StaticFiles):
    """results/ as static files, without the _ and . folders (indexes, scratch)."""

    async def get_response(self, path: str, scope):
        if re.split(r"[\\/]", path, maxsplit=1)[0].startswith(("_", ".")):
            raise HTTPException(404)
        return await super().get_response(path, scope)


app.mount("/walkthroughs", _WalkthroughFiles(directory=RESULTS, html=True), name="walkthroughs")
