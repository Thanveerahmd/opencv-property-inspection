# Spatially Aware Property Inspection

Team XCore's entry for the OpenCV AI Competition 2026 (powered by AWS), Agentic Vision path. From one phone
walkthrough video of a room, OpenCV 5 rebuilds the room in 3D, measures how much of every wall, floor and ceiling was
filmed well enough to inspect, finds and measures damage, and places every finding on the surface where it is. An
agent on Amazon Bedrock then works through that evidence with OpenCV tools, asks for missing parts of the room to be
filmed again, and proposes a decision that a person approves.

- Technical report: [docs/technical-report.md](docs/technical-report.md)
- Architecture: [docs/architecture.svg](docs/architecture.svg)
- What it looks like: [See it in action](#see-it-in-action) (screenshots and clips of every scenario)

<p align="center"><img src="docs/media/photoreal-splats.gif" width="760"
  alt="The phone video (left) and the room's Gaussian splats rendered from the same camera (right)"></p>
<p align="center"><sub>Left: the phone video. Right: the room's 3D Gaussian splats, rendered from the camera the 3D engine
  recovered for each frame.</sub></p>

![Architecture](docs/architecture.svg)

## See it in action

Everything below is the app running locally on a laptop (NVIDIA RTX 3080 Laptop GPU) on two real phone walkthroughs:
`demo_property_lite` (19.6 s, two rooms) and `tom_home` (21.1 s, an open-plan living room with a staircase). No frame
is staged: the screenshots come from the same results folders the evaluation in the technical report uses.

### 1. Upload a walkthrough and watch it being scanned

A person (an owner, tenant or field agent) uploads a phone video on the **Video walkthroughs** page, with the property
it is for and a quality setting (Advanced: 1.5 cm model; Lite: faster). The video plays under a scanning beam while the
server works through **20 steps in three parts**, ticked off live with the time left:

- **3D engine** (11 steps): Depth Anything 3 depth and pose on the GPU, floor and metric scale, OpenCV 5 camera tracking,
  depth filtering, ICP alignment and the TSDF mesh, gravity and Manhattan frame, room layout, furniture named by
  Grounding DINO, texturing from the video, floor plan, 3D exports.
- **Photoreal 3D**: Gaussian splats trained on the engine's cameras.
- **OpenCV analysis** (6 steps): quality and evidence integrity, privacy blurring, OpenCV TSDF and room planes,
  coverage and keyframes, defect candidates, labels and rectified walls; then the **agent** and the **viewer files**.

Walkthroughs are processed one at a time; the dashboard follows whichever one is running.

![Live scanning progress on the dashboard](docs/media/progress-dashboard.jpg)

| Upload form and walkthrough list | |
| --- | --- |
| ![Video walkthroughs page](docs/media/walkthroughs.jpg) | Each finished walkthrough is a card with its coverage map, grade, decision, coverage, findings and rooms; it opens the 3D viewer. |

### 2. Dashboard: every inspection at a glance

Condition grades, risk levels, underwriting decisions, repair exposure and average inspected coverage across all
walkthroughs, with the most recent ones as cards and a sortable table below. The sidebar shows the models in use (Depth
Anything 3 on the GPU, OpenCV 5, the inspection agent) and the GPU's memory.

![Dashboard](docs/media/dashboard.jpg)

### 3. Walk through the room in 3D

Each walkthrough opens in its own viewer: the decision, the approval state, and the key numbers (condition score,
inspectable coverage, findings, floor area, repair cost) above a 3D model with numbered finding pins.

![Walkthrough viewer](docs/media/viewer-tom.jpg)

The 3D model has several views of the same room, all in metres and in the same frame as the findings:

| Photoreal (Gaussian splats) | Photoreal walk inside |
| --- | --- |
| ![Photoreal](docs/media/photoreal.jpg) | ![Photoreal walk inside](docs/media/photoreal-walk.jpg) |
| Orbit the room as it looks in the video. The splats score 28.4 dB PSNR on frames they never trained on (the textured mesh: 20.5 dB). The blue line is the path the phone took. | The browser camera follows the phone: play or scrub the small video and the 3D view moves with it, frame by frame. |

| Walk inside the measured model | The clean model, orbiting |
| --- | --- |
| ![Walk inside](docs/media/walk-inside.jpg) | ![Clean model orbit](docs/media/clean-model-orbit.gif) |
| The textured TSDF model from the video camera: what the measurements and coverage are computed on. | Walls, floor and ceiling as clean planes textured from the video, with the furniture separated from the structure. |

Also in the tabs: **Coverage gaps** (red and amber patches that were not inspectable), the **SLAM model** (point layers
and mesh), the **clean room model** and **furniture only**. Every model downloads as a GLB.

### 4. How the 3D model is made

| Depth Anything 3: depth for every frame | OpenCV 5 tracking: verified feature tracks and poses |
| --- | --- |
| ![Depth Anything 3](docs/media/depth-anything-3.gif) | ![OpenCV tracking](docs/media/opencv-tracking.gif) |

| The video against the reconstructed model | What the engine measured for this walkthrough |
| --- | --- |
| ![Video vs model](docs/media/video-vs-model.gif) | ![3D engine and OpenCV pipeline panel](docs/media/engine-panel.jpg) |

Depth Anything 3 gives depth and an initial pose; OpenCV 5 checks every frame pair (LK optical flow, SIFT + MAGSAC
essential matrix), refines the trajectory and triangulates landmarks; point-to-plane ICP aligns each depth map before
the TSDF mesh is fused. The viewer's engine panel lists the numbers for each walkthrough: pair checks, ICP residuals,
reprojection error, wall flatness, photometric fit, objects and timings.

### 5. Coverage: what was actually inspected

Every 10 cm of wall, floor and ceiling is checked against every frame: **seen**, **inspectable** (within 3 m, at most
70° off square-on, sharp, not hidden) and **inspected** (in an analysed keyframe). The live map replays the walk with
coverage growing on the floor plan; the coverage coach shows, frame by frame, what is still to film and where to stand.

| Live map: the video beside the floor plan as coverage grows | Coverage at the end of the walk |
| --- | --- |
| ![Live map](docs/media/live-map.gif) | ![Coverage map](docs/media/coverage-map.jpg) |

![Coverage coach](docs/media/coach.jpg)

### 6. Findings: defect or object?

OpenCV finds candidates (cracks, stains and water damage, mould, rust, peeling paint), measures them in metres and
places them on the wall, floor or ceiling where they are. Candidates that are furniture, doors, pictures or a room seen
through a doorway are ruled out with the reason, by depth against the room planes, Grounding DINO and MobileNetV2. On
`demo_property_lite`, 66 of 67 candidates were ruled out; the one left (the edge of a dark door at the frame border)
waits for a person, who can **confirm** it with a severity, **dismiss** it, or **reopen** it later. The grade follows.

| Findings and review | Property risk and valuation |
| --- | --- |
| ![Findings](docs/media/findings.jpg) | ![Risk and valuation](docs/media/risk.jpg) |

### 7. The agent proposes, a person decides

The inspection agent works through the evidence with OpenCV tools (coverage report, integrity check, look at a
candidate, find a better view in other frames, measure, re-film requests, risk assessment) and proposes ACCEPT, REPAIR
or REFER. Below 60% inspected the proposal must be REFER, with instructions for what to film again. Every step is in
the trace; nothing is final until a person approves or overrides.

| Human approval and re-film requests | Evidence integrity |
| --- | --- |
| ![Human approval](docs/media/approval.jpg) | ![Evidence integrity](docs/media/integrity.jpg) |

![Agent trace](docs/media/agent-trace.jpg)

| Ask the inspection agent | The exported report |
| --- | --- |
| ![Ask](docs/media/ask.jpg) | ![Report](docs/media/report.jpg) |

The report exports as self-contained HTML (print to PDF), Markdown or JSON.

## How it works

One video goes through four parts. Each one is a module you can run on its own.

```
video ─► 3D engine (engine3d/, GPU)       Depth Anything 3 depth + pose ─► OpenCV 5 LK optical flow, SIFT + MAGSAC pose
                                          verification, trajectory refinement, landmarks, loop candidates ─► point-to-plane
                                          ICP frame-to-model alignment ─► 1.5 cm TSDF + free-space carving ─► gravity +
                                          Manhattan frame ─► clean room shell textured from the video ─► Grounding DINO
                                          object names ─► GLBs, floor plan, occupancy / navigation maps
      ─► OpenCV analysis (pipeline.py)    quality gate, evidence integrity, privacy, cv2.Volume TSDF, surface coverage,
                                          keyframes, defect candidates placed on the room planes, labels, rectified walls
      ─► agent (agent.py)                 looks again from better views, re-film requests, grade / risk / decision
      ─► viewer (walkthrough.py)          walkthrough.json, 3D model with finding pins, walk-inside camera path, live map,
                                          coverage map, coverage coach, report; a person reviews findings and approves
```

| Stage | What runs | Module |
| --- | --- | --- |
| Depth and pose | Depth Anything 3 (DA3-BASE, Apache-2.0) on the GPU, chunked with similarity alignment | `engine3d/v1prod/da3_infer.py` |
| Camera tracking | OpenCV 5: LK optical flow tracks, SIFT + MAGSAC essential-matrix checks per frame pair, trajectory refinement, ORB loop candidates, landmark reprojection | `engine3d/v1prod/slam/` |
| Dense geometry | depth filtering + joint-bilateral upsampling, point-to-plane ICP per frame, TSDF mesh, free-space carving | `engine3d/v1prod/stage_slam.py` |
| Clean room geometry | RANSAC floor, metric scale, Manhattan yaw, plane regularisation, walls / doors / openings / rooms | `engine3d/v1prod/layout.py`, `shell.py` |
| Real-world texturing | walls, floor and ceiling textured from the video with exposure and vignetting compensation | `engine3d/v1prod/pipeline.py`, `render.py` |
| Objects | furniture separated from the structure, named by multi-view voting of Grounding DINO boxes | `engine3d/v1prod/objects.py`, `labels.py`, `grounding.py` |
| Photoreal 3D | 3D Gaussian splats (gsplat) started on the engine's surface and trained on its cameras, per-frame exposure, held-out frames for scoring; shown in the viewer with the finding pins and the camera path | `engine3d/train_splats.py`, `render_splats.py`, `web/walkthrough.html` |
| Video and quality gate | `VideoCapture`, Laplacian sharpness, CLAHE | `video.py`, `pipeline.py` |
| Privacy | `FaceDetectorYN` (YuNet) and DB text detection, blurred before analysis | `privacy.py` |
| 3D fusion (OpenCV) | `cv2.Volume` hash TSDF on the engine's depth and poses, `fetchPointsNormals` | `fusion.py` |
| Coverage | 10 cm surface samples, occlusion against depth, 3 m / 70° / sharp rule, keyframes by set cover | `coverage.py`, `spatial/coverage.py` |
| Evidence integrity | `img_hash.PHash` against earlier walkthroughs, `compareHist` hard cuts, frozen and looped frames, `dft` screen moiré, `findContours` bezels | `integrity.py` |
| Defects | ridge filter, thinning, distance transform, Lab colour, morphology, contours; size in metres | `defects.py` |
| Defect or object? | each candidate's ray cast onto the room planes and checked against depth (furniture, doors, openings); Grounding DINO boxes (on a chair, frame, door …) and MobileNetV2 (`cv2.dnn`) rule objects out; Grounding DINO damage boxes corroborate | `localize.py`, `pipeline.py`, `objects.py` |
| Walls | plane homographies and `warpPerspective`: each wall rectified into one image | `ortho.py` |
| Labels | DB text detection + CRNN recognition on rating plates | `ocr.py` |
| Grade, risk, cost, decision | condition grade A–E, risk level, insurance perils, repair cost, valuation, underwriting decision with reasons | `risk.py` |
| Agent | Bedrock Converse with tool use; tools run the OpenCV stages above; rule-based policy without AWS | `agent.py` |
| Viewer | Photoreal (Gaussian splats) and textured 3D model with numbered finding pins, walk-inside mode on the camera path, live map video, coverage coach | `walkthrough.py`, `spatial/`, `web/walkthrough.html` |
| Report | self-contained HTML (print to PDF), Markdown and JSON | `report.py` |
| Web app | FastAPI: dashboard, upload with live progress, walkthrough viewer, finding review, agent trace, human approval | `server.py`, `web/` |

## Run it

Python 3.11. Dependencies are pinned in `requirements.txt` and `requirements-dev.txt`.

**One command on Windows** (NVIDIA GPU, [uv](https://docs.astral.sh/uv/) installed): builds `.venv` (and `.venv-splat`
with `-Splats`), installs the models and checks the GPU. With a model bundle made by
`python scripts/make_model_bundle.py` nothing has to be downloaded from Hugging Face:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle C:\path\spatial-inspection-models.zip -Splats
```

By hand:

```bash
uv venv --python 3.11
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m spatial_inspection.models download     # model files from their sources, SHA-256 checked
python -m pytest
```

Inspect a video end to end (3D engine on the GPU, OpenCV analysis, agent, viewer files in `results/<name>/`):

```bash
python -m spatial_inspection.walkthrough --video walkthrough.mp4 --name walkthrough --property "12 Lake Rd"
python -m spatial_inspection.agent results/walkthrough     # re-run the agent only
```

Photoreal splats run in a second environment, because gsplat's prebuilt Windows wheels need Python 3.10 and
PyTorch 2.4 (commands in `requirements-splat.txt`). When `.venv-splat` exists the build trains them after the 3D
engine (about 2.5 min on a laptop RTX 3080); without it the walkthrough is built without the Photoreal mode.

The model weights download on first use: Depth Anything 3 (`depth-anything/DA3-BASE`) and Grounding DINO
(`IDEA-Research/grounding-dino-tiny`) from Hugging Face; the OpenCV Zoo models with
`python -m spatial_inspection.models download`. Without an NVIDIA GPU, `python -m spatial_inspection.pipeline` runs the
OpenCV-only path (CPU odometry and depth through `cv2.dnn`).

Start the web endpoint and open http://localhost:8000:

```bash
uvicorn spatial_inspection.server:app --port 8000
```

`SI_THREADS` (default 4) caps OpenCV's threads per process and `SI_PRIORITY=low` (default) lowers the process
priority, so a laptop stays cool; the server sets both higher. With AWS credentials, `BEDROCK_MODEL_ID` and
`AWS_REGION` select the agent's model; without them the agent runs a rule-based policy over the same tools.

## Deploy to AWS

```bash
aws login
deploy/aws/deploy.sh
```

This creates an S3 bucket, an IAM role limited to that bucket and Bedrock, a security group, and a Graviton EC2
instance that installs the pinned dependencies and serves the app on port 80 (see `deploy/aws/`).

## Layout

```
src/spatial_inspection/   the package (one module per stage, see the table above)
  spatial/                the viewer's spatial layer: engine loader, coverage, pins, 2D maps, GLBs, coach
  web/                    the web app and the 3D walkthrough viewer (HTML, CSS, JS; three.js from a CDN)
engine3d/                 the 3D engine (Depth Anything 3 + OpenCV 5 SLAM + ICP + TSDF + room shell)
depth_anything_3/         Depth Anything 3 inference code (Apache-2.0, upstream)
tests/                    pytest; tests needing models or sample videos skip when they are absent
deploy/aws/               deploy.sh (run locally) and bootstrap.sh (EC2 user-data)
docs/                     technical report, architecture diagram, build plan
```

## Licences

The code is MIT-licensed ([LICENSE](LICENSE)), © 2026 Team XCore. The vendored `depth_anything_3/` keeps its
Apache-2.0 licence.

Every model and library allows commercial use: OpenCV 5 (Apache-2.0), Depth Anything 3 BASE (code and weights,
Apache-2.0, vendored in `depth_anything_3/`), Grounding DINO tiny (Apache-2.0), Open3D (MIT), YuNet (MIT), PP-OCRv3,
CRNN and MobileNetV2 (Apache-2.0), Depth Anything V2 Small (Apache-2.0), gsplat (Apache-2.0) for training the
splats and GaussianSplats3D (MIT, from a CDN) for showing them. No non-commercial weights (for example
LocateAnything-3B, DA3-GIANT or the Inria Gaussian Splatting code) are used. The 3D engine and the viewer come from
Team XCore's Property Inspection Platform.

## Working on it

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`), one topic per commit.
- `pre-commit run --all-files` and `pytest` must pass; CI runs both.
- No walkthrough videos, model weights or `.env` files in the repository.
