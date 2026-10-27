# Spatially Aware Property Inspection

Team XCore's entry for the OpenCV AI Competition 2026 (powered by AWS), Agentic Vision path. From one phone
walkthrough video of a room, OpenCV 5 rebuilds the room in 3D, measures how much of every wall, floor and ceiling was
filmed well enough to inspect, finds and measures damage, and places every finding on the surface where it is. An
agent on Amazon Bedrock then works through that evidence with OpenCV tools, asks for missing parts of the room to be
filmed again, and proposes a decision that a person approves.

- Technical report: [docs/technical-report.md](docs/technical-report.md)
- Architecture: [docs/architecture.svg](docs/architecture.svg)

![Architecture](docs/architecture.svg)

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
| Photoreal 3D | 3D Gaussian splats (gsplat) started on the engine's surface and trained on its cameras, per-frame exposure, held-out frames for scoring; shown in the viewer with the finding pins and the camera path | `engine3d/train_splats.py`, `web/walkthrough.html` |
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
