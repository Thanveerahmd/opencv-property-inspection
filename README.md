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

| Stage | OpenCV 5 | Module |
| --- | --- | --- |
| Video and quality gate | `VideoCapture`, Laplacian sharpness, `DISOpticalFlow`, CLAHE | `video.py` |
| Privacy | `FaceDetectorYN` (YuNet) and DB text detection, blurred before analysis | `privacy.py` |
| Camera tracking | LK optical flow, MAGSAC essential matrix, `recoverPose`, `triangulatePoints`, `solvePnPRansac` | `tracking.py`, `odometry.py` |
| Depth | `cv2.dnn` running Depth Anything V2 Small (ONNX) on the CPU | `depth.py` |
| 3D fusion | `cv2.Volume` hash TSDF, `fetchPointsNormals`, `savePointCloud` | `fusion.py` |
| Room layout | walls, floor and ceiling from fused points and normals | `layout.py` |
| Coverage | 10 cm samples, `projectPoints`, depth occlusion, 3 m / 70° / sharp rule | `coverage.py` |
| Defects | ridge filter, thinning, distance transform, Lab colour, morphology, contours; size in metres | `defects.py` |
| Walls | plane homographies and `warpPerspective`: each wall rectified into one image | `ortho.py` |
| Labels | DB text detection + CRNN recognition on rating plates | `ocr.py` |
| Agent | Bedrock Converse with tool use; tools run the OpenCV stages above | `agent.py` |
| Web endpoint | FastAPI: upload, results, 3D view, agent trace, human approval | `server.py`, `web/` |

## Run it

Python 3.11. Dependencies are pinned in `requirements.txt` and `requirements-dev.txt`.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m spatial_inspection.models download     # model files from their sources, SHA-256 checked
python -m pytest
```

Analyse a video (OpenCV-only path), then let the agent work through it:

```bash
python -m spatial_inspection.pipeline --video walkthrough.mp4 --out results/walkthrough --agent
python -m spatial_inspection.agent results/walkthrough     # re-run the agent only
```

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
  web/                    the web app (HTML, CSS, JS; three.js from a CDN)
tests/                    pytest; tests needing models or sample videos skip when they are absent
deploy/aws/               deploy.sh (run locally) and bootstrap.sh (EC2 user-data)
docs/                     technical report, architecture diagram, build plan
```

## Licences

Every model and library allows commercial use: OpenCV 5 (Apache-2.0), YuNet (MIT), PP-OCRv3 and CRNN (Apache-2.0),
Depth Anything V2 Small (Apache-2.0). No non-commercial weights (for example LocateAnything-3B or the Inria
Gaussian Splatting code) are used.

## Working on it

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`), one topic per commit.
- `pre-commit run --all-files` and `pytest` must pass; CI runs both.
- No walkthrough videos, model weights or `.env` files in the repository.
