# Spatially Aware Property Inspection

Team XCore's entry for the OpenCV AI Competition 2026 (powered by AWS): find damage in a phone walkthrough of a
property, place every finding in a photorealistic 3D model of the rooms, and produce a structured inspection report.

Status: **started**. The repository is set up and camera tracking works; the rest is planned in
[docs/plan.md](docs/plan.md).

## Technology stack

| Part | What it does | Built with |
| --- | --- | --- |
| 1. Damage detection | Finds and outlines defects (cracks, stains, damp, mould, broken fittings) and describes them | Qwen3-VL-2B + Grounding DINO: open-vocabulary localisation, defect identification and visual analysis |
| 2. Camera tracking and spatial understanding | Works out how the phone moved between frames, so every frame has a pose | OpenCV 5: corner tracking with optical flow, essential-matrix pose with MAGSAC, geometric pose refinement |
| 3. Photorealistic 3D reconstruction | Builds a 3D model of the property from the walkthrough | Multi-view reconstruction + 3D Gaussian Splatting (gsplat) |
| 4. Spatially aware investigation | Places each finding in the 3D model, shows it in context and writes the report | Defects projected through the tracked poses into the reconstruction; structured JSON and PDF report |

The competition asks for OpenCV 5 doing meaningful image or video analysis and a significant AWS component. OpenCV 5
carries parts 2 and 4 and the image handling throughout; AWS hosts storage, GPU inference and the processing pipeline
(see the plan).

## Run it

Python 3.11.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt -r requirements-dev.txt
.venv/Scripts/python -m pytest          # Windows; .venv/bin/python on Linux and macOS
.venv/Scripts/pre-commit install         # once per clone
```

Camera motion between two frames:

```python
import cv2
from spatial_inspection.tracking import frame_pose, intrinsics_guess

a, b = cv2.imread("frame_000.jpg"), cv2.imread("frame_005.jpg")
pose = frame_pose(a, b, intrinsics_guess(a.shape))
print(pose.rotation_deg, pose.t, pose.inliers)
```

## Layout

```
src/spatial_inspection/   the package
  tracking.py             camera motion between frames (OpenCV 5)
tests/                    pytest
docs/plan.md              build plan, AWS architecture, submission checklist
```

## Licences to keep clean

Every model and library must allow commercial use, because this work feeds Team XCore's product.

- OpenCV 5: Apache-2.0. Qwen3-VL: Apache-2.0. Grounding DINO (the open-source IDEA-Research release): Apache-2.0;
  the newer Grounding DINO 1.5 / 1.6 Pro models are a cloud API, not open weights.
- 3D Gaussian Splatting: use **gsplat** (Apache-2.0). The original Inria implementation is licensed for
  non-commercial research only.
- Do not bring in LocateAnything-3B from the inspection platform: its weights are NVIDIA non-commercial.

## Working on it

- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`), one topic per commit.
- Branches `feat/…`, `fix/…`, `exp/…`; pull requests into `main`.
- `pre-commit run --all-files` and `pytest` must pass; CI runs both.
- No walkthrough videos, model weights or `.env` files in the repository.
