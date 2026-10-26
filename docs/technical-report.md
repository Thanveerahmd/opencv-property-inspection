# Spatially Aware Property Inspection

Technical report · Team XCore · OpenCV AI Competition 2026 (powered by AWS) · Agentic Vision path

## 1. Problem

Insurers, lenders and landlords need to know the condition of a property, and more of that evidence now arrives as a
phone video filmed by an owner, tenant or field agent rather than as an inspector's visit. Two things go wrong:

1. **Coverage.** People film what they notice. In our earlier tests with phone walkthroughs, only 20–43% of the room
   surface was filmed well enough to inspect, and ceilings (where leaks show first) were almost never filmed.
   Nobody finds out until the video is reviewed, when the person has left.
2. **Evidence.** A defect in a video frame has no position, no size and no context: it cannot be found again,
   measured, or compared with the next inspection.

## 2. Users

- **Insurers and lenders** deciding whether a property is acceptable, needs repair, or needs more evidence.
- **Field agents, tenants and owners** filming the property, who need clear instructions on what is still missing.
- **Inspectors** who review the evidence and make the final call.

## 3. What the system does

From one phone video of a room:

1. OpenCV 5 checks every frame (sharpness, exposure, motion), blurs faces and private text, tracks the camera,
   estimates depth, and fuses everything into a 3D model of the room.
2. It fits the room's walls, floor and ceiling, measures how much of each surface was filmed well enough to inspect,
   and lists the largest gaps.
3. It finds defect candidates (cracks, stains and water damage, mould, rust) with classical computer vision, places
   each one in 3D on a named surface, and measures it in metres.
4. It rectifies every wall onto its own plane, so the whole wall can be read as one image with findings marked.
5. An agent on Amazon Bedrock works through the evidence with OpenCV tools: it looks at candidates, fetches better
   views from other frames, confirms or dismisses, asks for parts of the room to be filmed again, and proposes a
   decision (ACCEPT, REFER or REPAIR) that a person approves or overrides.

![Architecture](architecture.svg)

## 4. OpenCV 5 implementation

| Stage | OpenCV 5 functionality | Module |
| --- | --- | --- |
| Video and quality | `VideoCapture`, `Laplacian` variance, `DISOpticalFlow`, `calcHist`, `createCLAHE` | `video.py`, `pipeline.py` |
| Privacy | `FaceDetectorYN` (YuNet), `dnn.TextDetectionModel_DB` (PP-OCRv3), `GaussianBlur` | `privacy.py` |
| Camera tracking | `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` (forward-backward checked), `findEssentialMat` (USAC MAGSAC), `recoverPose`, `triangulatePoints`, `solvePnPRansac` | `tracking.py`, `odometry.py` |
| Depth | `dnn.readNet` + `blobFromImage` running Depth Anything V2 Small (ONNX) on the CPU, aligned to metric scale | `depth.py` |
| 3D fusion | `Volume` (hash TSDF), `VolumeSettings`, `fetchPointsNormals`, `savePointCloud`, `projectPoints` for colour | `fusion.py` |
| Room layout | normals folded every 90° for the room heading; wall bands with a "seen through a doorway" test | `layout.py` |
| Coverage | `projectPoints` of 10 cm surface samples into every frame, depth-buffer occlusion, 3 m / 70° / sharp rule | `coverage.py` |
| Defect evidence | `ximgproc.RidgeDetectionFilter`, `ximgproc.thinning`, `distanceTransform`, morphology, Lab colour statistics, `SimpleBlobDetector`, `HoughLinesP`, `connectedComponentsWithStats`, `findContours` | `defects.py` |
| Review crops | `createCLAHE`, `fastNlMeansDenoisingColored`, `detailEnhance` | `enhance.py` |
| Wall rectification | plane-induced homographies, `warpPerspective`, depth-based occlusion masks, weighted blending | `ortho.py` |
| Labels | `dnn.TextDetectionModel_DB` + `dnn.TextRecognitionModel` (CRNN), `getPerspectiveTransform` | `ocr.py` |

Every process caps OpenCV at a fixed number of threads (`setNumThreads`) and runs at reduced priority on a
workstation; the server raises both.

## 5. Agentic Vision: perception → decision → action

The agent (`agent.py`) is a tool-using model on Amazon Bedrock (Converse API, images in tool results). Its tools run
OpenCV 5 on the walkthrough, and their results decide what it does next:

- **Coverage → action.** If less than 60% of the surface was inspected, the decision must be REFER, and the agent
  writes re-film instructions for the largest gaps ("tilt up and sweep the whole main-room ceiling slowly").
- **Unclear evidence → new perception.** If a crop is blurred, far or ambiguous, `find_better_view` projects the
  finding's 3D position into every other frame, keeps the frames where the depth map shows it unoccluded, picks the
  closest, most square-on, sharpest view, re-runs OpenCV detection on it and shows the agent the new image.
- **Re-detection → decision.** The candidate is confirmed (with a severity) or dismissed (with a reason).
- **Human control.** The run ends with `request_human_approval`; the inspector approves or overrides in the web
  app, and that is recorded in the same trace.

Every step is written to `agent.json`: the model's reasoning, each tool call with its inputs and outputs, and the
images it looked at. When Bedrock is not reachable the same tools run under a rule-based policy, and the trace says so.

## 6. AWS deployment

| Service | Use |
| --- | --- |
| Amazon EC2 (Graviton, arm64) | The FastAPI web endpoint and the whole OpenCV 5 pipeline (CPU only) |
| Amazon S3 | Uploaded videos, results, rectified walls, 3D points, agent traces, the code bundle |
| Amazon Bedrock | The agent's model (Claude, Converse API with tool use and images) |
| AWS IAM + Systems Manager | Instance role limited to the bucket and Bedrock; shell access without SSH keys |

`deploy/aws/deploy.sh` creates the bucket, role, security group and instance; `deploy/aws/bootstrap.sh` installs
the pinned dependencies, downloads the models from their sources with SHA-256 checks, and starts the service.

## 7. Evaluation

_To be completed with the measured results: coverage against the inspection platform's GPU engine on the same
videos, trajectory error of the OpenCV-only path, defect-candidate precision on labelled sample images, runtime per
stage on the Graviton instance, and agent task success over the sample walkthroughs, with failure cases._

## 8. Limitations

_To be completed._

## 9. Responsible use

- Faces and private text are blurred before any analysis, before anything is stored, and before any image reaches
  the model.
- The agent proposes; a person decides. Every decision is recorded with the evidence behind it, and overrides are
  kept.
- Coverage is reported honestly: surfaces that could not be filmed (behind furniture) and surfaces that were not
  filmed are listed as not inspected, never assumed fine.
- Models: OpenCV Zoo YuNet (MIT), PP-OCRv3 and CRNN (Apache-2.0), Depth Anything V2 Small (Apache-2.0); no
  non-commercial weights.
