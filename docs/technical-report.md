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

1. **Hybrid visual SLAM and 3D reconstruction** (`engine3d/`, on the GPU). Depth Anything 3 predicts depth, camera
   pose and intrinsics for every sampled frame. OpenCV 5 then checks and corrects that geometry: LK optical-flow
   tracks, SIFT + MAGSAC essential-matrix verification of every frame pair, trajectory refinement, persistent
   landmarks with reprojection checks and ORB loop candidates. Each frame's depth is aligned to the growing model by
   point-to-plane ICP, fused into a 1.5 cm TSDF mesh with free-space carving, and put in a gravity + Manhattan frame.
   The clean room geometry (walls, doors, openings, floor, ceiling, rooms) is fitted on that mesh and textured from
   the video; furniture is separated from the structure and named by multi-view voting of Grounding DINO boxes.
2. **Coverage.** Every 10 cm of wall, floor and ceiling is checked against every frame's depth (occlusion-aware):
   seen, inspectable (within 3 m, at most 70° off straight-on, sharp) and inspected (in an analysed keyframe), with
   the largest gaps listed and a coverage coach that shows where to film next.
3. **Defects.** OpenCV finds candidates (cracks, stains and water damage, mould, rust, peeling paint) on the
   keyframes, places each one in 3D on a named surface and measures it in metres. Candidates that are furniture,
   doors, pictures or rooms seen through an opening are ruled out by depth against the room planes, by Grounding
   DINO boxes and by MobileNetV2; Grounding DINO damage boxes corroborate the rest.
4. **Evidence integrity.** Perceptual hashes against earlier walkthroughs, hard cuts, frozen or looped footage and a
   screen being filmed.
5. **Agent.** An agent on Amazon Bedrock works through the evidence with OpenCV tools: it fetches better views from
   other frames, re-detects, flags or dismisses, asks for parts of the room to be filmed again, grades the risk and
   proposes a decision (ACCEPT, REFER or REPAIR) that a person approves or overrides. Every finding can be confirmed,
   dismissed or reopened by the inspector, and the grade follows.
6. **Viewer.** A 3D model of the room with numbered finding pins, a walk-inside mode that puts the browser camera
   where the phone was at each moment of the video, a live map video of coverage growing as the person walked, the
   floor plan, the agent's trace and a report.

![Architecture](architecture.svg)

## 4. OpenCV 5 implementation

| Stage | OpenCV 5 functionality | Module |
| --- | --- | --- |
| Frames | `VideoCapture` sampling, `resize` (`INTER_AREA`), `cvtColor` | `engine3d/v1prod/da3_infer.py` |
| Camera tracking | `goodFeaturesToTrack`, `calcOpticalFlowPyrLK`, `undistortPoints`, `SIFT_create`, `ORB_create`, `BFMatcher`, `findEssentialMat` (USAC MAGSAC), `findHomography`, `recoverPose` | `engine3d/v1prod/slam/tracking.py`, `verify.py`, `loops.py` |
| Pose refinement | per-pair relative-pose agreement, outlier jumps replaced, trajectory refined against the verified pairs | `engine3d/v1prod/slam/refine.py` |
| Landmarks | `triangulatePoints`, `Rodrigues`, `projectPoints` reprojection RMSE per frame | `engine3d/v1prod/slam/landmarks.py` |
| Depth filtering | flying pixels (`dilate` / `erode` depth jumps, `Canny`), `bilateralFilter`, `ximgproc.jointBilateralFilter` upsampling to full resolution, `connectedComponentsWithStats`, grazing-angle cut | `engine3d/v1prod/slam/depth_filter.py`, `room.py` |
| Real-world texturing | walls / floor / ceiling textured from the video with per-frame exposure gains and a vignetting model; `inpaint` for small unseen holes, `dft` / `mulSpectrums` / `idft` pattern synthesis for unseen floor | `engine3d/v1prod/shell.py`, `render.py` |
| Floor plan and maps | occupancy grid, `morphologyEx`, `distanceTransform`, `findContours`, `approxPolyDP`, `minAreaRect`, `fillPoly`; A* path on the navigation grid | `engine3d/v1prod/maps2d.py`, `slam/occupancy.py` |
| Video and quality | `Laplacian` variance, `calcHist`, `createCLAHE` | `pipeline.py` |
| Privacy | `FaceDetectorYN` (YuNet), `dnn.TextDetectionModel_DB` (PP-OCRv3), `GaussianBlur` | `privacy.py` |
| 3D fusion | `Volume` (hash TSDF), `VolumeSettings`, `fetchPointsNormals`, `savePointCloud` | `fusion.py` |
| Coverage | surface samples projected into every frame, depth-buffer occlusion, 3 m / 70° / sharp rule | `coverage.py`, `spatial/coverage.py` |
| Evidence integrity | `img_hash.PHash`, `compareHist`, `dft` + `createHanningWindow` (moiré), `findContours` + `approxPolyDP` (bezels) | `integrity.py` |
| Defect evidence | `ximgproc.RidgeDetectionFilter`, `ximgproc.thinning`, `distanceTransform`, morphology, Lab colour statistics, `SimpleBlobDetector`, `HoughLinesP`, `connectedComponentsWithStats`, `findContours` | `defects.py` |
| Defect or object | ray cast onto the room planes vs depth; `dnn.readNet` MobileNetV2 (ImageNet) on the crop | `localize.py`, `objects.py` |
| Review crops | `createCLAHE`, `fastNlMeansDenoisingColored` | `enhance.py` |
| Wall rectification | plane-induced homographies, `warpPerspective`, depth-based occlusion masks, weighted blending | `ortho.py` |
| Labels | `dnn.TextDetectionModel_DB` + `dnn.TextRecognitionModel` (CRNN), `getPerspectiveTransform` | `ocr.py` |
| Viewer media | live map frames drawn with OpenCV (`polylines`, `fillPoly`, `putText`), annotated keyframes | `spatial/map2d.py`, `walkthrough.py` |

Learned models complement OpenCV where geometry alone is not enough: Depth Anything 3 (depth and an initial pose),
Grounding DINO (object names; "is this on a chair or the wall?"), MobileNetV2 through `cv2.dnn`. Every process caps
OpenCV at a fixed number of threads (`setNumThreads`) and runs at reduced priority on a workstation.

Without an NVIDIA GPU, an OpenCV-only path still runs (`pipeline.py`): LK + MAGSAC visual odometry with
`solvePnPRansac`, Depth Anything V2 Small through `cv2.dnn` on the CPU, and the same analysis.

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
- Models: Depth Anything 3 BASE (Apache-2.0), Grounding DINO tiny (Apache-2.0), OpenCV Zoo YuNet (MIT), PP-OCRv3,
  CRNN and MobileNetV2 (Apache-2.0), Depth Anything V2 Small (Apache-2.0); no non-commercial weights.
