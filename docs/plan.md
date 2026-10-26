# Build plan

The input is a phone walkthrough video of a property. The output is a 3D model with every finding placed in it and
a structured report. Each part below says what "done" means.

## 1. Damage detection

- Keyframes: pick sharp, well-spread frames with OpenCV (variance of the Laplacian for blur, optical-flow distance for
  spacing).
- Grounding DINO proposes boxes for open-vocabulary prompts per room type ("crack in wall", "water stain on ceiling",
  "mould", "peeling paint", "broken tile").
- Qwen3-VL-2B checks each box on a full-resolution crop, names the defect, rates severity and writes one sentence of
  evidence.
- Done when: precision and recall are measured on a labelled set of keyframes, with the threshold chosen from it.

## 2. Camera tracking and spatial understanding

- Done today: `tracking.py` tracks corners with pyramidal Lucas-Kanade optical flow (forward-backward checked) and
  estimates the relative pose with a MAGSAC essential matrix and `recoverPose`.
- Next: chain relative poses into a trajectory; geometric pose refinement with `solvePnP` against triangulated
  points and a small bundle adjustment over a sliding window; loop checks with ORB matches when a room is revisited.
- Done when: the trajectory of a sample walkthrough is drift-checked against a closed loop (start and end at the
  same doorway).

## 3. Photorealistic 3D reconstruction

- Camera poses from part 2 seed the reconstruction; multi-view reconstruction densifies the points.
- gsplat trains a Gaussian Splatting model of each walkthrough on a GPU.
- A plain mesh is kept alongside for measuring and for placing findings, because splats look right but are not
  reliable for measurement.
- Done when: a sample walkthrough renders as a navigable splat in the browser, and the mesh and splat line up.

## 4. Spatially aware investigation

- Each verified finding is projected from its keyframe through the camera pose onto the mesh, giving a position,
  a surface (wall, floor, ceiling) and a size in metres.
- A viewer shows the splat with the findings as markers; clicking one shows the crop and the evidence.
- The report lists findings by room with severity, position, photos and what could not be inspected.
- Done when: a sample walkthrough produces the report and the viewer end to end.

## AWS architecture

| Need | AWS service |
| --- | --- |
| Video uploads and results | Amazon S3 |
| GPU inference (Grounding DINO, Qwen3-VL-2B) and splat training | Amazon EC2 G5 / G6 instances, or SageMaker endpoints and jobs |
| Pipeline: keyframes → detection → tracking → reconstruction → report | AWS Step Functions with Lambda for the light steps |
| Results index | Amazon DynamoDB |
| Viewer and report | Amazon CloudFront in front of S3 |

## Submission checklist

Check every item against the official competition rules before submitting; secondary listings disagree on the dates.

- [ ] OpenCV 5 doing meaningful image or video analysis, visible in the code and the write-up
- [ ] A significant AWS component, described with an architecture diagram
- [ ] Working code in this repository (shared with the judges if the rules require it)
- [ ] Demo video
- [ ] Technical write-up
- [ ] Licences listed for every model and library
