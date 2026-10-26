"""Object names by multi-view voting of Grounding DINO boxes.

Grounding DINO (spatial_inspection.grounding, Apache-2.0, on the GPU) returns labelled boxes from a fixed furniture
vocabulary on frames spread over the video. For every object and every detection frame where the object is
visible (projected points inside the image, not occluded by the model depth), each box containing
most of the object's visible points votes for its label, weighted by how well the box fits the
object's projected extent. No SAM masks are produced, so objects are named but not re-split.
"""

import json
import os

import numpy as np

VOCAB = ["sofa", "armchair", "chair", "table", "coffee table", "dining table", "desk", "bed", "cabinet",
         "cupboard", "wardrobe", "shelf", "bookshelf", "drawers", "television", "fan", "lamp", "door",
         "curtain", "refrigerator", "washing machine", "sink", "toilet", "bathtub", "stool", "bench",
         "plant", "mirror", "picture frame", "clock", "air conditioner", "kitchen counter", "stove",
         "staircase", "bucket", "box", "bag"]
SYN = {"armchair": "armchair", "coffee table": "table", "desk": "table", "cupboard": "cabinet",
       "wardrobe": "wardrobe", "bookshelf": "shelf", "drawers": "cabinet", "kitchen counter": "counter"}
N_FRAMES = int(os.environ.get("PI_LABEL_FRAMES", "24"))


def log(msg):
    print(msg, flush=True)


def _canon(label, vocab):
    """Longest vocabulary match ('coffee table' before 'table')."""
    best = None
    for v in sorted(vocab, key=len, reverse=True):
        if v in label:
            best = v
            break
    if best is None:
        return None
    return SYN.get(best, best)


def _detect_frame(rgb):
    from spatial_inspection import grounding
    out = []
    for i in range(0, len(VOCAB), 19):          # Grounding DINO reads at most 256 text tokens per prompt
        dets = grounding.detect(rgb, VOCAB[i:i + 19], box_threshold=0.3, text_threshold=0.25) or []
        out += [d for d in dets if _canon(d["label"], VOCAB)]
    return out


def detect(cfg, st, work):
    out = f"{work}/detections.json"
    if os.path.exists(out):
        return json.load(open(out))
    from spatial_inspection import grounding
    if not grounding.available():
        log("  object names: Grounding DINO is not installed (torch + transformers) - objects stay numbered")
        return None
    # frames spread over the WHOLE video (not just its beginning)
    u = list(st["usable"])
    use = [u[i] for i in np.linspace(0, len(u) - 1, min(N_FRAMES, len(u))).round().astype(int)]
    log(f"  Grounding DINO ({grounding.MODEL_ID}) object detection on {len(use)} frames ...")
    res = [_detect_frame(st["frames"][k]) for k in use]
    grounding.release()
    dets = {str(k): r for k, r in zip(use, res)}
    n = sum(len(r) for r in res)
    if n == 0:
        log("  object names: Grounding DINO found no objects - objects stay numbered")
        return None
    data = {"vocab": VOCAB, "detections": dets, "source": grounding.MODEL_ID}
    with open(out, "w") as f:
        json.dump(data, f)
    log(f"  {n} labelled boxes")
    return data


def name_objects(cfg, st, objects, work, depth_model=None):
    if not objects:
        return
    res = detect(cfg, st, work)
    if res is None:
        return
    vocab, dets = res["vocab"], res["detections"]
    K, w2c = st["K_full"], st["w2c_model"]
    H, W = st["frames"][0].shape[:2]
    from .render import mesh_depths
    frames_used = sorted(int(k) for k in dets)
    if depth_model is None:
        depth_model = dict(zip(frames_used, mesh_depths(st["model"], K, w2c, W, H, frames=frames_used)))
    counts = {}
    for o in objects:
        P = np.asarray(o["mesh"].vertices)
        if len(P) > 1500:
            P = P[np.random.default_rng(0).choice(len(P), 1500, replace=False)]
        votes, seen = {}, 0
        for k in frames_used:
            Xc = P @ w2c[k][:3, :3].T + w2c[k][:3, 3]
            z = Xc[:, 2]
            ok = z > 0.2
            u = K[k][0, 0] * Xc[:, 0] / np.maximum(z, 1e-6) + K[k][0, 2]
            v = K[k][1, 1] * Xc[:, 1] / np.maximum(z, 1e-6) + K[k][1, 2]
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if ok.mean() < 0.5:
                continue
            idx = np.flatnonzero(ok)
            dm = depth_model[k][v[idx].astype(int), u[idx].astype(int)]
            vis = idx[(dm == 0) | (z[idx] < dm + 0.06)]
            if len(vis) < 0.4 * len(P):
                continue
            seen += 1
            uu, vv = u[vis], v[vis]
            ob = np.array([uu.min(), vv.min(), uu.max(), vv.max()])
            frame_best = {}
            for d in dets[str(k)] if str(k) in dets else dets.get(k, []):
                lab = _canon(d["label"], vocab)
                if lab is None:
                    continue
                x0, y0, x1, y1 = d["box"]
                inside = np.mean((uu >= x0) & (uu <= x1) & (vv >= y0) & (vv <= y1))
                if inside < 0.6:
                    continue
                ix = max(0, min(x1, ob[2]) - max(x0, ob[0])) * max(0, min(y1, ob[3]) - max(y0, ob[1]))
                ua = (x1 - x0) * (y1 - y0) + (ob[2] - ob[0]) * (ob[3] - ob[1]) - ix
                iou = ix / max(ua, 1e-6)
                if iou < 0.25:
                    continue
                s = d["score"] * iou
                frame_best[lab] = max(frame_best.get(lab, 0), s)
            for lab, s in frame_best.items():
                votes[lab] = votes.get(lab, 0) + s
        o["label_votes"] = {k_: round(v_ / max(seen, 1), 3) for k_, v_ in sorted(votes.items(), key=lambda x: -x[1])}
        o["label_frames"] = seen
        if votes:
            lab, v = max(votes.items(), key=lambda x: x[1])
            if seen >= 2 and v / seen >= 0.08:
                o["label"] = lab
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    log("  object names: " + ", ".join(f"{o['id']}={o['label']}" for o in objects))
