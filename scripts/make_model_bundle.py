"""Pack the model files (and sample videos) another machine needs, for scripts/setup_windows.ps1 -Bundle.

    python scripts/make_model_bundle.py --out ../share [--results]

spatial-inspection-models.zip
  models/opencv/*.onnx                  OpenCV Zoo models (YuNet, PP-OCRv3, CRNN, MobileNetV2, Depth Anything V2 Small)
  hf/hub/models--depth-anything--DA3-BASE        Hugging Face cache layout: refs + the resolved snapshot files
  hf/hub/models--IDEA-Research--grounding-dino-tiny
  data/samples/*.mp4                    the sample walkthroughs
spatial-inspection-demo-results.zip     (--results) finished walkthroughs from results/, to open without a GPU run

Model weights are stored, not compressed (they do not compress); links in the local Hugging Face cache are
resolved, so the bundle holds plain files that work on any machine. Python environments are not packed: they are
tied to the local Python install; setup_windows.ps1 rebuilds them from the pinned requirements.
"""

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_MODELS = ["depth-anything/DA3-BASE", "IDEA-Research/grounding-dino-tiny"]


def hf_hub() -> Path:
    home = os.environ.get("HF_HOME")
    return Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"


def add(z: zipfile.ZipFile, src: Path, arc: str) -> int:
    big = src.stat().st_size > 1 << 20
    z.write(src, arc, compress_type=zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED)
    return src.stat().st_size


def models_zip(out: Path) -> Path:
    path = out / "spatial-inspection-models.zip"
    total = 0
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        for f in sorted((ROOT / "models" / "opencv").glob("*.onnx")):
            total += add(z, f, f"models/opencv/{f.name}")
        for repo in HF_MODELS:
            d = hf_hub() / ("models--" + repo.replace("/", "--"))
            ref = (d / "refs" / "main").read_text().strip()
            z.writestr(f"hf/hub/{d.name}/refs/main", ref)
            snap = d / "snapshots" / ref
            for f in sorted(p for p in snap.rglob("*") if p.is_file()):
                total += add(z, f.resolve(), f"hf/hub/{d.name}/snapshots/{ref}/{f.relative_to(snap).as_posix()}")
        for f in sorted((ROOT / "data" / "samples").glob("*.mp4")):
            total += add(z, f, f"data/samples/{f.name}")
        z.writestr("README.txt", __doc__ + "\nInstall: powershell -ExecutionPolicy Bypass -File scripts\\setup_windows.ps1 "
                   "-Bundle <this zip or its extracted folder>\n")
    print(f"{path}  ({total / 2**30:.2f} GB of files)")
    return path


def results_zip(out: Path) -> Path:
    path = out / "spatial-inspection-demo-results.zip"
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        for d in sorted(p for p in (ROOT / "results").iterdir() if (p / "walkthrough.json").exists()):
            for f in sorted(p for p in d.rglob("*") if p.is_file() and p.name != "status.json"):
                add(z, f, f"{d.name}/{f.relative_to(d).as_posix()}")
    print(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT), help="the checkout whose models/, data/ and results/ are packed")
    ap.add_argument("--out", default=str(ROOT.parent / "spatial-inspection-share"))
    ap.add_argument("--results", action="store_true", help="also pack the finished walkthroughs in results/")
    a = ap.parse_args()
    globals()["ROOT"] = Path(a.repo).resolve()       # the functions above read the packed checkout from ROOT
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models_zip(out)
    if a.results:
        results_zip(out)


if __name__ == "__main__":
    main()
