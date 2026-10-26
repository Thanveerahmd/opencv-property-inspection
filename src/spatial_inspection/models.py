"""The model files the project uses: where they live, where they come from, their checksums and licences.

Files live under the repo's models/ folder (gitignored), or under the folder named by the
SPATIAL_INSPECTION_MODELS_DIR environment variable (not the generic MODELS_DIR, which other projects set too).
Fetch them all with:

    python -m spatial_inspection.models download
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
ENV_VAR = "SPATIAL_INSPECTION_MODELS_DIR"


@dataclass(frozen=True)
class ModelFile:
    relpath: str   # under the models folder
    url: str
    sha256: str
    licence: str


REGISTRY: dict[str, ModelFile] = {
    # OpenCV Zoo, YuNet face detector (cv2.FaceDetectorYN)
    "face_detection_yunet": ModelFile(
        "opencv/face_detection_yunet_2023mar.onnx",
        f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "MIT",
    ),
    # OpenCV Zoo, PP-OCRv3 DB text detector, English (cv2.dnn.TextDetectionModel_DB)
    "text_detection_ppocr": ModelFile(
        "opencv/text_detection_en_ppocrv3_2023may.onnx",
        f"{ZOO}/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
        "03f550c6b406fda8bf54bd8327815f6c7e2edd98cea02348c93d879254366587",
        "Apache-2.0",
    ),
    # OpenCV Zoo, CRNN text recogniser, English digits and lowercase letters (cv2.dnn.TextRecognitionModel)
    "text_recognition_crnn": ModelFile(
        "opencv/text_recognition_CRNN_EN_2021sep.onnx",
        f"{ZOO}/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx",
        "a84b1f6e11a65c2d733cb0cc1f014aae3f99051e3f11447dc282faa678eee544",
        "Apache-2.0",
    ),
    # Depth Anything V2 Small, ONNX export from onnx-community (monocular relative depth)
    "depth_anything_v2_small": ModelFile(
        "opencv/depth_anything_v2_small.onnx",
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
        "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
        "Apache-2.0",
    ),
}

_verified: set[Path] = set()


def models_dir() -> Path:
    """The folder named by SPATIAL_INSPECTION_MODELS_DIR if set, else the repo's models/ folder."""
    env = os.environ.get(ENV_VAR)
    return Path(env) if env else Path(__file__).resolve().parents[2] / "models"


def path(name: str) -> Path:
    """Where a registered model file lives (it may not exist yet)."""
    return models_dir() / REGISTRY[name].relpath


def available(name: str) -> bool:
    """True when the model file is on disk (not checked against its sha256)."""
    return path(name).is_file()


def sha256(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure(name: str, quiet: bool = True) -> Path:
    """Path to a model file, downloading it first if missing. Raises ValueError when the checksum does not match."""
    entry = REGISTRY[name]
    dest = path(name)
    if dest in _verified:
        return dest
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        if not quiet:
            print(f"downloading {name} from {entry.url}", file=sys.stderr)
        with urllib.request.urlopen(entry.url, timeout=60) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        got = sha256(part)
        if got != entry.sha256:
            part.unlink()
            raise ValueError(f"{name}: downloaded file has sha256 {got}, expected {entry.sha256}")
        part.replace(dest)
    else:
        got = sha256(dest)
        if got != entry.sha256:
            raise ValueError(f"{name}: {dest} has sha256 {got}, expected {entry.sha256}; delete it to re-download")
    _verified.add(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m spatial_inspection.models", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="download missing model files and verify every checksum")
    dl.add_argument("names", nargs="*", help=f"models to fetch (default: all of {', '.join(REGISTRY)})")
    sub.add_parser("list", help="show each model, its licence and whether it is on disk")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for name, entry in REGISTRY.items():
            state = "present" if available(name) else "missing"
            print(f"{name:26s} {entry.licence:11s} {state:8s} {path(name)}")
        return 0
    failed = 0
    for name in args.names or list(REGISTRY):
        try:
            print(f"{name}: ok {ensure(name, quiet=False)}")
        except (OSError, ValueError, KeyError) as e:
            print(f"{name}: FAILED {e}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
