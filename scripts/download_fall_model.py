"""Fetch the dedicated fall-detection model into models/fall_yolov11.pt.

Model: melihuzunoglu/human-fall-detection on the Hugging Face Hub — a YOLOv11
object detector fine-tuned for falls (classes: fallen / sitting / standing).
~5.5 MB, AGPL-3.0. Vigil runs it alongside the pose FSM for redundant fall/faint
coverage; if the file is absent, the pose FSM still covers falls.

    .venv/bin/python scripts/download_fall_model.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = "melihuzunoglu/human-fall-detection"
DEST = Path(__file__).resolve().parent.parent / "models" / "fall_yolov11.pt"


def main() -> None:
    from huggingface_hub import hf_hub_download

    DEST.parent.mkdir(exist_ok=True)
    print(f"[fall-model] downloading {REPO}/best.pt from Hugging Face …")
    src = hf_hub_download(REPO, "best.pt")
    shutil.copyfile(src, DEST)
    print(f"[fall-model] saved {DEST} ({DEST.stat().st_size} bytes)")

    from ultralytics import YOLO

    names = YOLO(str(DEST)).names
    print(f"[fall-model] OK — classes: {names}")


if __name__ == "__main__":
    main()
