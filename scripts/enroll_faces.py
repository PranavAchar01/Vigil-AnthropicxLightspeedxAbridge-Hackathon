"""Enroll faces so the camera can bind a person to their patient chart.

Drop a photo of each demo participant into data/faces/ and name the file after a
keyword in the patient they should map to (e.g. `covid.jpg`, `sepsis.jpg`,
`hypertension.jpg`, `healthy.jpg`) — the stem is matched against each cohort
patient's name + visit title. Optionally provide data/faces/map.json =
{"alice.jpg": "covid", ...} for explicit mapping. Only 512-d embeddings are saved
(data/face_gallery.json); the images are never stored or uploaded.

Run:  uv run python scripts/enroll_faces.py          (needs `.[faces]` deps)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vigil.chart import load_cohort  # noqa: E402
from vigil.config import settings  # noqa: E402
from vigil.perception.faces import FaceGallery, FaceRecognizer  # noqa: E402

FACES_DIR = settings.cohort_path.parent / "faces"


def resolve_patient(keyword: str, charts) -> tuple[str, str] | None:
    kw = keyword.lower()
    for c in charts:
        hay = f"{c.name} {c.visit_title}".lower()
        if kw in hay or all(w in hay for w in kw.split()):
            return c.patient_id, c.name
    return None


def main() -> None:
    charts = load_cohort(settings.cohort_path)
    if not FACES_DIR.exists():
        sys.exit(
            f"Create {FACES_DIR}/ and add one photo per participant "
            f"(name the file after a patient keyword, e.g. covid.jpg)."
        )
    mapping = {}
    map_file = FACES_DIR / "map.json"
    if map_file.exists():
        mapping = json.loads(map_file.read_text())

    rec = FaceRecognizer()
    gallery = FaceGallery(settings.face_match_threshold)
    images = [
        p
        for p in FACES_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name != "map.json"
    ]
    for img in images:
        keyword = mapping.get(img.name, img.stem)
        found = resolve_patient(keyword, charts)
        if not found:
            print(f"  ! {img.name}: no cohort patient matches '{keyword}' — skipping")
            continue
        emb = rec.embed_image(img)
        if emb is None:
            print(f"  ! {img.name}: no face detected — skipping")
            continue
        pid, name = found
        gallery.add(pid, name, emb)
        print(f"  + {img.name:22s} → {name}")

    if len(gallery) == 0:
        sys.exit("No faces enrolled.")
    settings.face_gallery_path.write_text(gallery.to_json())
    print(f"\nEnrolled {len(gallery)} → {settings.face_gallery_path}")


if __name__ == "__main__":
    main()
