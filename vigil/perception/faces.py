"""On-device face recognition to bind a detected person to their patient chart.

Open source, local only: InsightFace (ArcFace embeddings) runs via onnxruntime on
CPU/Apple-Silicon. We compare a live face's embedding against an enrolled gallery
(cosine similarity) and, above a threshold, resolve which patient is on camera so
the chart pulls up automatically. Only 512-d embeddings are stored — never images,
never uploaded. The cohort is synthetic; enroll consenting demo participants.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("vigil.faces")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


class FaceGallery:
    """Enrolled patient face embeddings + nearest-match identification."""

    def __init__(self, threshold: float = 0.45) -> None:
        self.threshold = threshold
        self.entries: list[tuple[str, str, np.ndarray]] = []  # (patient_id, name, embedding)

    def add(self, patient_id: str, name: str, embedding) -> None:
        self.entries.append((patient_id, name, np.asarray(embedding, dtype=float)))

    def identify(self, embedding) -> tuple[str, str, float] | None:
        """Return (patient_id, name, score) for the closest enrolled face, or None."""
        if not self.entries:
            return None
        emb = np.asarray(embedding, dtype=float)
        pid, name, best = max(self.entries, key=lambda e: cosine(emb, e[2]))
        score = cosine(emb, best)
        return (pid, name, score) if score >= self.threshold else None

    def __len__(self) -> int:
        return len(self.entries)

    def to_json(self) -> str:
        return json.dumps(
            [{"patient_id": p, "name": n, "embedding": e.tolist()} for p, n, e in self.entries]
        )

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.45) -> "FaceGallery":
        g = cls(threshold)
        for row in json.loads(Path(path).read_text()):
            g.add(row["patient_id"], row.get("name", ""), row["embedding"])
        return g


class FaceRecognizer:
    """Wraps InsightFace. Lazy heavy import so the app boots without the models."""

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def embed_largest(self, bgr) -> np.ndarray | None:
        """Embedding of the largest (nearest) face in the frame, or None."""
        faces = self.app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return f.normed_embedding

    def embed_image(self, path: str | Path) -> np.ndarray | None:
        import cv2

        img = cv2.imread(str(path))
        return None if img is None else self.embed_largest(img)
