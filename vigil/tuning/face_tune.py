"""Face-recognition threshold tuning: genuine-accept vs impostor-reject.

Uses the enrolled gallery's real ArcFace embeddings. Genuine samples = each
embedding plus intra-class noise (simulated re-capture variation across a spread of
severities); impostors = cross-identity real pairs; unknowns = random unit vectors.
Tunes the cosine match threshold to maximize (TPR − 2·FPR) and reports the EER.
"""

from __future__ import annotations

import numpy as np

from vigil.config import settings
from vigil.perception.faces import cosine
from vigil.tuning.optimizer import Param


def _load_embeddings() -> list[np.ndarray]:
    path = settings.face_gallery_path
    if path.exists():
        import json

        rows = json.loads(path.read_text())
        embs = [np.asarray(r["embedding"], float) for r in rows if r.get("embedding")]
        if len(embs) >= 2:
            return [e / (np.linalg.norm(e) + 1e-9) for e in embs]
    # Fallback: synthesize distinct identity prototypes so the tuner still runs.
    rng = np.random.default_rng(0)
    return [(v := rng.normal(0, 1, 512)) / np.linalg.norm(v) for _ in range(4)]


def build_scores(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (genuine_scores, impostor_scores) cosine arrays."""
    rng = np.random.default_rng(seed)
    embs = _load_embeddings()
    genuine, impostor = [], []
    for i, e in enumerate(embs):
        for _ in range(60):  # intra-class variation across a range of noise
            sigma = rng.uniform(0.01, 0.06)
            g = e + rng.normal(0, sigma, e.shape)
            genuine.append(cosine(e, g))
        for j, e2 in enumerate(embs):  # cross-identity impostors
            if j == i:
                continue
            impostor.append(cosine(e, e2))
            for _ in range(15):
                sigma = rng.uniform(0.01, 0.05)
                impostor.append(cosine(e, e2 + rng.normal(0, sigma, e2.shape)))
    for _ in range(200):  # unknown faces = random unit vectors
        v = rng.normal(0, 1, embs[0].shape)
        impostor.append(cosine(embs[rng.integers(len(embs))], v / np.linalg.norm(v)))
    return np.array(genuine), np.array(impostor)


def space():
    return [Param("threshold", 0.20, 0.80)]


def make_eval(scores):
    genuine, impostor = scores

    def eval_fn(cand: dict):
        thr = cand["threshold"]
        tpr = float((genuine >= thr).mean())
        fpr = float((impostor >= thr).mean())
        score = tpr - 2.0 * fpr
        return score, {
            "threshold": round(thr, 3),
            "genuine_accept": round(tpr, 3),
            "impostor_accept": round(fpr, 3),
            "genuine_min": round(float(genuine.min()), 3),
            "impostor_max": round(float(impostor.max()), 3),
        }

    return eval_fn


def equal_error_threshold(scores) -> float:
    """The classic EER operating point (where FAR ≈ FRR) — reported alongside."""
    genuine, impostor = scores
    best_thr, best_gap = 0.5, 1e9
    for thr in np.linspace(0.15, 0.85, 141):
        far = float((impostor >= thr).mean())
        frr = float((genuine < thr).mean())
        if abs(far - frr) < best_gap:
            best_gap, best_thr = abs(far - frr), float(thr)
    return round(best_thr, 3)
