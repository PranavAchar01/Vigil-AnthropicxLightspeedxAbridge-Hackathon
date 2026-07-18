"""Audio (AST scream) threshold calibration.

The AST model is fixed; what we tune is the decision threshold. Since realistic
screams can't be synthesized convincingly, we calibrate against a battery of
non-scream signals (silence, loud white/pink noise, speech-like harmonics, siren
tones, claps — exactly the sounds a naive loudness detector false-fires on) and
choose the LOWEST threshold that rejects all of them with margin (max sensitivity
at zero false positives). Recall is validated live with scripts/test_scream.py.
"""

from __future__ import annotations

import numpy as np

from vigil.tuning.optimizer import Param

SR = 16_000
N = 32_000  # 2.0 s window, matches the detector


def _signals(seed: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, N / SR, N, endpoint=False)
    out: list[tuple[str, np.ndarray]] = [("silence", np.zeros(N, np.float32))]
    for amp in (0.05, 0.2, 0.5):
        out.append((f"white_noise_{amp}", (amp * rng.standard_normal(N)).astype(np.float32)))
    # pink-ish noise
    w = rng.standard_normal(N)
    pink = np.cumsum(w)
    pink = 0.4 * pink / (np.abs(pink).max() + 1e-9)
    out.append(("pink_noise", pink.astype(np.float32)))
    # speech-like harmonic sweeps
    for f0 in (120, 180, 240):
        ph = 2 * np.pi * np.cumsum(f0 + 15 * np.sin(2 * np.pi * 3 * t)) / SR
        s = sum((1 / k) * np.sin(k * ph) for k in range(1, 12))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        out.append((f"speech_{f0}", (0.2 * s * env).astype(np.float32)))
    # siren / pure-tone sweeps (the loud+bright case)
    for base in (700, 1100, 1600):
        ph = 2 * np.pi * np.cumsum(base + 200 * np.sin(2 * np.pi * 0.7 * t)) / SR
        out.append((f"siren_{base}", (0.7 * np.sin(ph)).astype(np.float32)))
    # transients / claps
    clap = np.zeros(N, np.float32)
    for c in (4000, 12000, 20000, 28000):
        clap[c : c + 400] = rng.standard_normal(400) * 0.9
    out.append(("claps", clap))
    return out


def build_negative_scores(backend, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    sigs = _signals(seed)
    scores = np.array([backend.score(w) for _, w in sigs])
    return scores, [name for name, _ in sigs]


def space():
    return [Param("scream_threshold", 0.10, 0.60)]


def make_eval(neg_scores: np.ndarray):
    def eval_fn(cand: dict):
        thr = cand["scream_threshold"]
        fp = float((neg_scores >= thr).mean())
        # zero false positives dominates; among those, prefer the lowest threshold.
        score = -50.0 * fp - thr
        return score, {
            "threshold": round(thr, 3),
            "neg_false_positive_rate": round(fp, 3),
            "neg_score_max": round(float(neg_scores.max()), 4),
            "neg_score_mean": round(float(neg_scores.mean()), 4),
        }

    return eval_fn
