"""The loop engine — one reusable closed-loop optimizer for every AI in Vigil.

optimize() runs: seed → explore (random) → refine (local, annealed perturbation of
the incumbent best) → keep best → log every trial. Deterministic given `seed` so a
run reproduces. eval_fn(params: dict) -> (score: float, metrics: dict); higher score
is better. Returns the best params + metrics + full history.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Param:
    """One searchable dimension: a bounded (optionally integer) range."""

    name: str
    lo: float
    hi: float
    integer: bool = False

    def _round(self, v: float):
        v = min(max(v, self.lo), self.hi)
        return int(round(v)) if self.integer else float(v)

    def mid(self):
        return self._round((self.lo + self.hi) / 2.0)

    def sample(self, rng) -> float:
        return self._round(rng.uniform(self.lo, self.hi))

    def perturb(self, v: float, frac: float, rng) -> float:
        span = (self.hi - self.lo) * max(frac, 0.02)
        return self._round(v + rng.normal(0.0, span))


Space = list[Param]
EvalFn = Callable[[dict], "tuple[float, dict]"]


def optimize(
    space: Space,
    eval_fn: EvalFn,
    *,
    trials: int = 200,
    explore_frac: float = 0.5,
    seed: int = 0,
    name: str = "ai",
    log_path: str | Path | None = None,
    seed_params: dict | None = None,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()

    best = dict(seed_params) if seed_params else {p.name: p.mid() for p in space}
    best_score, best_metrics = eval_fn(best)
    history: list[dict] = [
        {
            "trial": -1,
            "score": round(best_score, 4),
            "best": round(best_score, 4),
            "params": dict(best),
            "metrics": best_metrics,
        }
    ]
    n_explore = int(trials * explore_frac)

    for t in range(trials):
        if t < n_explore:  # global exploration
            cand = {p.name: p.sample(rng) for p in space}
        else:  # local refinement around the incumbent, radius annealing to 0
            anneal = 1.0 - (t - n_explore) / max(trials - n_explore, 1)
            frac = 0.06 + 0.22 * anneal
            cand = dict(best)
            k = max(1, len(space) // 3)
            for idx in rng.choice(len(space), size=k, replace=False):
                p = space[idx]
                cand[p.name] = p.perturb(best[p.name], frac, rng)

        score, metrics = eval_fn(cand)
        improved = score > best_score
        if improved:
            best, best_score, best_metrics = cand, score, metrics
        history.append(
            {
                "trial": t,
                "score": round(score, 4),
                "best": round(best_score, 4),
                "params": cand,
                "metrics": metrics,
            }
        )

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for r in history:
                f.write(json.dumps(r) + "\n")

    result = {
        "name": name,
        "best_params": best,
        "best_score": round(best_score, 4),
        "best_metrics": best_metrics,
        "trials": trials,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"[tune:{name}] best score {result['best_score']} after {trials} trials "
            f"({result['elapsed_s']}s)  metrics={best_metrics}"
        )
    return result
