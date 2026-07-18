"""Closed-loop tuner for every AI in Vigil.

Runs the eval→search→promote loop for vision, fusion, face, audio (offline,
deterministic) and — with --reasoning — the Claude re-triage agent loop (bounded
live API calls). Each ROUND warm-starts from the previous best and re-draws the
synthetic benchmark with a fresh seed, so quality compounds across rounds; the
final params are validated on a held-out seed. Winners are written to
config/tuned.env, which vigil.config loads at startup.

    .venv/bin/python scripts/tune.py                     # one strong pass, offline AIs
    .venv/bin/python scripts/tune.py --reasoning         # + live Claude agent eval
    .venv/bin/python scripts/tune.py --minutes 300       # keep iterating for ~5 hours
    .venv/bin/python scripts/tune.py --vision-trials 600 # deeper search
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vigil.perception.vision import Params as VParams
from vigil.tuning import audio_tune, face_tune, fusion_tune, reasoning_tune, vision_tune
from vigil.tuning.optimizer import optimize

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
VAL_SEED = 9_999

ENV_MAP = {
    "vision": {
        "drop_frac": "VIGIL_DROP_FRAC",
        "fall_confirm_s": "VIGIL_FALL_CONFIRM_S",
        "low_cy": "VIGIL_LOW_CY",
        "torso_horiz_deg": "VIGIL_TORSO_HORIZ_DEG",
        "ar_wide": "VIGIL_AR_WIDE",
        "still_motion": "VIGIL_STILL_MOTION",
        "faint_s": "VIGIL_FAINT_S",
        "motionless_s": "VIGIL_MOTIONLESS_S",
        "unresponsive_s": "VIGIL_UNRESPONSIVE_S",
        "seizure_motion": "VIGIL_SEIZURE_MOTION",
        "seizure_osc": "VIGIL_SEIZURE_OSC",
        "seizure_s": "VIGIL_SEIZURE_S",
        "gesture_s": "VIGIL_GESTURE_S",
        "slump_min_deg": "VIGIL_SLUMP_MIN_DEG",
        "slump_s": "VIGIL_SLUMP_S",
    },
    "audio": {"scream_threshold": "VIGIL_SCREAM_THRESHOLD"},
    "face": {"threshold": "VIGIL_FACE_THRESHOLD"},
    "fusion": {
        "window_s": "VIGIL_FUSION_WINDOW_SECONDS",
        "cooldown_s": "VIGIL_FUSION_COOLDOWN_SECONDS",
    },
}


def _loop(space, eval_builder, dataset_builder, rounds, trials, name):
    """Iterated search: each round re-draws the benchmark and warm-starts from best."""
    best_params = None
    for r in range(rounds):
        ds = dataset_builder(seed=r)
        res = optimize(
            space(),
            eval_builder(ds),
            trials=trials,
            seed=1000 + r,
            name=f"{name}-r{r}",
            seed_params=best_params,
            verbose=False,
        )
        best_params = res["best_params"]
    val = eval_builder(dataset_builder(seed=VAL_SEED))
    val_score, val_metrics = val(best_params)
    base = {p.name: getattr(VParams(), p.name) for p in space()} if name == "vision" else None
    baseline = None
    if base:
        _, baseline = val(base)
    return {
        "best_params": best_params,
        "val_metrics": val_metrics,
        "val_score": round(val_score, 4),
        "baseline_val_metrics": baseline,
    }


def tune_vision(rounds, trials):
    print(f"[tune] vision — {rounds} round(s) x {trials} trials …")
    return _loop(
        vision_tune.space,
        vision_tune.make_eval,
        vision_tune.build_dataset,
        rounds,
        trials,
        "vision",
    )


def tune_fusion(rounds, trials):
    print(f"[tune] fusion — {rounds} round(s) x {trials} trials …")
    return _loop(
        fusion_tune.space,
        fusion_tune.make_eval,
        lambda seed: fusion_tune.streams(),
        rounds,
        trials,
        "fusion",
    )


def tune_face(rounds, trials):
    print(f"[tune] face — {rounds} round(s) x {trials} trials …")
    res = _loop(
        face_tune.space, face_tune.make_eval, face_tune.build_scores, rounds, trials, "face"
    )
    res["eer_threshold"] = face_tune.equal_error_threshold(face_tune.build_scores(seed=VAL_SEED))
    return res


def tune_audio(rounds, trials):
    print("[tune] audio — loading AST model + scoring non-scream battery …")
    from vigil.perception.audio import ASTBackend

    backend = ASTBackend()
    neg, names = audio_tune.build_negative_scores(backend)
    res = _loop(audio_tune.space, audio_tune.make_eval, lambda seed: neg, rounds, trials, "audio")
    # promote with a safety margin above the loudest non-scream we saw
    thr = max(res["best_params"]["scream_threshold"], float(neg.max()) + 0.15)
    res["best_params"]["scream_threshold"] = round(min(thr, 0.6), 3)
    res["neg_scores"] = {n: round(float(s), 4) for n, s in zip(names, neg)}
    return res


def _fmt(v):
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
        return str(int(v))
    return str(round(float(v), 4))


def write_tuned_env(promoted: dict) -> Path:
    """Merge promoted params into config/tuned.env (never clobber other AIs' values)."""
    CONFIG.mkdir(exist_ok=True)
    path = CONFIG / "tuned.env"
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                existing[k.strip()] = v.strip()
    for ai, params in promoted.items():
        for k, v in params.items():
            env = ENV_MAP.get(ai, {}).get(k)
            if env:
                existing[env] = _fmt(v)
    lines = [
        "# Auto-generated by scripts/tune.py — promoted best params. Loaded by vigil.config.\n"
    ]
    for ai in ("vision", "fusion", "face", "audio"):
        present = [(e, existing[e]) for e in ENV_MAP.get(ai, {}).values() if e in existing]
        if present:
            lines.append(f"# {ai}")
            lines += [f"{e}={v}" for e, v in present]
            lines.append("")
    path.write_text("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--vision-trials", type=int, default=220)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--minutes", type=float, default=0.0, help="keep iterating for N minutes")
    ap.add_argument("--reasoning", action="store_true", help="also run the live Claude agent eval")
    ap.add_argument("--only", default="", help="comma list: vision,fusion,face,audio")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else {"vision", "fusion", "face", "audio"}
    t0 = time.time()
    rounds = args.rounds
    report: dict = {"rounds": rounds, "results": {}}

    while True:
        if "vision" in only:
            report["results"]["vision"] = tune_vision(rounds, args.vision_trials)
        if "fusion" in only:
            report["results"]["fusion"] = tune_fusion(rounds, args.trials)
        if "face" in only:
            report["results"]["face"] = tune_face(rounds, args.trials)
        if "audio" in only:
            report["results"]["audio"] = tune_audio(rounds, args.trials)

        promoted = {ai: report["results"][ai]["best_params"] for ai in report["results"]}
        env_path = write_tuned_env(promoted)

        if args.reasoning:
            print("[tune] reasoning — live Claude re-triage eval (bounded) …")
            report["results"]["reasoning"] = reasoning_tune.run_eval()

        (CONFIG / "tuning_report.json").write_text(json.dumps(report, indent=2, default=str))
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[tune] wrote {env_path} and config/tuning_report.json ({elapsed_min:.1f} min)")

        if args.minutes and elapsed_min < args.minutes:
            rounds += 1
            print(
                f"[tune] continuing loop — round budget now {rounds} ({elapsed_min:.1f}/{args.minutes} min)"
            )
            continue
        break

    print("\n================= TUNING SUMMARY =================")
    for ai, res in report["results"].items():
        if ai == "reasoning":
            print(
                f"  reasoning : pass_rate {res.get('pass_rate')}  latency {res.get('mean_latency_s')}s "
                f"({res.get('cases')} cases)"
            )
        else:
            vm = res.get("val_metrics", {})
            print(f"  {ai:9}: val {vm}")
            if res.get("baseline_val_metrics"):
                print(f"             baseline {res['baseline_val_metrics']}")


if __name__ == "__main__":
    main()
