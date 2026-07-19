"""Seizure demo on localhost — run the real vision detector over a REAL seizure clip
and show it live in the dashboard, proving Vigil detects a convulsive seizure.

    uv run python scripts/demo_seizure.py           # -> http://localhost:8000

It points the vision source at data/demo/seizure_real.mov (a real seizure clip, kept
local and gitignored for privacy), turns the enrolled-face gate OFF (the person in the
clip isn't an enrolled patient), and disables mic detection. The dashboard streams the
annotated skeleton with the live seizure metrics and flashes SEIZURE DETECTED, then the
re-triage + escalation pipeline runs exactly as it would on the live camera.

Set VIGIL_SEIZURE_CLIP to use a different clip. Env is set BEFORE importing vigil so
config picks it up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIP = Path(os.environ.get("VIGIL_SEIZURE_CLIP", REPO / "data" / "demo" / "seizure_real.mov"))
PORT = int(os.environ.get("VIGIL_PORT", "8000"))


def main() -> None:
    if not CLIP.exists():
        sys.exit(
            f"[demo] seizure clip not found at {CLIP}\n"
            f"       Put a real seizure video there (e.g. cp your_clip.mov "
            f"{REPO / 'data' / 'demo' / 'seizure_real.mov'}), or set VIGIL_SEIZURE_CLIP."
        )
    # These must be set BEFORE vigil.config / the vision thread read them.
    os.environ["VIGIL_VIDEO_SOURCE"] = str(CLIP)
    os.environ["VIGIL_REQUIRE_ENROLLED_FACE"] = "0"  # the clip's subject isn't enrolled
    os.environ.setdefault("VIGIL_DISABLE_AUDIO", "1")  # no mic during the clip demo

    print("\n" + "=" * 64)
    print("  VIGIL — SEIZURE DEMO")
    print(f"  clip : {CLIP.name}")
    print(f"  open : http://localhost:{PORT}   (skeleton + live seizure metrics)")
    print("  The detector runs on the real clip and flashes SEIZURE DETECTED,")
    print("  then re-triage + escalation run as on the live camera.")
    print("=" * 64 + "\n")

    import uvicorn

    uvicorn.run("vigil.server.app:app", host="127.0.0.1", port=PORT, workers=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
