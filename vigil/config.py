"""Environment-driven configuration for Vigil.

Secrets come from the environment only (never committed). See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
    # Promoted params from the closed-loop tuner (scripts/tune.py) override defaults
    # for every AI. Loaded into os.environ BEFORE settings + the vision Params read it.
    _tuned = Path(__file__).resolve().parent.parent / "config" / "tuned.env"
    if _tuned.exists():
        load_dotenv(_tuned, override=True)
except ModuleNotFoundError:  # dotenv is a declared dep; tolerate running pre-`uv sync`
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Claude ---
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    model: str = field(default_factory=lambda: _env("VIGIL_MODEL", "claude-opus-4-8"))
    fast_model: str = field(
        default_factory=lambda: _env("VIGIL_FAST_MODEL", "claude-haiku-4-5-20251001")
    )

    # --- ElevenLabs ---
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))
    elevenlabs_agent_id: str = field(default_factory=lambda: _env("ELEVENLABS_AGENT_ID"))
    elevenlabs_phone_number_id: str = field(
        default_factory=lambda: _env("ELEVENLABS_PHONE_NUMBER_ID")
    )
    nurse_phone_number: str = field(default_factory=lambda: _env("NURSE_PHONE_NUMBER"))
    # Anti-spam: cap outbound nurse calls so a Twilio TRIAL is never spam-dialed.
    # Default = ONE call per server run. -1 = unlimited; a cooldown (seconds) can
    # instead space repeat calls when the cap is raised.
    max_nurse_calls: int = field(default_factory=lambda: int(_envf("VIGIL_MAX_NURSE_CALLS", 1)))
    nurse_call_cooldown_s: float = field(
        default_factory=lambda: _envf("VIGIL_NURSE_CALL_COOLDOWN_S", 0.0)
    )
    # optional: real voice check-in with the patient before paging (soft signals)
    patient_kiosk_number: str = field(default_factory=lambda: _env("PATIENT_KIOSK_NUMBER"))
    elevenlabs_checkin_agent_id: str = field(
        default_factory=lambda: _env("ELEVENLABS_CHECKIN_AGENT_ID")
    )

    # --- Twilio (direct-call transport; works on trial accounts where the
    #     ElevenLabs<->Twilio integration is blocked). Used as a fallback when
    #     ELEVENLABS_PHONE_NUMBER_ID isn't set. ---
    twilio_account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    twilio_from_number: str = field(default_factory=lambda: _env("TWILIO_FROM_NUMBER"))

    # --- Conversational agent live-status endpoint ---
    # Shared secret the agent's webhook tool sends in X-Vigil-Token; public base URL
    # (the tunnel, e.g. https://vigil.example.com) that ElevenLabs reaches us at.
    agent_token: str = field(default_factory=lambda: _env("VIGIL_AGENT_TOKEN"))
    public_url: str = field(default_factory=lambda: _env("VIGIL_PUBLIC_URL"))
    agent_voice_id: str = field(
        default_factory=lambda: _env("VIGIL_AGENT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    )

    # --- Face recognition (bind a detected face -> patient chart; on-device) ---
    face_gallery_path: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parent.parent / "data" / "face_gallery.json"
        )
    )
    face_match_threshold: float = field(default_factory=lambda: _envf("VIGIL_FACE_THRESHOLD", 0.45))
    face_identify_every_n: int = field(default_factory=lambda: int(_envf("VIGIL_FACE_EVERY_N", 15)))

    # --- Supabase observability backend ---
    supabase_url: str = field(default_factory=lambda: _env("SUPABASE_URL").rstrip("/"))
    supabase_secret_key: str = field(default_factory=lambda: _env("SUPABASE_SECRET_KEY"))
    supabase_publishable_key: str = field(default_factory=lambda: _env("SUPABASE_PUBLISHABLE_KEY"))

    # --- Data ---
    dataset_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "VIGIL_DATASET_PATH",
                "/Users/pranavachar/Downloads/synthetic-ambient-fhir-25/"
                "synthetic-ambient-fhir-25.jsonl",
            )
        )
    )
    cohort_path: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "demo_cohort.json"
    )
    # Dedicated pre-trained fall-detection YOLO (runs alongside the pose FSM).
    fall_model_path: Path = field(
        default_factory=lambda: Path(
            _env(
                "VIGIL_FALL_MODEL",
                str(Path(__file__).resolve().parent.parent / "models" / "fall_yolov11.pt"),
            )
        )
    )

    # --- Perception tuning ---
    # A fall is only confirmed after the person stays down this long (debounce).
    fall_confirm_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_FALL_CONFIRM_SECONDS", 1.0)
    )
    # Soft signal: no meaningful movement for this long -> voice check-in.
    motionless_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_MOTIONLESS_SECONDS", 540.0)
    )
    # AST scream classifier: sigmoid prob of the loudest AudioSet distress class
    # above which we emit a scream event. Non-scream sound scores ~0.001, so 0.30
    # gives near-zero false positives with high recall on real screams.
    scream_threshold: float = field(default_factory=lambda: _envf("VIGIL_SCREAM_THRESHOLD", 0.3))
    # Window (seconds) in which a scream and a fall count as one fused hard event.
    fusion_window_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_FUSION_WINDOW_SECONDS", 4.0)
    )
    # Cooldown (seconds) suppressing re-fires of the same-or-lower-severity emergency.
    fusion_cooldown_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_FUSION_COOLDOWN_SECONDS", 8.0)
    )

    def require(self, *names: str) -> None:
        """Raise if any required secret is unset (called lazily by modules that need it)."""
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"Missing required config: {', '.join(missing)}. Set them in .env "
                f"(see .env.example)."
            )


settings = Settings()
