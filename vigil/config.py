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

    # --- Perception tuning ---
    # A fall is only confirmed after the person stays down this long (debounce).
    fall_confirm_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_FALL_CONFIRM_SECONDS", 1.0)
    )
    # Soft signal: no meaningful movement for this long -> voice check-in.
    motionless_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_MOTIONLESS_SECONDS", 540.0)
    )
    # Scream classifier score above which we emit a scream event.
    scream_threshold: float = field(default_factory=lambda: _envf("VIGIL_SCREAM_THRESHOLD", 0.4))
    # Window (seconds) in which a scream and a fall count as one fused hard event.
    fusion_window_seconds: float = field(
        default_factory=lambda: _envf("VIGIL_FUSION_WINDOW_SECONDS", 4.0)
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
