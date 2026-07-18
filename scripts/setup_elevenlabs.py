"""One-time setup for the ElevenLabs outbound nurse-call agent.

Creates a Conversational AI agent whose prompt + first message reference the
dynamic variables Vigil fills at call time ({{patient_name}}, {{chart_summary}},
{{incident}}, {{esi}}) — no Security-tab toggles needed. Optionally imports a
Twilio number so you get the agent_phone_number_id.

Usage:
    export ELEVENLABS_API_KEY=sk_...
    uv run python scripts/setup_elevenlabs.py create-agent
    uv run python scripts/setup_elevenlabs.py import-number \\
        --phone +14155550123 --sid ACxxxx --token xxxx --label "Vigil ER line"

Put the printed agent_id -> ELEVENLABS_AGENT_ID and phone_number_id ->
ELEVENLABS_PHONE_NUMBER_ID in your .env (with NURSE_PHONE_NUMBER).
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

BASE = "https://api.elevenlabs.io/v1"

FIRST_MESSAGE = (
    "Charge nurse, this is the Vigil monitoring system with an urgent waiting-room "
    "alert. {{incident}} The patient is {{patient_name}}, {{esi}}. {{chart_summary}}"
)

SYSTEM_PROMPT = (
    "You are Vigil, an automated clinical escalation dispatcher calling a charge "
    "nurse about a waiting-room patient who has deteriorated. Be calm, terse, and "
    "specific. Lead with the action needed. Patient: {{patient_name}}. Acuity: "
    "{{esi}}. Chart: {{chart_summary}}. Event: {{incident}}. Answer the nurse's "
    "questions ONLY from this context; never invent clinical facts. If asked, "
    "repeat the location and the single most important reason to come now. Keep the "
    "call under 30 seconds unless the nurse asks for more."
)


def _headers() -> dict:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY first.")
    return {"xi-api-key": key, "Content-Type": "application/json"}


def create_agent(voice_id: str | None) -> None:
    conversation_config = {
        "agent": {
            "prompt": {"prompt": SYSTEM_PROMPT},
            "first_message": FIRST_MESSAGE,
            "language": "en",
        }
    }
    if voice_id:
        conversation_config["tts"] = {"voice_id": voice_id}
    body = {"name": "Vigil ER Escalation", "conversation_config": conversation_config}
    r = requests.post(f"{BASE}/convai/agents/create", headers=_headers(), json=body, timeout=20)
    r.raise_for_status()
    agent_id = r.json().get("agent_id")
    print("agent created")
    print("  ELEVENLABS_AGENT_ID=", agent_id, sep="")


def import_number(phone: str, sid: str, token: str, label: str) -> None:
    body = {"provider": "twilio", "phone_number": phone, "label": label, "sid": sid, "token": token}
    r = requests.post(f"{BASE}/convai/phone-numbers", headers=_headers(), json=body, timeout=20)
    r.raise_for_status()
    pid = r.json().get("phone_number_id")
    print("number imported")
    print("  ELEVENLABS_PHONE_NUMBER_ID=", pid, sep="")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-agent")
    a.add_argument("--voice", default=None, help="voice_id (see search_voices in the MCP)")
    n = sub.add_parser("import-number")
    n.add_argument("--phone", required=True)
    n.add_argument("--sid", required=True)
    n.add_argument("--token", required=True)
    n.add_argument("--label", default="Vigil ER line")
    args = ap.parse_args()
    if args.cmd == "create-agent":
        create_agent(args.voice)
    else:
        import_number(args.phone, args.sid, args.token, args.label)


if __name__ == "__main__":
    main()
