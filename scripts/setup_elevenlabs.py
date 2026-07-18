"""Set up the Vigil ElevenLabs Conversational AI agent.

Subcommands:
  create-agent      create the base agent (returns ELEVENLABS_AGENT_ID)
  import-number     import a Twilio number (returns ELEVENLABS_PHONE_NUMBER_ID)  [paid Twilio]
  create-tool       create the get_patient_status webhook tool (returns tool_id)
  configure-agent   upgrade the agent to a FULLY conversational, realistic voice agent:
                    v3-conversational TTS + a calm voice + native Claude brain +
                    turn-taking/barge-in, and attach the live-status tool.

Usage:
    export ELEVENLABS_API_KEY=sk_...
    uv run python scripts/setup_elevenlabs.py create-agent
    uv run python scripts/setup_elevenlabs.py create-tool \\
        --base-url https://vigil.example.com --token $VIGIL_AGENT_TOKEN
    uv run python scripts/setup_elevenlabs.py configure-agent \\
        --agent $ELEVENLABS_AGENT_ID --tool-id <tool_id> --voice EXAVITQu4vr4xnSDxMaL
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

BASE = "https://api.elevenlabs.io/v1"

FIRST_MESSAGE = (
    "Charge nurse, this is Vigil monitoring with an urgent waiting-room alert. "
    "{{incident}} That's {{patient_name}}, {{esi}}, in the waiting room. Do you have a second?"
)

AGENT_PROMPT = (
    "You are Vigil, an automated E D re-triage escalation dispatcher speaking LIVE with a "
    "charge nurse by phone about a waiting-room patient whose acuity just rose. Be calm, "
    "terse, and clinically specific; lead with the action needed. "
    "Patient={{patient_name}}, acuity={{esi}}, chart={{chart_summary}}, event={{incident}}, "
    "patient_id={{patient_id}}. RULES: "
    "(1) You do NOT hold live patient state in your context. Whenever the nurse asks what the "
    "patient is doing right now, how they look, whether anything changed, falls or screaming, "
    "or the current triage level, you MUST call get_patient_status and speak ONLY its returned "
    "fields. Never invent pose, movement, vitals, or E S I. "
    "(2) Always say how fresh the reading is (seconds ago), and if the triage direction is "
    "worsening or the E S I changed, flag that first. "
    "(3) One or two sentences per turn, then stop and let the nurse talk. "
    "(4) You are decision-support, not diagnosis; defer to the nurse. When they say they've "
    "got it or to hang up, thank them and end the call."
)


def _headers() -> dict:
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY first.")
    return {"xi-api-key": key, "Content-Type": "application/json"}


def _post(path: str, body: dict) -> dict:
    r = requests.post(f"{BASE}{path}", headers=_headers(), json=body, timeout=25)
    if not r.ok:
        sys.exit(f"POST {path} -> {r.status_code}\n{r.text[:800]}")
    return r.json()


def _patch(path: str, body: dict) -> dict:
    r = requests.patch(f"{BASE}{path}", headers=_headers(), json=body, timeout=25)
    if not r.ok:
        sys.exit(f"PATCH {path} -> {r.status_code}\n{r.text[:800]}")
    return r.json()


def create_agent(voice_id: str | None) -> None:
    cfg = {
        "agent": {
            "prompt": {"prompt": AGENT_PROMPT},
            "first_message": FIRST_MESSAGE,
            "language": "en",
        }
    }
    if voice_id:
        cfg["tts"] = {"voice_id": voice_id}
    data = _post(
        "/convai/agents/create", {"name": "Vigil ER Escalation", "conversation_config": cfg}
    )
    print("agent created\n  ELEVENLABS_AGENT_ID=" + str(data.get("agent_id")))


def import_number(phone: str, sid: str, token: str, label: str) -> None:
    data = _post(
        "/convai/phone-numbers",
        {"provider": "twilio", "phone_number": phone, "label": label, "sid": sid, "token": token},
    )
    print("number imported\n  ELEVENLABS_PHONE_NUMBER_ID=" + str(data.get("phone_number_id")))


def create_tool(base_url: str, token: str) -> None:
    # Point at the ACTIVE-patient endpoint (no path placeholder): the server resolves
    # which patient, so the LLM fills ZERO params -> no hallucinated IDs over the phone.
    url = base_url.rstrip("/") + "/agent/patient-status"
    body = {
        "tool_config": {
            "type": "webhook",
            "name": "get_patient_status",
            "description": (
                "Fetch the patient's LIVE status right now. Call this WHENEVER the nurse asks "
                "about the patient's current condition, posture, movement, distress, or triage "
                "level at this moment (e.g. 'is she still slumped?', 'has he moved?', 'what's "
                "her triage level now?', 'is he getting worse?'). Returns real-time pose/motion "
                "from the waiting-room camera, the latest re-triage E S I and how it changed, "
                "chart vitals, and a plain-language summary. Ground your answer in the returned "
                "fields and cite how fresh the data is."
            ),
            "response_timeout_secs": 15,
            "api_schema": {
                "url": url,
                "method": "GET",
                "content_type": "application/json",
                "request_headers": {"X-Vigil-Token": token},
            },
        }
    }
    data = _post("/convai/tools", body)
    tid = data.get("id") or data.get("tool_id") or (data.get("tool_config") or {}).get("id")
    print("tool created\n  TOOL_ID=" + str(tid))
    print("  (attach with: configure-agent --tool-id " + str(tid) + ")")


def configure_agent(agent_id: str, tool_id: str | None, voice: str, model: str, llm: str) -> None:
    prompt: dict = {"prompt": AGENT_PROMPT, "llm": llm, "temperature": 0.2, "max_tokens": 300}
    if tool_id:
        prompt["tool_ids"] = [tool_id]
    cfg = {
        "conversation_config": {
            "agent": {"language": "en", "first_message": FIRST_MESSAGE, "prompt": prompt},
            "asr": {"quality": "high"},
            "turn": {"turn_timeout": 6, "mode": "turn"},
            "tts": {
                "model_id": model,
                "voice_id": voice,
                "stability": 0.5,
                "similarity_boost": 0.8,
                "speed": 1.0,
                "optimize_streaming_latency": 3,
                "agent_output_audio_format": "ulaw_8000",
            },
            "conversation": {"max_duration_seconds": 600},
        }
    }
    data = _patch(f"/convai/agents/{agent_id}", cfg)
    eff = data.get("conversation_config") or {}
    print("agent configured ->", agent_id)
    print("  tts:", json.dumps((eff.get("tts") or {}), separators=(",", ":"))[:200])
    print("  llm:", ((eff.get("agent") or {}).get("prompt") or {}).get("llm"))
    print("  tools:", ((eff.get("agent") or {}).get("prompt") or {}).get("tool_ids"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("create-agent")
    a.add_argument("--voice", default=None)

    n = sub.add_parser("import-number")
    n.add_argument("--phone", required=True)
    n.add_argument("--sid", required=True)
    n.add_argument("--token", required=True)
    n.add_argument("--label", default="Vigil ER line")

    t = sub.add_parser("create-tool")
    t.add_argument(
        "--base-url", required=True, help="public tunnel base, e.g. https://vigil.example.com"
    )
    t.add_argument("--token", required=True, help="VIGIL_AGENT_TOKEN shared secret")

    c = sub.add_parser("configure-agent")
    c.add_argument("--agent", required=True)
    c.add_argument("--tool-id", default=None)
    c.add_argument("--voice", default="EXAVITQu4vr4xnSDxMaL")  # Sarah — calm, professional
    c.add_argument("--model", default="eleven_v3_conversational")
    c.add_argument("--llm", default="claude-haiku-4-5")

    args = ap.parse_args()
    if args.cmd == "create-agent":
        create_agent(args.voice)
    elif args.cmd == "import-number":
        import_number(args.phone, args.sid, args.token, args.label)
    elif args.cmd == "create-tool":
        create_tool(args.base_url, args.token)
    elif args.cmd == "configure-agent":
        configure_agent(args.agent, args.tool_id, args.voice, args.model, args.llm)


if __name__ == "__main__":
    main()
