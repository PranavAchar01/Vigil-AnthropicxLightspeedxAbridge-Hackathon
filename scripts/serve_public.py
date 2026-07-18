"""Serve Vigil publicly — RUN THIS IN YOUR TERMINAL (camera + mic live in your login
session). It opens a cloudflared HTTPS tunnel to the local backend, publishes the
tunnel URL to Supabase, and starts the server. The public Vercel page reads that URL
and connects automatically — so anyone who opens the Vercel link sees your live
camera + re-triage while this stays running.

    .venv/bin/python scripts/serve_public.py

Ctrl+C stops both the tunnel and the server. Each run gets a fresh tunnel URL; the
Vercel page follows it with no redeploy (it resolves the URL from Supabase live).

Prereqs (already done once): scripts/grant_camera.py for the camera, and the mic
needs Microphone permission for your terminal. cloudflared must be installed
(`brew install cloudflared`).
"""

from __future__ import annotations

import os
import re
import subprocess
import threading

PORT = 8080
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _free_port() -> None:
    subprocess.run(
        f"lsof -ti tcp:{PORT} | xargs kill -9",
        shell=True,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


def _start_tunnel() -> tuple[subprocess.Popen, str]:
    """Launch cloudflared and return (process, https_url). Keeps draining its output
    so the pipe never fills and blocks the tunnel."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    found: dict[str, str] = {}
    ready = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if "url" not in found:
                m = URL_RE.search(line)
                if m:
                    found["url"] = m.group(0)
                    ready.set()

    threading.Thread(target=reader, daemon=True).start()
    if not ready.wait(timeout=45):
        proc.terminate()
        raise RuntimeError("cloudflared did not report a tunnel URL within 45s")
    return proc, found["url"]


def main() -> None:
    _free_port()
    print("[serve] starting cloudflared tunnel…")
    tunnel, url = _start_tunnel()
    os.environ["VIGIL_PUBLIC_URL"] = url  # must be set BEFORE importing vigil.config

    # Publish immediately so the Vercel page resolves this backend even before boot.
    try:
        from vigil.server import supabase_sink as supa

        supa.set_backend_url(url)
    except Exception as e:  # noqa: BLE001
        print(f"[serve] (warn) could not pre-publish url: {e!r}")

    print("\n" + "=" * 66)
    print(f"  PUBLIC URL : {url}")
    print(f"  LOCAL      : http://localhost:{PORT}")
    print("  Share the Vercel link — it auto-connects to this backend via Supabase.")
    print("=" * 66 + "\n")

    try:
        import uvicorn

        uvicorn.run("vigil.server.app:app", host="127.0.0.1", port=PORT, workers=1)
    finally:
        tunnel.terminate()
        print("\n[serve] tunnel stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
