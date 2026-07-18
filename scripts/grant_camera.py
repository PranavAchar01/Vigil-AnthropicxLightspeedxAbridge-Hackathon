"""One-time macOS camera-permission grant. RUN THIS IN YOUR TERMINAL first.

Vigil opens the webcam from a worker thread, but macOS only lets the camera
*authorization prompt* appear from the MAIN thread. This tiny script opens the
camera on the main thread so the "Allow camera" prompt shows — click Allow. After
that, the Vigil server can use the camera.

    .venv/bin/python scripts/grant_camera.py

If no prompt appears / it fails: System Settings → Privacy & Security → Camera →
enable your terminal app (Terminal / Ghostty / iTerm), quit + reopen it, rerun.
"""

import sys

import cv2  # imported directly (NOT via vigil) so the auth request is NOT skipped

print("Requesting camera access on the main thread — click 'Allow' if macOS prompts…")
cap = cv2.VideoCapture(0)
ok = False
for _ in range(20):
    read, frame = cap.read()
    if read and frame is not None:
        ok = True
        break
cap.release()

if ok:
    print("\n✅ Camera works and permission is granted.")
    print("   Now run:  .venv/bin/python -m uvicorn vigil.server.app:app --port 8080")
    sys.exit(0)
else:
    print("\n❌ Camera did not open.")
    print("   Fix: System Settings → Privacy & Security → Camera → enable your")
    print("   terminal app, then quit + reopen the terminal and rerun this script.")
    sys.exit(1)
