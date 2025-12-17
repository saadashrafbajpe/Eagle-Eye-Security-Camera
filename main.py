import cv2
import numpy as np
import time
import threading
import platform
import subprocess
import shutil
import os

# ---------- SETTINGS ----------
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_MOTION_AREA = 80
VAR_THRESHOLD = 10
MOTION_PERSIST_SECONDS = 2
ALARM_WAV_PATH = "alert.wav"   # must be a WAV file
# -------------------------------

# Platform helpers
IS_WINDOWS = platform.system() == "Windows"

# Alarm control
_alarm_thread = None
_alarm_stop_event = threading.Event()


def _windows_alarm_loop(path):
    """Use winsound to play WAV in loop on Windows (no extra packages)."""
    import winsound
    # flags: SND_FILENAME to specify file, SND_ASYNC to play asynchronously, SND_LOOP to loop
    flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
    try:
        winsound.PlaySound(path, flags)
        # Wait until stop event set
        _alarm_stop_event.wait()
    finally:
        # Stop the looped sound
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            # fallback try stopping async play
            winsound.PlaySound(None, winsound.SND_ASYNC)


def _subprocess_alarm_loop(path):
    """
    Try available system tools for playing sound in a loop.
    - macOS: afplay (no loop flag) -> loop manually using a small loop
    - Linux: aplay or paplay or ffplay (ffplay requires ffmpeg)
    This tries to be robust but requires these tools to be installed.
    """
    # We'll just repeatedly call the system player while stop event not set.
    players = []

    # prefer these commands if available
    if shutil.which("ffplay"):
        # ffplay can play and exit; -nodisp -autoexit
        players.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
    if shutil.which("aplay"):
        players.append(["aplay", path])
    if shutil.which("paplay"):
        players.append(["paplay", path])
    if shutil.which("afplay"):  # macOS
        players.append(["afplay", path])
    # fallback: try Python's wave+pyaudio? (no, we avoid extra deps)

    if not players:
        # nothing available
        print("No external audio player found (ffplay/aplay/afplay). Alarm will not play on this OS.")
        return

    # use the first available player
    cmd = players[0]
    while not _alarm_stop_event.is_set():
        try:
            proc = subprocess.Popen(cmd)
            # wait until it finishes OR stop event set
            while proc.poll() is None:
                if _alarm_stop_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                time.sleep(0.05)
        except FileNotFoundError:
            # player disappeared; break to avoid busy loop
            print(f"Audio player {cmd[0]} not found at runtime.")
            break
        # tiny sleep to avoid immediate re-launch spin
        time.sleep(0.05)


def start_alarm():
    """Start alarm in a background thread (idempotent)."""
    global _alarm_thread, _alarm_stop_event
    if _alarm_thread and _alarm_thread.is_alive():
        return
    if not os.path.exists(ALARM_WAV_PATH):
        print(f"[ERROR] Alarm file not found: {ALARM_WAV_PATH}")
        return
    _alarm_stop_event.clear()
    if IS_WINDOWS:
        _alarm_thread = threading.Thread(target=_windows_alarm_loop, args=(ALARM_WAV_PATH,), daemon=True)
    else:
        _alarm_thread = threading.Thread(target=_subprocess_alarm_loop, args=(ALARM_WAV_PATH,), daemon=True)
    _alarm_thread.start()


def stop_alarm():
    """Signal alarm to stop and join thread briefly."""
    global _alarm_thread, _alarm_stop_event
    _alarm_stop_event.set()
    if _alarm_thread:
        _alarm_thread.join(timeout=1.0)
        _alarm_thread = None


def main():
    print("Starting security camera. Press 'q' to quit.")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check CAM_INDEX and permissions.")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    last_motion_time = 0
    alarm_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = back_sub.apply(gray)

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_MOTION_AREA:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        t_now = time.time()
        if motion_detected:
            last_motion_time = t_now
            if not alarm_on:
                print("[ALERT] Motion detected — starting alarm.")
                start_alarm()
                alarm_on = True
        else:
            if alarm_on and (t_now - last_motion_time) > MOTION_PERSIST_SECONDS:
                print("[INFO] No motion — stopping alarm.")
                stop_alarm()
                alarm_on = False

        status_text = f"Motion: {'YES' if motion_detected else 'NO'}    Alarm: {'ON' if alarm_on else 'OFF'}"
        cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if alarm_on else (255, 255, 255), 2)

        cv2.imshow("Security Camera", frame)
        cv2.imshow("Motion Mask", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
