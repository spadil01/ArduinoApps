# Impenetrable safe, "board decides" version -- runs on the UNO Q's Linux side.
#
# Here the UNO Q does ALL the thinking:
#   camera (plugged into the hub) -> neural network -> game rules -> lights + servo
#
# The laptop only shows the screen: safe_viewer.py connects to port 7200 and
# gets the camera picture plus everything the game decided, ~10 times a second.
# The laptop can also send "button" (its spacebar = pretend button press).
#
# Files that come from the laptop (push_to_board.py copies them here):
#   model.npz      the trained network's numbers (made from model.pt)
#   numpy_net.py   runs the network with NumPy (no PyTorch needed on the board)
#   pictures.py    how camera pictures are prepared (same as on the laptop)
#   safe_rules.py  the game rules (same file the laptop version uses)
import base64
import glob
import os
import json
import queue
import socketserver
import threading
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # hide noisy camera-search warnings
import cv2
from arduino.app_utils import *

from numpy_net import NumpyNet
from pictures import LABELS, prepare
from safe_rules import SURE, Safe

VIEW_PORT = 7200
SEND_EVERY_S = 0.1         # how often the laptop gets an update
PREVIEW_W = 384            # width of the camera picture sent to the laptop
MODEL_PATH = Path(__file__).parent / "model.npz"

presses = queue.Queue()    # button presses (real button or the laptop's spacebar)
viewers = []               # open connections to laptops
viewers_lock = threading.Lock()


class Hardware:
    """What the game rules use to control the safe: here it talks straight to the sketch."""

    def light(self, pin, on):
        Bridge.notify("light", pin, 1 if on else 0)

    def servo(self, angle):
        Bridge.notify("servo", angle)


def on_button(_):
    presses.put(True)


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")


class Viewer(socketserver.StreamRequestHandler):
    """One laptop watching. It may send 'button' lines."""

    def handle(self):
        print("viewer connected")
        with viewers_lock:
            viewers.append(self.connection)
        try:
            for raw in self.rfile:
                if raw.strip() == b"button":
                    presses.put(True)
        except OSError:
            pass                      # the laptop closed the window: that's fine
        finally:
            with viewers_lock:
                if self.connection in viewers:
                    viewers.remove(self.connection)
            print("viewer disconnected")


def send_to_viewers(message):
    data = (json.dumps(message) + "\n").encode()
    with viewers_lock:
        for conn in list(viewers):
            try:
                conn.sendall(data)
            except OSError:
                viewers.remove(conn)


def open_camera():
    """Try each camera the board has until one gives a picture."""
    for path in sorted(glob.glob("/dev/video*")):
        i = int(path.removeprefix("/dev/video"))
        cam = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            ok, frame = cam.read()
            if ok and frame is not None:
                print(f"camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                return cam
        cam.release()
    return None


def snapshot(safe, frame, chances, steady_s):
    """Everything the laptop needs to draw the screen."""
    now = time.time()
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (PREVIEW_W, PREVIEW_W * h // w), interpolation=cv2.INTER_AREA)
    jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
    return {
        "state": safe.state, "prev": safe.prev, "message": safe.message, "tone": safe.tone,
        "age": now - safe.since,                          # times are sent as "seconds ago"
        "entered": safe.entered, "entered_ages": [now - t for t in safe.entered_at],
        "need_clear": safe.need_clear,
        "lit": {str(pin): on for pin, on in safe.lit.items()},
        "chances": [round(float(c), 4) for c in chances], "steady": steady_s,
        "jpg": base64.b64encode(jpg.tobytes()).decode(),
    }


def brain():
    print("loading the network...")
    net = NumpyNet(MODEL_PATH)
    cam = open_camera()
    while cam is None:
        print("no camera found -- is it plugged into the hub? trying again in 3 s")
        time.sleep(3)
        cam = open_camera()

    safe = Safe(Hardware())
    seeing, seeing_since = None, time.time()   # what the network steadily sees, and since when
    last_sent, frames, fps_time = 0.0, 0, time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            print("camera: no picture")
            time.sleep(0.5)
            continue
        frame = cv2.flip(frame, 1)  # mirror, same as the training pictures

        chances = net.chances(prepare(frame))
        best = int(chances.argmax())

        # How long has the network been sure it sees this same thing?
        now_seeing = LABELS[best] if chances[best] >= SURE else None
        if now_seeing != seeing:
            seeing, seeing_since = now_seeing, time.time()
        steady_s = time.time() - seeing_since

        while not presses.empty():
            presses.get()
            safe.button()
        safe.update(seeing, steady_s)

        if time.time() - last_sent >= SEND_EVERY_S and viewers:
            send_to_viewers(snapshot(safe, frame, chances, steady_s))
            last_sent = time.time()

        frames += 1
        if time.time() - fps_time >= 10:
            print(f"{frames / (time.time() - fps_time):.1f} guesses/sec, sees {LABELS[best]} "
                  f"({100 * chances[best]:.0f}%), state {safe.state}")
            frames, fps_time = 0, time.time()


Bridge.provide("button", on_button)
Bridge.provide("status", on_status)
socketserver.ThreadingTCPServer.allow_reuse_address = True  # restart without "port in use"
socketserver.ThreadingTCPServer.daemon_threads = True
server = socketserver.ThreadingTCPServer(("0.0.0.0", VIEW_PORT), Viewer)
threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Thread(target=brain, daemon=True).start()

App.run()
