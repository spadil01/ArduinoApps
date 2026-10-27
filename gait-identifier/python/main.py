# Gait identifier -- runs on the UNO Q's Linux side.
#
#   sketch.ino -> Bridge "imu" (ms, ax, ay, az in g, gx, gy, gz in deg/s) -> this file
#   this file  -> Bridge "gait" ("stand", "walk", "run", "bike") -> LED panel
#   this file <-> WebSocket on port 8765 <-> the browser page (AI_Robotics/gait_identifier)
#
# The page connects straight to the board (no MQTT broker in between) and gets
# every sensor reading the moment it arrives. Recording examples: recorder.py.
# Training and guessing: gait_model.py. Trained models are saved in ../models/
# so they survive a restart.
import asyncio
import collections
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

from arduino.app_utils import *
from websockets.asyncio.server import broadcast, serve

import gait_model as gm
import recorder

WS_PORT = 8765
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GUESS_EVERY_S = 0.25   # a fresh guess 4 times a second (each looks at the last 2 s)
LED_AGREE = 2          # the LED panel switches after this many guesses in a row agree

lock = threading.Lock()
samples = []      # readings since the last report line
pages = set()     # browser pages connected right now
ws_loop = None    # the WebSocket thread's event loop

window = collections.deque(maxlen=gm.WINDOW)  # the last 2 s of readings (6 numbers each)
trained = {}      # spot -> what gait_model.train_spot returned
training = False
# Set by the page. agree/far: how strict "other" is (see gait_model.guess_all).
# preview: a picture to hold on the LED panel while not running (to show it off).
live = {"on": False, "spot": "ankle", "led": "all-tree",
        "agree": gm.AGREE_MIN, "far": gm.FAR_FACTOR, "preview": "idle"}
PREVIEWS = ("stand", "walk", "run", "bike", "idle", "other")


def send_to_pages(msg):
    """Send a message to every connected page. Safe to call from any thread."""
    if ws_loop and pages:
        ws_loop.call_soon_threadsafe(broadcast, pages, json.dumps(msg))


def show_gait(name):
    """Put a gait on the LED panel ("idle" shows "GAIT"; "other" or "" shows "?")."""
    Bridge.notify("gait", name)


def send_recordings():
    send_to_pages({"type": "recordings", "items": recorder.list_recordings()})


rec = recorder.Recorder(on_update=lambda st: send_to_pages(dict(st, type="recording")),
                        on_saved=send_recordings, show_gait=show_gait)


def on_imu(ms, ax, ay, az, gx, gy, gz):
    """Called by the sketch 50 times a second."""
    with lock:
        samples.append((ms, ax, ay, az, gx, gy, gz))
    rec.add((ms, ax, ay, az, gx, gy, gz))
    window.append((ax, ay, az, gx, gy, gz))
    send_to_pages({"type": "imu", "t": ms,
                   "a": [round(ax, 3), round(ay, 3), round(az, 3)],
                   "g": [round(gx, 1), round(gy, 1), round(gz, 1)]})


# ---------------- training ----------------

def load_models():
    for path in MODELS_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as f:
                result = pickle.load(f)
            if result.get("version") != gm.FEATURES_VERSION:
                print(f"{path.stem} models use old clues -- press Train")
                continue
            trained[path.stem] = result
            print(f"loaded {path.stem} models")
        except Exception as e:
            print(f"couldn't load {path.name} ({e}) -- press Train again")


def models_message():
    return {"type": "models", "training": training,
            "reports": {spot: t["report"] for spot, t in trained.items()}}


def train_all():
    """Train every spot that has recordings (in train_worker.py, a separate,
    lower-priority program so the live stream keeps flowing) and tell the pages."""
    global training
    training = True
    send_to_pages(models_message())
    try:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("train_worker.py")),
             str(recorder.DATA_DIR), str(MODELS_DIR)],
            stdout=subprocess.PIPE, text=True, preexec_fn=lambda: os.nice(10))
        for line in worker.stdout:
            done = json.loads(line)
            if "error" in done:
                send_to_pages({"type": "error", "text": done["error"]})
                continue
            with open(MODELS_DIR / f"{done['spot']}.pkl", "rb") as f:
                trained[done["spot"]] = pickle.load(f)
            scores = trained[done["spot"]]["report"]["models"]
            print(f"trained {done['spot']} in {done['seconds']}s: " +
                  ", ".join(f"{n} {scores[n]['score']:.0%}" for n in gm.MODEL_NAMES))
        if worker.wait() != 0:
            send_to_pages({"type": "error", "text": "training crashed -- see the board log"})
    finally:
        training = False
        send_to_pages(models_message())


# ---------------- live guessing ----------------

def live_loop():
    """While the page has Run on: guess 4 times a second and drive the LED panel."""
    recent = collections.deque(maxlen=LED_AGREE)  # the LED model's latest guesses
    shown = None
    last_idle = 0.0
    while True:
        time.sleep(GUESS_EVERY_S)
        models = trained.get(live["spot"])
        if rec.status()["phase"] != "idle":  # a recording owns the LED panel
            recent.clear()
            shown = None
            continue
        if not live["on"] or not models or len(window) < gm.WINDOW:
            recent.clear()
            shown = None
            # Not guessing: keep "GAIT" (or the picked preview) on the panel. Repeated
            # once a second so it also fixes a panel left showing an old gait
            # (the sketch ignores repeats).
            if time.monotonic() - last_idle >= 1:
                last_idle = time.monotonic()
                show_gait(live["preview"])
            continue
        t = time.perf_counter()
        guesses, features = gm.guess_all(models, gm.np.array(window), live["agree"], live["far"])
        recent.append(guesses[live["led"]]["gait"])
        if len(recent) == LED_AGREE and len(set(recent)) == 1 and recent[0] != shown:
            shown = recent[0]
            show_gait(shown)
        send_to_pages({"type": "guess", "spot": live["spot"], "led": live["led"], "shown": shown,
                       "guesses": guesses, "features": features,
                       "ms": round((time.perf_counter() - t) * 1000, 1)})


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")
    send_to_pages({"type": "status", "text": msg})


async def handle_page(ws):
    """One browser page: answer its messages until it disconnects."""
    pages.add(ws)
    print(f"page connected from {ws.remote_address[0]} ({len(pages)} open)")
    try:
        await ws.send(json.dumps({"type": "recordings", "items": recorder.list_recordings()}))
        await ws.send(json.dumps(dict(rec.status(), type="recording")))
        await ws.send(json.dumps(models_message()))
        await ws.send(json.dumps(dict(live, type="live")))
        async for text in ws:
            msg = json.loads(text)
            try:
                await handle_message(ws, msg)
            except Exception as e:  # tell the page instead of dropping the connection
                print(f"error handling {msg}: {e}")
                await ws.send(json.dumps({"type": "error", "text": str(e)}))
    finally:
        pages.discard(ws)
        print(f"page disconnected ({len(pages)} open)")


async def handle_message(ws, msg):
    kind = msg["type"]
    if kind == "ping":  # the page times how long this round trip takes
        await ws.send(json.dumps({"type": "pong", "t": msg["t"]}))
    elif kind == "record":
        rec.start(msg["person"], msg["spot"], msg["activity"], int(msg["seconds"]),
                  int(msg["countdown"]), msg["started"], msg.get("note", ""))
    elif kind == "stop":
        rec.stop()
    elif kind == "delete":
        recorder.delete_recording(msg["file"])
        send_recordings()
    elif kind == "train":
        if not training:
            threading.Thread(target=train_all, daemon=True).start()
    elif kind == "live":  # {"on": bool, "spot", "led": model name, "agree", "far", "preview"}, any of them
        if msg.get("spot") in recorder.SPOTS:
            live["spot"] = msg["spot"]
        if msg.get("led") in gm.MODEL_NAMES:
            live["led"] = msg["led"]
        if "on" in msg:
            live["on"] = bool(msg["on"])
            live["preview"] = "idle"
        if msg.get("preview") in PREVIEWS:  # showing a picture stops Run
            live["on"] = False
            live["preview"] = msg["preview"]
            if rec.status()["phase"] == "idle":
                show_gait(live["preview"])
        if "agree" in msg:
            live["agree"] = min(1.0, max(0.0, float(msg["agree"])))
        if "far" in msg:
            live["far"] = max(0.1, float(msg["far"]))
        send_to_pages(dict(live, type="live"))


async def ws_main():
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    async with serve(handle_page, "0.0.0.0", WS_PORT):
        await asyncio.Future()  # run forever


def report_loop():
    """Once a second: how many readings arrived, the latest one, and the total shake."""
    while True:
        time.sleep(1)
        with lock:
            batch = samples[:]
            samples.clear()
        if not batch:
            print("no readings from the sketch")
            continue
        _, ax, ay, az, gx, gy, gz = batch[-1]
        shake = [math.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2) for s in batch]
        print(f"{len(batch):2d}/s  accel(g)=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
              f"gyro(deg/s)=({gx:+7.1f},{gy:+7.1f},{gz:+7.1f})  "
              f"total shake min/max={min(shake):.2f}/{max(shake):.2f}  pages={len(pages)}")


print("Gait identifier starting")
load_models()
Bridge.provide("imu", on_imu)
Bridge.provide("status", on_status)
threading.Thread(target=lambda: asyncio.run(ws_main()), daemon=True).start()
threading.Thread(target=report_loop, daemon=True).start()
threading.Thread(target=live_loop, daemon=True).start()
App.run()
