# Step counter -- runs on the UNO Q's Linux side.
#
#   sketch.ino -> Bridge "imu" (ms, 6 raw IMU numbers, dist_mm) -> this file
#   this file  -> Bridge "show" (text for the LED panel)        -> sketch.ino
#   this file <-> WebSocket on port 8765 <-> the training page (AI_Robotics/step_counter)
#                                        <-> the results page (web/, served on port 7000)
#                                        <-> camera.py and get_data.py on the laptop
#
# It records trials (recorder.py) for training the neural network on the
# laptop, and counts steps live with that network (counter.py) for the
# results page -- open http://<board address>:7000 on a laptop or phone.
# The last trial setup survives a restart in ../settings.json.
import asyncio
import functools
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from arduino.app_utils import *
from websockets.asyncio.server import broadcast, serve

import counter
import recorder

WS_PORT = 8765
WEB_PORT = 7000       # the results page
WEB_DIR = Path(__file__).resolve().parent / "web"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"
ACCEL_PER_G = 2048.0  # must match the ranges set in sketch.ino
GYRO_PER_DPS = 16.4
COUNTDOWN_S = 10      # time to get in place after pressing Record

lock = threading.Lock()
count = 0         # readings since the last report line
latest = None
pages = set()     # browser pages (and laptop scripts) connected right now
counts_only = set()  # pages that asked for no sensor readings (the results page)
ws_loop = None    # the WebSocket thread's event loop

settings = {
    "setup": {"person": "me", "height_cm": "", "leg": "right", "activity": "walk", "note": ""},
}


def load_settings():
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        settings["setup"].update(saved.get("setup", {}))  # keeps new fields saved files don't have
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"couldn't read {SETTINGS_FILE.name} ({e}) -- using defaults")


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def send_to_pages(msg, to=lambda: pages):
    """Send a message to every connected page (or just those to() picks, worked
    out on the WebSocket thread). Safe to call from any thread."""
    if ws_loop and pages:
        ws_loop.call_soon_threadsafe(lambda: broadcast(to(), json.dumps(msg)))


def show(text):
    """Put text on the LED panel (numbers, and the letters in font.h)."""
    Bridge.notify("show", text)


def send_recordings():
    send_to_pages({"type": "recordings", "items": recorder.list_recordings()})


rec = recorder.Recorder(on_update=lambda st: send_to_pages(dict(st, type="recording")),
                        on_saved=send_recordings, show=show)

def on_count(st):
    send_to_pages(dict(st, type="count"))
    if not rec.busy():  # while recording, the panel shows the countdown instead
        show(str(st["steps"]))


try:
    live = counter.Counter(on_update=on_count)
except FileNotFoundError as e:  # no model/ yet: run train.py on the laptop, then deploy
    print(f"no trained network ({e.filename}) -- live counting is off")
    live = None


def on_imu(ms, ax, ay, az, gx, gy, gz, dist_mm):
    """Called by the sketch 100 times a second (raw whole numbers)."""
    global count, latest
    row = (ms, ax / ACCEL_PER_G, ay / ACCEL_PER_G, az / ACCEL_PER_G,
           gx / GYRO_PER_DPS, gy / GYRO_PER_DPS, gz / GYRO_PER_DPS,
           dist_mm / 10 if dist_mm >= 0 else dist_mm)  # -1 no echo, -2 sensor not answering
    with lock:
        count += 1
        latest = row
    rec.add(row)
    if live:
        live.add(row)
    send_to_pages({"type": "imu", "t": ms,
                   "a": [round(v, 3) for v in row[1:4]],
                   "g": [round(v, 1) for v in row[4:7]],
                   "d": round(row[7], 1)}, to=lambda: pages - counts_only)


def start_trial():
    rec.start(dict(settings["setup"], started=time.strftime("%Y-%m-%dT%H:%M:%S")), COUNTDOWN_S)


def on_status(msg):
    print(f"SKETCH SAYS: {msg}")
    send_to_pages({"type": "status", "text": msg})


async def handle_page(ws):
    """One browser page (or laptop script): answer its messages until it disconnects."""
    pages.add(ws)
    print(f"page connected from {ws.remote_address[0]} ({len(pages)} open)")
    try:
        await ws.send(json.dumps({"type": "recordings", "items": recorder.list_recordings()}))
        await ws.send(json.dumps(dict(rec.status(), type="recording")))
        await ws.send(json.dumps({"type": "settings", **settings}))
        if live:
            await ws.send(json.dumps(dict(live.status(), type="count")))
        async for text in ws:
            msg = json.loads(text)
            try:
                await handle_message(ws, msg)
            except Exception as e:  # tell the page instead of dropping the connection
                print(f"error handling {msg.get('type')}: {e}")
                await ws.send(json.dumps({"type": "error", "text": str(e)}))
    finally:
        pages.discard(ws)
        counts_only.discard(ws)
        print(f"page disconnected ({len(pages)} open)")


async def handle_message(ws, msg):
    kind = msg["type"]
    if kind == "ping":  # the page times how long this round trip takes
        await ws.send(json.dumps({"type": "pong", "t": msg["t"]}))
    elif kind == "setup":  # what the next trial is: {person, height_cm, leg, activity, note}
        settings["setup"].update({k: str(msg[k]) for k in settings["setup"] if k in msg})
        save_settings()
        send_to_pages({"type": "settings", **settings})
    elif kind == "record":  # the Record/Stop button
        rec.stop() if rec.busy() else start_trial()
    elif kind == "cancel":
        rec.cancel()
    elif kind == "counts_only":  # the results page: no 100 readings a second, please
        counts_only.add(ws)
    elif kind == "reset" and live:  # the results page's Start over button
        live.reset()
    elif kind in ("steps", "meters"):  # typed in after a trial: steps counted, meters walked
        recorder.set_answer(msg["file"], kind, msg[kind])
        send_recordings()
    elif kind == "delete":
        recorder.delete_recording(msg["file"])
        send_recordings()
    elif kind == "get":  # get_data.py downloading a trial
        await ws.send(json.dumps({"type": "file", "file": msg["file"],
                                  "text": recorder.read_recording(msg["file"])}))


async def ws_main():
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    # max_size=None: trial files can be bigger than the usual 1 MB message limit.
    async with serve(handle_page, "0.0.0.0", WS_PORT, max_size=None):
        await asyncio.Future()  # run forever


def web_main():
    """Hand out the results page (web/) to any laptop or phone that asks."""
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):  # don't fill the log with every page load
            pass
    ThreadingHTTPServer(("0.0.0.0", WEB_PORT),
                        functools.partial(Quiet, directory=str(WEB_DIR))).serve_forever()


def report_loop():
    """Once a second: how many readings arrived, and the latest one."""
    global count
    while True:
        time.sleep(1)
        with lock:
            n, row = count, latest
            count = 0
        # The results page gets no readings, so tell it once a second that they're coming.
        send_to_pages({"type": "sensor", "rate": n}, to=lambda: counts_only)
        if not n:
            print("no readings from the sketch")
            continue
        _, ax, ay, az, gx, gy, gz, d = row
        print(f"{n:3d}/s  accel(g)=({ax:+.2f},{ay:+.2f},{az:+.2f})  "
              f"gyro(deg/s)=({gx:+7.1f},{gy:+7.1f},{gz:+7.1f})  "
              f"dist={'no signal' if d == -2 else 'no echo' if d < 0 else f'{d:.1f} cm'}  pages={len(pages)}")


print("Step counter starting")
load_settings()
Bridge.provide("imu", on_imu)
Bridge.provide("status", on_status)
threading.Thread(target=lambda: asyncio.run(ws_main()), daemon=True).start()
threading.Thread(target=web_main, daemon=True).start()
threading.Thread(target=report_loop, daemon=True).start()
App.run()
