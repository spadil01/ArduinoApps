"""
Burden Lounge -> NOLOP walking guide: the Linux side of the UNO Q.

Data flow, 50 times a second:
  sketch (ankle sensor) -> detector.py (strides, turns) -> tracker.py (where am I on the route)
  -> navigator.py (what to say) -> phone (assets/index.html)

Roles (picked on the phone):
  expert     taps each landmark as it starts; walks are the "expert demonstrations"
  corrector  follows the instructions; "Not what I did" / "Wrong instruction" / "Re-sync"
             (DAgger ON = those corrections are training data; OFF = a plain test run)
  new        follows the instructions, no buttons (the real test)

Every walk is saved in data/<walk_id>/:
  raw.csv     every sensor sample (t_ms, ax..gz in g and deg/s) + phase + route part
  events.csv  everything that happened: detections, landmarks, instructions, taps, corrections
  meta.json   who/what/when: role, corrector number, DAgger, height, keep or discard, summary
"""

import csv
import json
import os
import threading
import time

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI

from detector import FS, Detector, calibrate, is_still
from tracker import Tracker
from navigator import Navigator
import navigator as navigator_module

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
with open(os.path.join(HERE, "route.json")) as f:
    ROUTE = json.load(f)
navigator = Navigator(ROUTE, os.path.join(HERE, "model.json"))

CALIB_N = int(2.0 * FS)        # 2 s of standing still
LINK_TIMEOUT_S = 3.0           # the phone pings every second
PUSH_EVERY_S = 0.25            # screen updates at most 4 times a second (plus on every event)
COUNT_HOLD_S = 0.5             # each step-countdown number stays at least this long (no flicker)
MIN_SHOW_S = 0.8               # every instruction stays on screen at least this long (no flashes)


# --------------------------------------------------------------------------
# One walk
# --------------------------------------------------------------------------
class Walk:
    def __init__(self, cfg):
        self.role = cfg["role"]
        self.corrector = (cfg.get("corrector") or 0) if self.role == "corrector" else 0
        self.dagger = bool(cfg.get("dagger")) and self.role == "corrector"
        self.height = cfg["height"]
        tag = {"expert": "expert", "new": "new"}.get(self.role) or \
            f"corrector{self.corrector}_{'dagger' if self.dagger else 'test'}"
        self.walk_id = time.strftime("%Y%m%d_%H%M%S") + "_" + tag
        self.dir = os.path.join(DATA_DIR, self.walk_id)
        os.makedirs(self.dir, exist_ok=True)

        self.tracker = Tracker(ROUTE, self.height, auto=(self.role != "expert"))
        self.detector = None
        self.calib = []
        self.phase = "CALIB"
        self.paused = False
        self.link_lost = False
        self.dropouts = 0
        self.expert_next = 0               # expert: index of the next landmark to tap
        self.decision = None
        self.n_corrections = 0
        self.stride_t = []                 # times of recent strides (walking rhythm)
        self.count = {"key": None, "n": None, "t": 0.0}   # what the step countdown is showing
        self.shown_t = 0.0                 # when the current screen text appeared
        self.landmarks = []                # (t_ms, landmark id, how) in order

        self.meta = {
            "walk_id": self.walk_id, "route": ROUTE["name"], "role": self.role,
            "corrector": self.corrector, "dagger": self.dagger, "height_cm": self.height,
            "step_m_guess": self.tracker.summary()["step_m"], "started": time.time(),
            "model": "model.json" if navigator.model else "route",
        }
        self._write_meta()
        self.raw_f = open(os.path.join(self.dir, "raw.csv"), "w", newline="")
        self.raw = csv.writer(self.raw_f)
        self.raw.writerow(["t_ms", "ax", "ay", "az", "gx", "gy", "gz", "phase", "leg"])
        self.ev_f = open(os.path.join(self.dir, "events.csv"), "w", newline="")
        self.ev = csv.writer(self.ev_f)
        self.ev.writerow(["t_ms", "source", "type", "value", "leg", "d", "action", "route_action", "say", "extra"])
        self.n_raw = 0
        self.last_sync = time.time()

    def _write_meta(self):
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def write_raw(self, t_ms, s):
        self.raw.writerow([t_ms] + [round(v, 4) for v in s] + [self.phase, self.tracker.i])
        self.n_raw += 1
        if self.n_raw % 50 == 0:
            self.raw_f.flush()
        if time.time() - self.last_sync > 5:        # push to the SD card every 5 s so a power cut loses little
            self.last_sync = time.time()
            for fh in (self.raw_f, self.ev_f):
                fh.flush()
                os.fsync(fh.fileno())

    def log(self, source, typ, value="", extra=None):
        s = self.tracker.summary()
        dec = self.decision or {}
        self.ev.writerow([int(time.time() * 1000), source, typ, value, s["leg"], s["d"],
                          dec.get("action", ""), dec.get("route_action", ""), dec.get("say", ""),
                          json.dumps(extra) if extra else ""])
        self.ev_f.flush()

    def note(self, notes):
        """Log what the tracker decided (landmarks reached, detections ignored)."""
        for n in notes:
            if n["type"] == "landmark":
                self.landmarks.append([int(time.time() * 1000), n["id"], n["how"]])
                self.log("tracker", "landmark", n["id"], {"how": n["how"]})
            else:
                self.log("tracker", n["type"], n.get("what", ""), {"why": n.get("why", "")})

    def decide(self):
        """Ask the navigator what to say; log it when it changes."""
        if self.role == "expert":
            self.decision = None
            return
        d = navigator.decide(self.smooth_summary())
        old = self.decision
        now = time.time()
        if old and old["say"] != d["say"] and now - self.shown_t < MIN_SHOW_S:
            return                  # the current screen hasn't been up long enough; try again next update
        if not old or old["say"] != d["say"]:
            self.shown_t = now
        self.decision = d
        if not old or old["say"] != d["say"] and not (old["action"] == d["action"] and "~" in d["say"]):
            self.log("navigator", "instruction", d["action"], {"source": d["source"], "conf": d["confidence"]})

    def smooth_summary(self):
        """Tracker summary with a steady step countdown.
        The tracker only moves once per stride (2 steps), and a little late, so its count jumps.
        Here the steps walked since the last stride are estimated from the walking rhythm, and the
        number shown goes down one step at a time, stays up at least COUNT_HOLD_S, never goes back up."""
        s = self.tracker.summary()
        now = time.time()
        est = float(s["steps_to"])
        walking = self.detector is not None and self.detector.activity != "Standing still"
        if walking and self.stride_t:
            gaps = sorted(b - a for a, b in zip(self.stride_t, self.stride_t[1:]))
            stride_s = gaps[len(gaps) // 2] if gaps else 1.15
            est -= min(2.0, (now - self.stride_t[-1]) / max(stride_s / 2, 0.25))
        est = max(0, round(est))
        c, key = self.count, (s["leg"], s["at_end"])
        if key != c["key"] or c["n"] is None:
            c.update(key=key, n=est, t=now)                 # new part of the route: start fresh
        elif est < c["n"] and now - c["t"] >= COUNT_HOLD_S:
            far = c["n"] > navigator_module.NEAR_STEPS
            c["n"] = est if (far or c["n"] - est >= 3) else c["n"] - 1
            c["t"] = now
        s["steps_to"] = c["n"]
        return s

    def finish_calib(self):
        bias, up = calibrate(self.calib[-CALIB_N:])
        self.detector = Detector(bias, up)
        self.phase = "WALK"
        self.meta.update(gyro_bias=bias, up=up)
        self._write_meta()
        self.log("app", "calibrated")
        self.decide()

    def close(self, keep):
        self.meta.update(keep=keep, ended=time.time(), dropouts=self.dropouts,
                         n_corrections=self.n_corrections, landmarks=self.landmarks,
                         final=self.tracker.summary())
        self._write_meta()
        self.raw_f.close()
        self.ev_f.close()


# --------------------------------------------------------------------------
# App state
# --------------------------------------------------------------------------
ui = WebUI()
lock = threading.Lock()
walk = None
status = "idle"                 # idle | calibrating | active
last_ping = 0.0
last_imu = 0.0
last_push = 0.0
imu_diag = {"text": "no report from sketch", "t": 0.0}


def link_ok():
    return time.time() - last_ping < LINK_TIMEOUT_S


def activity_label():
    """What the figure is doing. Stairs come from where you are on the route (the sensor alone can't tell)."""
    if not walk or not walk.detector:
        return "Standing still"
    act = walk.detector.activity
    leg = ROUTE["legs"][walk.tracker.i]
    if leg["kind"] == "stairs" and act == "Walking straight" and not walk.tracker.at_end:
        return "Going up stairs" if leg["dir"] == "up" else "Going down stairs"
    return act


def push(toast=None):
    global last_push
    last_push = time.time()
    msg = {"status": status}
    if walk:
        msg.update(walk.tracker.summary())
        msg.update(role=walk.role, corrector=walk.corrector, dagger=walk.dagger, walk_id=walk.walk_id,
                   paused=walk.paused, expert_next=walk.expert_next,
                   activity=activity_label())
        if walk.decision:
            msg.update({k: walk.decision[k] for k in ("say", "sub", "then", "ar")})
    if toast:
        msg["toast"] = toast
    ui.send_message("state", msg)


# --------------------------------------------------------------------------
# Sensor samples from the sketch (50 per second)
# --------------------------------------------------------------------------
def on_imu(t_ms, ax, ay, az, gx, gy, gz):
    global status, last_imu
    last_imu = time.time()
    s = (ax, ay, az, gx, gy, gz)
    toast, changed = None, False
    with lock:
        if walk is None:
            return

        # phone dropped off Wi-Fi: mark this stretch so it can be left out later
        if status == "active" and not link_ok():
            if not walk.link_lost:
                walk.link_lost, walk.dropouts = True, walk.dropouts + 1
                walk.log("app", "link_lost")
        elif walk.link_lost:
            walk.link_lost = False
            walk.log("app", "link_back")
            toast = "Link back. The gap is marked in the data."

        walk.phase = "CALIB" if status == "calibrating" else "PAUSE" if walk.paused else \
            "NOLINK" if walk.link_lost else "WALK"
        walk.write_raw(t_ms, s)

        if status == "calibrating":
            walk.calib.append(s)
            walk.calib = walk.calib[-CALIB_N:]
            if len(walk.calib) >= CALIB_N:
                if is_still(walk.calib):
                    walk.finish_calib()
                    status = "active"
                    toast, changed = "Calibrated. Go!", True
                elif len(walk.calib) == CALIB_N and walk.n_raw % CALIB_N == 0:
                    toast = "Hold still…"
        elif walk.detector:
            events = walk.detector.update(s)        # keep the detector running even when paused
            if not walk.paused:
                for e in events:
                    if e["type"] == "stride":
                        walk.stride_t = (walk.stride_t + [time.time()])[-6:]
                        walk.log("detector", "stride", e["cls"], {"dz": e["dz"], "dur": e["dur"], "peak": e["peak"]})
                        walk.note(walk.tracker.on_stride(e["cls"]))
                    elif e["type"] == "turn":
                        walk.log("detector", "turn", e["dir"], {"deg": e["deg"]})
                        walk.note(walk.tracker.on_turn(e["dir"]))
                if events:
                    walk.decide()
                    changed = True

        if changed or toast or time.time() - last_push > PUSH_EVERY_S:
            if not changed and status == "active" and walk.detector and not walk.paused:
                walk.decide()                       # keeps the step countdown moving between strides
            push(toast)


def on_imu_diag(bus_id, addr, whoami, ok, fail):
    """Once-a-second status from the sketch: where the sensor was found and how reads are going."""
    if bus_id < 0:
        text = "MPU-6050 not found on Wire or Wire1. Check wiring."
    else:
        bus_name = "Wire1 (Qwiic)" if bus_id == 1 else "Wire (A4/A5)"
        text = f"found on {bus_name} at 0x{addr:02X}, WHO_AM_I 0x{whoami:02X}, reads ok {ok} / failed {fail}"
    if text.split(",")[0] != imu_diag["text"].split(",")[0]:
        print("IMU:", text, flush=True)
    imu_diag.update(text=text, t=time.time())


# --------------------------------------------------------------------------
# Buttons on the phone
# --------------------------------------------------------------------------
def on_start(sid, data):
    global walk, status
    role = data.get("role")
    if role not in ("expert", "corrector", "new"):
        return
    try:
        height = int(data.get("height") or 0)
    except (TypeError, ValueError):
        height = 0
    if not 100 <= height <= 230:
        push("Enter a height between 100 and 230 cm")
        return
    with lock:
        if walk:
            walk.log("app", "aborted")
            walk.close(keep=False)
        walk = Walk({"role": role, "corrector": data.get("corrector"), "dagger": data.get("dagger"), "height": height})
        status = "calibrating"
        print("Walk started:", walk.walk_id, flush=True)
    push("Stand still, calibrating…")


def on_tap(sid, data):
    """Expert: the landmark that is starting right now."""
    with lock:
        if not walk or walk.role != "expert" or status != "active":
            return
        idx = data.get("idx")
        if idx != walk.expert_next:                  # double tap or out of order: ignore
            return
        walk.paused = False
        lm = ROUTE["legs"][idx]["end"]
        walk.log("expert", "tap", lm["id"])
        walk.note([walk.tracker.set_landmark(idx, "expert tap")])
        walk.expert_next = min(idx + 1, len(ROUTE["legs"]) - 1)
        toast = "Arrived! Tap ⚑ Arrived to save the walk" if lm["type"] == "arrive" else None
    push(toast)


def on_undo(sid, data):
    """Expert: take back the last landmark tap."""
    toast = None
    with lock:
        if not walk or walk.role != "expert" or not walk.landmarks:
            return
        if walk.tracker.undo():
            removed = walk.landmarks.pop()
            walk.expert_next = walk.tracker.i
            walk.log("expert", "undo", removed[1])
            toast = f"Removed {removed[1].replace('_', ' ')}"
    push(toast)


def on_pause(sid, data):
    with lock:
        if not walk or status != "active":
            return
        walk.paused = not walk.paused
        walk.log("app", "pause" if walk.paused else "resume")
        toast = "Paused: doors and waiting are ignored" if walk.paused else "Resumed"
    push(toast)


def on_correct(sid, data):
    """Corrector: kind 'did' = what I really did (fixes the detector),
                  kind 'say' = what the instruction should have been (fixes the navigator)."""
    with lock:
        if not walk or walk.role != "corrector" or status != "active":
            return
        kind, value = data.get("kind"), data.get("value")
        if kind not in ("did", "say"):
            return
        walk.n_corrections += 1
        walk.log("correction", kind, value, {"train": walk.dagger,
                                             "features_state": walk.tracker.summary()})
        # a turn the detector missed: tell the tracker so the walk can go on
        if kind == "did" and value in ("left", "right"):
            walk.note(walk.tracker.on_turn(value))
            walk.decide()
    push()


def on_resync(sid, data):
    with lock:
        if not walk or status != "active":
            return
        idx = data.get("idx")
        if not isinstance(idx, int) or not 0 <= idx < len(ROUTE["legs"]):
            return
        walk.log("correction", "resync", ROUTE["legs"][idx]["end"]["id"], {"train": walk.dagger})
        walk.note([walk.tracker.set_landmark(idx, "resync")])
        walk.decide()
    push()


def on_stop(sid, data):
    global walk, status
    keep = bool(data.get("keep", True))
    with lock:
        if walk:
            walk.log("app", "stop", "keep" if keep else "discard")
            walk.close(keep)
            print("Walk saved:" if keep else "Walk discarded:", walk.walk_id, flush=True)
            walk = None
        status = "idle"
    push("Walk saved" if keep else "Walk discarded")


def on_ping(sid, data):
    global last_ping
    last_ping = time.time()


def on_connect(sid):
    ui.send_message("route", ROUTE)
    push()


# --------------------------------------------------------------------------
# Heartbeat: tells the phone the board is alive and whether sensor data is arriving
# --------------------------------------------------------------------------
def sensor_wait_check():
    """Started a walk but no sensor data is arriving: say so instead of a silent 'Hold still'."""
    if walk and status == "calibrating" and walk.n_raw == 0:
        waited = time.time() - walk.meta["started"]
        if waited > 3 and int(waited) % 3 == 0:
            push("Waiting for the sensor… check that the IMU light is green")
            return True
    return False


def heartbeat_loop():
    while True:
        try:
            sensor_wait_check()
            imu_age = time.time() - last_imu if last_imu else None
            sketch_alive = time.time() - imu_diag["t"] < 3.0
            ui.send_message("hb", {
                "imu_ok": imu_age is not None and imu_age < 1.0,
                "imu_diag": imu_diag["text"] if sketch_alive else "sketch not running (check it built and uploaded)",
            })
            if walk and time.time() - last_push > 1.0:
                with lock:
                    push()
        except Exception as e:
            print("heartbeat error:", e, flush=True)
        time.sleep(1.0)


os.makedirs(DATA_DIR, exist_ok=True)
Bridge.provide("imu", on_imu)
Bridge.provide("imu_diag", on_imu_diag)
ui.on_connect(on_connect)
for name, fn in (("start", on_start), ("tap", on_tap), ("undo", on_undo), ("pause", on_pause),
                 ("correct", on_correct), ("resync", on_resync), ("stop", on_stop), ("ping", on_ping)):
    ui.on_message(name, fn)
threading.Thread(target=heartbeat_loop, daemon=True).start()
print("Ready:", ROUTE["name"], flush=True)

App.run()
