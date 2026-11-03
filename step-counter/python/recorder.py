# Records trials: the raw sensor readings while someone walks (or fakes it),
# saved as one CSV file per trial in ../data/.
#
# Each file starts with a "# {...}" line holding what it is (person, height_cm,
# leg, activity, when, note, plus two answers typed in after the trial:
# "steps", the steps the person counted, and "meters", the distance walked
# on a measured course -- None until typed in). Then one row per reading:
#   ms, ax, ay, az (g), gx, gy, gz (deg/s), dist_cm (-1 = no echo, -2 = sensor not answering)
# Nothing is trimmed off: every step counts, so the person stands still at the
# start and end instead.
import json
import re
import threading
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEGS = ["left", "right"]
ACTIVITIES = ["walk", "slow", "run", "stairs", "turns", "fake"]
NO_STEPS = "fake"  # sitting and tapping, swaying... counted as 0 steps automatically
RATE_HZ = 100
MIN_KEEP_S = 3    # shorter than this -> not worth saving
MAX_S = 300       # stop on its own after 5 minutes (forgot to press Stop)


class Recorder:
    def __init__(self, on_update, on_saved, show):
        """on_update(dict): progress for the page; on_saved(): the file list changed;
        show(text): what the LED panel should show."""
        self.on_update = on_update
        self.on_saved = on_saved
        self.show = show
        self.lock = threading.Lock()
        self.meta = None   # the trial in progress (None = idle)
        self.phase = None  # "countdown" or "recording"
        self.rows = []
        self.stop_flag = threading.Event()
        self.cancel_flag = threading.Event()
        DATA_DIR.mkdir(exist_ok=True)

    # ---- called from main.py ----

    def add(self, row):
        """Every sensor reading comes through here."""
        with self.lock:
            if self.phase == "recording":
                self.rows.append(row)

    def start(self, setup, countdown):
        """setup: {person, leg, activity, note, started}."""
        if setup["leg"] not in LEGS or setup["activity"] not in ACTIVITIES:
            raise ValueError(f"unknown leg/activity in {setup}")
        person = re.sub(r"[^A-Za-z0-9-]", "", setup["person"]) or "me"
        stamp = re.sub(r"[^0-9T]", "", setup["started"])[:15]
        with self.lock:
            if self.meta:
                raise RuntimeError("already recording")
            self.meta = {"person": person, "height_cm": setup.get("height_cm", ""), "leg": setup["leg"],
                         "activity": setup["activity"], "started": setup["started"],
                         "note": setup.get("note", ""),
                         "steps": 0 if setup["activity"] == NO_STEPS else None,
                         "meters": 0 if setup["activity"] == NO_STEPS else None,
                         # the file name, picked now so camera.py can name its frames folder
                         "file": f"{stamp}_{person}_{setup['activity']}.csv"}
            self.phase = "countdown"
            self.rows = []
        self.stop_flag.clear()
        self.cancel_flag.clear()
        threading.Thread(target=self._run, args=(countdown,), daemon=True).start()

    def stop(self):
        self.stop_flag.set()

    def cancel(self):
        self.cancel_flag.set()
        self.stop_flag.set()

    def busy(self):
        with self.lock:
            return self.meta is not None

    def status(self):
        with self.lock:
            return dict(self.meta or {}, phase=self.phase or "idle")

    # ---- the trial itself (its own thread) ----

    def _run(self, countdown):
        meta = self.meta
        for left in range(countdown, 0, -1):
            self.show(str(left))
            self.on_update(dict(meta, phase="countdown", left=left))
            if self.stop_flag.wait(1):
                return self._finish(None, "cancelled")
        with self.lock:
            self.phase = "recording"
        self.show("GO")
        start = time.monotonic()
        while (took := time.monotonic() - start) < MAX_S:
            self.on_update(dict(meta, phase="recording", took=round(took)))
            if took >= 1:
                self.show(str(int(took)))  # seconds so far, so you can see it's running
            if self.stop_flag.wait(1 - took % 1):
                break
        with self.lock:
            rows = self.rows
        self._finish(None if self.cancel_flag.is_set() else rows,
                     "cancelled" if self.cancel_flag.is_set() else "done")

    def _finish(self, rows, how):
        meta = self.meta
        rows = rows or []
        saved = None
        if len(rows) >= MIN_KEEP_S * RATE_HZ:
            with open(DATA_DIR / meta["file"], "w") as f:
                f.write("# " + json.dumps(meta) + "\n")
                for r in rows:
                    f.write(",".join(str(round(v, 4)) for v in r) + "\n")
            saved = meta["file"]
        with self.lock:
            self.meta = self.phase = None
        self.show("OK" if saved else "X")
        self.on_update(dict(meta, phase="idle", result=how, saved=saved,
                            seconds=round(len(rows) / RATE_HZ)))
        if saved:
            self.on_saved()


def _safe(name):
    path = DATA_DIR / Path(name).name  # .name: never step outside DATA_DIR
    if path.suffix != ".csv" or not path.exists():
        raise FileNotFoundError(name)
    return path


def read_meta(path):
    with open(path) as f:
        return json.loads(f.readline()[2:])


def list_recordings():
    """Every saved trial: its info plus length in seconds."""
    out = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        with open(path) as f:
            meta = json.loads(f.readline()[2:])
            n = sum(1 for _ in f)
        out.append(dict(meta, file=path.name, seconds=round(n / RATE_HZ)))
    return out


def set_answer(name, key, value):
    """Fill in (or fix) an answer typed in after the trial: "steps" (how many
    the person counted) or "meters" (distance walked). None = not known."""
    if key not in ("steps", "meters"):
        raise ValueError(f"unknown answer {key}")
    path = _safe(name)
    with open(path) as f:
        lines = f.readlines()
    meta = json.loads(lines[0][2:])
    if value is None:
        meta[key] = None
    else:
        meta[key] = max(0, int(value)) if key == "steps" else round(max(0.0, float(value)), 2)
    lines[0] = "# " + json.dumps(meta) + "\n"
    with open(path, "w") as f:
        f.writelines(lines)


def read_recording(name):
    with open(_safe(name)) as f:
        return f.read()


def delete_recording(name):
    _safe(name).unlink()
