# Records labelled examples: the raw sensor readings while someone does one
# activity, saved as one CSV file per recording in ../data/.
#
# Each file starts with a "# {...}" line holding what it is (person, spot,
# activity, when, note), then one row per reading: ms, ax, ay, az, gx, gy, gz.
# The first and last TRIM_S seconds are cut off -- that's the person getting
# going or stopping, not a clean example.
import json
import re
import threading
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPOTS = ["ankle", "waist"]
ACTIVITIES = ["stand", "walk", "run", "bike"]
RATE_HZ = 50
TRIM_S = 2
MIN_KEEP_S = 4  # shorter than this after trimming -> not worth saving


class Recorder:
    def __init__(self, on_update, on_saved, show_gait):
        """on_update(dict): progress for the page; on_saved(): the file list changed;
        show_gait(name): what the LED panel should show."""
        self.on_update = on_update
        self.on_saved = on_saved
        self.show_gait = show_gait
        self.lock = threading.Lock()
        self.meta = None   # the recording in progress (None = idle)
        self.phase = None  # "countdown" or "recording"
        self.rows = []
        self.stop_flag = threading.Event()
        DATA_DIR.mkdir(exist_ok=True)

    # ---- called from main.py ----

    def add(self, row):
        """Every sensor reading comes through here."""
        with self.lock:
            if self.phase == "recording":
                self.rows.append(row)

    def start(self, person, spot, activity, seconds, countdown, started, note=""):
        if spot not in SPOTS or activity not in ACTIVITIES:
            raise ValueError(f"unknown spot/activity {spot}/{activity}")
        person = re.sub(r"[^A-Za-z0-9-]", "", person) or "me"
        with self.lock:
            if self.meta:
                raise RuntimeError("already recording")
            self.meta = {"person": person, "spot": spot, "activity": activity,
                         "started": started, "note": note}
            self.phase = "countdown"
            self.rows = []
        self.stop_flag.clear()
        threading.Thread(target=self._run, args=(seconds, countdown), daemon=True).start()

    def stop(self):
        self.stop_flag.set()

    def status(self):
        with self.lock:
            return dict(self.meta or {}, phase=self.phase or "idle")

    # ---- the recording itself (its own thread) ----

    def _run(self, seconds, countdown):
        meta = self.meta
        self.show_gait(meta["activity"])
        for left in range(countdown, 0, -1):
            self.on_update(dict(meta, phase="countdown", left=left))
            if self.stop_flag.wait(1):
                return self._finish(None, "cancelled")
        with self.lock:
            self.phase = "recording"
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self.on_update(dict(meta, phase="recording", left=round(left)))
            if self.stop_flag.wait(min(1, left)):
                break
        with self.lock:
            rows = self.rows
        self._finish(rows, "stopped" if self.stop_flag.is_set() else "done")

    def _finish(self, rows, how):
        meta = self.meta
        trim = TRIM_S * RATE_HZ
        rows = (rows or [])[trim:-trim]
        saved = None
        if len(rows) >= MIN_KEEP_S * RATE_HZ:
            stamp = re.sub(r"[^0-9T]", "", meta["started"])[:15]
            name = f"{stamp}_{meta['person']}_{meta['spot']}_{meta['activity']}.csv"
            with open(DATA_DIR / name, "w") as f:
                f.write("# " + json.dumps(meta) + "\n")
                for r in rows:
                    f.write(",".join(str(round(v, 4)) for v in r) + "\n")
            saved = name
        with self.lock:
            self.meta = self.phase = None
        self.show_gait("idle")
        self.on_update(dict(meta, phase="idle", result=how, saved=saved,
                            seconds=round(len(rows) / RATE_HZ)))
        if saved:
            self.on_saved()


def list_recordings():
    """Every saved recording: its info plus file name and length in seconds."""
    out = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        with open(path) as f:
            meta = json.loads(f.readline()[2:])
            n = sum(1 for _ in f)
        out.append(dict(meta, file=path.name, seconds=round(n / RATE_HZ)))
    return out


def delete_recording(name):
    path = DATA_DIR / Path(name).name  # .name: never step outside DATA_DIR
    if path.suffix == ".csv" and path.exists():
        path.unlink()
