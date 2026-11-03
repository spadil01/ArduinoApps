# Counts steps live with the network trained on the laptop
# (AI_Robotics/step_counter/train.py), using numpy only -- no PyTorch here.
#
# model/stepnet.npz   the network's numbers
# model/report.json   its cutoff, its fair-test error (the ± on the steps),
#                     and step length (meters per step, with its own ±)
#
# Twice a second it runs the network over the last 4 s of readings. The
# network looks ~0.65 s both ways around each reading, so a step only counts
# once 0.66 s of readings have come in after it.
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "model"
RATE_HZ = 100
WINDOW = 400          # readings the network sees each time (4 s)
EDGE = 66             # readings it looks either way (7-wide start + 5 blocks spreading 1..16)
MIN_GAP = 25          # nobody takes two steps closer together than 0.25 s
EVERY_S = 0.5


class Net:
    """StepNet from stepnet.py, rewritten in numpy (inference only)."""

    def __init__(self, path):
        self.w = dict(np.load(path))

    def conv(self, x, name, dilation=1):
        """1D convolution with "same" padding. x: [channels, time]."""
        w, b = self.w[name + ".weight"], self.w[name + ".bias"]
        k = w.shape[2]
        pad = dilation * (k // 2)
        xp = np.pad(x, ((0, 0), (pad, pad)))
        t = x.shape[1]
        return sum(w[:, :, i] @ xp[:, i * dilation:i * dilation + t] for i in range(k)) + b[:, None]

    def __call__(self, x):
        """x: [7, time] raw readings (ax, ay, az, gx, gy, gz, dist_cm) -> [time] 0..1."""
        a, g, d = x[0:3], x[3:6] / 200, x[6:7]
        ok = (d >= 0).astype(np.float32)
        h = np.concatenate([a, g, np.linalg.norm(a, axis=0, keepdims=True) - 1,
                            np.linalg.norm(g, axis=0, keepdims=True),
                            np.clip(d, 0, 60) / 30 * ok + 2 * (1 - ok), ok])
        h = np.maximum(self.conv(h, "inp"), 0)
        for i, dil in enumerate((1, 2, 4, 8, 16)):
            h = h + self.conv(np.maximum(self.conv(h, f"blocks.{i}.conv", dil), 0), f"blocks.{i}.mix")
        return 1 / (1 + np.exp(-self.conv(h, "out")[0]))


class Counter:
    def __init__(self, on_update):
        """on_update(dict): the latest count, for the results page."""
        self.on_update = on_update
        report = json.loads((MODEL_DIR / "report.json").read_text())
        self.net = Net(MODEL_DIR / "stepnet.npz")
        self.threshold = report.get("threshold", 0.5)
        self.steps_pct = report["error"]["network"]["percent"]
        self.step_m = report["step_length"]["meters"]
        self.step_m_pct = report["step_length"]["percent"]
        self.lock = threading.Lock()
        self.rows = deque(maxlen=WINDOW)
        self.n = 0         # readings since the last reset
        self.done = 0      # readings already looked at for steps
        self.last_step = -MIN_GAP
        self.steps = 0
        self.resets = 0    # goes up on every reset, so a count already underway gets dropped
        self.since = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, row):
        """row: (ms, ax, ay, az, gx, gy, gz, dist_cm), 100 a second."""
        with self.lock:
            self.rows.append(row[1:8])
            self.n += 1

    def reset(self):
        with self.lock:
            self.rows.clear()
            self.n = self.done = self.steps = 0
            self.last_step = -MIN_GAP
            self.resets += 1
            self.since = time.time()
        self.on_update(self.status())

    def status(self):
        """Steps and meters, each with a ± from how far off the fair test was."""
        with self.lock:
            steps, since = self.steps, self.since
        meters = steps * self.step_m
        meters_pct = (self.steps_pct ** 2 + self.step_m_pct ** 2) ** 0.5  # both errors together
        return {"steps": steps, "steps_pm": round(steps * self.steps_pct / 100),
                "meters": round(meters, 1), "meters_pm": round(meters * meters_pct / 100, 1),
                "steps_pct": self.steps_pct, "meters_pct": round(meters_pct, 1),
                "step_m": self.step_m, "seconds": round(time.time() - since)}

    def _run(self):
        while True:
            time.sleep(EVERY_S)
            try:
                if self._count():
                    self.on_update(self.status())
            except Exception as e:
                print(f"counter error: {e}")

    def _count(self):
        """Look for steps in readings not looked at yet. True if any were found."""
        with self.lock:
            x = np.array(self.rows, dtype=np.float32).T
            n, done, resets = self.n, self.done, self.resets
        if n - done < RATE_HZ * EVERY_S or len(x) == 0:
            return False
        start = n - x.shape[1]                      # reading number of x's first column
        prob = self.net(x)
        lo = max(done, start + EDGE if start > 0 else 0) - start
        hi = x.shape[1] - EDGE                      # past here the network can't see enough ahead yet
        found = []
        last = self.last_step
        for i in range(max(lo, 1), hi):
            if prob[i] >= self.threshold and prob[i] > prob[i - 1] and prob[i] >= prob[i + 1] \
                    and start + i - last >= MIN_GAP:
                found.append(start + i)
                last = start + i
        with self.lock:
            if self.resets != resets:  # reset while the network was working: throw this away
                return False
            self.done = max(self.done, start + hi)
            self.last_step = last
            self.steps += len(found)
        return bool(found)
