"""
detector.py: turns raw ankle-sensor readings into events.

Feed it one sample at a time (50 per second). It returns a list of events:
  {"type": "stride", "dz": 0.31, "cls": "up"}    one full swing of the sensor leg (= 2 steps).
                                                 dz = how far the ankle went up/down (meters).
  {"type": "turn", "dir": "left", "deg": 88}     the body turned about 90 degrees.
It also keeps `activity` up to date for the screen ("Walking straight", "Turning left", ...).

How it works, in short:
  - "Which way is up" is learned while standing still, then followed as the leg swings
    (the gyro says how the sensor rotated; gravity gently corrects it when the leg is calm).
  - Turns  = rotation around "up", added up over the last 2 seconds.
  - Strides = forward swings of the shin. The swing axis (the shin's side-to-side axis) is learned
    from the walking itself, so it doesn't matter how the sensor is twisted on the leg.
  - Between swings the ankle is almost still; that is used to measure how far it moved up or down
    during each stride (stairs: about +-0.33 m, flat: about 0).

Pure Python (only math), so it runs on the board as-is.
"""

import math
from collections import deque

FS = 50.0
DT = 1.0 / FS
G = 9.81

# Tunable numbers (check_walks.py on the laptop is used to tune these)
SWING_START_DPS = 100.0   # leg swing must pass this rotation speed to count as a stride
SWING_END_FRAC = 0.4      # swing is over when speed drops below 40 % of its peak
MIN_STRIDE_S = 0.75       # two strides can't be closer than this (real walking strides take ~0.9-1.4 s)
TURN_DEG = 50.0           # body rotation needed to call a turn
TURN_WINDOW_S = 2.0
TURN_STEP_DEG = 90.0      # a long continuous turn sends one more event every extra 90 degrees
TURN_DONE_DEG = 10.0      # turn is over when less than this was turned in the last second
STAIR_DZ = 0.20           # ankle height change per stride that counts as stairs (logged only: not reliable yet)
STILL_DPS = 20.0
STILL_S = 0.6


def calibrate(samples):
    """samples: list of (ax, ay, az, gx, gy, gz) while standing still -> (gyro bias, unit 'up' vector)."""
    n = len(samples)
    m = [sum(s[i] for s in samples) / n for i in range(6)]
    norm = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2) or 1.0
    return (m[3], m[4], m[5]), (m[0] / norm, m[1] / norm, m[2] / norm)


def is_still(samples):
    """True if these samples look like someone standing still (used to check calibration)."""
    for i in range(3, 6):
        v = [s[i] for s in samples]
        if max(v) - min(v) > 8.0:
            return False
    mags = [math.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2) for s in samples]
    return max(mags) - min(mags) < 0.08


class Detector:
    def __init__(self, bias, up):
        self.bias = bias
        self.u = list(up)                  # "up", in the sensor's own axes
        self.n = 0                         # sample counter
        self.yawq = deque(maxlen=int(TURN_WINDOW_S * FS))
        self.recent_yaw = deque(maxlen=int(1.0 * FS))
        self.spin = 0.0                    # degrees turned in the current continuous turn
        self.spin_fired = 0                # turn events already sent for it
        # stride tracking
        self.av = []                       # vertical acceleration (m/s^2) since the last calm moment
        self.in_swing = False
        self.swing_peak = 0.0
        self.since_swing = 10 ** 6
        self.calm_i = 0                    # index (into self.av) of the calmest sample since the swing ended
        self.calm_val = 1e9
        self.swung = False                 # a swing happened since the last calm moment
        self.still_n = 0
        self.last_stride_n = -10 ** 6
        # the leg's swing axis (side-to-side axis of the shin), learned while walking
        self.cov = [[0.0] * 3 for _ in range(3)]
        self.axis = None
        self.pos_peak = 0.0
        self.neg_peak = 0.0
        # stairs (for the activity label only; the tracker decides what really counts)
        self.stair_run = 0
        self.flat_run = 0
        self.stairs = None                 # None, "up" or "down"
        self.activity = "Standing still"

    # ---- one sample in, events out ----
    def update(self, s):
        ax, ay, az, gx, gy, gz = s
        self.n += 1
        events = []
        wx, wy, wz = gx - self.bias[0], gy - self.bias[1], gz - self.bias[2]   # deg/s
        gmag = math.sqrt(wx * wx + wy * wy + wz * wz)

        # 1) follow "up" as the leg rotates: du/dt = -w x u
        rx, ry, rz = math.radians(wx), math.radians(wy), math.radians(wz)
        u = self.u
        cx, cy, cz = ry * u[2] - rz * u[1], rz * u[0] - rx * u[2], rx * u[1] - ry * u[0]
        u = [u[0] - cx * DT, u[1] - cy * DT, u[2] - cz * DT]
        amag = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        if abs(amag - 1.0) < 0.08 and gmag < 60:           # leg calm: let gravity correct drift
            k = 0.02
            u = [(1 - k) * u[0] + k * ax / amag, (1 - k) * u[1] + k * ay / amag, (1 - k) * u[2] + k * az / amag]
        un = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2) or 1.0
        self.u = u = [u[0] / un, u[1] / un, u[2] / un]

        yaw = wx * u[0] + wy * u[1] + wz * u[2]             # deg/s around "up", + = left
        swing = self._swing(wx - yaw * u[0], wy - yaw * u[1], wz - yaw * u[2])   # + = leg swinging forward
        a_up = ((ax * u[0] + ay * u[1] + az * u[2]) - 1.0) * G
        self.av.append(a_up)

        # 2) turns
        self.yawq.append(yaw * DT)
        self.recent_yaw.append(yaw * DT)
        if self.spin_fired:
            # in the middle of a turn: one more event for every extra 90 degrees (a U-turn = 2 turns)
            self.spin += yaw * DT
            if abs(self.spin) >= TURN_DEG + TURN_STEP_DEG * self.spin_fired and self.spin * self.spin_dir > 0:
                events.append({"type": "turn", "dir": "left" if self.spin > 0 else "right", "deg": round(abs(self.spin))})
                self.spin_fired += 1
            if len(self.recent_yaw) == self.recent_yaw.maxlen and abs(sum(self.recent_yaw)) < TURN_DONE_DEG:
                self.spin_fired, self.spin = 0, 0.0      # turn finished
                self.yawq.clear()
        else:
            total = sum(self.yawq)
            if abs(total) >= TURN_DEG:
                events.append({"type": "turn", "dir": "left" if total > 0 else "right", "deg": round(abs(total))})
                self.spin, self.spin_fired, self.spin_dir = total, 1, (1 if total > 0 else -1)
                self.recent_yaw.clear()

        # 3) strides
        self.since_swing += 1
        if not self.in_swing:
            if gmag < self.calm_val:                         # remember the calmest moment
                self.calm_val, self.calm_i = gmag, len(self.av) - 1
            if swing > SWING_START_DPS and self.since_swing > MIN_STRIDE_S * FS:
                # a new swing begins: the stride before it ended at the calm moment
                if self.swung:
                    events.append(self._finish_stride(self.calm_i))
                else:
                    self.av = self.av[self.calm_i:]
                self.in_swing, self.swing_peak, self.since_swing = True, swing, 0
        else:
            self.swing_peak = max(self.swing_peak, swing)
            if swing < SWING_END_FRAC * self.swing_peak:
                self.in_swing, self.swung = False, True
                self.calm_val, self.calm_i = 1e9, len(self.av) - 1

        # standing still: close the last stride
        self.still_n = self.still_n + 1 if gmag < STILL_DPS else 0
        if self.still_n == int(STILL_S * FS):
            if self.swung and not self.in_swing:
                events.append(self._finish_stride(self.calm_i))
            self.av = self.av[-1:]
            self.calm_val, self.calm_i = 1e9, 0
        if len(self.av) > 10 * FS and not self.swung:        # keep memory bounded while standing
            self.av = self.av[-int(FS):]
            self.calm_i = 0

        self._update_activity(events)
        return events

    def _swing(self, x, y, z):
        """Rotation speed of the shin around its side-to-side axis, signed so the forward swing is positive.
        The axis is the direction the gyro spins the most (learned from the walking itself)."""
        c, k = self.cov, 0.995
        v = (x, y, z)
        for i in range(3):
            for j in range(3):
                c[i][j] = k * c[i][j] + v[i] * v[j]
        if self.n % 25 == 0 or self.axis is None:
            e = list(self.axis or (1.0, 0.3, 0.2))
            for _ in range(8):                              # power iteration -> main axis
                e = [sum(c[i][j] * e[j] for j in range(3)) for i in range(3)]
                m = math.sqrt(sum(q * q for q in e)) or 1.0
                e = [q / m for q in e]
            if self.axis and sum(a * b for a, b in zip(e, self.axis)) < 0:
                e = [-q for q in e]                         # keep the same sign as before
            self.axis = e
        sw = x * self.axis[0] + y * self.axis[1] + z * self.axis[2]
        # forward swing is the biggest, sharpest peak: flip the sign if the negative side is bigger
        self.pos_peak = max(self.pos_peak * 0.998, sw)
        self.neg_peak = max(self.neg_peak * 0.998, -sw)
        if self.neg_peak > 1.3 * self.pos_peak and self.neg_peak > SWING_START_DPS:
            self.axis = [-q for q in self.axis]
            self.pos_peak, self.neg_peak = self.neg_peak, self.pos_peak
            sw = -sw
        return sw

    def _finish_stride(self, end):
        """Vertical travel of the ankle from the last calm moment to `end` (meters)."""
        a = self.av[:end + 1]
        self.av = self.av[end:]
        self.calm_val, self.calm_i, self.swung = 1e9, 0, False
        n = len(a)
        dz = 0.0
        if n > 2:
            v, vel = [], 0.0
            for x in a:
                vel += x * DT
                v.append(vel)
            drift = v[-1]
            dz = sum(v[i] - drift * (i + 1) / n for i in range(n)) * DT   # ankle is ~still at both ends
        cls = "up" if dz > STAIR_DZ else "down" if dz < -STAIR_DZ else "flat"
        self.last_stride_n = self.n
        # stairs label for the screen: 2 stair strides in a row to enter, 2 flat to leave
        if cls == "flat":
            self.flat_run += 1; self.stair_run = 0
            if self.flat_run >= 2:
                self.stairs = None
        else:
            self.stair_run += 1; self.flat_run = 0
            if self.stair_run >= 2:
                self.stairs = cls
        return {"type": "stride", "dz": round(dz, 3), "cls": cls,
                "dur": round(n / FS, 2), "peak": round(self.swing_peak)}   # kept for training a stairs detector later

    def _update_activity(self, events):
        turn = sum(self.recent_yaw)
        if self.still_n >= int(STILL_S * FS):
            self.activity = "Standing still"
            self.stairs = None
        elif abs(turn) > 30:
            self.activity = "Turning left" if turn > 0 else "Turning right"
        elif self.n - self.last_stride_n < 2.5 * FS or self.in_swing:
            self.activity = "Walking straight"
