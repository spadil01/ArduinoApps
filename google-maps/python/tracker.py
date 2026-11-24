"""
tracker.py: works out where you are on the route.

It only ever gets two kinds of news from detector.py:
  on_stride(cls)  one stride happened; cls = "flat", "up" or "down" (did the ankle go up/down a stair?)
  on_turn(dir)    the body turned left/right
and the route (route.json) tells it what SHOULD happen next. A turn or a staircase only counts
if the route expects one nearby, so one wrong detection can't throw the whole walk off.

Where you are = which part of the route ("leg", 0..8) + how far into it ("d", meters).
Distance comes from strides x stride length. Stride length starts as a guess from height
and is corrected every time you reach a turn at the end of a long hallway.

Expert walks: the expert's taps decide the landmarks (set_landmark); strides only move the dot.
Re-sync from the corrector uses set_landmark too.
"""

import copy
import math

STEP_PER_HEIGHT = 0.36       # first guess: step length ~ 0.36 x height. People following a phone walk with
                             # shorter steps (0.32-0.42 x height measured); a guess on the short side makes
                             # turn warnings come a little late instead of far too early. Corrected at the first turn.
TURN_WINDOW_M = 2.5          # accept a turn this close to where the route has one (or 35 % of the hallway)
TURN_WINDOW_FRAC = 0.35
STAIRS_WINDOW_M = 3.0        # same idea for the start of stairs
SMALL_STAIRS = 6             # stairs this short can't be detected reliably: assume them
BIG_STAIRS_DONE = 0.75       # big stairs end only after ~75 % of the steps (so the landing doesn't end them)
MISSED_M = 3.0               # walked this far past a turn: "you may have passed the turn"
USE_STAIR_DZ = False         # True = also trust the detector's up/down reading of each stride.
                             # Off: an ankle sensor can't tell stairs from flat reliably yet, so stairs
                             # are placed by distance and counted by strides (2 steps per stride).


class Tracker:
    def __init__(self, route, height_cm, auto=True):
        self.legs = route["legs"]
        self.step_m = STEP_PER_HEIGHT * height_cm / 100.0
        self.scale = 1.0            # learned correction to the step length
        self.auto = auto            # False for expert walks: only taps move between landmarks
        self.i = 0                  # current leg
        self.d = 0.0                # meters into the current leg
        self.stair_strides = 0      # strides counted on the current stairs
        self.flat_run = 0           # flat strides in a row while on stairs
        self.leg_strides = 0        # strides since this leg started
        self.missed = False
        self.at_end = False
        self._undo = []

    # ---------------- helpers ----------------
    @property
    def stride_m(self):
        return 2.0 * self.step_m * self.scale

    def _leg(self, i=None):
        return self.legs[self.i if i is None else i]

    def _remaining(self):
        leg = self._leg()
        return max(0.0, leg["m"] - self.d) if leg["kind"] == "walk" else leg["m"] * (1 - self._stairs_frac())

    def _stairs_frac(self):
        leg = self._leg()
        return min(1.0, 2.0 * self.stair_strides / max(1, leg.get("steps", 1)))

    def _advance(self, how, carry=0.0):
        """Landmark at the end of the current leg reached; move to the next leg."""
        idx = self.i
        if self.i >= len(self.legs) - 1:
            self.at_end = True
            return {"type": "landmark", "idx": idx, "id": self._leg()["end"]["id"], "how": how}
        self.i += 1
        self.d = min(max(carry, 0.0), self._leg()["m"]) if self._leg()["kind"] == "walk" else 0.0
        self.stair_strides = self.flat_run = self.leg_strides = 0
        self.missed = False
        return {"type": "landmark", "idx": idx, "id": self.legs[idx]["end"]["id"], "how": how}

    def _learn_scale(self, leg_m):
        """Reached the end of a long hallway: compare the distance we counted with the real one."""
        if leg_m >= 5.0 and self.d > 1.0:
            ratio = max(0.6, min(1.6, leg_m / self.d))
            self.scale = max(0.6, min(1.6, self.scale * ratio ** 0.7))

    # ---------------- news from the detector ----------------
    def on_stride(self, cls):
        """cls: 'flat', 'up' or 'down'. Returns a list of notes (landmarks reached, things ignored)."""
        notes = []
        if self.at_end:
            return notes
        leg = self._leg()
        self.leg_strides += 1

        if leg["kind"] == "walk":
            end = leg["end"]
            # walking into the stairs the route expects
            if (self.auto and USE_STAIR_DZ and end["type"] == "stairs_start" and cls == end["dir"]
                    and self._remaining() <= max(STAIRS_WINDOW_M, 0.5 * leg["m"])):
                notes.append(self._advance("stairs"))
                self.stair_strides = 1
                return notes
            # stairs again right after we thought they ended (e.g. a landing): go back onto them
            prev = self.legs[self.i - 1] if self.i else None
            if (self.auto and USE_STAIR_DZ and prev and prev["kind"] == "stairs" and cls == prev["dir"]
                    and self.d < 2.0 and prev["steps"] > SMALL_STAIRS):
                self.i -= 1
                self.stair_strides = max(1, int(BIG_STAIRS_DONE * prev["steps"] / 2) - 1)
                self.flat_run = 0
                notes.append({"type": "back_on_stairs", "idx": self.i})
                return notes
            if USE_STAIR_DZ and cls != "flat" and end["type"] != "stairs_start":
                notes.append({"type": "ignored", "what": "stairs_" + cls, "why": "no stairs expected here"})
            self.d += self.stride_m
            if not self.auto:
                return notes
            if end["type"] == "stairs_start":
                nxt = self._leg(self.i + 1)
                if (nxt["steps"] <= SMALL_STAIRS or not USE_STAIR_DZ) and self.d >= leg["m"]:
                    notes.append(self._advance("reached by distance"))
                    self.stair_strides = 0                 # the stride that got here was still on flat ground
                elif self.d > leg["m"] + STAIRS_WINDOW_M + 1:
                    self.missed = True
            elif end["type"] == "turn" and self.d > leg["m"] + max(MISSED_M, 0.3 * leg["m"]) \
                    and not (prev and prev["kind"] == "stairs"):   # right after stairs: too unsure to warn
                self.missed = True
            elif end["type"] == "arrive" and self.d >= leg["m"]:
                self.d = leg["m"]
                notes.append(self._advance("distance"))
            return notes

        # ---- on stairs ----
        if not USE_STAIR_DZ or cls == leg["dir"]:
            self.stair_strides += 1
            self.flat_run = 0
        else:
            self.flat_run += 1
        if not self.auto:
            return notes
        done = 2.0 * self.stair_strides / leg["steps"]
        small = leg["steps"] <= SMALL_STAIRS
        if not USE_STAIR_DZ:
            # short stairs: done once the steps are counted. Big stairs: people go down at very
            # different speeds, so counting can end them halfway; they end at the turn at the bottom
            # (on_turn), or only after twice the expected strides as a backstop.
            if done >= (1.0 if small else 2.0):
                notes.append(self._advance("stairs counted"))
        elif (small and (done >= 0.5 or self.leg_strides >= math.ceil(leg["steps"] / 2)) and (self.flat_run >= 1 or done >= 1.0)) \
                or (not small and done >= BIG_STAIRS_DONE and self.flat_run >= 2):
            notes.append(self._advance("stairs ended", carry=self.flat_run * self.stride_m))
        return notes

    def on_turn(self, direction):
        notes = []
        if self.at_end or not self.auto:
            return notes
        # look ahead: the matching turn may be at the end of this leg, or just past a short leg / the
        # end of the stairs (a turn at the bottom proves the stairs are over)
        dist = self._remaining()
        j = self.i
        while j < len(self.legs):
            leg = self.legs[j]
            end = leg["end"]
            window = max(TURN_WINDOW_M, TURN_WINDOW_FRAC * leg["m"])
            if end["type"] == "turn":
                if end["dir"] == direction and dist <= window:
                    if j == self.i:
                        self._learn_scale(leg["m"])
                    while self.i < j:
                        notes.append(self._advance("skipped past"))
                    notes.append(self._advance("turn"))
                    return notes
                break                                   # can't skip past a different turn
            if end["type"] == "arrive":
                break
            j += 1
            if j < len(self.legs) and self.legs[j]["kind"] == "walk":   # stairs length is unsure: don't count it
                dist += self.legs[j]["m"]
            if dist > 6.0:
                break
        notes.append({"type": "ignored", "what": "turn_" + direction, "why": "no such turn expected here"})
        return notes

    # ---------------- taps from the phone ----------------
    def set_landmark(self, idx, how):
        """Expert tap or corrector re-sync: 'I'm at landmark idx right now'."""
        self._undo.append(copy.deepcopy(self.__dict__))
        self._undo = self._undo[-20:]
        if idx == self.i and self._leg()["kind"] == "walk" and self._leg()["end"]["type"] == "turn":
            self._learn_scale(self._leg()["m"])
        self.i = max(0, min(idx, len(self.legs) - 1))
        self.d = self._leg()["m"] if self._leg()["kind"] == "walk" else 0.0
        self.at_end = False                       # a re-sync can move you back from "arrived"
        return self._advance(how)

    def undo(self):
        if self._undo:
            state = self._undo.pop()
            keep = self._undo
            self.__dict__.update(state)
            self._undo = keep
            return True
        return False

    # ---------------- for the phone and the log ----------------
    def summary(self):
        leg = self._leg()
        before = sum(l["m"] for l in self.legs[:self.i])
        here = leg["m"] * self._stairs_frac() if leg["kind"] == "stairs" else min(self.d, leg["m"])
        rem = self._remaining()
        steps_to = round(rem / (self.stride_m / 2)) if leg["kind"] == "walk" \
            else max(0, leg["steps"] - 2 * self.stair_strides)
        return {
            "leg": self.i,
            "d": round(here, 2),
            "progress_m": round(before + here, 1),
            "total_m": round(sum(l["m"] for l in self.legs), 1),
            "steps_to": steps_to,
            "missed": self.missed,
            "at_end": self.at_end,
            "step_m": round(self.stride_m / 2, 3),
        }
