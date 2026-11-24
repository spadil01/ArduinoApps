"""
navigator.py: decides what instruction to show ("Turn left in ~6 steps", "Then stairs down").

Two ways to decide the next move:
  1. model.json (behavior cloning): a decision tree / small forest trained on the laptop from the
     expert's walks plus the correctors' DAgger corrections. Used when the file exists.
  2. The route itself (route.json). Used until a model has been trained, and as a fallback.

The decision is one of ACTIONS. The words on screen are built from it plus the tracker's
"steps to the next landmark".

`features()` is the exact list of numbers the model sees. train_navigator.py (laptop) must use
the same function, so this file is copied there too.
"""

import json
import os

ACTIONS = ["straight", "left", "right", "stairs_down", "stairs_up", "arrive"]

# what the model is allowed to look at (all from the tracker; nothing it can't know while walking)
FEATURES = [
    "leg",                # which part of the route (0..8)
    "d",                  # meters into this part
    "remaining_m",        # meters left in this part (by the route)
    "frac",               # how far through this part, 0..1
    "turns_done",         # turns passed so far
    "stairs_done",        # staircases passed so far
    "last_was_turn",      # 1 if the last landmark passed was a turn
    "last_was_stairs",    # 1 if it was the end of stairs
    "step_m",             # this walker's step length estimate
    "missed",             # 1 if the tracker thinks a turn was missed
]

SYM = {"left": "↰", "right": "↱", "stairs_down": "⤵", "stairs_up": "⤴"}
MIN_CONFIDENCE = 0.6     # model must have at least this share of its trees agreeing, else use the route
NEAR_STEPS = 6          # farther than this: show "Go straight" (the turn is written underneath)


def route_action(route, s):
    """What the route says the next move is, for tracker summary s."""
    legs = route["legs"]
    leg = legs[min(s["leg"], len(legs) - 1)]
    end = leg["end"]
    if s.get("at_end"):
        return "arrive"
    if leg["kind"] == "stairs":
        return "stairs_" + leg["dir"]
    if end["type"] == "turn":
        return end["dir"]
    if end["type"] == "stairs_start":
        return "stairs_" + end["dir"]
    return "arrive" if s.get("steps_to", 99) <= 1 else "straight"


def plausible_actions(route, s):
    """Moves that make sense here: this part's move or the next part's (the model decides WHEN to
    switch). Anything else (e.g. 'stairs down' at the start) means the model is guessing."""
    legs = route["legs"]
    i = min(s["leg"], len(legs) - 1)
    ok = {route_action(route, s), route_action(route, dict(s, leg=min(i + 1, len(legs) - 1), d=0.0, steps_to=99))}
    if legs[i]["end"]["type"] == "arrive" or i == len(legs) - 1:
        ok |= {"straight", "arrive"}
    return ok


def features(route, s):
    legs = route["legs"]
    i = min(s["leg"], len(legs) - 1)
    m = legs[i]["m"]
    passed = [l["end"]["type"] for l in legs[:i]]
    d = s["d"]
    if legs[i]["kind"] == "stairs":
        d = 0.0          # how far along the stairs isn't reliable (people take them at very different
                         # speeds), so the model only knows "on the stairs"; the turn at the bottom moves it on
    return {
        "leg": i,
        "d": d,
        "remaining_m": max(0.0, m - d),
        "frac": min(1.0, d / m) if m else 1.0,
        "turns_done": passed.count("turn"),
        "stairs_done": passed.count("stairs_end"),
        "last_was_turn": int(bool(passed) and passed[-1] == "turn"),
        "last_was_stairs": int(bool(passed) and passed[-1] == "stairs_end"),
        "step_m": s.get("step_m", 0.7),
        "missed": int(bool(s.get("missed"))),
    }


class TreeModel:
    """Decision tree or random forest saved as JSON:
    {"features": [...], "classes": [...], "trees": [{"f": [...], "t": [...], "l": [...], "r": [...], "v": [[...]]}]}"""

    def __init__(self, spec):
        self.features, self.classes, self.trees = spec["features"], spec["classes"], spec["trees"]

    def predict(self, x):
        vec = [x[k] for k in self.features]
        votes = [0.0] * len(self.classes)
        for t in self.trees:
            n = 0
            while t["l"][n] != -1:
                n = t["l"][n] if vec[t["f"][n]] <= t["t"][n] else t["r"][n]
            v = t["v"][n]
            tot = sum(v) or 1.0
            for k, c in enumerate(v):
                votes[k] += c / tot
        best = max(range(len(votes)), key=lambda k: votes[k])
        return self.classes[best], votes[best] / len(self.trees)


class Navigator:
    def __init__(self, route, model_path=None):
        self.route = route
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path) as f:
                    self.model = TreeModel(json.load(f))
                print(f"Navigator: using trained model {os.path.basename(model_path)}", flush=True)
            except Exception as e:
                print("Navigator: could not load model, using the route:", e, flush=True)
        if self.model is None:
            print("Navigator: no trained model yet, using the route", flush=True)

    def decide(self, s):
        """s = tracker.summary(). Returns the decision and the words for the screen."""
        by_route = route_action(self.route, s)
        action, source, conf = by_route, "route", 1.0
        if self.model is not None and not s.get("at_end"):
            try:
                m_action, m_conf = self.model.predict(features(self.route, s))
                if m_conf >= MIN_CONFIDENCE and m_action in plausible_actions(self.route, s):
                    action, conf, source = m_action, m_conf, "model"
                else:
                    conf, source = m_conf, "route (model unsure)"
            except Exception as e:
                print("Navigator: model failed, using the route:", e, flush=True)
        words = self.words(action, s, action == by_route)
        return dict(words, action=action, route_action=by_route, source=source, confidence=round(conf, 2))

    # ---------- words on the screen ----------
    def words(self, action, s, know_distance):
        legs = self.route["legs"]
        i = min(s["leg"], len(legs) - 1)
        leg = legs[i]
        steps = s.get("steps_to", 0) if know_distance else 0
        dest = self.route["dest"]
        sub, then = "", None

        if action == "arrive":
            if s.get("at_end") or steps <= 1:
                return {"ar": "flag", "say": "You have arrived", "sub": f"Welcome to {dest}", "then": None}
            say, sub, ar = f"{dest} in ~{steps} steps", "keep going straight", "straight"
        elif action == "straight":
            if leg["end"]["type"] == "arrive":
                steps = s.get("steps_to", 0)                 # same place either way: the steps are known
            say, ar = (f"{dest} in ~{steps} steps" if leg["end"]["type"] == "arrive" else "Go straight"), "straight"
            if leg["end"]["type"] == "arrive" and steps <= 1:
                say = f"{dest} is right here"
            sub = "keep going straight" if leg["end"]["type"] == "arrive" else ""
        elif action in ("left", "right"):
            ar = action
            if not know_distance or steps < 1:
                say = f"Turn {action}"
            else:
                say = f"Turn {action} in ~{steps} step{'s' if steps != 1 else ''}"
            if steps > NEAR_STEPS:                           # far away: keep walking straight for now
                return {"ar": "straight", "say": "Go straight",
                        "sub": f"{SYM[action]} Turn {action} in ~{steps} steps", "then": None}
        else:                                                # stairs
            way = action.split("_")[1]
            ar = way
            if leg["kind"] == "stairs":
                say, sub = f"Go {way} the stairs", f"{leg['steps']} steps"
                nxt_leg = legs[i + 1] if i + 1 < len(legs) else None
                if nxt_leg and nxt_leg["kind"] == "walk":
                    # what comes after the stairs: a few steps forward first (you can't turn on the last step)
                    n = max(1, round(nxt_leg["m"] / max(s.get("step_m", 0.65), 0.3)))
                    fwd = f"{n} step{'s' if n > 1 else ''} forward"
                    end_word = "bottom" if way == "down" else "top"
                    if nxt_leg["end"]["type"] == "turn" and nxt_leg["m"] < 3:
                        t_dir = nxt_leg["end"]["dir"]
                        if s.get("d", 0) >= leg["m"]:           # all steps counted: you're at the bottom
                            return {"ar": "straight", "say": f"At the {end_word}: {fwd}, then turn {t_dir}",
                                    "sub": "", "then": None}
                        return {"ar": ar, "say": say, "sub": sub,
                                "then": f"Then: {fwd}, then turn {SYM[t_dir]} {t_dir}"}
                    if nxt_leg["end"]["type"] == "arrive":
                        return {"ar": ar, "say": say, "sub": sub,
                                "then": f"Then: ~{n} steps straight to {dest}"}
            else:
                if not know_distance or steps < 1:
                    say = f"Take the stairs {way}"
                else:
                    say = f"Stairs {way} in ~{steps} step{'s' if steps != 1 else ''}"
                if steps > NEAR_STEPS:
                    return {"ar": "straight", "say": "Go straight",
                            "sub": f"{SYM[action]} Stairs {way} in ~{steps} steps", "then": None}

        # a second move coming right after this one
        nxt = legs[i + 1] if i + 1 < len(legs) else None
        if nxt and know_distance and (nxt["m"] < 3 or leg["kind"] == "stairs"):
            if nxt["kind"] == "stairs":
                if action != "stairs_" + nxt["dir"]:          # don't repeat "stairs up ... then stairs up"
                    then = f"Then {SYM['stairs_' + nxt['dir']]} stairs {nxt['dir']}"
            elif nxt["end"]["type"] == "turn":
                again = " again" if action == nxt["end"]["dir"] else ""
                then = f"Then {SYM[nxt['end']['dir']]} turn {nxt['end']['dir']}{again}"
            elif nxt["end"]["type"] == "stairs_start":
                then = f"Then {SYM['stairs_' + nxt['end']['dir']]} stairs {nxt['end']['dir']}"
            elif nxt["end"]["type"] == "arrive":
                then = f"Then {dest}"
        return {"ar": ar, "say": say, "sub": sub, "then": then}
