# Turns recordings into models and models into guesses. No neural nets: only
# decision trees (one tree, or a "random forest" of many) and k-nearest neighbours.
#
# A recording is cut into WINDOW_S-second slices. Each slice is boiled down to a
# few numbers ("features"), and the models learn which numbers go with which activity.
#
#   Model 1 ("peak"): ONE number -- the biggest shove along one direction (X, Y
#                     or Z). Training tries all three and keeps the best.
#   Model 2 ("all"):  ~20 numbers from all six sensor readings.
#
# Each model is built two ways -- "tree" and "knn" -- so we can compare them.
#
# Any model can also answer "other" instead of one of the four:
#   "split":      its vote is too divided (the top answer gets < AGREE_MIN of it)
#   "unfamiliar": the movement is far from everything it learned. "Far" is
#                 measured like KNN does (distance to the nearest examples) and
#                 compared to how far real activities from a body it never
#                 trained on usually are (from the fair test). A strong vote
#                 (>= SURE_MIN) is trusted unless it's VERY_FAR times farther still,
#                 because new people always look a bit unlike the trainer.
#
# Fair testing: models are always scored on recordings they were NOT trained
# on, grouped so the test is like meeting someone new (e.g. train on the left
# leg, test on the right; later, leave out one whole person).
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RATE_HZ = 50
WINDOW_S = 2
WINDOW = WINDOW_S * RATE_HZ
TRAIN_HOP = RATE_HZ // 2  # a new training slice every 0.5 s (they overlap)
ACTIVITIES = ["stand", "walk", "run", "bike"]
AXES = ["X", "Y", "Z"]
KINDS = ["tree", "knn"]
MODEL_NAMES = ["peak-tree", "peak-knn", "all-tree", "all-knn"]
TOP_FEATURES = 6  # how many of Model 2's clues the page shows
OTHER = "other"
AGREE_MIN = 0.6   # default: "other" when the top answer has less of the vote than this
FAR_FACTOR = 1.0  # default: "other" when this many times farther than usual from the examples
SURE_MIN = 0.8   # a vote this strong only becomes "unfamiliar" when...
VERY_FAR = 3.0   # ...it's this many times past the far setting
USUAL_PCT = 99    # "usual" distance = what 99% of fair-test slices stay under


# ---------------- loading ----------------

def load_recordings(data_dir):
    """[(info dict, array of rows ms, ax, ay, az, gx, gy, gz), ...]"""
    out = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        with open(path) as f:
            meta = json.loads(f.readline()[2:])
        meta["file"] = path.name
        out.append((meta, np.loadtxt(path, delimiter=",", ndmin=2)))
    return out


def group_of(meta):
    """Which 'body' a recording belongs to, for fair testing: the person, plus
    left/right if the note says so (a different leg is a bit like a different person)."""
    note = meta.get("note", "").lower()
    side = "left" if "left" in note else "right" if "right" in note else ""
    return f"{meta['person']}-{side}" if side else meta["person"]


def slices(rows, hop=TRAIN_HOP):
    """Overlapping WINDOW-long slices of the 6 sensor columns."""
    data = rows[:, 1:7]
    return [data[i:i + WINDOW] for i in range(0, len(data) - WINDOW + 1, hop)]


# ---------------- features ----------------

def peak_features(win):
    """Model 1 candidates: the biggest shove (away from the slice's average,
    so gravity doesn't count) along each of X, Y, Z, in g."""
    acc = win[:, :3]
    return np.abs(acc - acc.mean(axis=0)).max(axis=0)


def rhythm(signal):
    """How long until the movement repeats (seconds), and how strongly it repeats
    (0 = not at all, 1 = perfectly). Found by sliding the signal over itself."""
    s = signal - signal.mean()
    energy = np.dot(s, s)
    if energy < 1e-9:
        return 0.0, 0.0
    lo, hi = int(0.25 * RATE_HZ), int(1.6 * RATE_HZ)  # steps/pedal turns every 0.25-1.6 s
    n = len(s)
    full = np.correlate(s, s, mode="full")[n - 1:]    # overlap at every shift, all at once
    lags = np.arange(lo, hi)
    corr = full[lo:hi] / energy * n / (n - lags)       # fair to long shifts (less overlap)
    best = int(np.argmax(corr))
    return lags[best] / RATE_HZ, max(0.0, float(corr[best]))


# "Tilt" (which way gravity pulls on the sensor) shows body position, not
# movement. At the waist that's a real clue: sitting on a bike tilts the hips
# differently from standing up. At the ankle it misleads: biking lying down
# (air-biking) vs sitting up points the shin completely differently.
USE_TILT = {"ankle": False, "waist": True}

ALL_FEATURE_NAMES = (
    [f"shake {a}" for a in AXES]              # how much it moves along each direction
    + [f"twist {a}" for a in AXES]            # how much it turns around each direction
    + ["total shake avg", "total shake spread", "total shake max", "total shake min",
       "total twist avg", "total twist spread", "total twist max",
       "twist / shake", "repeat time", "repeat strength",
       "jerk", "fast shaking share", "spikiness"]  # thuds: walking/running have them, pedaling doesn't
)
TILT_NAMES = [f"tilt {a}" for a in AXES]


def feature_names(spot):
    return ALL_FEATURE_NAMES + (TILT_NAMES if USE_TILT[spot] else [])


FEATURES_VERSION = 4  # bump when the features change, so old saved models aren't used


def thuds(acc_mag):
    """How 'thuddy' the movement is: (jerk = how fast the shove changes, in g/s;
    share of the shaking that is fast, above 3 per second; spikiness = 3 for
    smooth-ish random movement, higher when there are sharp spikes)."""
    jerk = np.abs(np.diff(acc_mag)).mean() * RATE_HZ
    d = acc_mag - acc_mag.mean()
    power = np.abs(np.fft.rfft(d)) ** 2
    freqs = np.fft.rfftfreq(len(d), 1 / RATE_HZ)
    fast = power[freqs >= 3].sum() / (power.sum() + 1e-9)
    spiky = (d ** 4).mean() / ((d ** 2).mean() ** 2 + 1e-9)
    return jerk, fast, spiky


def all_features(win, spot):
    """Model 2: ~20 numbers from all six readings (plus tilt where USE_TILT says).
    Signs are dropped wherever a left vs right leg (a mirrored sensor) would flip them."""
    acc, gyr = win[:, :3], win[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyr_mag = np.linalg.norm(gyr - gyr.mean(axis=0), axis=1)  # minus the sensor's drift
    period, strength = rhythm(acc_mag)
    return np.array([
        *acc.std(axis=0),
        *gyr.std(axis=0),
        acc_mag.mean(), acc_mag.std(), acc_mag.max(), acc_mag.min(),
        gyr_mag.mean(), gyr_mag.std(), gyr_mag.max(),
        gyr_mag.std() / (100 * acc_mag.std() + 1),  # (+1: stays sensible when perfectly still)
        period, strength,
        *thuds(acc_mag),
        *(np.abs(acc.mean(axis=0)) if USE_TILT[spot] else []),
    ])


def feature_table(recordings, spot):
    """Every training slice for one spot: (peak features, all features, labels, groups)."""
    P, A, y, g = [], [], [], []
    for meta, rows in recordings:
        if meta["spot"] != spot:
            continue
        for win in slices(rows):
            P.append(peak_features(win))
            A.append(all_features(win, spot))
            y.append(meta["activity"])
            g.append(group_of(meta))
    return np.array(P), np.array(A), np.array(y), np.array(g)


# ---------------- models ----------------

def new_model(which, kind):
    if kind == "knn":
        # KNN compares distances, so every number is first put on the same scale.
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    if which == "peak":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0)
    return RandomForestClassifier(n_estimators=50, min_samples_leaf=3, n_jobs=-1, random_state=0)


def neighbor_distance(knn, X):
    """How far each row is from its nearest training examples (on KNN's own scale)."""
    scaler, neighbors = knn[0], knn[-1]
    dist, _ = neighbors.kneighbors(scaler.transform(X))
    return dist.mean(axis=1)


def fair_test(X, y, groups, which, kind):
    """Train without one group, test on that group, for every group.
    Returns (accuracy 0-1, confusion counts [true][guessed], distances) --
    distances (KNN only) say how far each held-out slice was from the examples
    it was compared against: real activities on a body the model never saw."""
    confusion = np.zeros((len(ACTIVITIES), len(ACTIVITIES)), dtype=int)
    distances = []
    for held_out in np.unique(groups):
        train, test = groups != held_out, groups == held_out
        if len(np.unique(y[train])) < 2 or not test.any():
            continue
        model = new_model(which, kind).fit(X[train], y[train])
        for truth, guess in zip(y[test], model.predict(X[test])):
            confusion[ACTIVITIES.index(truth), ACTIVITIES.index(guess)] += 1
        if kind == "knn":
            distances.extend(neighbor_distance(model, X[test]))
    total = confusion.sum()
    return (np.trace(confusion) / total if total else 0.0), confusion, distances


def zones(model, values):
    """Model 1 only looks at one number, so its whole decision fits on a number
    line: [{"from", "to", "gait"}, ...] covering 0 to a bit past the biggest value seen."""
    grid = np.linspace(0, values.max() * 1.2, 400)
    labels = model.predict(grid.reshape(-1, 1))
    out = []
    for x, label in zip(grid, labels):
        if out and out[-1]["gait"] == label:
            out[-1]["to"] = float(x)
        else:
            out.append({"from": float(x), "to": float(x), "gait": str(label)})
    return out


def train_spot(recordings, spot):
    """Train and fairly test all four models for one spot.
    Returns {"peak-tree": {...}, "peak-knn": ..., "all-tree": ..., "all-knn": ...},
    each with the fitted model, its score and confusion table, plus "report":
    everything the page shows about this training (plain JSON)."""
    P, A, y, groups = feature_table(recordings, spot)
    if len(np.unique(y)) < 2:
        raise ValueError(f"need recordings of at least 2 activities on the {spot}")
    out = {}
    usual = {}  # "peak"/"all" -> usual distance from the examples (for "unfamiliar")
    for kind in KINDS:
        # Model 1: try each direction, keep the one that scores best on the fair test.
        tries = {axis: fair_test(P[:, [i]], y, groups, "peak", kind) for i, axis in enumerate(AXES)}
        axis = max(tries, key=lambda a: tries[a][0])
        i = AXES.index(axis)
        model = new_model("peak", kind).fit(P[:, [i]], y)
        if kind == "knn":
            usual["peak"] = usual_distance(tries[axis][2])
        out[f"peak-{kind}"] = {
            "model": model, "axis": axis,
            "score": tries[axis][0], "confusion": tries[axis][1],
            "axis_scores": {a: tries[a][0] for a in AXES},
            "zones": zones(model, P[:, i]),
        }
        score, confusion, distances = fair_test(A, y, groups, "all", kind)
        if kind == "knn":
            usual["all"] = usual_distance(distances)
        model = new_model("all", kind).fit(A, y)
        if kind == "tree":
            model.set_params(n_jobs=1)  # one guess at a time: extra helpers only slow it down
        out[f"all-{kind}"] = {"model": model, "score": score, "confusion": confusion}

    # Model 2's most useful clues (by the forest), and each activity's typical value.
    importance = out["all-tree"]["model"].feature_importances_
    top = [int(i) for i in np.argsort(importance)[::-1][:TOP_FEATURES]]
    out["version"] = FEATURES_VERSION
    out["spot"] = spot
    out["usual"] = usual
    out["report"] = {
        "spot": spot,
        "slices": int(len(y)),
        "groups": [str(x) for x in sorted(set(groups))],
        "activities": [a for a in ACTIVITIES if a in set(y)],
        "usual": usual,
        "models": {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in out[name].items() if k != "model"}
                   for name in MODEL_NAMES},
        "clues": [{
            "index": i, "name": feature_names(spot)[i], "importance": float(importance[i]),
            "typical": {a: float(np.median(A[y == a, i])) for a in ACTIVITIES if a in set(y)},
            "max": float(np.percentile(A[:, i], 99) * 1.1),
        } for i in top],
    }
    for m in out["report"]["models"].values():
        m["score"] = float(m["score"])
    return out


def usual_distance(distances):
    return max(float(np.percentile(distances, USUAL_PCT)), 1e-6) if len(distances) else 1e-6


def guess_all(models, win, agree_min=AGREE_MIN, far_factor=FAR_FACTOR):
    """Run all four trained models on one slice. Returns
    ({name: {"gait" (may be "other"), "best" (top of the four), "sure", "why" ("", "split"
     or "unfamiliar"), "shares": {activity: 0-1}}},
     {"peak": {axis: g}, "all": [...], "far": {"peak"/"all": distance / usual distance}})"""
    peaks = peak_features(win)
    feats = all_features(win, models["spot"])
    inputs = {"peak": peaks[[AXES.index(models["peak-knn"]["axis"])]], "all": feats}
    far = {fam: float(neighbor_distance(models[f"{fam}-knn"]["model"], x.reshape(1, -1))[0]
                      / models["usual"][fam]) for fam, x in inputs.items()}
    guesses = {}
    for name in MODEL_NAMES:
        m, fam = models[name], name.split("-")[0]
        x = peaks[[AXES.index(m["axis"])]] if fam == "peak" else feats
        probs = m["model"].predict_proba(x.reshape(1, -1))[0]
        shares = {str(a): round(float(p), 3) for a, p in zip(m["model"].classes_, probs)}
        best = max(shares, key=shares.get)
        too_far = far_factor * (VERY_FAR if shares[best] >= SURE_MIN else 1)
        why = "unfamiliar" if far[fam] > too_far else "split" if shares[best] < agree_min else ""
        guesses[name] = {"gait": OTHER if why else best, "best": best, "sure": shares[best],
                         "why": why, "shares": shares}
    return guesses, {"peak": dict(zip(AXES, peaks.round(3).tolist())),
                     "all": feats.round(4).tolist(),
                     "far": {k: round(v, 2) for k, v in far.items()}}
