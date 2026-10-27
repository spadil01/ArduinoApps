# Trains the models in a separate program, started by main.py when the page
# presses Train. Kept apart so the heavy maths can't hold up main.py's live
# sensor stream. Saves <models dir>/<spot>.pkl and prints one JSON line per spot.
#
#   python train_worker.py <data dir> <models dir>
import json
import os
import pickle
import sys
import time

import gait_model as gm
import recorder

data_dir, models_dir = sys.argv[1:3]
os.makedirs(models_dir, exist_ok=True)
recordings = gm.load_recordings(data_dir)
for spot in recorder.SPOTS:
    t = time.time()
    try:
        result = gm.train_spot(recordings, spot)
    except ValueError as e:
        print(json.dumps({"spot": spot, "error": str(e)}), flush=True)
        continue
    result["report"]["trained"] = time.strftime("%Y-%m-%d %H:%M")
    path = os.path.join(models_dir, f"{spot}.pkl")
    with open(path + ".tmp", "wb") as f:
        pickle.dump(result, f)
    os.replace(path + ".tmp", path)  # main.py never sees a half-written file
    print(json.dumps({"spot": spot, "seconds": round(time.time() - t, 1)}), flush=True)
