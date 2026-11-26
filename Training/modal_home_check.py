"""Does the home-night REM clue (a long quiet, still stretch) hold on the sleep-lab people?

Pulls the same simple sound features used on the home nights from every PSG-Audio recording, then checks three things:
the same small model trained and tested inside the lab, a home-trained model tested on lab people, and a lab-trained
model tested on the home nights.

    modal run modal_home_check.py                 # features for every recording, then the report
    modal run modal_home_check.py --stage report  # the report only
Home-night feature files (from home_night_features.py) go on the volume under root/nights/features/.
"""
import modal
from modal_ears import image, volume

app = modal.App("sleep-home-check", image=image.add_local_python_source(
    "modal_ears", "modal_distill", "home_night_features", "home_night_train"))
ROOT, RUN = "/vol/root", "/vol/root/runs/home-check-v1"


@app.function(volumes={"/vol": volume}, cpu=2, memory=8192, timeout=3600, max_containers=60)
def features(recording: str):
    import os
    import subprocess
    from pathlib import Path
    import numpy as np
    import home_night_features as hf
    out = f"{RUN}/features/{recording}.npz"
    if os.path.exists(out):
        return recording
    per = int(30 * hf.FPS)
    mels, centroid, flat, stage, usable, starts = [], [], [], [], [], []
    for path in sorted(Path(f"{ROOT}/data/blocks").glob(f"{recording}[[]*.npz")):
        with np.load(path, allow_pickle=False) as b:
            rows, audio_file = len(b["stage"]), str(b["audioFile"])
            stage.append(b["stage"].astype(np.int64)); usable.append(b["usable"].astype(bool))
        pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{ROOT}/data/audio/{audio_file}", "-f", "f32le",
                              "-ac", "1", "-ar", "16000", "-"], check=True, capture_output=True).stdout
        audio = np.frombuffer(pcm, dtype=np.float32)
        audio = np.pad(audio, (0, max(0, rows * 480000 + hf.N - len(audio))))[: rows * 480000 + hf.N]
        m, c, f = hf.frames(audio, 1.0)
        base = sum(len(x) for x in mels)
        mels.append(m[: rows * per]); centroid.append(c[: rows * per]); flat.append(f[: rows * per])
        starts += [base + i * per for i in range(rows)]
    mels = np.concatenate(mels)
    mels = mels - np.median(mels, axis=0)  # relative to this recording's own quiet level, as at home
    x = hf.epoch_rows(mels, np.concatenate(centroid), np.concatenate(flat), starts)
    y = np.concatenate(stage)
    y[~np.concatenate(usable)] = -1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, x=x, y=y, hours=np.arange(len(y)) / 120.0)
    volume.commit()
    return recording


@app.function(volumes={"/vol": volume}, cpu=8, memory=32768, timeout=3600)
def report():
    import json
    import warnings
    from pathlib import Path
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
    from home_night_train import context
    from modal_distill import blocks_and_split
    warnings.filterwarnings("ignore")
    _, validation = blocks_and_split()

    def load(path):
        d = np.load(path)
        keep = d["y"] >= 0
        return context(d["x"], d["hours"], False)[keep], d["y"][keep]

    lab = {p.stem: load(p) for p in sorted(Path(f"{RUN}/features").glob("*.npz"))}
    home = {p.stem: load(p) for p in sorted(Path(f"{ROOT}/nights/features").glob("*.npz"))}
    is_val = lambda name: name.split("-")[0] in validation
    stack = lambda group: (np.concatenate([x for x, _ in group]), np.concatenate([y for _, y in group]))
    xt, yt = stack([v for k, v in lab.items() if not is_val(k)])
    xv, yv = stack([v for k, v in lab.items() if is_val(k)])
    xh, yh = stack(list(home.values()))

    def scores(y, proba):
        pred = proba.argmax(1)
        return {"remAUC": round(float(roc_auc_score(y == 3, proba[:, 3])), 3),
                "deepAUC": round(float(roc_auc_score(y == 2, proba[:, 2])), 3),
                "awakeAUC": round(float(roc_auc_score(y == 0, proba[:, 0])), 3),
                "balanced": round(float(balanced_accuracy_score(y, pred)), 3),
                "recall": recall_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0).round(3).tolist()}

    simple = lambda: LogisticRegression(C=0.02, max_iter=2000, class_weight="balanced")
    trees = lambda: HistGradientBoostingClassifier(max_depth=6, max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    out = {"recordings": len(lab), "validationRecordings": sum(is_val(k) for k in lab),
           "rows": {"labTrain": len(yt), "labValidation": len(yv), "home": len(yh)},
           "labStages": np.bincount(yv, minlength=4).tolist()}
    lab_simple, lab_trees, home_simple = simple().fit(xt, yt), trees().fit(xt, yt), simple().fit(xh, yh)
    out["lab to lab, simple"] = scores(yv, lab_simple.predict_proba(xv))
    out["lab to lab, trees"] = scores(yv, lab_trees.predict_proba(xv))
    out["home to lab, simple"] = scores(yv, home_simple.predict_proba(xv))
    for name, (x, y) in home.items():
        out[f"lab to home {name}, simple"] = scores(y, lab_simple.predict_proba(x))
        out[f"lab to home {name}, trees"] = scores(y, lab_trees.predict_proba(x))
    # Each feature on its own: does REM pull it the same way in the lab as at home?
    single = []
    for i in range(xv.shape[1]):
        single.append([i, round(float(roc_auc_score(yv == 3, xv[:, i])), 3), round(float(roc_auc_score(yh == 3, xh[:, i])), 3)])
    out["singleFeatureRemAUC"] = single
    per_person = [roc_auc_score(y == 3, lab_simple.predict_proba(x)[:, 3]) for k, (x, y) in lab.items()
                  if is_val(k) and 0 < (y == 3).sum() < len(y)]
    out["labValidationPerPersonRemAUC"] = {"median": round(float(np.median(per_person)), 3), "people": len(per_person),
                                           "shareAboveHalf": round(float(np.mean(np.array(per_person) > 0.5)), 3)}
    with open(f"{RUN}/report.json", "w") as f:
        json.dump(out, f, indent=1)
    volume.commit()
    return out


@app.local_entrypoint()
def main(stage: str = "all", limit: int = 0):
    import json
    if stage == "all":
        names = sorted({e.path.split("/")[-1].split("[")[0] for e in volume.listdir("root/data/blocks")})
        names = names[:limit] if limit else names
        for done, name in enumerate(features.map(names, order_outputs=False), 1):
            if done % 20 == 0 or done == len(names): print(f"{done}/{len(names)} recordings", flush=True)
    if not limit:
        result = report.remote()
        result.pop("singleFeatureRemAUC")
        print(json.dumps(result, indent=1))
