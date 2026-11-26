"""Lab recordings plus home nights in one model, as a morning (whole-night) and a live (past-only) version.

Run from a folder holding features/ (lab, from modal_home_check.py) and one feature file per home night:
    python home_night_combined.py night-a.npz night-b.npz
"""
import sys, glob, hashlib, warnings
import numpy as np
from home_night_train import context, roll
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
warnings.filterwarnings("ignore")

def live(x):
    """Past-only version: quiet level and context use only what has been heard so far."""
    out = np.empty_like(x); 
    for i in range(len(x)):
        seen = x[max(0, i - 239): i + 1]
        out[i] = (x[i] - np.median(seen, 0)) / (seen.std(0) + 1e-3 if i > 9 else 1.0)
    out = np.clip(out, -8, 8)
    past = lambda z, n, fn: np.array([fn(z[max(0, i - n): i + 1], axis=0) for i in range(len(z))])
    core = out[:, 48:]
    return np.hstack([out, past(out, 10, np.mean), past(out, 40, np.mean), past(core, 10, np.std), past(core, 40, np.std)])

def load(path, mode):
    d = np.load(path); k = d["y"] >= 0
    x = context(d["x"], d["hours"], False) if mode == "morning" else live(d["x"])
    return x[k], d["y"][k]

files = sorted(glob.glob("features/*.npz"))
people = sorted({f.split("/")[-1].split("-")[0] for f in files})
val = set(sorted(people, key=lambda p: hashlib.sha256(f"teacher-v12:{p}".encode()).hexdigest())[:40])
nights = sys.argv[1:3]
for mode in ("morning", "live"):
    lab = [(f.split("/")[-1].split("-")[0] in val, load(f, mode)) for f in files]
    cat = lambda g: (np.concatenate([x for x, _ in g]), np.concatenate([y for _, y in g]))
    xt, yt = cat([v for iv, v in lab if not iv]); xv, yv = cat([v for iv, v in lab if iv])
    home = {n: load(n, mode) for n in nights}
    def cw(y):
        c = np.bincount(y, minlength=4).astype(float); return (len(y) / (4 * np.maximum(c, 1)))[y]
    def fit(groups):  # groups: list of (x, y, share)
        x = np.concatenate([g[0] for g in groups]); y = np.concatenate([g[1] for g in groups])
        w = np.concatenate([cw(g[1]) * g[2] / len(g[1]) for g in groups]); w *= len(y) / w.sum()
        return LogisticRegression(C=0.02, max_iter=2000).fit(x, y, sample_weight=w)
    def show(tag, m, x, y):
        p = m.predict_proba(x); pred = p.argmax(1)
        r = recall_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
        print(f"  {tag:34s} balanced {balanced_accuracy_score(y, pred):.2f}  REM AUC {roc_auc_score(y == 3, p[:, 3]):.2f}  awake {r[0]:.0%} light {r[1]:.0%} deep {r[2]:.0%} REM {r[3]:.0%}")
    print("==", mode)
    m = fit([(xt, yt, 1)]); show("lab only -> lab validation", m, xv, yv)
    for test, other in ((nights[1], nights[0]), (nights[0], nights[1])):
        xo, yo = home[other]; x, y = home[test]
        print(f" test on {test}")
        show("lab only", m, x, y)
        show("other home night only", fit([(xo, yo, 1)]), x, y)
        for share in (0.2, 0.5):
            show(f"lab + other home night ({int(share*100)}%)", fit([(xt, yt, 1 - share), (xo, yo, share)]), x, y)
