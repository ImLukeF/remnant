"""Train on one home night's Watch stages, test on the other."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score

def roll(x, half, fn):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - half): i + half + 1], axis=0)
    return out

def context(x, hours, clock):
    x = (x - np.median(x, 0)) / (x.std(0) + 1e-6)          # each night scaled to itself
    core = x[:, 48:]                                        # loudness, movement and breathing
    parts = [
        x,
        roll(x, 5, np.mean), roll(x, 20, np.mean),
        roll(core, 5, np.std), roll(core, 20, np.std),
        roll(core, 50, np.mean), roll(core, 50, np.std),
        roll(core, 100, np.mean), roll(core, 100, np.std),
    ]
    if clock:
        parts.append(hours[:, None])
    return np.hstack(parts)


def smooth_probabilities(proba, width=5):
    if width <= 1 or len(proba) == 0:
        return proba
    half = width // 2
    out = np.empty_like(proba)
    for i in range(len(proba)):
        out[i] = proba[max(0, i - half) : min(len(proba), i + half + 1)].mean(axis=0)
    return out

def build(path, clock):
    d = np.load(path)
    return context(d["x"], d["hours"], clock), d["y"], d["hours"], d["phone"]

def score(name, y, proba):
    pred = proba.argmax(1)
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    print(f"  {name:26s} REM AUC {roc_auc_score(y == 3, proba[:, 3]):.2f}  balanced {balanced_accuracy_score(y, pred):.2f}"
          f"  awake {rec[0]:.0%} light {rec[1]:.0%} deep {rec[2]:.0%} REM {rec[3]:.0%}  REM minutes called {int((pred == 3).sum() / 2)}")

if __name__ == "__main__":
    pairs = [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
    for train, test in pairs:
        print(train.split("/")[-1], "->", test.split("/")[-1])
        for clock in (False, True):
            xa, ya, ha, _ = build(train, clock)
            xb, yb, hb, phone = build(test, clock)
            ka, kb = ya >= 0, yb >= 0
            tag = "sound + clock" if clock else "sound only"
            lr = LogisticRegression(C=0.02, max_iter=3000, class_weight="balanced").fit(xa[ka], ya[ka])
            score(f"{tag}, simple", yb[kb], lr.predict_proba(xb[kb]))
            gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, class_weight="balanced", random_state=0).fit(xa[ka], ya[ka])
            score(f"{tag}, trees", yb[kb], gb.predict_proba(xb[kb]))
        lr = LogisticRegression(class_weight="balanced").fit(ha[ka, None], ya[ka])
        score("clock only", yb[kb], lr.predict_proba(hb[kb, None]))
        one = np.eye(4)[np.clip(phone[kb], 0, 3)]
        score("phone model today", yb[kb], one)
