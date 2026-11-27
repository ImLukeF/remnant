"""Lab recordings plus home nights in one model, as a morning (whole-night) and a live (past-only) version.

Run from a folder holding features/ (lab, from modal_home_check.py) and one feature file per home night:
    python home_night_combined.py night-a.npz night-b.npz
    python home_night_combined.py --export /path/to/remnant night-a.npz night-b.npz
"""
import argparse, glob, hashlib, json, os, sys, warnings
import numpy as np
from home_night_train import context, roll, smooth_probabilities
import home_night_features as hf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
warnings.filterwarnings("ignore")

SMOOTH_WIDTH = 5


def live(x):
    """Past-only version: quiet level and context use only what has been heard so far."""
    out = np.empty_like(x)
    for i in range(len(x)):
        seen = x[max(0, i - 239) : i + 1]
        out[i] = (x[i] - np.median(seen, 0)) / (seen.std(0) + 1e-3 if i > 9 else 1.0)
    out = np.clip(out, -8, 8)
    past = lambda z, n, fn: np.array([fn(z[max(0, i - n) : i + 1], axis=0) for i in range(len(z))])
    core = out[:, 48:]
    return np.hstack([out, past(out, 10, np.mean), past(out, 40, np.mean), past(core, 10, np.std), past(core, 40, np.std)])


def load(path, mode):
    d = np.load(path)
    k = d["y"] >= 0
    x = context(d["x"], d["hours"], False) if mode == "morning" else live(d["x"])
    return x[k], d["y"][k]


def cw(y):
    c = np.bincount(y, minlength=4).astype(float)
    return (len(y) / (4 * np.maximum(c, 1)))[y]


def fit(groups):  # groups: list of (x, y, share)
    x = np.concatenate([g[0] for g in groups])
    y = np.concatenate([g[1] for g in groups])
    w = np.concatenate([cw(g[1]) * g[2] / len(g[1]) for g in groups])
    w *= len(y) / w.sum()
    return LogisticRegression(C=0.02, max_iter=2000).fit(x, y, sample_weight=w)


def show(tag, m, x, y, smooth=1):
    p = smooth_probabilities(m.predict_proba(x), smooth)
    pred = p.argmax(1)
    r = recall_score(y, pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    k4 = cohen_kappa_score(y, pred)
    rem_f1 = f1_score(y == 3, pred == 3, zero_division=0)
    rem_p = precision_score(y == 3, pred == 3, zero_division=0)
    print(
        f"  {tag:34s} bal {balanced_accuracy_score(y, pred):.3f}  k4 {k4:.3f}  "
        f"REM[AUC {roc_auc_score(y == 3, p[:, 3]):.3f} F1 {rem_f1:.3f} R {r[3]:.0%} P {rem_p:.0%}]  "
        f"W {r[0]:.0%} L {r[1]:.0%} D {r[2]:.0%}"
    )


def export_model(repo_root, model):
    payload = {
        "identifier": "morning-v3",
        "stages": ["awake", "light", "deep", "rem"],
        "smoothWidth": SMOOTH_WIDTH,
        "coefficients": [[round(float(v), 6) for v in row] for row in model.coef_],
        "intercepts": [round(float(v), 6) for v in model.intercept_],
    }
    raw_json = json.dumps(payload, separators=(",", ":"))
    swift_path = os.path.join(repo_root, "Sources", "SleepWakeKit", "MorningModelData.swift")
    with open(swift_path, "w") as f:
        f.write(
            f"/// The morning model's numbers: {model.coef_.shape[1]} measurements per half-minute, weighed for each of the four stages.\n"
            "enum MorningModelData {\n"
            '    static let json = #"""\n'
            f"{raw_json}\n"
            '"""#\n'
            "}\n"
        )
    # Regenerate synthetic 10-minute test reference matching SleepWakeKitTests.swift exactly
    epochs = 20
    count = epochs * 480_000 + 1_024
    s = 12_345
    rng_vals = np.empty(count, dtype=np.int32)
    for i in range(count):
        s = (s * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        rng_vals[i] = ((s >> 16) & 0xFFFF) - 32_768
    idx = np.arange(count, dtype=np.int64)
    epoch = idx // 480_000
    gain = 1 + (epoch % 5) + np.where((epoch % 3) == 0, 6 * ((idx // 32_000) % 2), 0) + np.where((idx % 1_440_000) < 8_000, 40, 0)
    samples = (rng_vals * gain).astype(np.float32) / np.float32(8_388_608)
    mels, centroid, flat, hi_band = hf.frames(samples, 1.0)
    mels = np.maximum(mels - np.percentile(mels, 10, axis=0), 0.0)
    hi_rel = hi_band - np.median(hi_band)
    starts = [int(i * 30 * hf.FPS) for i in range(epochs)]
    rows = hf.epoch_rows(mels, centroid, flat, hi_rel, starts)
    feats = context(rows, np.arange(epochs) / 120.0, False)
    probs = smooth_probabilities(model.predict_proba(feats), SMOOTH_WIDTH)
    ref_path = os.path.join(repo_root, "Tests", "SleepWakeKitTests", "morning-reference.json")
    with open(ref_path, "w") as f:
        json.dump(
            {
                "rows": [[round(float(v), 6) for v in r] for r in rows],
                "probabilities": [[round(float(v), 6) for v in p] for p in probs],
            },
            f,
            separators=(",", ":"),
        )
    print(f"Exported morning-v3 ({model.coef_.shape[1]} features, smooth={SMOOTH_WIDTH}) to {swift_path} and {ref_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="Repo root to write MorningModelData.swift and morning-reference.json")
    ap.add_argument("--features-dir", default="features", help="Folder of lab feature .npz files")
    ap.add_argument("nights", nargs="+", help="Home night .npz files")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    people = sorted({os.path.basename(f).split("-")[0] for f in files})
    val = set(sorted(people, key=lambda p: hashlib.sha256(f"teacher-v12:{p}".encode()).hexdigest())[:40])

    if args.export:
        lab_all = [load(f, "morning") for f in files]
        x_lab = np.concatenate([x for x, _ in lab_all])
        y_lab = np.concatenate([y for _, y in lab_all])
        home_all = [load(n, "morning") for n in args.nights]
        x_home = np.concatenate([x for x, _ in home_all])
        y_home = np.concatenate([y for _, y in home_all])
        model = fit([(x_lab, y_lab, 0.5), (x_home, y_home, 0.5)])
        export_model(args.export, model)
        return

    for mode in ("morning", "live"):
        lab = [(os.path.basename(f).split("-")[0] in val, load(f, mode)) for f in files]
        cat = lambda g: (np.concatenate([x for x, _ in g]), np.concatenate([y for _, y in g]))
        xt, yt = cat([v for iv, v in lab if not iv])
        xv, yv = cat([v for iv, v in lab if iv])
        home = {n: load(n, mode) for n in args.nights}
        smooth = SMOOTH_WIDTH if mode == "morning" else 1
        print("==", mode)
        m = fit([(xt, yt, 1)])
        show("lab only -> lab validation", m, xv, yv, smooth=smooth)
        if len(args.nights) >= 2:
            for test, other in ((args.nights[1], args.nights[0]), (args.nights[0], args.nights[1])):
                xo, yo = home[other]
                x, y = home[test]
                print(f" test on {test}")
                show("lab only", m, x, y, smooth=smooth)
                show("other home night only", fit([(xo, yo, 1)]), x, y, smooth=smooth)
                for share in (0.2, 0.5):
                    show(f"lab + other home night ({int(share*100)}%)", fit([(xt, yt, 1 - share), (xo, yo, share)]), x, y, smooth=smooth)


if __name__ == "__main__":
    main()
