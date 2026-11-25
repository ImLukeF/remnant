"""Weekly home fine-tune: the phone's GRU learns Luke's room from his own nights, never at the sleep lab's expense.

Three kinds of home signal, all from the microphone:
- every usable epoch: the model gives the same answer with extra noise in its numbers (consistency);
- epochs the current model is sure about (>= .85): its own answer, as a soft teacher;
- epochs Luke was plainly awake: before "Trying to sleep", around "I'm awake" and "TV off", before the time he says he
  fell asleep and after he woke. These are the only real labels.
A quarter of nights (fixed by their id) are never trained on. The new model is kept only if, on those nights, it
catches at least as many awake moments as the current one without calling more of the night awake, and if the sleep lab
check (the 40 PSG-Audio validation people, clean and with noise; the current model learned from them too, so this
checks for forgetting) does not get worse. The reserved final-test people are
never opened. Morning wake ratings are reported against the stage seen before waking, for both models.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

from array_gru import layout, encode_matrix, normalized, batch, forward
from expand_cohort import atomic_json
from expanded_data import load
from features import STAGES
from improve_model import metrics
from run_dataset import checksum
from train_consistency import consistency
from train_sequence import make_network, export

APPLE_EPOCH = 978307200.0  # Swift encodes dates as seconds since 1 Jan 2001.
CONFIDENT = 0.85
LAB_TOLERANCE = 0.005
MORE_AWAKE_ALLOWED = 0.05
MIN_TRAINING_NIGHTS = 7
MIN_HELD_BACK_NIGHTS = 3
MIN_AWAKE_EPOCHS = 60
MIN_ASLEEP_EPOCHS = 120
RATED_MORNINGS_NEEDED = 20
RECIPE = {"history": 10, "hidden": 32, "consistency": 0.25, "noiseShare": 0.1, "classPower": 0.9}  # v11 recipe-0, the current model


def held_back(night_id):
    return int(hashlib.sha256(night_id.upper().encode()).hexdigest(), 16) % 4 == 0


def home_night(body):
    """One uploaded night as arrays, with its awake labels, its likely-asleep window and its wake rating."""
    import numpy as np
    night = body["night"]
    epochs = sorted(body["epochs"], key=lambda e: e["start"])
    if not epochs: return None
    when = lambda value: None if value is None else float(value) + APPLE_EPOCH
    starts = np.array([when(e["start"]) for e in epochs]); ends = starts + 30
    features = np.zeros((len(epochs), 58))
    usable = np.zeros(len(epochs), dtype=bool)
    for row, epoch in enumerate(epochs):
        values = epoch.get("features") or []
        if len(values) == 58 and all(isinstance(v, (int, float)) and np.isfinite(v) for v in values):
            features[row] = values; usable[row] = bool(epoch.get("usable"))
    awake = np.zeros(len(epochs), dtype=bool); near_awake = np.zeros(len(epochs), dtype=bool)
    bedtime = None
    for marker in night.get("markers", []):
        at = when(marker["date"])
        if marker["kind"] in ("awake", "tvOff"):
            awake |= (ends > at - 60) & (starts < at + 60)
            near_awake |= (ends > at - 600) & (starts < at + 600)
        if marker["kind"] == "tryingToSleep":
            awake |= starts < at
        if marker["kind"] in ("tryingToSleep", "tvOff"):
            bedtime = max(bedtime or at, at)
    feedback = night.get("feedback", {})
    fell, woke = when(feedback.get("fellAsleepAt")), when(feedback.get("wokeAt"))
    if fell: awake |= ends <= fell - 300
    if woke: awake |= starts >= woke + 60
    onset = fell + 900 if fell else (bedtime + 1800 if bedtime else None)
    asleep = np.zeros(len(epochs), dtype=bool)
    if onset and woke and woke - 900 > onset:
        asleep = (starts >= onset) & (ends <= woke - 900) & ~awake & ~near_awake
    before_waking = (ends > woke - 600) & (ends <= woke - 60) if woke else np.zeros(len(epochs), dtype=bool)
    return {"id": night["id"].upper(), "started": starts[0], "features": features, "usable": usable,
            "startSeconds": starts - starts[0], "awake": awake & usable, "asleep": asleep & usable,
            "beforeWaking": before_waking & usable, "wakeDifficulty": feedback.get("wakeDifficulty")}


def home_arrays(nights):
    import numpy as np
    keys = ("features", "usable", "startSeconds", "awake", "asleep", "beforeWaking")
    arrays = {key: np.concatenate([n[key] for n in nights]) for key in keys}
    arrays["subject"] = np.concatenate([np.full(len(n["usable"]), "home-" + n["id"], dtype="U48") for n in nights])
    arrays["augmentedFeatures"] = arrays["features"]; arrays["augmentedUsable"] = arrays["usable"]
    arrays["stage"] = np.where(arrays["awake"], 0, -1)
    return arrays


def prepared(arrays, model, history, noisy=True):
    data = layout(arrays, history)
    data["clean"] = normalized(encode_matrix(arrays["features"][data["indexes"]]), model["means"], model["scales"])
    if noisy:
        data["noisy"] = normalized(encode_matrix(arrays["augmentedFeatures"][data["indexes"]]), model["means"], model["scales"])
    return data


def probabilities(model, data, take, view="clean"):
    import numpy as np
    out = [forward(model, batch(data[view], data["windows"], data["lengths"], take[s:s + 512]), data["lengths"][take[s:s + 512]])
           for s in range(0, len(take), 512)]
    return np.concatenate(out) if out else np.zeros((0, 4))


def lab_scores(model, data, people):
    import numpy as np
    take = np.flatnonzero(np.isin(data["people"], people))
    rows = [{"subject": str(p), "stage": STAGES[int(s)]} for p, s in zip(data["people"][take], data["labels"][take])]
    return {view: metrics(rows, probabilities(model, data, take, view)) for view in ("clean", "noisy")}


def home_scores(model, data, arrays, nights):
    """Awake moments caught and share of the likely-asleep window called awake, on the given nights."""
    import numpy as np
    subjects = ["home-" + n["id"] for n in nights]
    take = np.flatnonzero(np.isin(data["people"], subjects))
    if not len(take): return {"awakeEpochs": 0, "asleepEpochs": 0, "awakeCaught": None, "asleepCalledAwake": None}
    called_awake = probabilities(model, data, take).argmax(axis=1) == 0
    awake = arrays["awake"][data["indexes"][take]]; asleep = arrays["asleep"][data["indexes"][take]]
    return {"awakeEpochs": int(awake.sum()), "asleepEpochs": int(asleep.sum()),
            "awakeCaught": float(called_awake[awake].mean()) if awake.any() else None,
            "asleepCalledAwake": float(called_awake[asleep].mean()) if asleep.any() else None}


def wake_ratings(model, data, arrays, nights):
    """For rated mornings: the chance of deep and of light sleep in the ten minutes before waking."""
    import numpy as np
    from scipy.stats import spearmanr
    rows = []
    for night in nights:
        if night["wakeDifficulty"] not in (1, 2, 3, 4, 5): continue
        take = np.flatnonzero((data["people"] == "home-" + night["id"]) & arrays["beforeWaking"][data["indexes"]])
        if not len(take): continue
        p = probabilities(model, data, take).mean(axis=0)
        rows.append({"night": night["id"], "difficulty": night["wakeDifficulty"], "deep": float(p[2]), "light": float(p[1])})
    result = {"ratedMornings": len(rows), "needed": RATED_MORNINGS_NEEDED, "mornings": rows}
    if len(rows) >= 5 and len({r["difficulty"] for r in rows}) > 1:
        for stage in ("deep", "light"):
            value = float(spearmanr([r[stage] for r in rows], [r["difficulty"] for r in rows]).statistic)
            result[stage + "VsDifficulty"] = value if np.isfinite(value) else None
    return result


def network_from(model):
    import torch
    network = make_network(model["hiddenSize"])
    with torch.no_grad():
        for parameter, key in ((network.gru.weight_ih_l0, "inputWeights"), (network.gru.weight_hh_l0, "recurrentWeights"),
                               (network.gru.bias_ih_l0, "inputBiases"), (network.gru.bias_hh_l0, "recurrentBiases"),
                               (network.output.weight, "outputWeights"), (network.output.bias, "outputBiases")):
            value = torch.tensor(model[key], dtype=torch.float64)
            if value.shape != parameter.shape: raise ValueError("The current model's shape doesn't match the GRU.")
            parameter.copy_(value)
    return network


def fine_tune(baseline, lab, lab_people, home, home_train, seconds, passes=3):
    """Start from the current model and keep its normalisation, so the phone reads the new one the same way."""
    import numpy as np
    import torch
    began = time.monotonic(); torch.manual_seed(7); rng = np.random.default_rng(7)
    network = network_from(baseline)
    lab_take = np.flatnonzero(np.isin(lab["people"], lab_people))
    labels = lab["labels"][lab_take].astype(np.int64); counts = np.bincount(labels, minlength=4)
    _, inverse, person_counts = np.unique(lab["people"][lab_take], return_inverse=True, return_counts=True)
    weights = (len(labels) / (4 * counts[labels])) ** RECIPE["classPower"] / person_counts[inverse]
    weights = torch.from_numpy(weights / weights.mean())
    home_take = np.flatnonzero(np.isin(home["people"], ["home-" + n["id"] for n in home_train]))
    teacher = probabilities(baseline, home, home_take)
    awake = home["labels"][home_take] == 0
    pseudo = np.where(awake, 0, np.where(teacher.max(axis=1) >= CONFIDENT, teacher.argmax(axis=1), -1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.0001, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.02)
    log = []; completed = 0
    steps = (len(lab_take) + 255) // 256
    for number in range(1, passes + 1):
        if log and time.monotonic() - began + log[-1]["seconds"] > seconds: break
        network.train(); total = 0.0; started = time.monotonic()
        home_order = rng.permutation(len(home_take))
        for step, local in enumerate(np.array_split(rng.permutation(len(lab_take)), steps)):
            if time.monotonic() - began > seconds: break
            take = lab_take[local]; lengths = torch.from_numpy(lab["lengths"][take]); target = torch.from_numpy(labels[local])
            optimizer.zero_grad(set_to_none=True)
            clean = network(torch.from_numpy(batch(lab["clean"], lab["windows"], lab["lengths"], take)), lengths)
            noisy = network(torch.from_numpy(batch(lab["noisy"], lab["windows"], lab["lengths"], take)), lengths)
            loss = (((1 - RECIPE["noiseShare"]) * loss_function(clean, target) + RECIPE["noiseShare"] * loss_function(noisy, target)
                     + RECIPE["consistency"] * consistency(clean, noisy)) * weights[local]).mean()
            if len(home_take):
                pick = home_order[(step * 128) % len(home_take):][:128]
                rows = home_take[pick]; home_lengths = torch.from_numpy(home["lengths"][rows])
                x = batch(home["clean"], home["windows"], home["lengths"], rows)
                shifted = np.clip(x + rng.normal(0, 0.2, (len(rows), 1, 1)) + rng.normal(0, 0.1, x.shape), -8, 8)
                shifted[np.arange(x.shape[1])[None, :] >= home["lengths"][rows, None]] = 0
                home_clean = network(torch.from_numpy(x), home_lengths)
                home_noisy = network(torch.from_numpy(shifted), home_lengths)
                home_loss = RECIPE["consistency"] * consistency(home_clean, home_noisy).mean()
                target = torch.from_numpy(pseudo[pick])
                if (target >= 0).any():
                    keep = target >= 0
                    weight = torch.from_numpy(np.where(awake[pick][keep.numpy()], 1.0, 0.5))
                    home_loss = home_loss + (loss_function(home_clean[keep], target[keep]) * weight).mean()
                loss = loss + home_loss
            if not torch.isfinite(loss): raise ValueError("Non-finite home objective.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 1, error_if_nonfinite=True); optimizer.step()
            total += float(loss.detach())
        completed = number
        log.append({"pass": number, "seconds": time.monotonic() - started, "loss": total / steps})
        print(json.dumps({"event": "home_pass", **log[-1]}), flush=True)
    if not completed: raise TimeoutError("No complete pass within the weekly budget.")
    network.eval()
    model = export(network, RECIPE, np.asarray(baseline["means"]), np.asarray(baseline["scales"]), [])
    probe = lab_take[:128]
    with torch.no_grad():
        expected = torch.softmax(network(torch.from_numpy(batch(lab["clean"], lab["windows"], lab["lengths"], probe)),
                                         torch.from_numpy(lab["lengths"][probe])), dim=1).numpy()
    difference = float(np.max(np.abs(probabilities(model, lab, probe) - expected)))
    if difference > 2e-10: raise ValueError("Exported GRU differs from PyTorch.")
    return model, {"passes": log, "homeEpochs": int(len(home_take)), "awakeLabels": int(awake.sum()),
                   "teacherLabels": int(((pseudo >= 0) & ~awake).sum()), "exportMaximumDifference": difference}


def percent(value):
    return "none" if value is None else f"{round(value * 100)}%"


def run(args):
    import numpy as np
    import torch
    torch.set_num_threads(3); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    began = time.monotonic(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text())
    if baseline.get("modelKind") != "gru" or baseline.get("featureProfile") != "spectral-v4":
        raise ValueError("The weekly fine-tune starts from the phone's spectral GRU.")
    nights = [n for n in (home_night(json.loads(p.read_text())) for p in sorted(args.nights.glob("*.json"))) if n]
    nights = [n for n in nights if n["usable"].sum() >= 120]
    train_nights = [n for n in nights if not held_back(n["id"])]; check_nights = [n for n in nights if held_back(n["id"])]
    report = {"baselineModelID": baseline["modelID"], "baselineModelSHA256": checksum(args.baseline),
              "nights": len(nights), "trainingNights": len(train_nights), "heldBackNights": len(check_nights),
              "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "kept": False}
    if len(train_nights) < MIN_TRAINING_NIGHTS or len(check_nights) < MIN_HELD_BACK_NIGHTS:
        report.update(status="waiting", summary=f"Waiting for more nights: {len(nights)} so far, about 10 needed.")
        atomic_json(root / "report.json", report); print(json.dumps(report), flush=True); return report
    home = home_arrays(nights); home_data = prepared(home, baseline, RECIPE["history"], noisy=False)
    arrays, _, manifest = load(args.development); previous = json.loads((args.previous / "protocol.json").read_text())
    if previous["development"] != manifest: raise ValueError("Use exactly the verified sleep-lab corpus.")
    lab = prepared(arrays, baseline, RECIPE["history"])
    # The current model learned from all 199 lab people, so the fine-tune keeps all of them and the lab check asks
    # whether it still does as well on the 40 it was once validated on: a check for forgetting, both models alike.
    validation = previous["validationSubjects"]; training = previous["refitSubjects"]
    if not set(validation) <= set(training) or set(training) & set(previous["reservedTestSubjects"]):
        raise ValueError("The lab check must come from the lab people, and the reserved final test stays closed.")
    model, fit = fine_tune(baseline, lab, training, home_data, train_nights, args.max_seconds - (time.monotonic() - began) - 120)
    model.update(modelID="parley-home-" + time.strftime("%Y%m%d"),
                 dataSource=baseline["dataSource"] + "; fine-tuned on Luke's own home nights (microphone numbers only)")
    lab_before, lab_after = lab_scores(baseline, lab, validation), lab_scores(model, lab, validation)
    model["balancedAccuracy"] = lab_after["clean"]["balancedAccuracy"]
    path = root / "candidate-model.json"; atomic_json(path, model, compact=True)
    if path.stat().st_size > 508000: raise ValueError("The model exceeds the phone's import budget.")
    home_before, home_after = (home_scores(m, home_data, home, check_nights) for m in (baseline, model))
    lab_held = (all(r > 0 for view in lab_after for r in lab_after[view]["recallPerClass"])
                and all(lab_after[v][m] >= lab_before[v][m] - LAB_TOLERANCE for v in ("clean", "noisy") for m in ("macroF1", "balancedAccuracy")))
    enough = home_after["awakeEpochs"] >= MIN_AWAKE_EPOCHS and home_after["asleepEpochs"] >= MIN_ASLEEP_EPOCHS
    home_better = enough and (home_after["awakeCaught"] >= home_before["awakeCaught"]
                              and home_after["asleepCalledAwake"] <= home_before["asleepCalledAwake"] + MORE_AWAKE_ALLOWED)
    caught = f"awake moments caught {percent(home_before['awakeCaught'])} → {percent(home_after['awakeCaught'])}"
    if not enough: status, summary = "not kept", "Not kept: the held-back nights need more marked awake times and check-ins."
    elif not lab_held: status, summary = "not kept", f"Not kept: the sleep-lab check got worse ({caught})."
    elif not home_better: status, summary = "not kept", f"Not kept: {caught}, asleep called awake {percent(home_before['asleepCalledAwake'])} → {percent(home_after['asleepCalledAwake'])}."
    else: status, summary = "kept", f"Kept: {caught}; sleep-lab check held."
    report.update(status=status, summary=summary, kept=status == "kept", modelID=model["modelID"],
                  modelSHA256=checksum(path), fit=fit, lab={"before": lab_before, "after": lab_after, "tolerance": LAB_TOLERANCE},
                  home={"before": home_before, "after": home_after},
                  wakeRatings={"before": wake_ratings(baseline, home_data, home, nights), "after": wake_ratings(model, home_data, home, nights)},
                  approvedForAlarmTiming=False, finalTestRead=False, seconds=time.monotonic() - began)
    atomic_json(root / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "summary", "nights", "seconds")}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "previous", "baseline", "nights", "output"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=1800)
    run(p.parse_args())
