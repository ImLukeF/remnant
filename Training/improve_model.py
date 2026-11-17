"""Select improved models only on development people; no final-test files are read here."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from causal_model import names, predict, transform, load_observations
from expand_cohort import atomic_json
from features import FEATURE_NAMES, STAGES
from train import load_rows


def metrics(rows, probabilities):
    matrix = [[0] * 4 for _ in STAGES]
    for row, values in zip(rows, probabilities):
        matrix[STAGES.index(row["stage"])][max(range(4), key=values.__getitem__)] += 1
    if any(sum(row) == 0 for row in matrix):
        raise ValueError("Evaluation requires all four stages.")
    recall = [matrix[k][k] / sum(matrix[k]) for k in range(4)]
    f1 = [2 * matrix[k][k] / (sum(matrix[k]) + sum(row[k] for row in matrix)) for k in range(4)]
    predicted_light = sum(row[1] for row in matrix)
    return {"confusionMatrix": matrix, "classOrder": STAGES, "recallPerClass": recall,
            "balancedAccuracy": sum(recall) / 4, "ordinaryAccuracy": sum(matrix[k][k] for k in range(4)) / len(rows),
            "lightPrecision": matrix[1][1] / predicted_light if predicted_light else 0,
            "f1PerClass": f1, "macroF1": sum(f1) / 4,
            "epochs": len(rows), "subjects": sorted({row["subject"] for row in rows})}


def linear(train, validation, history, penalty):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    x = np.asarray([row["modelFeatures"] for row in train])
    labels = np.asarray([STAGES.index(row["stage"]) for row in train])
    means, scales = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    x = (x - means) / scales
    sample_weights = len(labels) / (4 * np.bincount(labels, minlength=4)[labels])
    count, dim = x.shape
    def objective(parameters):
        coefficients = parameters.reshape(4, dim + 1)
        weights, bias = coefficients[:, :dim], coefficients[:, dim]
        logits = x @ weights.T + bias
        loss = (sample_weights * (logsumexp(logits, axis=1) - logits[np.arange(count), labels])).mean() + penalty * (weights * weights).sum() / 2
        probabilities = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        probabilities[np.arange(count), labels] -= 1
        residual = probabilities * sample_weights[:, None] / count
        gradient = np.column_stack((residual.T @ x + penalty * weights, residual.sum(axis=0)))
        return loss, gradient.ravel()
    result = minimize(objective, np.zeros(4 * (dim + 1)), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-7, "ftol": 1e-11})
    if not result.success:
        raise ValueError("Optimizer did not converge: " + str(result.message))
    coefficients = result.x.reshape(4, dim + 1)
    return {"modelKind": "linear", "means": means.tolist(), "scales": scales.tolist(),
            "weights": coefficients[:, :dim].tolist(), "biases": coefficients[:, dim].tolist()}, {"converged": True, "iterations": result.nit, "objective": result.fun}


def boosted(train, validation, history, depth, balance, primary="balancedAccuracy", fixed_rounds=None):
    import numpy as np
    import xgboost as xgb
    x = np.asarray([row["modelFeatures"] for row in train], dtype=np.float32)
    y = np.asarray([STAGES.index(row["stage"]) for row in train])
    v = np.asarray([row["modelFeatures"] for row in validation], dtype=np.float32)
    vy = np.asarray([STAGES.index(row["stage"]) for row in validation])
    weights = (len(y) / (4 * np.bincount(y, minlength=4)[y])) ** balance
    weights /= weights.mean()
    source, check = xgb.DMatrix(x, label=y, weight=weights, nthread=3), xgb.DMatrix(v, label=vy, nthread=3)
    def balanced(values, data):
        actual, expected = values.argmax(axis=1), data.get_label().astype(int)
        if primary == "macroF1":
            return "macro_f1", float(np.mean([2 * np.sum((actual == k) & (expected == k)) /
                                               (np.sum(actual == k) + np.sum(expected == k)) for k in range(4)]))
        return "balanced", float(np.mean([np.mean(actual[expected == k] == k) for k in range(4)]))
    stopping = {"evals": [(check, "validation")], "custom_metric": balanced, "maximize": True,
                "early_stopping_rounds": 40} if fixed_rounds is None else {"evals": []}
    booster = xgb.train({"objective": "multi:softprob", "num_class": 4, "tree_method": "hist", "max_depth": depth,
                         "eta": 0.05, "min_child_weight": 10, "lambda": 10, "subsample": 1, "colsample_bytree": 1,
                         "seed": 7, "nthread": 3, "base_score": 0}, source,
                        num_boost_round=300 if fixed_rounds is None else fixed_rounds,
                        verbose_eval=False, **stopping)
    if fixed_rounds is None:
        booster = booster[:booster.best_iteration + 1]
    model = export_booster(booster)
    # Independent portable inference must match the training library before selection.
    expected = booster.predict(check)
    difference = float(max(abs(a - b) for row, values in zip(validation, expected) for a, b in zip(predict(model, row["modelFeatures"]), values)))
    if difference > 2e-6:
        raise ValueError(f"Tree export differs from XGBoost: {difference}")
    return model, {"rounds": booster.num_boosted_rounds(), "nativeExportMaximumDifference": difference}


def export_booster(booster):
    obj = json.loads(bytes(booster.save_raw(raw_format="json")))
    learner = obj["learner"]
    if learner["gradient_booster"]["name"] != "gbtree" or learner["objective"]["name"] != "multi:softprob":
        raise ValueError("Unsupported native model representation.")
    tree_model = learner["gradient_booster"]["model"]
    base = json.loads(learner["learner_model_param"]["base_score"])
    if not isinstance(base, list):
        base = [base] * 4
    trees = []
    for tree, class_index in zip(tree_model["trees"], tree_model["tree_info"]):
        if any(tree["split_type"]):
            raise ValueError("Only numerical split trees can be exported.")
        trees.append({"classIndex": class_index, "left": tree["left_children"], "right": tree["right_children"],
                      "features": [feature if left != -1 else -1 for feature, left in zip(tree["split_indices"], tree["left_children"])],
                      "values": tree["split_conditions"]})
    return {"modelKind": "trees", "biases": list(map(float, base)), "trees": trees}


def parameters_digest(model):
    fields = {key: value for key, value in model.items() if key not in {"heldOutSubjects", "balancedAccuracy"}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def choose_candidate(candidates, primary):
    eligible = [entry for entry in candidates if entry["fitsNativeImporter"]
                and all(recall > 0 for recall in entry["validation"]["recallPerClass"])]
    if not eligible:
        raise ValueError("No importable candidate recognizes every stage on validation; no final test may be read.")
    return max(eligible, key=lambda entry: (entry["validation"][primary], entry["validation"]["balancedAccuracy"],
                                           entry["validation"]["recallPerClass"][1], -entry["id"]))


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "selection.json").exists():
        raise ValueError("Selection is frozen; use its checkpoint rather than rerunning selection.")
    previous = json.loads((args.development / "cohort-plan.json").read_text())
    hashes = json.loads((args.development / "feature-checksums.json").read_text())
    combined = hashlib.sha256()
    for item in previous["files"]:
        data = (args.development / "features" / (Path(item["path"]).stem + ".jsonl")).read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes[item["path"]]:
            raise ValueError("Development features changed after their verified extraction.")
        combined.update(data)
    if combined.hexdigest() != hashlib.sha256((args.development / "epochs.jsonl").read_bytes()).hexdigest():
        raise ValueError("Combined development epochs do not match verified feature files.")
    if args.profile == "spectral":
        from spectral import SPECTRAL_NAMES
        raw_names = SPECTRAL_NAMES
        rows = load_observations(args.development / "epochs.jsonl", raw_names)
    else:
        raw_names = FEATURE_NAMES
        rows = load_rows(args.development / "epochs.jsonl")
    roles = {subject: "train" if role == "train" else "validation" for subject, role in previous["splits"].items()}
    observed = sorted(roles)
    present = {row["subject"] for row in rows}
    excluded = sorted(set(roles) - present)
    roles = {person: role for person, role in roles.items() if person in present}
    primary = "macroF1" if args.profile == "spectral" else "balancedAccuracy"
    config = {"trainingSubjects": sorted(s for s, role in roles.items() if role == "train"),
              "validationSubjects": sorted(s for s, role in roles.items() if role != "train"),
              "finalTestNotRead": True, "seed": 7, "histories": [1, 10], "linearPenalties": [0.01, 0.1],
              "treeDepths": [2, 4], "treeBalancePowers": [0.5, 1.0], "treeMaxRounds": 300,
              "maximumPayloadBytes": 512000, "evaluationMetadataReserveBytes": 4096,
              "developmentEpochsSHA256": combined.hexdigest(), "developmentExtractorSHA256": previous["extractorSHA256"],
              "featureProfile": args.profile, "previouslyObservedSubjects": observed, "excludedForNoUsableAudio": excluded,
              "refitOnAllDevelopment": args.refit_development,
              "selectionEligibility": "Fits native import budget and has positive validation recall for every stage.",
              "selectionMetric": primary + "; balanced accuracy and light recall break ties", "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    atomic_json(root / "protocol.json", config)
    candidates = []
    for history in config["histories"]:
        contextual = transform(rows, history, raw_names)
        train = [row for row in contextual if roles[row["subject"]] == "train"]
        validation = [row for row in contextual if roles[row["subject"]] == "validation"]
        if any({row["stage"] for row in subset} != set(STAGES) for subset in (train, validation)):
            raise ValueError("Usable development audio must cover all four stages in both groups.")
        for kind, settings in [("linear", {"penalty": p}) for p in config["linearPenalties"]] + [("trees", {"depth": d, "balance": b}) for d in config["treeDepths"] for b in config["treeBalancePowers"]]:
            fitted, detail = linear(train, validation, history, **settings) if kind == "linear" else boosted(train, validation, history, primary=primary, **settings)
            model = {"schemaVersion": 2, "modelID": "psg-audio-spectral-v4" if args.profile == "spectral" else "psg-audio-improved-v3", "dataSource": "PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345",
                     "rawFeatureNames": raw_names, "featureNames": names(history, raw_names), "classes": STAGES,
                     "contextEpochs": history, "heldOutSubjects": [], "balancedAccuracy": 0, **fitted}
            if args.profile == "spectral":
                model["featureProfile"] = "spectral-v4"
            score = metrics(validation, [predict(model, row["modelFeatures"]) for row in validation])
            payload_bytes = len((json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n").encode())
            entry = {"id": len(candidates), "history": history, "kind": kind, "settings": settings, "fit": detail, "validation": score,
                     "payloadBytes": payload_bytes,
                     "fitsNativeImporter": payload_bytes <= config["maximumPayloadBytes"] - config["evaluationMetadataReserveBytes"]}
            atomic_json(root / f"candidate-{entry['id']}.json", model, compact=True)
            candidates.append(entry)
            atomic_json(root / "candidates.json", candidates)
            print(json.dumps({"event": "candidate", "id": entry["id"], "history": history, "kind": kind,
                              "balancedAccuracy": score["balancedAccuracy"], "lightRecall": score["recallPerClass"][1]}), flush=True)
    winner = choose_candidate(candidates, primary)
    chosen = json.loads((root / f"candidate-{winner['id']}.json").read_text())
    refit = None
    if args.refit_development:
        contextual = transform(rows, winner["history"], raw_names)
        validation = [row for row in contextual if roles[row["subject"]] == "validation"]
        if winner["kind"] == "linear":
            fitted, detail = linear(contextual, validation, winner["history"], **winner["settings"])
        else:
            fitted, detail = boosted(contextual, validation, winner["history"], **winner["settings"],
                                      fixed_rounds=winner["fit"]["rounds"])
        chosen.update(fitted)
        payload_bytes = len((json.dumps(chosen, separators=(",", ":"), allow_nan=False) + "\n").encode())
        if payload_bytes > config["maximumPayloadBytes"] - config["evaluationMetadataReserveBytes"]:
            raise ValueError("Final refit exceeds the native import budget; no final test may be read.")
        refit = {"trainingSubjects": sorted(roles), "epochs": len(contextual), "fit": detail,
                 "payloadBytes": payload_bytes, "validationScoreBelongsToSelectionCheckpoint": True,
                 "fixedSettingsBeforeRefit": winner["settings"]}
    atomic_json(root / "selected-model.json", chosen, compact=True)
    atomic_json(root / "selection.json", {**winner, "parametersSHA256": parameters_digest(chosen), "finalTestRead": False,
                                         "trainingSubjects": config["trainingSubjects"], "validationSubjects": config["validationSubjects"]})
    saved = json.loads((root / "selection.json").read_text())
    saved.update(previouslyObservedSubjects=observed, excludedForNoUsableAudio=excluded, selectionMetric=primary,
                 finalRefit=refit)
    atomic_json(root / "selection.json", saved)
    print(json.dumps({"event": "selected", "candidate": winner["id"], "kind": winner["kind"], "history": winner["history"],
                      "balancedAccuracy": winner["validation"]["balancedAccuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("basic", "spectral"), default="basic")
    parser.add_argument("--refit-development", action="store_true",
                        help="After selection, refit fixed settings on all development people before reserving a new test.")
    run(parser.parse_args())
