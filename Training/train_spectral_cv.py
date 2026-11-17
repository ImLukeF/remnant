"""Refine the existing spectral trees using fixed participant folds and paired admission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from causal_model import names, predict, transform
from expand_cohort import atomic_json
from features import STAGES
from improve_model import boosted, metrics, parameters_digest
from prepare_sequence_test import admit
from spectral import SPECTRAL_NAMES
from train_sequence import load_corpus, fold_for


SETTINGS = [
    {"depth": 1, "rounds": 160, "balance": 1.0},
    {"depth": 2, "rounds": 80, "balance": 1.0},
    {"depth": 2, "rounds": 240, "balance": 1.0},
    {"depth": 2, "rounds": 320, "balance": 1.0},
    {"depth": 3, "rounds": 80, "balance": 1.0},
    {"depth": 3, "rounds": 160, "balance": 1.0},
    {"depth": 3, "rounds": 240, "balance": 1.0},
    {"depth": 2, "rounds": 160, "balance": 0.75},
    {"depth": 3, "rounds": 160, "balance": 0.75},
    {"depth": 4, "rounds": 80, "balance": 1.0},
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    import numpy as np
    import xgboost
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if (root/"selection.json").exists() or (root/"development-review.json").exists():
        raise ValueError("This development experiment is complete.")
    rows, manifest = load_corpus(args.development, args.retired_test, args.previous_evaluation, args.registry)
    folds = {p: fold_for(p) for p in manifest["people"]}
    previous = json.loads((args.baseline_cache/"protocol.json").read_text())
    if previous["development"] != manifest or previous["folds"] != folds:
        raise ValueError("Paired baseline lineage or participant folds changed.")
    baseline = [json.loads((args.baseline_cache/f"baseline-fold-{f}.json").read_text()) for f in range(3)]
    baseline_f1 = sum(r["validation"]["macroF1"] for r in baseline)/3
    baseline_accuracy = sum(r["validation"]["balancedAccuracy"] for r in baseline)/3
    protocol = {"experiment": "spectral-trees-cv-v7", "development": manifest, "folds": folds,
                "configurations": SETTINGS, "contextEpochs": 10, "usesFutureAudio": False,
                "developmentExtractorSHA256": manifest["extractorSHA256"],
                "baselineProtocolSHA256": sha(args.baseline_cache/"protocol.json"),
                "baselineReportsSHA256": {str(f): sha(args.baseline_cache/f"baseline-fold-{f}.json") for f in range(3)},
                "xgboostVersion": xgboost.__version__, "numpyVersion": np.__version__,
                "selectionMetric": "Both mean paired metrics must improve; every stage recognized in every fold; then maximize macro F1.",
                "refit": "The selected fixed configuration is refit on all development people without early stopping.",
                "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_text()) != protocol:
        raise ValueError("Resume only the identical frozen development experiment.")
    atomic_json(root/"protocol.json", protocol)
    groups = []
    for fold in range(3):
        train = transform([r for r in rows if folds[r["subject"]] != fold], 10, SPECTRAL_NAMES)
        validation = transform([r for r in rows if folds[r["subject"]] == fold], 10, SPECTRAL_NAMES)
        score = baseline[fold]["validation"]
        if (score["subjects"] != sorted({r["subject"] for r in validation}) or score["epochs"] != len(validation)
                or any(sum(score["confusionMatrix"][i]) != sum(r["stage"] == stage for r in validation) for i,stage in enumerate(STAGES))):
            raise ValueError("Paired baseline observations changed.")
        groups.append((train, validation))
    candidates = []
    for index, settings in enumerate(SETTINGS):
        reports = []
        for fold, (train, validation) in enumerate(groups):
            report_path = root/f"candidate-{index}-fold-{fold}.json"
            model_path = root/f"weights-{index}-fold-{fold}.json"
            if report_path.exists() and model_path.exists():
                report = json.loads(report_path.read_text())
                if report["modelFileSHA256"] != sha(model_path): raise ValueError("Cached tree checkpoint changed.")
            else:
                model, fit = boosted(train, validation, 10, settings["depth"], settings["balance"], fixed_rounds=settings["rounds"])
                atomic_json(model_path, model, compact=True)
                report = {"fold": fold, "validation": metrics(validation, [predict(model, r["modelFeatures"]) for r in validation]),
                          "fit": fit, "modelFileSHA256": sha(model_path)}
                atomic_json(report_path, report)
            reports.append(report)
            print(json.dumps({"event": "spectral_fold", "candidate": index, "fold": fold,
                              "macroF1": report["validation"]["macroF1"], "balancedAccuracy": report["validation"]["balancedAccuracy"]}), flush=True)
        candidate = {"id": index, "settings": settings, "folds": reports,
                     "meanMacroF1": sum(r["validation"]["macroF1"] for r in reports)/3,
                     "meanBalancedAccuracy": sum(r["validation"]["balancedAccuracy"] for r in reports)/3,
                     "everyFoldRecognizesEveryStage": all(all(x > 0 for x in r["validation"]["recallPerClass"]) for r in reports),
                     "baselineMeanMacroF1": baseline_f1, "baselineMeanBalancedAccuracy": baseline_accuracy}
        candidates.append(candidate); atomic_json(root/"candidates.json", candidates)
    eligible = []
    for candidate in candidates:
        try: admit(candidate)
        except ValueError: continue
        eligible.append(candidate)
    if not eligible:
        atomic_json(root/"development-review.json", {"status": "not_improved", "freshTestUntouched": True,
                    "baselineMeanMacroF1": baseline_f1, "baselineMeanBalancedAccuracy": baseline_accuracy})
        print(json.dumps({"event": "not_improved", "freshTestUntouched": True}), flush=True); return
    winner = max(eligible, key=lambda c: (c["meanMacroF1"], c["meanBalancedAccuracy"], -c["id"]))
    transformed = transform(rows, 10, SPECTRAL_NAMES); settings = winner["settings"]
    # The probe is fitting data used only for independent export math, not a validation score.
    model, fit = boosted(transformed, transformed[:128], 10, settings["depth"], settings["balance"], fixed_rounds=settings["rounds"])
    model.update(schemaVersion=2, modelID="psg-audio-spectral-cv-v7", dataSource="PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345",
                 featureProfile="spectral-v4", rawFeatureNames=SPECTRAL_NAMES, featureNames=names(10, SPECTRAL_NAMES),
                 contextEpochs=10, classes=STAGES, heldOutSubjects=[], balancedAccuracy=0)
    atomic_json(root/"selected-model.json", model, compact=True)
    if (root/"selected-model.json").stat().st_size > 508000: raise ValueError("Candidate exceeds native import budget.")
    selection = {**winner, "parametersSHA256": parameters_digest(model), "trainingSubjects": manifest["people"],
                 "validationSubjects": [], "previouslyObservedSubjects": manifest["people"], "finalTestRead": False,
                 "finalRefit": {"trainingSubjects": manifest["people"], "settings": settings, "fit": fit},
                 "validation": {"kind": "three participant folds; development only", "macroF1": winner["meanMacroF1"],
                                "balancedAccuracy": winner["meanBalancedAccuracy"]}}
    atomic_json(root/"selection.json", selection)
    print(json.dumps({"event": "selected", "id": winner["id"], "macroF1": winner["meanMacroF1"],
                      "balancedAccuracy": winner["meanBalancedAccuracy"], "parametersSHA256": selection["parametersSHA256"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("development", "retired-test", "previous-evaluation", "registry", "baseline-cache", "output"):
        p.add_argument("--"+flag, type=Path, required=True)
    run(p.parse_args())
