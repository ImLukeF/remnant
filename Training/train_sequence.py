"""Subject-separated CPU sequence experiment; the next final-test cohort is never read here."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from causal_model import load_observations, transform, predict
from expand_cohort import atomic_json
from features import STAGES
from improve_model import boosted, metrics, parameters_digest
from run_spectral_corpus import feature_fingerprint
from sequence_model import encoded, sequence_indexes, SequencePredictor
from spectral import SPECTRAL_NAMES


CONFIGURATIONS = [
    {"history": 1, "hidden": 32, "gainDB": 0},
    {"history": 10, "hidden": 32, "gainDB": 0},
    {"history": 20, "hidden": 32, "gainDB": 0},
    {"history": 20, "hidden": 32, "gainDB": 12},
]
BALANCED_CONFIGURATIONS = [
    {"history": history, "hidden": hidden, "gainDB": 0, "classPower": 1.0}
    for hidden in (16,32) for history in (10,20)
] + [{"history":20,"hidden":hidden,"gainDB":12,"classPower":1.0} for hidden in (16,32)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(development, retired, previous_evaluation, registry):
    plan = json.loads((development / "cohort-plan.json").read_text())
    test_plan = json.loads((retired / "download-plan.json").read_text())
    expected = feature_fingerprint()
    if any(p["extractorSHA256"] != expected for p in (plan, test_plan)) or plan["revision"] != test_plan["revision"]:
        raise ValueError("Only matching, pinned spectral corpora may be combined.")
    old = json.loads(previous_evaluation.read_text())
    hashes = [json.loads((development / "feature-checksums.json").read_text()), old["testFeatureSHA256"]]
    people, rows, sources = set(), [], []
    for root, manifest, checks in zip((development, retired), (plan, test_plan), hashes):
        combined = hashlib.sha256()
        for item in manifest["files"]:
            source = root / "features" / (Path(item["path"]).stem + ".jsonl")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != checks[item["path"]]:
                raise ValueError("Cached clinical observations changed after extraction.")
            combined.update(data)
        if combined.hexdigest() != digest(root / "epochs.jsonl"):
            raise ValueError("Combined observations disagree with their verified blocks.")
        subset = load_observations(root / "epochs.jsonl", SPECTRAL_NAMES)
        subjects = {row["subject"] for row in subset}
        if subjects & people:
            raise ValueError("Development sources overlap participants.")
        people |= subjects
        rows.extend(subset)
        sources.append({"root": root.name, "planSHA256": digest(root / ("cohort-plan.json" if root == development else "download-plan.json")),
                        "epochsSHA256": combined.hexdigest(), "people": len(subjects), "epochs": len(subset)})
    if people != set(json.loads(registry.read_text())):
        raise ValueError("Every previously observed participant must remain in the exclusion registry.")
    return sorted(rows, key=lambda r: (r["subject"], r["startSeconds"])), {"sources": sources,
        "extractorSHA256": expected, "revision": plan["revision"], "people": sorted(people)}


def fold_for(subject):
    return int.from_bytes(hashlib.sha256(subject.encode()).digest()[:4], "big") % 3


def make_network(hidden, input_size=len(SPECTRAL_NAMES)):
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_size, hidden, batch_first=True)
            self.dropout = torch.nn.Dropout(0.15)
            self.output = torch.nn.Linear(hidden, 4)

        def forward(self, values, lengths):
            packed = torch.nn.utils.rnn.pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
            _, hidden_state = self.gru(packed)
            return self.output(self.dropout(hidden_state[-1]))
    return Network().double()


def arrays(rows, history, means, scales, clip=True):
    import numpy as np
    raw = np.asarray([encoded(row["features"]) for row in rows], dtype=np.float64)
    normalized = (raw-means)/scales
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite sequence normalization.")
    if clip:
        np.clip(normalized, -8, 8, out=normalized)
    windows = sequence_indexes(rows, history)
    values = np.zeros((len(rows), history, len(SPECTRAL_NAMES)), dtype=np.float64)
    for index, sequence in enumerate(windows):
        values[index, :len(sequence)] = normalized[sequence]
    return values, np.asarray([len(x) for x in windows]), np.asarray([STAGES.index(row["stage"]) for row in rows])


def torch_probabilities(network, values, lengths):
    import numpy as np
    import torch
    network.eval()
    result = []
    with torch.no_grad():
        for index in range(0, len(values), 512):
            logits = network(torch.from_numpy(values[index:index+512]), torch.from_numpy(lengths[index:index+512]))
            result.append(torch.softmax(logits, dim=1).numpy())
    probabilities = np.concatenate(result)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite sequence probabilities.")
    return probabilities


def export(network, config, means, scales, held_out):
    return {"schemaVersion": 3, "modelKind": "gru", "modelID": "psg-audio-causal-gru-v5",
            "dataSource": "PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345",
            "featureProfile": "spectral-v4", "rawFeatureNames": SPECTRAL_NAMES, "featureNames": SPECTRAL_NAMES,
            "classes": STAGES, "heldOutSubjects": held_out, "balancedAccuracy": 0,
            "contextEpochs": config["history"], "hiddenSize": config["hidden"], "normalizedClip": 8,
            "means": means.tolist(), "scales": scales.tolist(),
            "inputWeights": network.gru.weight_ih_l0.detach().tolist(),
            "recurrentWeights": network.gru.weight_hh_l0.detach().tolist(),
            "inputBiases": network.gru.bias_ih_l0.detach().tolist(),
            "recurrentBiases": network.gru.bias_hh_l0.detach().tolist(),
            "outputWeights": network.output.weight.detach().tolist(),
            "outputBiases": network.output.bias.detach().tolist()}


def fit(train, validation, config, seed, maximum_epochs, patience, fixed_epochs=None, reference_score=None):
    import numpy as np
    import torch
    from collections import Counter
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    raw = np.asarray([encoded(row["features"]) for row in train], dtype=np.float64)
    means, scales = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-6)
    x, lengths, labels = arrays(train, config["history"], means, scales, clip=False)
    vx, vl, _ = arrays(validation, config["history"], means, scales) if validation else (None, None, None)
    counts = np.bincount(labels, minlength=4)
    if any(counts == 0):
        raise ValueError("Training needs all four independently scored stages.")
    subject_counts = Counter(row["subject"] for row in train)
    sample_weights = np.asarray([(len(train)/(4*counts[label]))**config.get("classPower",0.75) * len(train)/(len(subject_counts)*subject_counts[row["subject"]])
                                 for row, label in zip(train, labels)])
    sample_weights /= sample_weights.mean()
    network = make_network(config["hidden"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.02)
    best, best_key, best_epoch, stale = None, None, 0, 0
    log = []
    for epoch in range(1, (fixed_epochs or maximum_epochs)+1):
        started = time.monotonic()
        network.train()
        for indexes in np.array_split(rng.permutation(len(train)), (len(train)+255)//256):
            batch = x[indexes].copy()
            if config["gainDB"]:
                offsets = rng.uniform(-config["gainDB"], config["gainDB"], len(indexes))
                for local, original in enumerate(indexes):
                    for feature in (0, 2, 3):
                        batch[local, :lengths[original], feature] += offsets[local]/scales[feature]
            np.clip(batch, -8, 8, out=batch)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_function(network(torch.from_numpy(batch), torch.from_numpy(lengths[indexes])), torch.from_numpy(labels[indexes]))
            loss = (losses*torch.from_numpy(sample_weights[indexes])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite sequence training objective.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        entry = {"epoch": epoch, "seconds": time.monotonic()-started}
        if validation:
            score = metrics(validation, torch_probabilities(network, vx, vl).tolist())
            primary = (min(score["macroF1"]/reference_score["macroF1"], score["balancedAccuracy"]/reference_score["balancedAccuracy"])
                       if reference_score else score["macroF1"])
            key = (all(x > 0 for x in score["recallPerClass"]), primary, score["macroF1"], score["balancedAccuracy"])
            entry.update(macroF1=score["macroF1"], balancedAccuracy=score["balancedAccuracy"], recalls=score["recallPerClass"])
            if best_key is None or key > best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best = copy.deepcopy(network.state_dict())
            else:
                stale += 1
        log.append(entry)
        print(json.dumps({"event": "epoch", **entry}), flush=True)
        if validation and epoch >= 8 and stale >= patience:
            break
    if validation:
        network.load_state_dict(best)
    else:
        best_epoch = fixed_epochs
    network.eval()
    model = export(network, config, means, scales, sorted({r["subject"] for r in validation}))
    score = metrics(validation, torch_probabilities(network, vx, vl).tolist()) if validation else None
    # Check the independent portable equations against PyTorch before a checkpoint is accepted.
    check_rows = validation[:128] if validation else train[:128]
    ax, al, _ = arrays(check_rows, config["history"], means, scales)
    expected = torch_probabilities(network, ax, al)
    portable = SequencePredictor(model)
    actual = [portable.append(row) for row in check_rows]
    error = float(np.max(np.abs(expected-np.asarray(actual))))
    if error > 2e-10:
        raise ValueError(f"Portable recurrent equations differ from PyTorch: {error}")
    return model, score, {"bestEpoch": best_epoch, "epochsRun": len(log), "history": log, "portableMaximumDifference": error}


def run(args):
    import numpy as np
    import torch
    if torch.version.cuda is not None:
        raise ValueError("This bounded research job requires the CPU-only PyTorch build.")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"selection.json").exists():
        raise ValueError("Selection is already frozen.")
    rows, manifest = load_corpus(args.development, args.retired_test, args.previous_evaluation, args.registry)
    folds = {p: fold_for(p) for p in manifest["people"]}
    settings_list = BALANCED_CONFIGURATIONS if args.configuration_set == "balanced" else CONFIGURATIONS
    cached_reports, cache_record = {}, None
    if args.baseline_cache:
        prior = json.loads((args.baseline_cache/"protocol.json").read_text())
        if prior["development"] != manifest or prior["folds"] != folds:
            raise ValueError("Reuse baseline results only for the identical verified observations and subject folds.")
        for fold in range(3):
            path = args.baseline_cache/f"baseline-fold-{fold}.json"
            report = json.loads(path.read_text())
            expected_rows = [r for r in rows if folds[r["subject"]]==fold]
            if (report["fold"] != fold or report["validation"]["subjects"] != sorted({r["subject"] for r in expected_rows})
                    or report["validation"]["epochs"] != len(expected_rows) or report["fit"]["rounds"] != 160
                    or not report["fit"]["nativeExportMaximumDifference"] <= 2e-6
                    or any(sum(report["validation"]["confusionMatrix"][k]) != sum(r["stage"]==stage for r in expected_rows) for k,stage in enumerate(STAGES))):
                raise ValueError("Cached baseline does not match this fold's fitting/evaluation contract.")
            cached_reports[fold] = report
        cache_record = {"sourceProtocolSHA256":digest(args.baseline_cache/"protocol.json"),"sourceSha":prior["sourceSha"],
                        "reportsSHA256":{str(f):digest(args.baseline_cache/f"baseline-fold-{f}.json") for f in range(3)}}
    for fold in range(3):
        for held_out in (False, True):
            if {r["stage"] for r in rows if (folds[r["subject"]] == fold) == held_out} != set(STAGES):
                raise ValueError("Every fold needs training and validation coverage of all four stages.")
    config = {"schemaVersion": 1, "experiment": "causal-gru-v5", "development": manifest,
              "previouslyObservedSubjects": manifest["people"], "folds": folds, "configurations": settings_list,
              "seed": 7, "maximumEpochs": args.max_epochs, "patience": args.patience,
              "classWeightPower": 1.0 if args.configuration_set == "balanced" else 0.75, "subjectBalancedLoss": True, "labelSmoothing": 0.02,
              "optimizer": "AdamW", "learningRate": 0.001, "weightDecay": 0.01, "dropout": 0.15,
              "developmentExtractorSHA256": manifest["extractorSHA256"], "futureAudio": False,
              "previousEvaluationSHA256": digest(args.previous_evaluation), "exclusionRegistrySHA256": digest(args.registry),
              "baselineCache": cache_record,
              "checkpointMetric": "Minimum relative macro F1 and balanced-accuracy improvement over the paired baseline" if args.configuration_set=="balanced" else "Macro F1",
              "selectionMetric": "Mean three-fold macro F1, then balanced accuracy; every fold must recognize every stage.",
              "finalRefitEpochs": "Median best epoch over three validation folds, fixed before fitting all development people.",
              "torchVersion": torch.__version__, "numpyVersion": np.__version__, "torchCPUOnly": True,
              "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_text()) != config:
        raise ValueError("Resume only the identical frozen development experiment.")
    atomic_json(root/"protocol.json", config)
    baseline, candidates = [], []
    for fold in range(3):
        train = [r for r in rows if folds[r["subject"]] != fold]
        validation = [r for r in rows if folds[r["subject"]] == fold]
        saved = root/f"baseline-fold-{fold}.json"
        if fold in cached_reports:
            report = cached_reports[fold]
            atomic_json(saved,report)
        elif saved.exists():
            report = json.loads(saved.read_text())
        else:
            transformed_train, transformed_val = transform(train, 10, SPECTRAL_NAMES), transform(validation, 10, SPECTRAL_NAMES)
            model, detail = boosted(transformed_train, transformed_val, 10, depth=2, balance=1, fixed_rounds=160)
            report = {"fold": fold, "validation": metrics(transformed_val, [predict(model,r["modelFeatures"]) for r in transformed_val]), "fit": detail}
            atomic_json(saved, report)
        baseline.append(report)
        print(json.dumps({"event": "baseline", "fold": fold, "macroF1": report["validation"]["macroF1"]}), flush=True)
    for index, settings in enumerate(settings_list):
        reports = []
        for fold in range(3):
            report_path, model_path = root/f"candidate-{index}-fold-{fold}.json", root/f"weights-{index}-fold-{fold}.json"
            if report_path.exists() and model_path.exists():
                report = json.loads(report_path.read_text())
                if digest(model_path) != report["modelFileSHA256"]:
                    raise ValueError("Cached sequence checkpoint changed.")
            else:
                train = [r for r in rows if folds[r["subject"]] != fold]
                validation = [r for r in rows if folds[r["subject"]] == fold]
                print(json.dumps({"event": "fit", "candidate": index, "fold": fold, "settings": settings}), flush=True)
                model, score, detail = fit(train, validation, settings, 7+fold, args.max_epochs, args.patience,
                                          reference_score=baseline[fold]["validation"] if args.configuration_set=="balanced" else None)
                atomic_json(model_path, model, compact=True)
                report = {"fold": fold, "validation": score, "fit": detail, "modelFileSHA256": digest(model_path)}
                atomic_json(report_path, report)
            reports.append(report)
        summary = {"id": index, "settings": settings, "folds": reports,
                   "meanMacroF1": float(np.mean([r["validation"]["macroF1"] for r in reports])),
                   "meanBalancedAccuracy": float(np.mean([r["validation"]["balancedAccuracy"] for r in reports])),
                   "everyFoldRecognizesEveryStage": all(all(x>0 for x in r["validation"]["recallPerClass"]) for r in reports)}
        candidates.append(summary)
        atomic_json(root/"candidates.json", candidates)
        print(json.dumps({"event": "candidate_complete", "id": index, "macroF1": summary["meanMacroF1"],
                          "balancedAccuracy": summary["meanBalancedAccuracy"]}), flush=True)
    eligible = [r for r in candidates if r["everyFoldRecognizesEveryStage"]]
    if not eligible:
        raise ValueError("No recurrent candidate recognizes every stage across all validation folds.")
    baseline_f1 = float(np.mean([r["validation"]["macroF1"] for r in baseline]))
    baseline_accuracy = float(np.mean([r["validation"]["balancedAccuracy"] for r in baseline]))
    improved = [r for r in eligible if r["meanMacroF1"]>baseline_f1 and r["meanBalancedAccuracy"]>baseline_accuracy]
    if not improved:
        atomic_json(root/"development-review.json",{"status":"not_improved","baselineMeanMacroF1":baseline_f1,
                    "baselineMeanBalancedAccuracy":baseline_accuracy,"freshTestUntouched":True})
        print(json.dumps({"event":"not_improved","freshTestUntouched":True}),flush=True)
        return
    winner = max(improved, key=lambda r:(r["meanMacroF1"],r["meanBalancedAccuracy"],-r["id"]))
    epochs = int(np.median([r["fit"]["bestEpoch"] for r in winner["folds"]]))
    model, _, detail = fit(rows, [], winner["settings"], 7, args.max_epochs, args.patience, fixed_epochs=epochs)
    encoded_bytes = len((json.dumps(model,separators=(",",":"))+"\n").encode())
    if encoded_bytes > 508000:
        raise ValueError("Final recurrent model exceeds the native import budget.")
    atomic_json(root/"selected-model.json", model, compact=True)
    selection = {**winner, "parametersSHA256": parameters_digest(model), "finalTestRead": False,
                 "trainingSubjects": manifest["people"], "validationSubjects": [],
                 "previouslyObservedSubjects": manifest["people"], "finalRefit": {"epochs": epochs, "fit": detail, "trainingSubjects": manifest["people"]},
                 "baselineCrossValidation": baseline, "payloadBytes": encoded_bytes,
                 "baselineMeanMacroF1": float(np.mean([r["validation"]["macroF1"] for r in baseline])),
                 "baselineMeanBalancedAccuracy": float(np.mean([r["validation"]["balancedAccuracy"] for r in baseline]))}
    atomic_json(root/"selection.json", selection)
    print(json.dumps({"event": "selected", "id": winner["id"], "meanMacroF1": winner["meanMacroF1"],
                      "meanBalancedAccuracy": winner["meanBalancedAccuracy"], "refitPeople": len(manifest["people"]),
                      "parametersSHA256": selection["parametersSHA256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--retired-test", type=Path, required=True)
    parser.add_argument("--previous-evaluation", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--configuration-set",choices=("initial","balanced"),default="initial")
    parser.add_argument("--baseline-cache",type=Path)
    args = parser.parse_args()
    if not 8 <= args.max_epochs <= 60 or not 3 <= args.patience <= 12:
        parser.error("Keep the CPU experiment within its declared epoch bounds.")
    run(args)
