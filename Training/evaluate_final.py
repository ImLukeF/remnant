"""Evaluate a frozen selection once; fitting is deliberately absent from this entry point."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from causal_model import predict, transform, load_observations
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics, parameters_digest
from train import evaluate, load_rows


def compare_legacy(model, rows):
    if list(model["classes"]) != list(STAGES):
        raise ValueError("Comparison model uses an incompatible class order.")
    result = evaluate(model, rows)
    matrix = result["confusionMatrix"]
    f1 = [2 * matrix[k][k] / (sum(matrix[k]) + sum(row[k] for row in matrix)) for k in range(4)]
    result.update(f1PerClass=f1, macroF1=sum(f1) / 4)
    return result


def run(args):
    root = args.selection.resolve()
    if (root / "evaluation.json").exists():
        raise ValueError("Final evaluation is already recorded; do not tune and re-test this cohort.")
    selection = json.loads((root / "selection.json").read_text())
    model = json.loads((root / "selected-model.json").read_text())
    plan = json.loads((args.test / "download-plan.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["developmentExtractorSHA256"] != plan["extractorSHA256"]:
        raise ValueError("Training and final-test signal processing must match.")
    if parameters_digest(model) != selection["parametersSHA256"] or plan["selectedParametersSHA256"] != selection["parametersSHA256"]:
        raise ValueError("Final testing must use the parameters frozen before test preparation.")
    known = set(selection["trainingSubjects"] + selection["validationSubjects"])
    known.update(selection.get("previouslyObservedSubjects", []))
    if known & set(plan["testSubjects"]) or not plan["fittingForbidden"]:
        raise ValueError("Final test does not meet participant isolation requirements.")
    combined = hashlib.sha256()
    feature_hashes = {}
    for item in plan["files"]:
        data = (args.test / "features" / (Path(item["path"]).stem + ".jsonl")).read_bytes()
        feature_hashes[item["path"]] = hashlib.sha256(data).hexdigest()
        combined.update(data)
    if combined.hexdigest() != hashlib.sha256((args.test / "epochs.jsonl").read_bytes()).hexdigest():
        raise ValueError("Final test epochs do not match extracted feature files.")
    rows = load_observations(args.test / "epochs.jsonl", model["rawFeatureNames"])
    if {row["subject"] for row in rows} != set(plan["testSubjects"]):
        raise ValueError("One or more final-test people have no usable observations.")
    if {row["stage"] for row in rows} != set(STAGES):
        raise ValueError("Usable final-test observations must cover all four stages before predictions.")
    contextual = transform(rows, model["contextEpochs"], model["rawFeatureNames"])
    values = [predict(model, row["modelFeatures"]) for row in contextual]
    score = metrics(contextual, values)
    older = {}
    for name, path in (("original14", args.baseline14), ("expanded40", args.baseline40)):
        old = json.loads(path.read_text())
        older[name] = {"modelSHA256": hashlib.sha256(path.read_bytes()).hexdigest(), **compare_legacy(old, rows)}
    if args.baseline_causal:
        old = json.loads(args.baseline_causal.read_text())
        past = transform(rows, old["contextEpochs"], old["rawFeatureNames"])
        older["causalV3"] = {"modelSHA256": hashlib.sha256(args.baseline_causal.read_bytes()).hexdigest(),
                             **metrics(past, [predict(old, row["modelFeatures"]) for row in past])}
    exported = copy.deepcopy(model)
    exported.update(heldOutSubjects=plan["testSubjects"], balancedAccuracy=score["balancedAccuracy"])
    if parameters_digest(exported) != selection["parametersSHA256"]:
        raise ValueError("Evaluation modified model parameters.")
    samples = []
    for subject in sorted(plan["testSubjects"])[:2]:
        count = 0
        for row, probabilities in zip(contextual, values):
            if row["subject"] == subject and count < 60:
                samples.append({key: row[key] for key in ("subject", "startSeconds", "features", "usable")}
                              | {"expectedProbabilities": dict(zip(STAGES, probabilities))})
                count += 1
    atomic_json(root / "model.json", exported, compact=True)
    atomic_json(root / "export-check.json", samples)
    atomic_json(root / "evaluation.json", {"status": "experimental", "approvedForAlarmTiming": False, "usesFutureAudio": False,
                "freshTest": score, "comparisons": older, "selectedCandidate": selection["id"],
                "validation": selection["validation"], "trainingPeople": len(selection["trainingSubjects"]),
                "validationPeople": len(selection["validationSubjects"]), "parametersSHA256": selection["parametersSHA256"],
                "testEpochsSHA256": combined.hexdigest(), "testFeatureSHA256": feature_hashes,
                "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "testStageParticipantCounts": {stage: len({row["subject"] for row in rows if row["stage"] == stage}) for stage in STAGES}})
    atomic_json(root / "known-subjects-after-test.json", sorted(set(plan["previouslyObservedSubjects"]) | set(plan["testSubjects"])))
    print(json.dumps({"event": "evaluated", "people": len(plan["testSubjects"]), "balancedAccuracy": score["balancedAccuracy"],
                      "lightRecall": score["recallPerClass"][1], "original14BalancedAccuracy": older["original14"]["balancedAccuracy"],
                      "expanded40BalancedAccuracy": older["expanded40"]["balancedAccuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--baseline14", type=Path, required=True)
    parser.add_argument("--baseline40", type=Path, required=True)
    parser.add_argument("--baseline-causal", type=Path)
    run(parser.parse_args())
