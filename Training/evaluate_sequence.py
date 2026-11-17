"""One independent test for a frozen recurrent model; no optimization imports or calls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from causal_model import load_observations, transform, predict
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics, parameters_digest
from sequence_model import SequencePredictor


def run(args):
    root = args.selection.resolve()
    if (root/"evaluation.json").exists():
        raise ValueError("This independent cohort has already been evaluated; do not tune against it.")
    selection = json.loads((root/"selection.json").read_text())
    model = json.loads((root/"selected-model.json").read_text())
    protocol = json.loads((root/"protocol.json").read_text())
    plan = json.loads((args.test/"download-plan.json").read_text())
    if (parameters_digest(model) != selection["parametersSHA256"]
            or plan["selectedParametersSHA256"] != selection["parametersSHA256"]
            or plan["extractorSHA256"] != protocol["developmentExtractorSHA256"]):
        raise ValueError("Frozen parameters and matching signal processing are required.")
    if not plan["fittingForbidden"] or set(plan["testSubjects"]) & set(selection["previouslyObservedSubjects"]):
        raise ValueError("Final participants overlap development or permit fitting.")
    combined, hashes = hashlib.sha256(), {}
    for item in plan["files"]:
        data = (args.test/"features"/(Path(item["path"]).stem+".jsonl")).read_bytes()
        combined.update(data)
        hashes[item["path"]] = hashlib.sha256(data).hexdigest()
    if combined.hexdigest() != hashlib.sha256((args.test/"epochs.jsonl").read_bytes()).hexdigest():
        raise ValueError("Final-test observations differ from their extracted blocks.")
    rows = sorted(load_observations(args.test/"epochs.jsonl",model["rawFeatureNames"]),key=lambda r:(r["subject"],r["startSeconds"]))
    if {r["subject"] for r in rows} != set(plan["testSubjects"]) or {r["stage"] for r in rows} != set(STAGES):
        raise ValueError("Usable observations must cover every test participant and every scored stage before predictions.")
    predictor, person, values = SequencePredictor(model), None, []
    for row in rows:
        if person != row["subject"]:
            predictor.reset()
            person = row["subject"]
        values.append(predictor.append(row))
    score = metrics(rows,values)
    old = json.loads(args.baseline.read_text())
    if old["featureProfile"] != model["featureProfile"] or old["classes"] != model["classes"]:
        raise ValueError("Baseline stages and raw signal profile must match.")
    contextual = transform(rows,old["contextEpochs"],old["rawFeatureNames"])
    previous = metrics(contextual,[predict(old,row["modelFeatures"]) for row in contextual])
    exported = copy.deepcopy(model)
    exported.update(heldOutSubjects=plan["testSubjects"],balancedAccuracy=score["balancedAccuracy"])
    if parameters_digest(exported) != selection["parametersSHA256"]:
        raise ValueError("Evaluation changed inference parameters.")
    samples = []
    for subject in sorted(plan["testSubjects"])[:2]:
        pairs = [(row,p) for row,p in zip(rows,values) if row["subject"] == subject][:60]
        for row,probabilities in pairs:
            samples.append({k:row[k] for k in ("subject","startSeconds","features","usable")}
                           | {"expectedProbabilities":dict(zip(STAGES,probabilities))})
    atomic_json(root/"model.json",exported,compact=True)
    atomic_json(root/"export-check.json",samples)
    atomic_json(root/"evaluation.json",{"status":"experimental","approvedForAlarmTiming":False,"usesFutureAudio":False,
        "freshTest":score,"previousSpectralV4":previous,"baselineModelSHA256":hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "crossValidation":{"macroF1":selection["meanMacroF1"],"balancedAccuracy":selection["meanBalancedAccuracy"],
                           "baselineMacroF1":selection["baselineMeanMacroF1"],"baselineBalancedAccuracy":selection["baselineMeanBalancedAccuracy"]},
        "parametersSHA256":selection["parametersSHA256"],"testEpochsSHA256":combined.hexdigest(),"testFeatureSHA256":hashes,
        "trainingPeople":len(selection["trainingSubjects"]),"testPeople":len(plan["testSubjects"]),
        "stageParticipantCounts":{s:len({r["subject"] for r in rows if r["stage"]==s}) for s in STAGES},
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()})
    atomic_json(root/"known-subjects-after-test.json",sorted(set(selection["previouslyObservedSubjects"])|set(plan["testSubjects"])))
    print(json.dumps({"event":"evaluated","freshPeople":len(plan["testSubjects"]),"balancedAccuracy":score["balancedAccuracy"],
                      "macroF1":score["macroF1"],"baselineBalancedAccuracy":previous["balancedAccuracy"],"baselineMacroF1":previous["macroF1"]}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--test",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    run(parser.parse_args())
