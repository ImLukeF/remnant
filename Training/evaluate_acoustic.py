"""Score a frozen acoustic model once on newly reserved clinical participants; never fit here."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from acoustic_data import load
from acoustic_model import predict_rows
from causal_model import transform,predict
from evaluation_contracts import participant_registry
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics,parameters_digest
from spectral import SPECTRAL_NAMES


def run(args):
    root=args.selection.resolve()
    if (root/"evaluation.json").exists(): raise ValueError("This independent cohort has already been scored; do not tune and re-test it.")
    selection=json.loads((root/"selection.json").read_text()); model=json.loads((root/"selected-model.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text()); plan=json.loads((args.test/"download-plan.json").read_text())
    registry=participant_registry(selection,plan)
    if (parameters_digest(model)!=selection["parametersSHA256"] or plan["selectedParametersSHA256"]!=selection["parametersSHA256"]
            or plan["extractorSHA256"]!=protocol["developmentExtractorSHA256"]):
        raise ValueError("Inference parameters and signal processing must remain frozen through final testing.")
    rows,frames,manifest=load(args.test)
    if (set(manifest["people"])!=set(plan["testSubjects"]) or {r["stage"] for r in rows}!=set(STAGES)
            or manifest["frameNames"]!=model["rawFeatureNames"]):
        raise ValueError("Every reserved person and scored stage must have usable matching observations before predictions.")
    values=predict_rows(model,rows,frames); score=metrics(rows,values)
    baseline=json.loads(args.baseline.read_text())
    if baseline["rawFeatureNames"]!=SPECTRAL_NAMES or baseline["classes"]!=list(STAGES):
        raise ValueError("Baseline summary measurements or stage order are incompatible.")
    contextual=transform(rows,baseline["contextEpochs"],baseline["rawFeatureNames"])
    previous=metrics(contextual,[predict(baseline,r["modelFeatures"]) for r in contextual])
    exported=copy.deepcopy(model); exported.update(heldOutSubjects=plan["testSubjects"],balancedAccuracy=score["balancedAccuracy"])
    if parameters_digest(exported)!=selection["parametersSHA256"]: raise ValueError("Final evaluation modified model parameters.")
    samples=[]
    for subject in sorted(plan["testSubjects"])[:2]:
        indexes=[i for i,r in enumerate(rows) if r["subject"]==subject][:60]
        for index in indexes:
            row=rows[index]
            samples.append({k:row[k] for k in ("subject","startSeconds","features","usable")}
                           | {"acousticFrames":frames[index].reshape(-1).tolist(),"expectedProbabilities":dict(zip(STAGES,values[index]))})
    atomic_json(root/"model.json",exported,compact=True)
    atomic_json(root/"export-check.json",samples,compact=True)
    atomic_json(root/"evaluation.json",{"status":"experimental","approvedForAlarmTiming":False,"usesFutureAudio":False,
        "freshTest":score,"previousSpectralV4":previous,"baselineModelSHA256":hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "crossValidation":{"macroF1":selection["meanMacroF1"],"balancedAccuracy":selection["meanBalancedAccuracy"],
                           "baselineMacroF1":selection["baselineMeanMacroF1"],"baselineBalancedAccuracy":selection["baselineMeanBalancedAccuracy"]},
        "trainingPeople":len(selection["trainingSubjects"]),"testPeople":len(plan["testSubjects"]),"testManifest":manifest,
        "parametersSHA256":selection["parametersSHA256"],"stageParticipantCounts":{s:len({r["subject"] for r in rows if r["stage"]==s}) for s in STAGES},
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()})
    atomic_json(root/"known-subjects-after-test.json",registry)
    print(json.dumps({"event":"evaluated","freshPeople":len(plan["testSubjects"]),"balancedAccuracy":score["balancedAccuracy"],
                      "macroF1":score["macroF1"],"baselineBalancedAccuracy":previous["balancedAccuracy"],"baselineMacroF1":previous["macroF1"]}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path,required=True); p.add_argument("--test",type=Path,required=True); p.add_argument("--baseline",type=Path,required=True)
    run(p.parse_args())
