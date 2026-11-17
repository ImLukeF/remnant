"""Once-only frozen comparison on reserved new people; this entry point never fits a model."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from array_trees import probabilities
from expanded_data import load,context
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics,parameters_digest
from run_dataset import checksum


def score(view,values):
    rows=[{"subject":str(person),"stage":STAGES[int(label)]} for person,label in zip(view["people"],view["labels"])]
    return metrics(rows,values)


def paired_intervals(view,new,old,samples=1000):
    import numpy as np
    people=np.unique(view["people"]);matrices=[]
    for person in people:
        mask=view["people"]==person;labels=view["labels"][mask].astype(int)
        matrices.append([np.bincount(4*labels+values[mask].argmax(axis=1),minlength=16).reshape((4,4)) for values in (new,old)])
    matrices=np.asarray(matrices);rng=np.random.default_rng(7);changes=[]
    for _ in range(samples):
        chosen=matrices[rng.integers(0,len(people),len(people))].sum(axis=0)
        row_totals=chosen.sum(axis=2)
        if (row_totals==0).any(): continue
        recall=np.diagonal(chosen,axis1=1,axis2=2)/row_totals
        f1=2*np.diagonal(chosen,axis1=1,axis2=2)/(row_totals+chosen.sum(axis=1))
        changes.append([float(recall[0].mean()-recall[1].mean()),float(f1[0].mean()-f1[1].mean())])
    bounds=np.quantile(np.asarray(changes),[0.025,0.975],axis=0)
    return {"method":"Paired participant bootstrap, 1000 resamples, seed 7","validResamples":len(changes),
            "balancedAccuracyDifference95":bounds[:,0].tolist(),"macroF1Difference95":bounds[:,1].tolist()}


def run(args):
    import numpy as np
    root=args.selection.resolve()
    if (root/"evaluation.json").exists(): raise ValueError("This independent cohort was already scored; do not tune and re-test it.")
    selection=json.loads((root/"selection.json").read_text());protocol=json.loads((root/"protocol.json").read_text())
    model=json.loads((root/"selected-model.json").read_text());plan=json.loads((args.test/"development-plan.json").read_text())
    held_out=set(plan["roles"]);known=set(protocol["previouslyObservedSubjects"])
    if (not plan["fittingForbidden"] or plan["phase"]!="final" or set(plan["roles"].values())!={"test"}
            or held_out!=set(protocol["reservedTestSubjects"]) or len(held_out)!=30 or held_out&known
            or set(plan["previouslyObservedSubjects"])!=known
            or parameters_digest(model)!=selection["parametersSHA256"]
            or set(protocol.get("refitSubjects",known))!=set(selection.get("trainingSubjects",known))
            or plan["selectedParametersSHA256"]!=selection["parametersSHA256"]
            or checksum(args.baseline)!=protocol["baselineModelSHA256"]):
        raise ValueError("Final participants, incumbent and selected inference parameters must remain frozen.")
    arrays,_,manifest=load(args.test,phase="final")
    if (manifest["extractorSHA256"]!=protocol["development"]["extractorSHA256"]
            or manifest["noiseManifestSHA256"]!=protocol["development"]["noiseManifestSHA256"]):
        raise ValueError("Training and final measurements use different signal processing or noise sources.")
    views={"clean":context(arrays),"noisy":context(arrays,noisy=True)}
    reference_path=root/"full-night-original-84-weights.json"
    reference_report=json.loads((root/"full-night-original-84-report.json").read_text())
    if checksum(reference_path)!=reference_report["modelFileSHA256"]: raise ValueError("The full-night control weights changed.")
    trained=len(protocol.get("refitSubjects",known))
    reference_people=reference_report.get("fit",{}).get("trainingPeople",84)
    expanded_key=f"expanded{trained}";reference_key=f"fullNight{reference_people}"
    models={expanded_key:model,reference_key:json.loads(reference_path.read_text()),"incumbent72":json.loads(args.baseline.read_text())}
    results={};predictions={}
    for name,parameters in models.items():
        results[name]={};predictions[name]={}
        for condition,view in views.items():
            values=probabilities(parameters,view["inputs"])
            predictions[name][condition]=values;result=score(view,values)
            matrix=result["confusionMatrix"];nonlight=sum(matrix[k][j] for k in (2,3) for j in range(4))
            result["deepOrREMFalseLightRate"]=(matrix[2][1]+matrix[3][1])/nonlight if nonlight else None
            results[name][condition]=result
    intervals={condition:paired_intervals(view,predictions[expanded_key][condition],predictions["incumbent72"][condition])
               for condition,view in views.items()}
    exported=copy.deepcopy(model);exported.update(heldOutSubjects=sorted(held_out),balancedAccuracy=results[expanded_key]["clean"]["balancedAccuracy"])
    if parameters_digest(exported)!=selection["parametersSHA256"]: raise ValueError("Final scoring modified inference parameters.")
    samples=[];view=views["clean"]
    for person in sorted(held_out)[:2]:
        indexes=np.flatnonzero(view["people"]==person)[:60]
        for index in indexes:
            original=view["indexes"][index]
            samples.append({"subject":person,"startSeconds":float(view["starts"][index]),"features":arrays["features"][original].tolist(),
                "usable":True,"expectedProbabilities":dict(zip(STAGES,predictions[expanded_key]["clean"][index].tolist()))})
    atomic_json(root/"model.json",exported,compact=True)
    atomic_json(root/"export-check.json",samples,compact=True)
    atomic_json(root/"known-subjects-after-test.json",sorted(known|held_out))
    atomic_json(root/"evaluation.json",{"status":"experimental","approvedForAlarmTiming":False,"usesFutureAudio":False,
        "trainingPeople":trained,"originalControlTrainingPeople":reference_people,"testPeople":30,"results":results,"pairedParticipantIntervals":intervals,
        "qualityExcludedSubjects":protocol.get("qualityExcludedSubjects",{}),
        "testManifest":manifest,"parametersSHA256":selection["parametersSHA256"],"modelFileSHA256":checksum(root/"model.json"),
        "stageParticipantCounts":{s:len(set(views["clean"]["people"][views["clean"]["labels"]==k])) for k,s in enumerate(STAGES)},
        "controlledNoiseIsSynthetic":True,"realPhoneOvernightValidated":False,
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()})
    print(json.dumps({"event":"expanded_final_evaluated","people":30,"cleanBalancedAccuracy":results[expanded_key]["clean"]["balancedAccuracy"],
                      "incumbentCleanBalancedAccuracy":results["incumbent72"]["clean"]["balancedAccuracy"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("selection","test","baseline"): p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
