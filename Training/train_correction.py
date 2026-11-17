"""Bounded cached-data weighting/noise/calibration experiment; reserved final people are never opened."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from array_trees import probabilities
from causal_model import names
from expanded_contracts import admitted
from expanded_data import load,context
from expand_cohort import atomic_json
from features import STAGES
from improve_model import parameters_digest
from run_dataset import checksum
from spectral import SPECTRAL_NAMES
from train_expanded import fit_arrays,quality_key,score_views,PARAMETERS

RECIPES=[{"classPower":0.5,"noiseShare":0.25},{"classPower":0.75,"noiseShare":0.25},{"classPower":0.75,"noiseShare":0.5}]


def calibration_split(people):
    ordered=sorted(people,key=lambda p:hashlib.sha256(("correction-v9-calibration:"+p).encode()).hexdigest())
    if len(set(ordered))!=159: raise ValueError("Use the 159 usable fitting participants from the frozen larger cohort.")
    return sorted(ordered[24:]),sorted(ordered[:24])


def calibrate(model,views,people):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp,softmax
    logits=[];labels=[];weights=[]
    for view in views.values():
        mask=np.isin(view["people"],people);ids=view["people"][mask]
        if set(ids)!=set(people): raise ValueError("Every calibration person must have both measured conditions.")
        _,inverse,counts=np.unique(ids,return_inverse=True,return_counts=True)
        logits.append(np.log(np.maximum(probabilities(model,view["inputs"][mask]),1e-12)))
        labels.append(view["labels"][mask]);weights.append(1/counts[inverse])
    logits=np.concatenate(logits);labels=np.concatenate(labels);weights=np.concatenate(weights);weights/=weights.sum()
    if set(labels)!=set(range(4)): raise ValueError("Calibration requires measured examples of every stage.")
    def objective(bias):
        values=logits+bias;probs=softmax(values,axis=1)
        loss=float(np.sum(weights*(logsumexp(values,axis=1)-values[np.arange(len(labels)),labels]))+0.1*np.mean(bias*bias))
        probs[np.arange(len(labels)),labels]-=1
        return loss,(probs*weights[:,None]).sum(axis=0)+0.05*bias
    result=minimize(objective,np.zeros(4),jac=True,method="L-BFGS-B",bounds=[(-0.75,0.75)]*4)
    if not result.success: raise ValueError("Calibration did not converge.")
    bias=result.x-result.x.mean()
    return bias.tolist(),{"people":len(people),"epochs":len(labels),"iterations":int(result.nit),"objective":float(result.fun),
        "beforeObjective":objective(np.zeros(4))[0],"usesValidationPeople":False,"regularization":0.1,"bias":bias.tolist()}


def adjusted(model,bias):
    import numpy as np
    result=copy.deepcopy(model)
    result["biases"]=(np.asarray(result["biases"],dtype=np.float32)+np.asarray(bias,dtype=np.float32)).tolist()
    return result


def wake_errors(score):
    output={}
    for condition,result in score.items():
        matrix=result["confusionMatrix"];nonlight=sum(sum(matrix[k]) for k in (2,3))
        output[condition]={"lightPrecision":result["lightPrecision"],"lightRecall":result["recallPerClass"][1],
            "deepOrREMCalledLight":(matrix[2][1]+matrix[3][1])/nonlight}
    return output


def run(args):
    import numpy as np
    began=time.monotonic();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()): raise ValueError("Use a new private experiment output; never overwrite completed results.")
    arrays,plan,manifest=load(args.development)
    previous=json.loads((args.previous/"protocol.json").read_text())
    if previous["development"]!=manifest or checksum(args.baseline)!=previous["baselineModelSHA256"]:
        raise ValueError("Reuse exactly the verified larger cohort and incumbent.")
    training=list(previous["trainingSubjects"]);validation=list(previous["validationSubjects"]);refit=list(previous["refitSubjects"])
    inner,calibration=calibration_split(training)
    if (len(validation)!=40 or len(refit)!=199 or set(inner)&set(calibration) or (set(inner)|set(calibration))&set(validation)
            or set(refit)!=set(training)|set(validation) or set(refit)&set(plan["reservedTestSubjects"])):
        raise ValueError("Fitting, calibration, validation and final people must remain separated.")
    protocol={"experiment":"cached-correction-v9","development":manifest,"trainingSubjects":training,
        "innerFittingSubjects":inner,"calibrationSubjects":calibration,"validationSubjects":validation,"refitSubjects":refit,
        "previouslyObservedSubjects":previous["previouslyObservedSubjects"],"reservedTestSubjects":plan["reservedTestSubjects"],
        "qualityExcludedSubjects":previous["qualityExcludedSubjects"],"baselineModelSHA256":checksum(args.baseline),
        "parameters":PARAMETERS,"rounds":160,"recipes":RECIPES,"calibration":"Four regularized biases, inner calibration people only; raw and calibrated checkpoints compared.",
        "selection":"Unchanged paired clean/noisy F1 and balanced-recall admission; every stage recognized; same minimum relative gain ranking.",
        "usesFutureAudio":False,"finalTestRead":False,"maximumSeconds":args.max_seconds,
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    atomic_json(root/"protocol.json",protocol)
    for name in ("full-night-original-84-weights.json","full-night-original-84-report.json"):
        shutil.copyfile(args.previous/name,root/name)
    reference=json.loads((root/"full-night-original-84-report.json").read_text())
    if checksum(root/"full-night-original-84-weights.json")!=reference["modelFileSHA256"]:
        raise ValueError("The retained original-cohort control changed.")
    baseline=json.loads(args.baseline.read_text())
    print(json.dumps({"event":"building_correction_views","fittingPeople":len(inner),"calibrationPeople":len(calibration),"validationPeople":len(validation)}),flush=True)
    views={"clean":context(arrays),"noisy":context(arrays,noisy=True)}
    baseline_scores=score_views(baseline,views,validation);reports={};candidates={}
    def fit(people,recipe,probe_people):
        masks=[np.isin(v["people"],people) for v in views.values()]
        inputs=np.concatenate([v["inputs"][m] for v,m in zip(views.values(),masks)])
        labels=np.concatenate([v["labels"][m] for v,m in zip(views.values(),masks)])
        ids=np.concatenate([v["people"][m] for v,m in zip(views.values(),masks)])
        flags=np.concatenate([np.full(int(m.sum()),i==1,dtype=bool) for i,m in enumerate(masks)])
        if set(ids)!=set(people): raise ValueError("A fitting participant disappeared.")
        probe=views["clean"]["inputs"][np.isin(views["clean"]["people"],probe_people)][:256]
        return fit_arrays(inputs,labels,ids,probe,class_power=recipe["classPower"],noise_flags=flags,noise_share=recipe["noiseShare"])
    for number,recipe in enumerate(RECIPES):
        if time.monotonic()-began>args.max_seconds-180: raise TimeoutError("The bounded experiment has insufficient time for another fixed fit.")
        print(json.dumps({"event":"correction_fit","recipe":number,"settings":recipe}),flush=True)
        model,fit_report=fit(inner,recipe,calibration);bias,calibration_report=calibrate(model,views,calibration)
        for suffix,parameters,offset in (("raw",model,[0]*4),("calibrated",adjusted(model,bias),bias)):
            name=f"recipe-{number}-{suffix}";score=score_views(parameters,views,validation)
            path=root/(name+"-weights.json");atomic_json(path,parameters,compact=True)
            report={"fit":fit_report,"calibration":calibration_report if suffix=="calibrated" else None,
                "validation":score,"wakeErrors":wake_errors(score),"admitted":admitted(score,baseline_scores),"modelFileSHA256":checksum(path)}
            reports[name]=report;candidates[name]=(recipe,offset)
            atomic_json(root/(name+"-report.json"),report)
            print(json.dumps({"event":"correction_validation","id":name,"admitted":report["admitted"],
                "cleanF1":score["clean"]["macroF1"],"noisyF1":score["noisy"]["macroF1"]}),flush=True)
        atomic_json(root/"progress.json",{"validation":reports,"baselineValidation":baseline_scores,"finalTestRead":False})
    eligible=[name for name,r in reports.items() if r["admitted"]]
    summary={"status":"admitted" if eligible else "trained_not_improved","validation":reports,"baselineValidation":baseline_scores,
        "fittingPeople":135,"innerFittingPeople":135,"calibrationPeople":24,"validationPeople":40,"finalTestRead":False,"approvedForAlarmTiming":False,
        "qualityExcludedSubjects":previous["qualityExcludedSubjects"],"sourceSha":protocol["sourceSha"]}
    if eligible:
        chosen=max(eligible,key=lambda name:quality_key(reports[name]["validation"],baseline_scores));recipe,bias=candidates[chosen]
        print(json.dumps({"event":"correction_refit","chosen":chosen,"people":len(refit)}),flush=True)
        model,fit_report=fit(refit,recipe,refit[:2]);model=adjusted(model,bias)
        model.update(schemaVersion=2,modelID="psg-audio-correction-v9",dataSource="PSG-Audio V3, CC BY 4.0; audited MUSAN augmentation",
            featureProfile="spectral-v4",rawFeatureNames=SPECTRAL_NAMES,featureNames=names(10,SPECTRAL_NAMES),contextEpochs=10,
            classes=STAGES,heldOutSubjects=[],balancedAccuracy=0)
        path=root/"selected-model.json";atomic_json(path,model,compact=True)
        if path.stat().st_size>508000: raise ValueError("The model exceeds the native import budget.")
        digest=parameters_digest(model)
        summary.update(chosenRecipe=chosen,parametersSHA256=digest,finalRefitPeople=len(refit),refit={"fit":fit_report,"bias":bias,"modelFileSHA256":checksum(path)})
        atomic_json(root/"training-summary.json",summary)
        atomic_json(root/"selection.json",{"id":chosen,"trainingSubjects":refit,"validationSubjects":[],
            "qualityExcludedSubjects":previous["qualityExcludedSubjects"],"previouslyObservedSubjects":previous["previouslyObservedSubjects"],
            "parametersSHA256":digest,"finalTestRead":False,"controlledNoiseAdmissionPassed":True})
    else:
        summary["finalRefitPeople"]=0
        atomic_json(root/"training-summary.json",summary);atomic_json(root/"development-review.json",summary)
    print(json.dumps({"event":"correction_complete","status":summary["status"],"seconds":time.monotonic()-began,"finalTestRead":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("development","previous","baseline","output"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--max-seconds",type=int,default=1100);args=p.parse_args()
    if not 300<=args.max_seconds<=1200: p.error("Use a bounded 5-20 minute experiment.")
    run(args)
