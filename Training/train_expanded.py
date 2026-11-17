"""Train larger full-night cohorts with fixed trees and a genuinely untouched final reservation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from array_trees import probabilities
from causal_model import names
from expanded_data import load, context
from expanded_contracts import admitted,participant_identities
from expand_cohort import atomic_json
from features import STAGES
from improve_model import export_booster, metrics, parameters_digest
from run_dataset import checksum
from spectral import SPECTRAL_NAMES

PARAMETERS={"objective":"multi:softprob","num_class":4,"tree_method":"hist","max_depth":2,
            "eta":0.05,"min_child_weight":10,"lambda":10,"subsample":1,"colsample_bytree":1,
            "seed":7,"nthread":3,"base_score":0}


def fit_arrays(inputs,labels,people,probe,rounds=160,class_power=1.0,noise_flags=None,noise_share=0.5):
    import numpy as np
    import xgboost as xgb
    if not 1<=rounds<=320: raise ValueError("Bound the fixed tree fitting rounds.")
    if not 0<=class_power<=1: raise ValueError("Use a bounded class-weight power.")
    counts=np.bincount(labels,minlength=4)
    if len(counts)!=4 or (counts==0).any(): raise ValueError("Every fitting group must include all four stages.")
    unique,inverse,person_counts=np.unique(people,return_inverse=True,return_counts=True)
    weights=(len(labels)/(4*counts[labels]))**class_power*(len(labels)/(len(unique)*person_counts[inverse]))
    if noise_flags is not None:
        flags=np.asarray(noise_flags,dtype=bool)
        if flags.shape!=labels.shape or not flags.any() or flags.all() or not 0<noise_share<1:
            raise ValueError("Weighted noise training requires both conditions and a bounded mixture.")
        weights[flags]*=noise_share/weights[flags].sum()
        weights[~flags]*=(1-noise_share)/weights[~flags].sum()
    weights/=weights.mean()
    began=time.monotonic()
    source=xgb.DMatrix(inputs,label=labels,weight=weights,nthread=3)
    booster=xgb.train(PARAMETERS,source,num_boost_round=rounds,verbose_eval=False)
    model=export_booster(booster)
    expected=booster.predict(xgb.DMatrix(probe,nthread=3))
    difference=float(np.max(np.abs(probabilities(model,probe)-expected)))
    if difference>2e-6: raise ValueError("Native-compatible export differs from the training library.")
    return model,{"rounds":rounds,"trainingPeople":len(unique),"trainingEpochs":len(labels),"stageCounts":counts.tolist(),
                  "subjectBalancedLoss":True,"classBalancedLoss":class_power==1,"classWeightPower":class_power,
                  "noiseWeightShare":noise_share if noise_flags is not None else None,"exportMaximumDifference":difference,
                  "actualNoiseWeightShare":float(weights[flags].sum()/weights.sum()) if noise_flags is not None else None,
                  "seconds":time.monotonic()-began}


def rows(view,mask):
    return [{"subject":str(p),"stage":STAGES[int(k)]} for p,k in zip(view["people"][mask],view["labels"][mask])]


def score_views(model,views,validation):
    import numpy as np
    result={}
    for name,view in views.items():
        mask=np.isin(view["people"],validation)
        result[name]=metrics(rows(view,mask),probabilities(model,view["inputs"][mask]))
    return result


def quality_key(score,baseline):
    ratios=[score[name][metric]/max(baseline[name][metric],1e-12) for name in ("clean","noisy") for metric in ("macroF1","balancedAccuracy")]
    return min(ratios),sum(score[name]["macroF1"] for name in ("clean","noisy"))/2


def run(args):
    import numpy as np
    import xgboost
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/"selection.json").exists() or (root/"development-review.json").exists(): raise ValueError("This expanded experiment is complete.")
    arrays,plan,manifest=load(args.development)
    identities=participant_identities(args.development/"labels",set(plan["roles"]))
    reservation=json.loads((args.development/"participants.json").read_text())
    known=json.loads(args.original_registry.read_text())
    if checksum(args.original_registry)!=reservation["knownRegistrySHA256"] or any(plan["roles"].get(p)!="train" for p in known):
        raise ValueError("Previously seen people must remain in training, outside both validation and final testing.")
    training=sorted(p for p,r in plan["roles"].items() if r=="train")
    validation=sorted(p for p,r in plan["roles"].items() if r=="validation")
    if len(training)!=160 or len(validation)!=40 or len(plan["reservedTestSubjects"])!=30:
        raise ValueError("Use the frozen 160/40/30 participant protocol.")
    excluded=manifest.get("qualityExcludedSubjects",{})
    if not set(excluded).issubset(training): raise ValueError("Quality exclusions cannot change independent validation or final-test people.")
    training=[p for p in training if p not in excluded]
    refit_people=sorted(set(plan["roles"])-set(excluded))
    reference_people=sorted(set(known)-set(excluded))
    baseline=json.loads(args.baseline.read_text())
    if baseline["rawFeatureNames"]!=SPECTRAL_NAMES or baseline["contextEpochs"]!=10 or baseline["classes"]!=list(STAGES):
        raise ValueError("The incumbent input/stage contract changed.")
    protocol={"experiment":"expanded-full-nights-v8","development":manifest,"trainingSubjects":training,"validationSubjects":validation,
        "reservedTestSubjects":plan["reservedTestSubjects"],"previouslyObservedSubjects":sorted(plan["roles"]),
        "baselineModelSHA256":checksum(args.baseline),"originalRegistrySHA256":checksum(args.original_registry),
        "participantIdentityHashes":identities,
        "qualityExcludedSubjects":excluded,"refitSubjects":refit_people,
        "parameters":PARAMETERS,"rounds":160,"contextEpochs":10,"subjectBalancedLoss":True,"classBalancedLoss":True,
        "candidates":["clean","clean-plus-room-noise"],"reference":"Full-night original 84, same fixed fitting recipe",
        "admission":"Both macro F1 and balanced accuracy must improve on both clean and controlled-noise validation; every stage recognized.",
        "selection":"Best minimum relative metric improvement, then mean clean/noisy macro F1.",
        "refit":f"Selected fixed recipe on all {len(refit_people)} usable development people. Unadmitted weights remain experimental and final-test people stay untouched.",
        "xgboostVersion":xgboost.__version__,"numpyVersion":np.__version__,"usesFutureAudio":False,"finalTestRead":False,
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_text())!=protocol:
        raise ValueError("Resume only the identical frozen expanded fitting experiment.")
    atomic_json(root/"protocol.json",protocol)
    print(json.dumps({"event":"building_causal_views","people":len(plan["roles"]),"scoredEpochs":len(arrays["stage"])}),flush=True)
    views={"clean":context(arrays),"noisy":context(arrays,noisy=True)}
    baseline_scores=score_views(baseline,views,validation);reports={}
    def fit_cached(name,people,augmented,final_refit=False):
        path=root/(name+"-weights.json");report_path=root/(name+"-report.json")
        if path.exists() and report_path.exists():
            model=json.loads(path.read_text());report=json.loads(report_path.read_text())
            if checksum(path)!=report["modelFileSHA256"]: raise ValueError("A cached model checkpoint changed.")
            return model,report
        chosen=[views["clean"]]+([views["noisy"]] if augmented else [])
        masks=[np.isin(view["people"],people) for view in chosen]
        inputs=np.concatenate([v["inputs"][m] for v,m in zip(chosen,masks)])
        labels=np.concatenate([v["labels"][m] for v,m in zip(chosen,masks)])
        person_ids=np.concatenate([v["people"][m] for v,m in zip(chosen,masks)])
        if set(person_ids)!=set(people): raise ValueError("A fitting participant disappeared without the frozen quality exclusion.")
        probe=inputs[:128] if final_refit else views["clean"]["inputs"][np.isin(views["clean"]["people"],validation)]
        print(json.dumps({"event":"expanded_fit","name":name,"people":len(people),"rows":len(labels)}),flush=True)
        model,fit=fit_arrays(inputs,labels,person_ids,probe)
        atomic_json(path,model,compact=True)
        report={"fit":fit,"modelFileSHA256":checksum(path)}
        if not final_refit: report["validation"]=score_views(model,views,validation)
        atomic_json(report_path,report)
        return model,report
    _,reference=fit_cached("full-night-original-84",reference_people,False)
    for label,augmented in (("clean",False),("clean-plus-room-noise",True)):
        _,report=fit_cached(label,training,augmented)
        report["admitted"]=admitted(report["validation"],baseline_scores);reports[label]=report
        print(json.dumps({"event":"expanded_validation","candidate":label,"admitted":report["admitted"],
            "cleanBalancedAccuracy":report["validation"]["clean"]["balancedAccuracy"],"noisyBalancedAccuracy":report["validation"]["noisy"]["balancedAccuracy"]}),flush=True)
    eligible=[name for name,report in reports.items() if report["admitted"]]
    chosen=max(eligible or list(reports),key=lambda name:quality_key(reports[name]["validation"],baseline_scores))
    model,refit=fit_cached(f"refit-{len(refit_people)}",refit_people,chosen=="clean-plus-room-noise",final_refit=True)
    data_source="PSG-Audio V3; full available nights; CC BY 4.0"
    if chosen=="clean-plus-room-noise": data_source+="; MUSAN speech/music/ambient augmentation with per-source attribution"
    model.update(schemaVersion=2,modelID="psg-audio-expanded-v8",dataSource=data_source,
        featureProfile="spectral-v4",rawFeatureNames=SPECTRAL_NAMES,featureNames=names(10,SPECTRAL_NAMES),contextEpochs=10,
        classes=STAGES,heldOutSubjects=[],balancedAccuracy=0)
    output=root/("selected-model.json" if eligible else "experimental-model.json")
    atomic_json(output,model,compact=True)
    if output.stat().st_size>508000: raise ValueError("The expanded model exceeds the native import budget.")
    summary={"status":"admitted" if eligible else "trained_not_improved","chosenRecipe":chosen,"parametersSHA256":parameters_digest(model),
        "validation":reports,"baselineValidation":baseline_scores,"fullNight84Validation":reference,
        "fittingPeople":len(training),"validationPeople":len(validation),"finalRefitPeople":len(refit_people),"finalTestRead":False,
        "reservedDevelopmentPeople":len(plan["roles"]),"qualityExcludedSubjects":excluded,
        "approvedForAlarmTiming":False,"sourceSha":protocol["sourceSha"],"refit":refit}
    atomic_json(root/"training-summary.json",summary)
    if eligible:
        score=reports[chosen]["validation"]["clean"];previous=baseline_scores["clean"]
        atomic_json(root/"selection.json",{"id":chosen,"trainingSubjects":refit_people,"validationSubjects":[],
            "qualityExcludedSubjects":excluded,
            "previouslyObservedSubjects":sorted(plan["roles"]),"parametersSHA256":parameters_digest(model),"finalTestRead":False,
            "meanMacroF1":score["macroF1"],"meanBalancedAccuracy":score["balancedAccuracy"],
            "baselineMeanMacroF1":previous["macroF1"],"baselineMeanBalancedAccuracy":previous["balancedAccuracy"],
            "everyValidationViewRecognizesEveryStage":True,"controlledNoiseAdmissionPassed":True})
    else: atomic_json(root/"development-review.json",summary)
    print(json.dumps({"event":"expanded_training_complete","status":summary["status"],"refitPeople":len(refit_people),"finalTestRead":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("development","output","baseline","original-registry"): p.add_argument("--"+flag,type=Path,required=True)
    run(p.parse_args())
