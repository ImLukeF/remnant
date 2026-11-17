"""Bounded past-only GRU with paired clean/noisy consistency; final people are never opened."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

from array_gru import layout,encode_matrix,normalized,batch,forward
from expanded_contracts import admitted
from expanded_data import load,context
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics,parameters_digest
from run_dataset import checksum
from train_correction import calibration_split,wake_errors
from train_expanded import score_views,quality_key
from train_sequence import make_network,export

RECIPES=[{"history":10,"hidden":32,"consistency":0.25,"noiseShare":0.25,"classPower":1.0},
         {"history":10,"hidden":32,"consistency":0.0,"noiseShare":0.25,"classPower":1.0}]
RELATIVE_RECIPES=[{"history":10,"hidden":32,"consistency":0.25,"noiseShare":0.1,"classPower":0.9},
                  {"history":10,"hidden":32,"consistency":0.25,"noiseShare":0.1,"classPower":1.0}]


def frozen_refit(root,arrays,manifest,baseline_sha,seconds):
    """Refit a previously admitted recipe; never tune on final people or replace a completed refit."""
    import numpy as np
    protocol=json.loads((root/"protocol.json").read_text());summary=json.loads((root/"training-summary.json").read_text())
    chosen=summary.get("chosenRecipe")
    if ((root/"selected-model.json").exists() or (root/"selection.json").exists()
            or summary["status"]!="admitted_pending_refit" or protocol["development"]!=manifest
            or protocol["baselineModelSHA256"]!=baseline_sha or not chosen
            or not admitted(summary["validation"][chosen]["validation"],summary["baselineValidation"])):
        raise ValueError("Refit only an admitted frozen checkpoint on the unchanged development corpus.")
    report=json.loads((root/(chosen+"-report.json")).read_text())
    if report!=summary["validation"][chosen] or checksum(root/(chosen+"-weights.json"))!=report["modelFileSHA256"]:
        raise ValueError("The admitted checkpoint or its measured report changed.")
    recipe=protocol["recipes"][int(chosen.rsplit("-",1)[1])]
    if protocol["recipes"] not in (RECIPES,RELATIVE_RECIPES): raise ValueError("Refit cannot change the frozen recipes.")
    people=protocol["refitSubjects"]
    if (len(people)!=199 or set(people)&set(protocol["reservedTestSubjects"])
            or set(people)|set(protocol["qualityExcludedSubjects"])!=set(protocol["previouslyObservedSubjects"])):
        raise ValueError("Final refit must use exactly the 199 usable development people.")
    model,network,data,fit_report=fit(arrays,people,[],recipe,seconds,fixed_epochs=report["fit"]["bestEpoch"])
    model["modelID"]=json.loads((root/(chosen+"-weights.json")).read_text())["modelID"]
    path=root/"selected-model.json";atomic_json(path,model,compact=True)
    if path.stat().st_size>508000: raise ValueError("The model exceeds the native import budget.")
    digest=parameters_digest(model)
    summary.update(status="admitted",finalRefitPeople=len(people),parametersSHA256=digest,
        refit={"fit":fit_report,"modelFileSHA256":checksum(path)},
        refitSourceSha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    atomic_json(root/"training-summary.json",summary)
    atomic_json(root/"selection.json",{"id":chosen,"trainingSubjects":people,"validationSubjects":[],
        "qualityExcludedSubjects":protocol["qualityExcludedSubjects"],"previouslyObservedSubjects":protocol["previouslyObservedSubjects"],
        "parametersSHA256":digest,"finalTestRead":False,"controlledNoiseAdmissionPassed":True})
    take=np.concatenate([np.flatnonzero(data["people"]==p)[:60] for p in sorted(people)[:2]])
    values=predictions(network,data,take)["clean"]
    samples=[{"subject":str(data["people"][index]),"startSeconds":float(data["starts"][index]),"usable":True,
        "features":arrays["features"][data["indexes"][index]].tolist(),"expectedProbabilities":dict(zip(STAGES,value.tolist()))}
        for index,value in zip(take,values)]
    atomic_json(root/"export-check.json",samples,compact=True)
    print(json.dumps({"event":"consistency_refit_complete","people":len(people),"epochs":fit_report["bestEpoch"],"finalTestRead":False}),flush=True)


def consistency(clean,noisy):
    """Jensen-Shannon divergence: symmetric, bounded and differentiable through both views."""
    import math
    import torch
    left=torch.log_softmax(clean,dim=1);right=torch.log_softmax(noisy,dim=1)
    middle=torch.logaddexp(left,right)-math.log(2)
    return 0.5*((left.exp()*(left-middle)).sum(dim=1)+(right.exp()*(right-middle)).sum(dim=1))


def prepare(arrays,people,history):
    import numpy as np
    if not np.array_equal(arrays["usable"],arrays["augmentedUsable"]):
        raise ValueError("Paired training requires identical clean/noisy usable intervals.")
    data=layout(arrays,history);mask=np.isin(data["people"],people)
    if set(data["people"][mask])!=set(people): raise ValueError("Every declared fitting person is required.")
    clean=encode_matrix(arrays["features"][data["indexes"]]);noisy=encode_matrix(arrays["augmentedFeatures"][data["indexes"]])
    means=clean[mask].mean(axis=0);scales=np.maximum(clean[mask].std(axis=0),1e-6)
    data.update(clean=normalized(clean,means,scales),noisy=normalized(noisy,means,scales),means=means,scales=scales)
    return data


def predictions(network,data,indexes):
    import numpy as np
    import torch
    network.eval();result={}
    with torch.no_grad():
        for condition in ("clean","noisy"):
            values=[]
            for start in range(0,len(indexes),512):
                take=indexes[start:start+512]
                x=batch(data[condition],data["windows"],data["lengths"],take)
                values.append(torch.softmax(network(torch.from_numpy(x),torch.from_numpy(data["lengths"][take])),dim=1).numpy())
            result[condition]=np.concatenate(values)
    return result


def measured(data,indexes,values):
    rows=[{"subject":str(person),"stage":STAGES[int(stage)]} for person,stage in zip(data["people"][indexes],data["labels"][indexes])]
    return {name:metrics(rows,probabilities) for name,probabilities in values.items()}


def checkpoint_key(score,reference=None):
    complete=all(r>0 for name in score for r in score[name]["recallPerClass"])
    if reference is not None: return complete,*quality_key(score,reference)
    return complete,min(score[name]["macroF1"] for name in score),sum(score[name]["balancedAccuracy"] for name in score)


def fit(arrays,people,calibration,recipe,seconds,maximum_epochs=8,fixed_epochs=None,reference_scores=None):
    import numpy as np
    import torch
    began=time.monotonic();torch.manual_seed(7);rng=np.random.default_rng(7)
    if set(people)&set(calibration): raise ValueError("Checkpoint selection cannot use fitting people.")
    data=prepare(arrays,people,recipe["history"])
    indexes=np.flatnonzero(np.isin(data["people"],people));check=np.flatnonzero(np.isin(data["people"],calibration))
    if calibration and set(data["people"][check])!=set(calibration): raise ValueError("Calibration person disappeared.")
    labels=data["labels"][indexes].astype(np.int64);counts=np.bincount(labels,minlength=4)
    if (counts==0).any(): raise ValueError("Training requires every sleep stage.")
    _,inverse,person_counts=np.unique(data["people"][indexes],return_inverse=True,return_counts=True)
    weights=(len(labels)/(4*counts[labels]))**recipe["classPower"]/person_counts[inverse];weights/=weights.mean()
    weights=torch.from_numpy(weights);network=make_network(recipe["hidden"])
    optimizer=torch.optim.AdamW(network.parameters(),lr=0.001,weight_decay=0.01)
    loss_function=torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.02)
    best=None;best_key=None;best_epoch=0;stale=0;log=[]
    for epoch in range(1,(fixed_epochs or maximum_epochs)+1):
        if time.monotonic()-began>seconds-10: break
        started=time.monotonic();network.train();total=0.0
        for local in np.array_split(rng.permutation(len(indexes)),(len(indexes)+255)//256):
            if time.monotonic()-began>seconds: break
            take=indexes[local];lengths=torch.from_numpy(data["lengths"][take]);target=torch.from_numpy(labels[local])
            optimizer.zero_grad(set_to_none=True)
            clean=network(torch.from_numpy(batch(data["clean"],data["windows"],data["lengths"],take)),lengths)
            noisy=network(torch.from_numpy(batch(data["noisy"],data["windows"],data["lengths"],take)),lengths)
            loss=(((1-recipe["noiseShare"])*loss_function(clean,target)+recipe["noiseShare"]*loss_function(noisy,target)
                   +recipe["consistency"]*consistency(clean,noisy))*weights[local]).mean()
            if not torch.isfinite(loss): raise ValueError("Non-finite consistency objective.")
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),1,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(local)
        else:
            entry={"epoch":epoch,"seconds":time.monotonic()-started,"loss":total/len(indexes)}
            if calibration:
                score=measured(data,check,predictions(network,data,check))
                key=checkpoint_key(score,reference_scores)
                entry["calibration"]=score
                if best_key is None or key>best_key:
                    best_key=key;best=copy.deepcopy(network.state_dict());best_epoch=epoch;stale=0
                else: stale+=1
            else: best=copy.deepcopy(network.state_dict());best_epoch=epoch
            log.append(entry)
            print(json.dumps({"event":"consistency_epoch","epoch":epoch,"beta":recipe["consistency"],"seconds":entry["seconds"],
                "innerScores":{k:{m:s[m] for m in ("macroF1","balancedAccuracy","recallPerClass")} for k,s in entry.get("calibration",{}).items()}}),flush=True)
            if calibration and epoch>=4 and stale>=2: break
            continue
        break  # Discard an incomplete epoch when the declared budget expires.
    if best is None: raise TimeoutError("No complete epoch within the bounded fitting budget.")
    if fixed_epochs and best_epoch!=fixed_epochs: raise TimeoutError("Final refit did not finish its frozen epoch count.")
    network.load_state_dict(best);network.eval()
    model=export(network,recipe,data["means"],data["scales"],list(calibration))
    model.update(modelID="psg-audio-causal-consistency-v10",dataSource="PSG-Audio V3 CC BY 4.0; audited MUSAN paired waveform augmentation")
    probe=indexes[:128];expected=predictions(network,data,probe)["clean"]
    actual=forward(model,batch(data["clean"],data["windows"],data["lengths"],probe),data["lengths"][probe])
    difference=float(np.max(np.abs(actual-expected)))
    if difference>2e-10: raise ValueError("Independent GRU export differs from PyTorch.")
    report={"trainingPeople":len(people),"trainingEpochs":len(indexes),"calibrationPeople":len(calibration),"bestEpoch":best_epoch,
        "history":log,"stageCounts":counts.tolist(),"exportMaximumDifference":difference,"seconds":time.monotonic()-began}
    return model,network,data,report


def run(args):
    import numpy as np
    import torch
    torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.version.cuda is not None: raise ValueError("Use the pinned CPU-only PyTorch package.")
    began=time.monotonic();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    arrays,plan,manifest=load(args.development);previous=json.loads((args.previous/"protocol.json").read_text())
    if previous["development"]!=manifest or checksum(args.baseline)!=previous["baselineModelSHA256"]:
        raise ValueError("Reuse exactly the verified expanded cohort and incumbent.")
    if args.refit:
        return frozen_refit(root,arrays,manifest,checksum(args.baseline),args.max_seconds-(time.monotonic()-began)-15)
    if any(root.iterdir()): raise ValueError("Use a new output; never overwrite frozen results.")
    training=previous["trainingSubjects"];validation=previous["validationSubjects"];refit=previous["refitSubjects"]
    inner,calibration=calibration_split(training)
    if (len(validation)!=40 or len(refit)!=199 or (set(inner)|set(calibration))&set(validation)
            or set(refit)!=set(training)|set(validation) or set(refit)&set(plan["reservedTestSubjects"])):
        raise ValueError("Fitting, checkpoint selection, validation and final people must remain separated.")
    relative=args.relative_checkpoints;recipes=RELATIVE_RECIPES if relative else RECIPES
    protocol={"experiment":"causal-paired-relative-v11" if relative else "causal-paired-consistency-v10","development":manifest,"trainingSubjects":training,"innerFittingSubjects":inner,
        "calibrationSubjects":calibration,"validationSubjects":validation,"refitSubjects":refit,"recipes":recipes,"maximumEpochs":8,
        "previouslyObservedSubjects":previous["previouslyObservedSubjects"],"reservedTestSubjects":plan["reservedTestSubjects"],
        "qualityExcludedSubjects":previous["qualityExcludedSubjects"],"baselineModelSHA256":checksum(args.baseline),
        "selection":"Checkpoint on 24 inner people; unchanged paired four-metric/all-stage admission on 40 separate people; stop on first admitted fixed recipe.",
        "checkpointMetric":"Minimum relative gain in all four paired metrics" if relative else "Minimum absolute clean/noisy macro F1",
        "usesFutureAudio":False,"finalTestRead":False,"maximumSeconds":args.max_seconds,
        "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    atomic_json(root/"protocol.json",protocol)
    for name in ("full-night-original-84-weights.json","full-night-original-84-report.json"):
        shutil.copyfile(args.previous/name,root/name)
    reference=json.loads((root/"full-night-original-84-report.json").read_text())
    if checksum(root/"full-night-original-84-weights.json")!=reference["modelFileSHA256"]: raise ValueError("The retained control changed.")
    views={"clean":context(arrays),"noisy":context(arrays,noisy=True)}
    baseline_model=json.loads(args.baseline.read_text());baseline=score_views(baseline_model,views,validation);reports={};chosen=None
    inner_baseline=score_views(baseline_model,views,calibration) if relative else None
    if inner_baseline is not None: atomic_json(root/"inner-reference.json",inner_baseline)
    for number,recipe in enumerate(recipes):
        budget=min(475,args.max_seconds-(time.monotonic()-began)-30)
        if budget<90: break
        print(json.dumps({"event":"consistency_fit","id":number,"people":len(inner),"budgetSeconds":budget,"recipe":recipe}),flush=True)
        model,network,data,fit_report=fit(arrays,inner,calibration,recipe,budget,reference_scores=inner_baseline)
        if relative: model["modelID"]="psg-audio-causal-relative-v11"
        take=np.flatnonzero(np.isin(data["people"],validation));score=measured(data,take,predictions(network,data,take))
        name=f"recipe-{number}";path=root/(name+"-weights.json");atomic_json(path,model,compact=True)
        report={"fit":fit_report,"validation":score,"wakeErrors":wake_errors(score),"admitted":admitted(score,baseline),"modelFileSHA256":checksum(path)}
        reports[name]=report;atomic_json(root/(name+"-report.json"),report)
        print(json.dumps({"event":"consistency_validation","id":name,"admitted":report["admitted"],"scores":score}),flush=True)
        atomic_json(root/"progress.json",{"validation":reports,"baselineValidation":baseline,"finalTestRead":False})
        if report["admitted"]: chosen=name;break
    summary={"status":"admitted_pending_refit" if chosen else "trained_not_improved","chosenRecipe":chosen,"validation":reports,"baselineValidation":baseline,
        "innerFittingPeople":135,"calibrationPeople":24,"validationPeople":40,"finalRefitPeople":0,"finalTestRead":False,"approvedForAlarmTiming":False,
        "qualityExcludedSubjects":previous["qualityExcludedSubjects"],"sourceSha":protocol["sourceSha"],"seconds":time.monotonic()-began}
    atomic_json(root/"training-summary.json",summary)
    print(json.dumps({"event":"consistency_complete","status":summary["status"],"seconds":summary["seconds"],"finalTestRead":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("development","previous","baseline","output"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--refit",action="store_true",help="Refit only a frozen, admitted completed comparison; never run another selection.")
    p.add_argument("--relative-checkpoints",action="store_true",help="Fixed v11 recipes; inner checkpoint selection uses all four relative metrics.")
    p.add_argument("--max-seconds",type=int,default=1100);args=p.parse_args()
    if not 300<=args.max_seconds<=1200: p.error("Use a bounded 5-20 minute experiment.")
    run(args)
