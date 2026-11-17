"""Review the app's fixed 85%-confidence/two-epoch signal on development data, without approving alarms."""
import argparse
import json
from pathlib import Path

from array_gru import predict
from array_trees import probabilities
from expanded_data import load,context
from expand_cohort import atomic_json
from run_dataset import checksum


def review(people,starts,labels,values):
    import numpy as np
    high=(values.argmax(axis=1)==1)&(values[:,1]>=0.85)
    contiguous=(people[1:]==people[:-1])&(np.abs(starts[1:]-starts[:-1]-30)<=0.5)
    take=np.flatnonzero(contiguous&high[1:]&high[:-1])+1
    return {"eligibleAdjacentPairs":int(contiguous.sum()),"candidatePairs":len(take),"peopleWithCandidates":len(set(people[take])),
        "currentLightFraction":float(np.mean(labels[take]==1)) if len(take) else None,
        "bothEpochsLightFraction":float(np.mean((labels[take]==1)&(labels[take-1]==1))) if len(take) else None,
        "currentDeepOrREMCount":int(np.isin(labels[take],[2,3]).sum()),"currentAwakeCount":int((labels[take]==0).sum())}


def run(args):
    import numpy as np
    root=args.selection;protocol=json.loads((root/"protocol.json").read_text());summary=json.loads((root/"training-summary.json").read_text())
    arrays,_,manifest=load(args.development)
    if manifest!=protocol["development"] or checksum(args.baseline)!=protocol["baselineModelSHA256"]:
        raise ValueError("Only the unchanged development cache and incumbent may be reviewed.")
    people=protocol["validationSubjects"]
    if len(people)!=40 or set(people)&set(protocol["reservedTestSubjects"]): raise ValueError("Review only the existing 40 validation people.")
    mask=np.isin(arrays["subject"],people);arrays={k:v[mask] for k,v in arrays.items()}
    models={"incumbent":json.loads(args.baseline.read_text())}
    for name,report in summary["validation"].items():
        path=root/(name+"-weights.json")
        if checksum(path)!=report["modelFileSHA256"]: raise ValueError("A measured checkpoint changed.")
        models[name]=json.loads(path.read_text())
    results={name:{} for name in models}
    for condition in ("clean","noisy"):
        view=context(arrays,noisy=condition=="noisy")
        for name,model in models.items():
            if model["modelKind"]=="gru":
                values,indexes=predict(model,arrays,noisy=condition=="noisy")
                if not np.array_equal(indexes,view["indexes"]): raise ValueError("Signal review intervals differ.")
            else: values=probabilities(model,view["inputs"])
            results[name][condition]=review(view["people"],view["starts"],view["labels"],values)
    output={"validationPeople":40,"confidenceThreshold":0.85,"requiredContiguousEpochs":2,"results":results,
        "finalTestRead":False,"approvedForAlarmTiming":False,"realPhoneOvernightValidated":False,
        "limitation":"All available adjacent development intervals, not scheduled wake windows, independent accuracy, wake comfort or real-TV validation."}
    atomic_json(root/"wake-signal-review.json",output);print(json.dumps(output),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("selection","development","baseline"): p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
