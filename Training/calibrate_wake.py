"""Inner-person calibration of light-sleep probability; diagnostic only, never alarm approval."""
import argparse
import json
from pathlib import Path

from array_gru import predict
from expanded_data import load
from expand_cohort import atomic_json
from review_wake_signals import review
from run_dataset import checksum


def light_logit(values):
    import numpy as np
    from scipy.special import logit
    return logit(np.clip(np.asarray(values)[:,1],1e-8,1-1e-8))


def apply(parameters,values):
    from scipy.special import expit
    return expit(parameters["slope"]*light_logit(values)+parameters["intercept"])


def fit(people,labels,values):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    x=light_logit(values);y=(np.asarray(labels)==1).astype(float)
    unique,inverse,counts=np.unique(people,return_inverse=True,return_counts=True)
    weights=1/counts[inverse];weights/=weights.sum()
    if len(np.unique(y))!=2: raise ValueError("Calibration needs measured light and non-light epochs.")
    def objective(parameters):
        slope,bias=parameters;z=slope*x+bias
        loss=np.sum(weights*(np.logaddexp(0,z)-y*z))+0.001*((slope-1)**2+bias*bias)
        residual=weights*(expit(z)-y)
        return float(loss),np.asarray([np.sum(residual*x)+0.002*(slope-1),residual.sum()+0.002*bias])
    result=minimize(objective,[1,0],jac=True,method="L-BFGS-B",bounds=[(0,8),(-12,12)])
    if not result.success: raise ValueError("Light-probability calibration did not converge.")
    return {"slope":float(result.x[0]),"intercept":float(result.x[1]),"people":len(unique),
        "objective":float(result.fun),"rawObjective":objective([1,0])[0],"innerLightPrior":float(np.sum(weights*y)),
        "personBalanced":True,"usesValidationPeople":False}


def score(people,starts,labels,values,parameters):
    import numpy as np
    calibrated=apply(parameters,values);raw=np.asarray(values)[:,1];y=np.asarray(labels)==1
    # The same two-epoch probability signal is tested without requiring an uncalibrated argmax.
    paired=np.column_stack((1-calibrated,calibrated,np.zeros(len(y)),np.zeros(len(y))))
    result=review(people,starts,labels,paired)
    result.update(rawBrier=float(np.mean((raw-y)**2)),calibratedBrier=float(np.mean((calibrated-y)**2)),
        observedLightPrior=float(y.mean()),constantInnerPriorBrier=float(np.mean((parameters["innerLightPrior"]-y)**2)))
    return result


def run(args):
    import numpy as np
    root=args.selection;protocol=json.loads((root/"protocol.json").read_text());summary=json.loads((root/"training-summary.json").read_text())
    if (root/"wake-calibration-review.json").exists(): raise ValueError("This diagnostic is already frozen.")
    arrays,_,manifest=load(args.development)
    if manifest!=protocol["development"]: raise ValueError("Use the unchanged private development measurements.")
    inner=protocol["calibrationSubjects"];validation=protocol["validationSubjects"]
    if (len(inner)!=24 or len(validation)!=40 or set(inner)&set(validation)
            or (set(inner)|set(validation))&set(protocol["reservedTestSubjects"])
            or set(inner)&set(protocol["innerFittingSubjects"])): raise ValueError("Calibration must exclude fitting, validation and final people.")
    mask=np.isin(arrays["subject"],inner+validation);arrays={k:v[mask] for k,v in arrays.items()};results={}
    for name,report in summary["validation"].items():
        path=root/(name+"-weights.json")
        if checksum(path)!=report["modelFileSHA256"]: raise ValueError("A frozen backbone changed.")
        model=json.loads(path.read_text());views={}
        for condition in ("clean","noisy"):
            values,indexes=predict(model,arrays,noisy=condition=="noisy")
            views[condition]={"values":values,"people":arrays["subject"][indexes],"labels":arrays["stage"][indexes],"starts":arrays["startSeconds"][indexes]}
        fitting=[np.isin(view["people"],inner) for view in views.values()]
        parameters=fit(np.concatenate([v["people"][m] for v,m in zip(views.values(),fitting)]),
            np.concatenate([v["labels"][m] for v,m in zip(views.values(),fitting)]),
            np.concatenate([v["values"][m] for v,m in zip(views.values(),fitting)]))
        measured={}
        for condition,view in views.items():
            m=np.isin(view["people"],validation)
            if set(view["people"][m])!=set(validation): raise ValueError("Validation person disappeared.")
            measured[condition]=score(view["people"][m],view["starts"][m],view["labels"][m],view["values"][m],parameters)
        results[name]={"parameters":parameters,"validation":measured,"backboneSHA256":checksum(path)}
    result={"calibrationPeople":24,"validationPeople":40,"results":results,"approvedForAlarmTiming":False,"finalTestRead":False,
        "limitation":"Reused development validation, all adjacent intervals; no independent final test, scheduled-window or phone validation. Parameters cannot transfer to a refitted backbone without new calibration."}
    atomic_json(root/"wake-calibration-review.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("selection","development"): p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
