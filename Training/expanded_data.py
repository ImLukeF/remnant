"""Load verified full-night arrays with bounded causal context and no pickled objects."""
import json
from pathlib import Path

from causal_model import Context
from expanded_features import fingerprint
from run_dataset import checksum


def load(root,phase="development"):
    import numpy as np
    root=Path(root);plan_path=root/"development-plan.json"
    plan=json.loads(plan_path.read_text())
    if (plan.get("phase","development")!=phase or phase=="development" and
            (plan.get("fittingForbidden",False) or "test" in plan["roles"].values())):
        raise ValueError("Final-test measurements must never be opened by development fitting.")
    if set(plan["roles"])&set(plan["reservedTestSubjects"]): raise ValueError("Independent people leaked into development.")
    protocol=json.loads((root/"extraction-protocol.json").read_text())
    complete=json.loads((root/"development-complete.json").read_text())
    checks_path=root/"feature-checksums.json";checks=json.loads(checks_path.read_text())
    if (protocol["extractorSHA256"]!=fingerprint() or protocol["developmentPlanSHA256"]!=checksum(plan_path)
            or complete["featureChecksumsSHA256"]!=checksum(checks_path)
            or complete["extractorSHA256"]!=protocol["extractorSHA256"]
            or complete["noiseManifestSHA256"]!=protocol["noiseManifestSHA256"]
            or set(checks)!={f["path"] for f in plan["files"]}):
        raise ValueError("Require the unchanged completed full-night corpus.")
    parts={key:[] for key in ("features","augmentedFeatures","startSeconds","stage","usable","augmentedUsable","subject")}
    for item in plan["files"]:
        path=root/"features"/(Path(item["path"]).stem+".npz")
        if checksum(path)!=checks[item["path"]]: raise ValueError("Clinical observations changed after extraction.")
        with np.load(path,allow_pickle=False) as block:
            n=len(block["stage"])
            if (block["features"].shape!=(n,58) or block["augmentedFeatures"].shape!=(n,58)
                    or any(block[k].shape!=(n,) for k in ("startSeconds","stage","usable","augmentedUsable"))
                    or block["subject"].item()!=item["subject"] or block["sourcePath"].item()!=item["path"]
                    or block["sourceSHA256"].item()!=item["sha256"]
                    or block["noiseRole"].item()!=plan["roles"][item["subject"]]
                    or block["noiseManifestSHA256"].item()!=protocol["noiseManifestSHA256"]
                    or block["extractorSHA256"].item()!=protocol["extractorSHA256"]
                    or not np.isfinite(block["features"]).all() or not np.isfinite(block["augmentedFeatures"]).all()
                    or not np.isfinite(block["startSeconds"]).all()
                    or ((block["stage"]<0)|(block["stage"]>3)).any()
                    or (block["augmentedUsable"]&~block["usable"]).any()
                    or (block["startSeconds"]<0).any()
                    or (block["startSeconds"]+30>float(block["recordingDuration"])+1e-6).any()):
                raise ValueError("Invalid full-night measurement provenance or shape.")
            for key in parts:
                parts[key].append(np.full(n,item["subject"],dtype="U32") if key=="subject" else block[key].copy())
    arrays={key:np.concatenate(values) for key,values in parts.items()}
    order=np.lexsort((arrays["startSeconds"],arrays["subject"]))
    arrays={key:value[order] for key,value in arrays.items()}
    if ((arrays["subject"][1:]==arrays["subject"][:-1]) & (arrays["startSeconds"][1:]==arrays["startSeconds"][:-1])).any():
        raise ValueError("Duplicated clinical intervals would leak or overweight observations.")
    if set(arrays["subject"])!=set(plan["roles"]): raise ValueError("A reserved person disappeared from the complete cache.")
    if int(arrays["usable"].sum())!=complete["usableEpochs"]: raise ValueError("Full-night quality counts changed.")
    excluded=complete.get("qualityExcludedSubjects",{})
    if excluded:
        from corpus_quality import exclusions
        summaries=json.loads((root/"block-summary.json").read_text())
        if (exclusions(root,plan,checks,summaries)!=excluded
                or checksum(root/"quality-amendment.json")!=complete["qualityAmendmentSHA256"]):
            raise ValueError("The explicit pre-fitting quality exclusion changed.")
    usable=set(arrays["subject"][arrays["usable"]])
    if set(plan["roles"])-usable!=set(excluded): raise ValueError("A participant disappeared without a verified quality amendment.")
    manifest={"developmentPlanSHA256":checksum(plan_path),"extractorSHA256":protocol["extractorSHA256"],
        "featureChecksumsSHA256":checksum(checks_path),"noiseManifestSHA256":protocol["noiseManifestSHA256"],
        "people":sorted(plan["roles"]),"epochs":len(order)}
    if excluded: manifest.update(qualityExcludedSubjects=excluded,usableSubjects=sorted(usable),qualityAmendmentSHA256=complete["qualityAmendmentSHA256"])
    return arrays,plan,manifest


def context(arrays,noisy=False):
    import numpy as np
    indexes=np.flatnonzero(arrays["augmentedUsable"] if noisy else arrays["usable"])
    inputs=arrays["augmentedFeatures"] if noisy else arrays["features"]
    values=np.empty((len(indexes),232),dtype=np.float32);person=None;history=None
    for output,index in enumerate(indexes):
        if arrays["subject"][index]!=person:
            person=arrays["subject"][index];history=Context(10,58)
        values[output]=history.append(float(arrays["startSeconds"][index]),inputs[index].tolist())
    return {"inputs":values,"labels":arrays["stage"][indexes],"people":arrays["subject"][indexes],
            "starts":arrays["startSeconds"][indexes],"indexes":indexes}
