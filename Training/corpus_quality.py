"""Explicit pre-fitting exclusion of wholly silent training participants; no score-based selection."""
import argparse
import json
from pathlib import Path

from expand_cohort import atomic_json
from features import STAGES
from run_dataset import checksum


def people_summary(summaries):
    people={}
    for detail in summaries.values():
        person=people.setdefault(detail["subject"],{"role":detail["role"],"usableEpochs":0,"stageCounts":[0]*4})
        if person["role"]!=detail["role"]: raise ValueError("A participant changed corpus roles.")
        person["usableEpochs"]+=detail["usableEpochs"]
        person["stageCounts"]=[a+b for a,b in zip(person["stageCounts"],detail["cleanStageCounts"])]
    return people


def verify_silent(root,plan,checks,people):
    import numpy as np
    excluded={p for p,value in people.items() if value["usableEpochs"]==0}
    evidence={}
    for person in sorted(excluded):
        if plan["roles"][person]!="train": raise ValueError("This amendment cannot exclude validation or final-test people.")
        epochs=0;blocks={}
        for item in plan["files"]:
            if item["subject"]!=person: continue
            path=root/"features"/(Path(item["path"]).stem+".npz")
            if checksum(path)!=checks[item["path"]]: raise ValueError("Silent participant observations changed.")
            with np.load(path,allow_pickle=False) as values:
                count=len(values["stage"]);epochs+=count
                silent=[-120,0,-120,-120,0,0,0]
                if (values["subject"].item()!=person or values["noiseRole"].item()!="train"
                        or values["sourceSHA256"].item()!=item["sha256"]
                        or values["usable"].any() or values["augmentedUsable"].any()
                        or values["features"].shape!=(count,58)
                        or not np.array_equal(values["features"][:,:7],np.broadcast_to(silent,(count,7)))
                        or not np.array_equal(values["features"],values["augmentedFeatures"])):
                    raise ValueError("Only verified wholly silent measurements qualify; never bypass other quality failures.")
            blocks[item["path"]]=checks[item["path"]]
        if not epochs: raise ValueError("Missing labels or coverage are not a silence-only exclusion.")
        evidence[person]={"role":"train","reason":"Every scored interval contains silent microphone measurements.",
            "scoredEpochs":epochs,"blocks":blocks}
    return evidence


def exclusions(root,plan,checks,summaries):
    path=root/"quality-amendment.json"
    if not path.exists(): return {}
    amendment=json.loads(path.read_text())
    if (plan.get("phase","development")!="development" or plan.get("fittingForbidden",False)
            or amendment["policy"]!="wholly-silent-training-only-v1"
            or not amendment["beforeModelFitting"]
            or amendment["developmentPlanSHA256"]!=checksum(root/"development-plan.json")
            or amendment["featureChecksumsSHA256"]!=checksum(root/"feature-checksums.json")):
        raise ValueError("A quality amendment must preserve the frozen development plan and observations.")
    evidence=verify_silent(root,plan,checks,people_summary(summaries))
    if evidence!=amendment["excludedSubjects"] or not evidence:
        raise ValueError("The explicit silence-only exclusion evidence changed.")
    return evidence


def complete(root,plan,checks,summaries,protocol):
    if set(checks)!=set(summaries) or set(checks)!={f["path"] for f in plan["files"]}:
        raise ValueError("Every verified block needs a matching quality summary.")
    people=people_summary(summaries)
    excluded=exclusions(root,plan,checks,summaries)
    if (set(people)!=set(plan["roles"])
            or {p for p,value in people.items() if value["usableEpochs"]==0}!=set(excluded)):
        raise ValueError("A reserved development participant has no usable labelled audio.")
    quality={"people":people,"blocks":len(checks),"usableEpochs":sum(p["usableEpochs"] for p in people.values()),
        "stageOrder":STAGES,"sourceBlocksExcluded":plan.get("unusableSourceBlocks",[]),
        "extractorSHA256":protocol["extractorSHA256"],"noiseManifestSHA256":protocol["noiseManifestSHA256"],
        "featureChecksumsSHA256":checksum(root/"feature-checksums.json"),
        "developmentPlanSHA256":checksum(root/"development-plan.json")}
    if excluded:
        quality.update(qualityExcludedSubjects=excluded,usablePeople=len(people)-len(excluded),
            qualityAmendmentSHA256=checksum(root/"quality-amendment.json"))
    return quality


def amend(root,model_root):
    root=Path(root);model_root=Path(model_root)
    if model_root.exists() and any(model_root.iterdir()):
        raise ValueError("Quality exclusions must be frozen before any model fitting or validation results.")
    if (root/"quality-amendment.json").exists(): raise ValueError("The quality amendment is already frozen.")
    plan=json.loads((root/"development-plan.json").read_text())
    if plan.get("phase","development")!="development" or plan.get("fittingForbidden",False):
        raise ValueError("This amendment never opens or alters final-test participants.")
    checks=json.loads((root/"feature-checksums.json").read_text());summaries=json.loads((root/"block-summary.json").read_text())
    if set(checks)!=set(summaries) or set(checks)!={f["path"] for f in plan["files"]}:
        raise ValueError("Finish every frozen source before auditing participant quality.")
    for item in plan["files"]:
        if checksum(root/"features"/(Path(item["path"]).stem+".npz"))!=checks[item["path"]]:
            raise ValueError("Clinical observations changed before the quality amendment.")
    evidence=verify_silent(root,plan,checks,people_summary(summaries))
    if not evidence: raise ValueError("There are no wholly silent training participants to exclude.")
    atomic_json(root/"quality-amendment.json",{"policy":"wholly-silent-training-only-v1","beforeModelFitting":True,
        "developmentPlanSHA256":checksum(root/"development-plan.json"),
        "featureChecksumsSHA256":checksum(root/"feature-checksums.json"),"excludedSubjects":evidence,
        "validationSubjectsUnchanged":True,"reservedTestSubjectsUnchanged":True,
        "observedParticipantsRemainExcludedFromFutureTests":True})
    protocol=json.loads((root/"extraction-protocol.json").read_text())
    quality=complete(root,plan,checks,summaries,protocol)
    atomic_json(root/"development-complete.json",quality)
    print(json.dumps({"event":"silent_training_exclusion_frozen","excludedPeople":len(evidence),
        "usablePeople":quality["usablePeople"],"usableHours":quality["usableEpochs"]/120}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development",type=Path,required=True);parser.add_argument("--model-root",type=Path,required=True)
    args=parser.parse_args();amend(args.development,args.model_root)
