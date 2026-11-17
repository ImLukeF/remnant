"""Pure admission and participant-isolation checks; no fitting or final-test observations."""
import json
import math
import hashlib
import xml.etree.ElementTree as ET
from improve_model import parameters_digest


def participant_identities(folder,expected):
    result={};seen={"PatientID":set(),"PatientDBID":set()}
    for path in sorted(folder.glob("*.rml")):
        subject=path.stem.split("-",1)[0]
        if subject not in expected: continue
        patients=[node for node in ET.parse(path).getroot().iter() if node.tag.rsplit("}",1)[-1]=="Patient"]
        if len(patients)!=1 or subject in result: raise ValueError("Exactly one night and patient identity is required per person.")
        fields={child.tag.rsplit("}",1)[-1]:child.text for child in patients[0]}
        hashes={}
        for key in seen:
            if not fields.get(key) or not fields[key].strip(): raise ValueError("A clinical participant identity is missing.")
            value=hashlib.sha256(fields[key].strip().encode()).hexdigest()
            if value in seen[key]: raise ValueError("Distinct recording names repeat the same clinical participant.")
            seen[key].add(value);hashes[key]=value
        result[subject]=hashes
    if set(result)!=set(expected): raise ValueError("A reserved clinical participant is missing its identity record.")
    return result


def admitted(score,baseline):
    for views in (score,baseline):
        for name in ("clean","noisy"):
            view=views[name]
            if (len(view["recallPerClass"])!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in
                    [view["macroF1"],view["balancedAccuracy"],*view["recallPerClass"]])):
                return False
    return (all(all(r>0 for r in score[name]["recallPerClass"]) for name in ("clean","noisy"))
            and all(score[name][metric]>baseline[name][metric] for name in ("clean","noisy") for metric in ("macroF1","balancedAccuracy")))


def final_admission(root,reservation,known):
    selection=json.loads((root/"selection.json").read_text())
    summary=json.loads((root/"training-summary.json").read_text())
    model=json.loads((root/"selected-model.json").read_text())
    parameters=parameters_digest(model)
    excluded=selection.get("qualityExcludedSubjects",{})
    trained=set(selection["trainingSubjects"])
    if (summary["status"]!="admitted" or summary["chosenRecipe"]!=selection["id"]
            or not selection["controlledNoiseAdmissionPassed"]
            or not admitted(summary["validation"][selection["id"]]["validation"],summary["baselineValidation"])
            or parameters!=selection["parametersSHA256"] or parameters!=summary["parametersSHA256"]
            or trained&set(excluded) or trained|set(excluded)!=set(reservation["roles"])
            or excluded!=summary.get("qualityExcludedSubjects",{})
            or any(reservation["roles"].get(p)!="train" for p in excluded)
            or summary["refit"]["fit"]["trainingPeople"]!=len(trained)
            or set(known)!=set(reservation["roles"])
            or set(reservation["reservedTestSubjects"])&set(known)
            or len(set(reservation["reservedTestSubjects"]))!=30 or len(reservation["reservedTestSubjects"])!=30):
        raise ValueError("Only an admitted frozen model may read the reserved new clinical participants.")
    return parameters
