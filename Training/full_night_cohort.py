"""Freeze a larger participant-separated cohort before reading labels or fitting models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error

from expand_cohort import atomic_json, header_timing, scored_counts
from extract_psg import annotations
from run_dataset import REVISION, REPOSITORY, checksum, licence_evidence, listing, read_url, resolve, select_subjects
from run_spectral_corpus import feature_fingerprint


def participant_roles(available, known, development=200, validation=40, test=30):
    available, known = set(available), set(known)
    if not known.issubset(available) or not 0 < validation < development or development-validation < len(known) or test < 1:
        raise ValueError("Known people must fit inside the training group.")
    unseen = sorted(available-known, key=lambda p: hashlib.sha256(("expanded-full-nights-v8:"+p).encode()).hexdigest())
    required = development-len(known)+test
    if len(unseen) < required: raise ValueError("Insufficient unique new participants.")
    held_out = unseen[validation:validation+test]
    training = sorted(known | set(unseen[validation+test:validation+test+development-validation-len(known)]))
    roles = {p: "train" for p in training}
    roles.update({p: "validation" for p in unseen[:validation]})
    assert len(roles) == development and not set(roles).intersection(held_out)
    return roles, sorted(held_out)


def retry(operation):
    for attempt in range(4):
        try: return operation()
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 3: raise
            time.sleep(2**attempt)


def prepare(args):
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    labels=root/"labels"; labels.mkdir(exist_ok=True)
    people=select_subjects(retry(lambda: listing("APNEA_RML")),1000)
    known=json.loads(args.known_subjects.read_text())
    roles,held_out=participant_roles([p[0] for p in people],known,args.development_people,args.validation_people,args.test_people)
    reservation={"schemaVersion":1,"experiment":"expanded-full-nights-v8","repository":REPOSITORY,"revision":REVISION,
                 "roles":roles,"reservedTestSubjects":held_out,"previouslyObservedSubjects":sorted(known),
                 "knownRegistrySHA256":checksum(args.known_subjects),"selectionUsesLabels":False,
                 "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    reservation_path=root/"participants.json"
    if reservation_path.exists():
        saved=json.loads(reservation_path.read_text())
        if any(saved[k]!=reservation[k] for k in ("revision","roles","reservedTestSubjects","knownRegistrySHA256")):
            raise ValueError("Resume only the identical participant reservation.")
    else: atomic_json(reservation_path,reservation)
    # Conservative exclusion: every reserved development person is retired from future independent tests.
    atomic_json(root/"observed-subjects.json",sorted(roles))
    plan_path=root/"development-plan.json"
    if plan_path.exists():
        saved=json.loads(plan_path.read_text())
        if (saved["roles"]!=roles or saved["revision"]!=REVISION or saved["extractorSHA256"]!=feature_fingerprint()
                or saved["participantReservationSHA256"]!=checksum(reservation_path)):
            raise ValueError("Full-night source plan changed.")
        if sum(f["bytes"] for f in saved["files"])>args.max_download_gib*1024**3: raise ValueError("Source plan exceeds budget.")
        print(json.dumps({"event":"full_night_plan_reused","blocks":len(saved["files"])}),flush=True)
        return saved
    evidence=licence_evidence()
    selected=[item for item in people if item[0] in roles]
    def inventory(person):
        subject,recording,rml=person
        data=retry(lambda: read_url(resolve(rml["path"])))
        if hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()!=rml["oid"]:
            raise ValueError("Source annotations changed.")
        label_path=labels/(recording+".rml"); label_path.write_bytes(data)
        source_files=retry(lambda: listing("APNEA_EDF/"+recording))
        files=[]
        for item in sorted(source_files,key=lambda f:f["path"]):
            if item["type"]!="file" or not item["path"].endswith(".edf"): continue
            if "lfs" not in item: raise ValueError("EDF has no immutable source checksum.")
            files.append({"subject":subject,"recording":recording,"rml":rml["path"],"rmlGitOID":rml["oid"],
                          "path":item["path"],"bytes":item["size"],"sha256":item["lfs"]["oid"]})
        if not files: raise ValueError("A reserved development participant has no audio.")
        return files
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups=list(pool.map(inventory,selected))
    flat=[item for group in groups for item in group]
    atomic_json(root/"source-inventory.json",flat)
    if sum(f["bytes"] for f in flat)>args.max_download_gib*1024**3:
        raise ValueError("Complete available development recordings exceed the download bound.")
    progress_path=root/"header-checks.json"
    checked=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pending=[f for f in flat if f["path"] not in checked]
    def inspect(item): return retry(lambda: header_timing(item,labels/(item["recording"]+".rml"),allow_empty=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect,pending):
            checked[result["path"]]=result
            atomic_json(progress_path,checked)
            if len(checked)%50==0: print(json.dumps({"event":"full_night_headers","complete":len(checked),"total":len(flat)}),flush=True)
    files=[]; tails=[]; coverage={}; gaps=[]; unusable=[]
    for group in groups:
        staged=annotations(labels/(group[0]["recording"]+".rml"))
        records=[checked[item["path"]] for item in group]
        for item in records:
            original=next(old for old in group if old["path"]==item["path"])
            if any(original[k]!=item[k] for k in original): raise ValueError("Cached header source changed.")
        unusable.extend(item for item in records if "unusableReason" in item)
        ordered=sorted((item for item in records if "unusableReason" not in item),key=lambda f:f["offsetSeconds"])
        if not ordered: raise ValueError("A reserved development participant has no valid source headers.")
        end=None; counts={}
        for item in ordered:
            original=next(old for old in group if old["path"]==item["path"])
            if any(original[k]!=item[k] for k in original): raise ValueError("Cached header source changed.")
            if end is not None:
                if item["offsetSeconds"]<end-0.01: raise ValueError("Overlapping full-night blocks would duplicate observations.")
                if item["offsetSeconds"]>end+0.01: gaps.append({"subject":item["subject"],"seconds":item["offsetSeconds"]-end})
            end=item["offsetSeconds"]+item["durationSeconds"]
            if item["durationSeconds"]<30: tails.append(item); continue
            for key,value in scored_counts(item,staged).items(): counts[key]=counts.get(key,0)+value
            files.append(item)
        coverage[group[0]["subject"]]=counts
    saved={"schemaVersion":1,"repository":REPOSITORY,"revision":REVISION,"featureProfile":"spectral-v4",
           "extractorSHA256":feature_fingerprint(),"roles":roles,"reservedTestSubjects":held_out,
           "participantReservationSHA256":checksum(reservation_path),"licenceEvidence":evidence,"files":files,
           "shortTailsExcluded":tails,"unusableSourceBlocks":unusable,"gaps":gaps,"annotatedStageCountsByPerson":coverage,
           "allAvailableBlocks":True,"sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    atomic_json(plan_path,saved)
    print(json.dumps({"event":"full_night_plan_frozen","developmentPeople":len(roles),"trainingPeople":sum(r=="train" for r in roles.values()),
                      "validationPeople":sum(r=="validation" for r in roles.values()),"reservedFinalPeople":len(held_out),
                      "blocks":len(files),"GiB":sum(f["bytes"] for f in files)/1024**3,
                      "audioHours":sum(f["durationSeconds"] for f in files)/3600,"shortTails":len(tails)}),flush=True)
    return saved


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--known-subjects",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--development-people",type=int,default=200); p.add_argument("--validation-people",type=int,default=40)
    p.add_argument("--test-people",type=int,default=30); p.add_argument("--max-download-gib",type=float,default=768)
    args=p.parse_args()
    if not 0<args.max_download_gib<=1024 or not 1<=args.test_people<=40 or args.development_people>240:
        p.error("Bound participant count and total source download size.")
    prepare(args)
