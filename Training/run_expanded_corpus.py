"""Bounded resumable clean/noisy extraction for a frozen expanded development cohort."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess

from expanded_features import extract, fingerprint
from expand_cohort import atomic_json
from features import STAGES
from room_noise import NoiseBank
from run_dataset import checksum, prefetched_audio


def extraction_bank(root):
    bank=NoiseBank(root)
    # Retain the immutable resampled public clips across epochs. The original DSP body is unchanged;
    # the cache proof verifies byte-identical mixtures. At 48 kHz this stays below one GiB.
    resample=NoiseBank.resampled.__wrapped__
    bank.resampled=lru_cache(maxsize=40)(lambda index,rate:resample(bank,index,rate))
    return bank


_worker_bank=None


def initialize_worker(noise_root):
    global _worker_bank
    _worker_bank=extraction_bank(noise_root)


def extract_block(item,source,labels,temporary,role,bank=None):
    import hashlib
    import numpy as np
    data=labels.read_bytes()
    if hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()!=item["rmlGitOID"]:
        raise ValueError("Verified clinical annotations changed.")
    detail=extract(item,source,labels,temporary,bank if bank is not None else _worker_bank,role)
    with np.load(temporary,allow_pickle=False) as values:
        if (values["sourceSHA256"].item()!=item["sha256"] or values["sourcePath"].item()!=item["path"]
                or values["noiseRole"].item()!=role):
            raise ValueError("Extracted source identity or noise role changed.")
        detail.update(subject=item["subject"],role=role,
            cleanStageCounts=np.bincount(values["stage"][values["usable"]],minlength=4).tolist(),
            noisyStageCounts=np.bincount(values["stage"][values["augmentedUsable"]],minlength=4).tolist())
    return detail


def extracted_blocks(ready,root,bank,roles,workers):
    def arguments(item,source):
        return item,source,root/"labels"/(item["recording"]+".rml"),root/"features"/(Path(item["path"]).stem+".partial"),roles[item["subject"]]
    if workers==1:
        for _,item,source in ready:
            if shutil.disk_usage(root).free<20*1024**3: raise ValueError("Keep 20 GiB of free disk headroom during extraction.")
            yield item,source,extract_block(*arguments(item,source),bank=bank)
        return
    # Only the parent publishes checkpoints. Each child owns one distinct temporary block;
    # at most `workers` blocks and the existing bounded download queue can be in flight.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,initargs=(bank.root,)) as pool:
        active={}
        def completed():
            finished,_=wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                item,source=active.pop(future)
                yield item,source,future.result()
        for _,item,source in ready:
            if shutil.disk_usage(root).free<20*1024**3: raise ValueError("Keep 20 GiB of free disk headroom during extraction.")
            active[pool.submit(extract_block,*arguments(item,source))]=(item,source)
            if len(active)>=workers: yield from completed()
        while active: yield from completed()


def run(args):
    import numpy as np
    root=args.output.resolve();plan_path=root/"development-plan.json"
    plan=json.loads(plan_path.read_text());bank=extraction_bank(args.noise_bank)
    workers=getattr(args,"extract_workers",1)
    if workers not in (1,2,3): raise ValueError("Use one to three bounded extraction workers.")
    if set(plan["roles"])&set(plan["reservedTestSubjects"]): raise ValueError("Final-test people leaked into development.")
    files=plan["files"]
    if sum(f["bytes"] for f in files)>args.max_download_gib*1024**3: raise ValueError("Frozen source plan exceeds the total download bound.")
    cache=root/"raw-cache";features=root/"features";cache.mkdir(exist_ok=True);features.mkdir(exist_ok=True)
    protocol={"developmentPlanSHA256":checksum(plan_path),"noiseManifestSHA256":bank.sha,"extractorSHA256":fingerprint(),
              "featureProfile":"spectral-v4-clean-and-room-noise","noiseRolesFollowParticipantRoles":True,
              "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    path=root/"extraction-protocol.json"
    if path.exists():
        previous=json.loads(path.read_text())
        if any(previous[k]!=protocol[k] for k in ("developmentPlanSHA256","noiseManifestSHA256","extractorSHA256")):
            raise ValueError("Source, noise or extraction math changed during the expanded run.")
    else: atomic_json(path,protocol)
    progress_path=root/"feature-checksums.json";progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    summary_path=root/"block-summary.json";summaries=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    pending=[]
    for item in files:
        out=features/(Path(item["path"]).stem+".npz")
        if item["path"] in progress:
            if not out.exists() or checksum(out)!=progress[item["path"]]: raise ValueError("Completed clinical observations changed.")
            residue=cache/(Path(item["path"]).stem+".edf")
            if residue.exists(): residue.unlink()
        else: pending.append(item)
    selected=[];size=0
    for item in pending:
        if selected and size+item["bytes"]>args.batch_gib*1024**3: break
        selected.append(item);size+=item["bytes"]
    print(json.dumps({"event":"expanded_batch","alreadyComplete":len(progress),"total":len(files),
                      "batchBlocks":len(selected),"batchGiB":size/1024**3,"extractWorkers":workers}),flush=True)
    ready=prefetched_audio(selected,cache,features,args.download_workers)
    for item,source,detail in extracted_blocks(ready,root,bank,plan["roles"],workers):
        output=features/(Path(item["path"]).stem+".npz");temporary=output.with_suffix(".partial")
        temporary.replace(output);summaries[item["path"]]=detail
        atomic_json(summary_path,summaries)
        progress[item["path"]]=checksum(output);atomic_json(progress_path,progress);source.unlink()
        print(json.dumps({"event":"expanded_extracted","complete":len(progress),"total":len(files),
                          "usableEpochs":detail["usableEpochs"]}),flush=True)
    if len(progress)==len(files):
        if set(progress)!=set(summaries): raise ValueError("Every verified block needs a matching quality summary.")
        people={}
        for detail in summaries.values():
            person=people.setdefault(detail["subject"],{"role":detail["role"],"usableEpochs":0,"stageCounts":[0]*4})
            person["usableEpochs"]+=detail["usableEpochs"]
            person["stageCounts"]=[a+b for a,b in zip(person["stageCounts"],detail["cleanStageCounts"])]
        if set(people)!=set(plan["roles"]) or any(p["usableEpochs"]==0 for p in people.values()):
            raise ValueError("A reserved development participant has no usable labelled audio.")
        quality={"people":people,"blocks":len(progress),"usableEpochs":sum(p["usableEpochs"] for p in people.values()),
                 "stageOrder":STAGES,"sourceBlocksExcluded":plan.get("unusableSourceBlocks",[]),
                 "extractorSHA256":protocol["extractorSHA256"],"noiseManifestSHA256":bank.sha,
                 "featureChecksumsSHA256":checksum(progress_path),"developmentPlanSHA256":checksum(plan_path)}
        atomic_json(root/"development-complete.json",quality)
        print(json.dumps({"event":"expanded_complete","people":len(people),"usableHours":quality["usableEpochs"]/120}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--noise-bank",type=Path,required=True)
    p.add_argument("--download-workers",type=int,choices=(1,2,3,4),default=4)
    p.add_argument("--extract-workers",type=int,choices=(1,2,3),default=1)
    p.add_argument("--max-download-gib",type=float,default=768);p.add_argument("--batch-gib",type=float,default=96)
    args=p.parse_args()
    if not 0<args.batch_gib<=128 or not 0<args.max_download_gib<=1024: p.error("Keep expanded extraction bounded.")
    run(args)
