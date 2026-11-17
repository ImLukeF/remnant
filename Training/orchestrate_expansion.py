"""Lightweight VPS controller; every heavy phase runs as an exact-source bounded ai-team job."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def save(path,value):
    temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def cli(*argv):
    result=subprocess.run(["/usr/local/bin/ai-team",*argv],capture_output=True,text=True,timeout=120)
    if result.returncode: raise RuntimeError((result.stderr or result.stdout)[-4000:])
    return json.loads(result.stdout)


def run(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/"controller-state.json"
    state=json.loads(path.read_text()) if path.exists() else {"featureSource":args.feature_source,"jobs":[],"completedPhases":[]}
    if state["featureSource"]!=args.feature_source: raise ValueError("Resume the same extraction source.")
    if args.initial_job and not state.get("activeJob") and not state["jobs"]:
        state.update(activeJob=args.initial_job,activePhase="development_batch_0")
    save(path,state)
    def count(folder):
        file=folder/"feature-checksums.json"
        return len(json.loads(file.read_text())) if file.exists() else 0

    def execute(name,request,progress_root=None):
        if name in state["completedPhases"]: return
        source=request["sourceSha"]
        if len(source)!=40 or any(c not in "0123456789abcdef" for c in source): raise ValueError("Use an exact pushed source SHA.")
        if state.get("activeJob"):
            if state["activePhase"]!=name: raise ValueError("A different phase already owns the active job.")
            job=state["activeJob"]
        else:
            receipt=cli("run","--repository","ImLukeF/parley","--commit",source,"--lane",request["lane"],
                        "--prepare","none","--timeout","3600","--",*request["argv"])
            job=receipt["jobId"];state.update(activeJob=job,activePhase=name);save(path,state)
        last=-1
        while True:
            receipt=cli("status",job)
            declared=receipt.get("sourceSha") or receipt.get("request",{}).get("sourceSha")
            if declared!=source: raise ValueError("The job receipt does not match the frozen source.")
            if progress_root:
                number=count(progress_root);state["completedBlocks"]=number
                if number//10!=last:
                    last=number//10;print(json.dumps({"event":"pipeline_progress","phase":name,"blocks":number,"jobId":job}),flush=True)
            state["jobStatus"]=receipt["status"];save(path,state)
            if receipt.get("finishedAt") or receipt["status"] in {"passed","failed","cancelled","timed_out","timeout"}: break
            time.sleep(30)
        state["jobs"].append(receipt);state.pop("activeJob",None);state.pop("activePhase",None)
        if receipt["status"]!="passed" or receipt.get("sourceDirtyAfter",False):
            state["status"]="stopped_on_job_failure";save(path,state)
            raise RuntimeError(f"Inspect {name} job {job}: {receipt['status']}; do not silently retry changed inputs.")
        state["completedPhases"].append(name);save(path,state)

    def extract_batches(folder,request,prefix):
        for number in range(args.max_batches):
            # An attached job must finish and have a verified receipt even if it has just written the completion marker.
            if (folder/"development-complete.json").exists() and not state.get("activeJob"): return
            phase=f"{prefix}_batch_{number}"
            if phase in state["completedPhases"]: continue
            before=count(folder);execute(phase,request,folder)
            if count(folder)<=before and not (folder/"development-complete.json").exists() and not (args.initial_job and number==0):
                raise RuntimeError("A successful batch made no extraction progress.")
        if not (folder/"development-complete.json").exists(): raise RuntimeError("The bounded batch count was exhausted.")

    feature_request={"sourceSha":args.feature_source,"lane":"codex-sleep-expanded-features-v8","argv":[
        "env","OPENBLAS_NUM_THREADS=2","OMP_NUM_THREADS=2","uv","run","--python","3.12","--with-requirements",
        "Packages/SleepWakeKit/Training/requirements-research-linux.txt","python","Packages/SleepWakeKit/Training/run_expanded_corpus.py",
        "--output",str(root),"--noise-bank",str(args.noise_bank),"--max-download-gib","768","--batch-gib","64","--download-workers","4"]}
    try:
        state["status"]="extracting_development";save(path,state)
        extract_batches(root,feature_request,"development")
        state["status"]="development_ready";save(path,state)
        config=root/"training-request.json"
        if not config.exists(): return
        steps=json.loads(config.read_text())
        state["status"]="training";save(path,state);execute("training",steps["training"])
        model_root=Path(steps["modelRoot"])
        if not (model_root/"selection.json").exists():
            state["status"]="trained_not_admitted";save(path,state);return
        state["status"]="preparing_final";save(path,state);execute("prepare_final",steps["prepareFinal"])
        final_root=Path(steps["finalRoot"])
        state["status"]="extracting_final";save(path,state)
        extract_batches(final_root,steps["extractFinal"],"final")
        state["status"]="evaluating";save(path,state);execute("evaluate",steps["evaluate"])
        state["status"]="complete";save(path,state)
    except BaseException as error:
        state["error"]=str(error);save(path,state);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--noise-bank",type=Path,required=True)
    parser.add_argument("--feature-source",required=True);parser.add_argument("--initial-job")
    parser.add_argument("--max-batches",type=int,default=20)
    args=parser.parse_args()
    if not 1<=args.max_batches<=20: parser.error("Bound the number of one-hour extraction batches.")
    run(args)
