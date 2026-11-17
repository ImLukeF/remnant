"""Bounded reprocessing of the known development corpus; no new final-test people are read."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from expand_cohort import atomic_json
from run_dataset import checksum, prefetched_audio, REPOSITORY, annotation_start, read_url, resolve
from acoustic_frames import FRAME_NAMES
from train_sequence import load_corpus


def fingerprint():
    code=Path(__file__).parent
    return hashlib.sha256(b"".join((code/name).read_bytes() for name in
        ("spectral.py","features.py","extract_psg.py","acoustic_frames.py","extract_acoustic_psg.py"))).hexdigest()


def extract_files(root,files,workers):
    cache=root/"raw-cache"; cache.mkdir(exist_ok=True)
    features=root/"features"; features.mkdir(exist_ok=True)
    progress_path=root/"feature-checksums.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pending=[]
    for item in files:
        out=features/(Path(item["path"]).stem+".npz")
        if item["path"] in progress:
            if not out.exists() or checksum(out)!=progress[item["path"]]: raise ValueError("Previously extracted acoustic features changed.")
        else: pending.append(item)
    import pyedflib
    for _,item,source in prefetched_audio(pending,cache,features,workers):
        if "previousFeatures" in item:
            label_arguments=["--previous-features",item["previousFeatures"]]
        else:
            labels=cache/(item["recording"]+".rml")
            data=labels.read_bytes() if labels.exists() else read_url(resolve(item["rml"]))
            if hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()!=item["rmlGitOID"]:
                raise ValueError("Independent annotation source checksum changed.")
            labels.write_bytes(data); label_arguments=["--stages-rml",str(labels)]
        with pyedflib.EdfReader(str(source)) as edf:
            if edf.getFileDuration()!=item["durationSeconds"] or "Mic" not in edf.getSignalLabels(): raise ValueError("Verified source timing/channel changed.")
            if "previousFeatures" not in item and (edf.getStartdatetime()-annotation_start(labels)).total_seconds()!=item["offsetSeconds"]:
                raise ValueError("Independent audio timing differs from its verified annotation offset.")
        output=features/(Path(item["path"]).stem+".npz"); temporary=output.with_suffix(".partial")
        subprocess.run([sys.executable,str(Path(__file__).with_name("extract_acoustic_psg.py")),"--audio-edf",str(source),
                        *label_arguments,"--audio-offset-seconds",str(item["offsetSeconds"]),"--subject",item["subject"],"--output",str(temporary)],check=True)
        temporary.replace(output); progress[item["path"]]=checksum(output)
        atomic_json(progress_path,progress); source.unlink()
        print(json.dumps({"event":"acoustic_extracted","complete":len(progress),"total":len(files)}),flush=True)
    atomic_json(root/"complete.json",{"blocks":len(files),"extractorSHA256":fingerprint()})


def run(args):
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    cache=root/"raw-cache"; cache.mkdir(exist_ok=True)
    features=root/"features"; features.mkdir(exist_ok=True)
    # Verify the complete cached cohort, hashes and exclusion registry before any download.
    _, prior=load_corpus(args.development,args.retired_test,args.previous_evaluation,args.registry)
    plans=[json.loads((args.development/"cohort-plan.json").read_text()),json.loads((args.retired_test/"download-plan.json").read_text())]
    files=[]
    for source,plan in zip((args.development,args.retired_test),plans):
        for item in plan["files"]:
            files.append({**item,"previousFeatures":str(source/"features"/(Path(item["path"]).stem+".jsonl"))})
    if sum(item["bytes"] for item in files)>args.max_download_gib*1024**3:
        raise ValueError("Development reprocessing exceeds its download bound.")
    metadata={"featureProfile":"mel-sequence-v6","extractorSHA256":fingerprint(),"previous":prior,
              "files":files,"frameShape":[240,25],"frameNames":FRAME_NAMES,"framesPerSecond":8,"decimalPlaces":4,
              "repository":REPOSITORY,"licenceEvidence":plans[0]["licenceEvidence"],
              "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    path=root/"cohort-plan.json"
    if path.exists():
        saved=json.loads(path.read_text())
        if any(saved[key]!=metadata[key] for key in ("featureProfile","extractorSHA256","previous","files","frameShape")):
            raise ValueError("Resume only the identical time-resolved development corpus.")
    else: atomic_json(path,metadata)
    progress_path=root/"feature-checksums.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    print(json.dumps({"event":"acoustic_plan","people":len(prior["people"]),"blocks":len(files),
                      "complete":len(progress),"bytes":sum(x["bytes"] for x in files)}),flush=True)
    extract_files(root,files,args.download_workers)
    atomic_json(root/"complete.json",{"blocks":len(files),"people":len(prior["people"]),"extractorSHA256":fingerprint()})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development",type=Path,required=True)
    p.add_argument("--retired-test",type=Path,required=True)
    p.add_argument("--previous-evaluation",type=Path,required=True)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-download-gib",type=float,default=128)
    p.add_argument("--download-workers",type=int,choices=(1,2,3,4),default=4)
    args=p.parse_args()
    if not 0<args.max_download_gib<=128: p.error("Keep development reprocessing within 128 GiB.")
    run(args)
