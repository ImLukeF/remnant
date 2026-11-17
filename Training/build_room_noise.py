"""Build a small role-separated noise bank from individually audited MUSAN source groups."""
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import tarfile
import wave

from expand_cohort import atomic_json
from prepare_room_noise import ARCHIVE
from run_dataset import checksum


def build(root):
    import numpy as np
    root=Path(root);audit_path=root/"audit.json";audit=json.loads(audit_path.read_text())
    if audit["archive"]!=ARCHIVE or checksum(root/ARCHIVE["path"])!=ARCHIVE["sha256"]:
        raise ValueError("The original audited noise archive changed.")
    if (root/"bank.json").exists(): raise ValueError("This noise bank is already frozen.")
    sources={"musan/music/rfm":("music","CC-BY-3.0"),
             "musan/speech/us-gov":("speech","Public-Domain"),
             "musan/noise/free-sound":("noise","Public-Domain")}
    texts={prefix:audit["metadata"][prefix+"/LICENSE"] for prefix in sources}
    if (not texts["musan/music/rfm"].startswith("CC BY 3.0\n")
            or "US Public Domain" not in texts["musan/speech/us-gov"]
            or "Public Domain" not in texts["musan/noise/free-sound"]):
        raise ValueError("Previously inspected source-group licences changed.")
    candidates={}
    for prefix in sources:
        names=[f["path"] for f in audit["audioFiles"] if str(PurePosixPath(f["path"]).parent)==prefix and f["bytes"]>=60*16000*2+44]
        candidates[prefix]=sorted(names,key=lambda name:hashlib.sha256(("room-noise-bank-v8:"+name).encode()).hexdigest())[:32]
        if len(candidates[prefix])<16: raise ValueError("Insufficient licensed long-form background audio.")
    wanted={name for names in candidates.values() for name in names}
    clips_dir=root/"clips";clips_dir.mkdir(exist_ok=True);valid={prefix:[] for prefix in sources}
    with tarfile.open(root/ARCHIVE["path"],mode="r|gz") as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile(): continue
            # A bounded prefix contains the PCM header and 60 seconds, even for a much longer source file.
            prefix=archive.extractfile(member).read(4*1024**2)
            try:
                with wave.open(BytesIO(prefix),"rb") as audio:
                    if audio.getnchannels()!=1 or audio.getsampwidth()!=2 or audio.getframerate()!=16000 or audio.getcomptype()!="NONE": continue
                    duration=audio.getnframes()/16000
                    payload=audio.readframes(60*16000)
                if len(payload)!=60*16000*2: continue
            except wave.Error: continue
            values=(np.frombuffer(payload,dtype="<i2").astype(np.float32)/32768)
            if float(np.std(values.astype(np.float64)))<1e-6: continue
            source_prefix=str(PurePosixPath(member.name).parent)
            path=clips_dir/(PurePosixPath(member.name).stem+".npy");np.save(path,values)
            valid[source_prefix].append({"path":str(path.relative_to(root)),"sha256":checksum(path),
                "sourcePath":member.name,"sourceArchiveSHA256":ARCHIVE["sha256"],"sourceDurationSeconds":duration,
                "excerptStartSeconds":0,"excerptDurationSeconds":60,"kind":sources[source_prefix][0],"licence":sources[source_prefix][1],
                "licenceDocumentSHA256":hashlib.sha256(texts[source_prefix].encode()).hexdigest()})
    selected=[];seen=set()
    for prefix,names in candidates.items():
        options={clip["sourcePath"]:clip for clip in valid[prefix]}
        ordered=[]
        for name in names:
            if name in options and options[name]["sha256"] not in seen:
                clip=options[name];seen.add(clip["sha256"]);ordered.append(clip)
        if len(ordered)<16: raise ValueError("Noise quality/deduplication left insufficient independent files.")
        for index,clip in enumerate(ordered[:16]): selected.append({**clip,"role":"train" if index<8 else "validation" if index<12 else "test"})
    retained={clip["path"] for clip in selected}
    for file in clips_dir.glob("*.npy"):
        if str(file.relative_to(root)) not in retained: file.unlink()
    atomic_json(root/"bank.json",{"schemaVersion":1,"approvedForAugmentation":True,"sampleRate":16000,
        "archive":ARCHIVE,"auditSHA256":checksum(audit_path),"sourceLicenceDocuments":texts,
        "attribution":"MUSAN: David Snyder, Guoguo Chen and Daniel Povey (2015); music by Kevin MacLeod, CC BY 3.0; speech and environmental selections marked Public Domain in the original source-group licences.",
        "licenceEvidence":"https://www.openslr.org/17/","clips":selected,
        "description":"Synthetic speech/music/ambient interference; not an actual home-TV validation corpus."})
    print(json.dumps({"event":"noise_bank_frozen","clips":len(selected),"roles":dict(Counter(c["role"] for c in selected)),
                      "bankSHA256":checksum(root/"bank.json")}),flush=True)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    build(p.parse_args().output)
