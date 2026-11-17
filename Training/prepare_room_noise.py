"""Download a pinned MUSAN archive and audit its original per-source licence metadata."""
import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile

from expand_cohort import atomic_json
from run_dataset import checksum, download

ARCHIVE = {"path":"musan.tar.gz","bytes":11086114085,
           "sha256":"86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1"}
SOURCE = "https://openslr.trmal.net/resources/17/musan.tar.gz"
MIRROR = "https://huggingface.co/datasets/Emulated-Inc/enhancement/resolve/e4272bcb5f7757fae1fd1f58267f19c6ffe8c860/musan.tar.gz"


def audit(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    archive=root/ARCHIVE["path"]
    download(ARCHIVE,archive,source_url=MIRROR if args.mirror else SOURCE)
    if (root/"audit.json").exists():
        previous=json.loads((root/"audit.json").read_text())
        if previous["archive"]["sha256"]!=checksum(archive): raise ValueError("Audited noise source changed.")
        return
    metadata={}; files=[]; expanded=0
    with tarfile.open(archive,mode="r|gz") as source:
        for member in source:
            if not member.isfile(): continue
            expanded+=member.size
            if expanded>20*1024**3: raise ValueError("Noise archive exceeds its uncompressed bound.")
            path=PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts: raise ValueError("Unsafe noise archive member.")
            if path.suffix==".wav": files.append({"path":str(path),"bytes":member.size})
            elif path.name in {"LICENSE","LICENSE.txt","README","README.txt","ANNOTATIONS","ANNOTATIONS.txt"}:
                if member.size>1_000_000: raise ValueError("Noise licence metadata exceeds bound.")
                metadata[str(path)]=source.extractfile(member).read().decode("utf-8")
    atomic_json(root/"audit.json",{"archive":ARCHIVE,"originalSource":SOURCE,"downloadSource":MIRROR if args.mirror else SOURCE,
                "licenceEvidence":"https://www.openslr.org/17/","declaredLicence":"CC BY 4.0",
                "audioApprovedForUse":False,"metadata":metadata,"audioFiles":files})
    print(json.dumps({"event":"noise_archive_audited","audioFiles":len(files),"licenceFiles":list(metadata),
                      "audioApprovedForUse":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mirror",action="store_true",help="Use the pinned byte-identical public mirror.")
    audit(p.parse_args())
