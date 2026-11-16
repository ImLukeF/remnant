"""Re-extract frequency observations for development people only, with bounded downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from expand_cohort import atomic_json
from run_dataset import REVISION, REPOSITORY, annotation_start, checksum, download, licence_evidence, read_url, resolve
from spectral import SPECTRAL_NAMES


def feature_fingerprint():
    code = Path(__file__).parent
    return hashlib.sha256(b"".join((code / name).read_bytes() for name in
        ("features.py", "extract_psg.py", "spectral.py", "extract_spectral_psg.py"))).hexdigest()


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache, features = root / "raw-cache", root / "features"
    cache.mkdir(exist_ok=True)
    features.mkdir(exist_ok=True)
    previous = json.loads((args.development / "cohort-plan.json").read_text())
    retired = json.loads((args.retired_test / "download-plan.json").read_text())
    if previous["revision"] != REVISION or retired["revision"] != REVISION:
        raise ValueError("Development source revisions must remain pinned.")
    roles = {person: "train" if role == "train" else "validation" for person, role in previous["splits"].items()}
    for person in retired["testSubjects"]:
        if person in roles:
            raise ValueError("Retired test overlaps development people.")
        roles[person] = "validation"
    files = previous["files"] + retired["files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("Duplicated source audio in development plan.")
    fingerprint = feature_fingerprint()
    path = root / "cohort-plan.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["extractorSHA256"] != fingerprint or saved["splits"] != roles or saved["files"] != files:
            raise ValueError("Resume only the exact frozen frequency-feature plan.")
    else:
        saved = {"revision": REVISION, "repository": REPOSITORY, "featureProfile": "spectral-v4", "featureNames": SPECTRAL_NAMES,
                 "extractorSHA256": fingerprint, "splits": roles, "files": files, "licenceEvidence": licence_evidence(),
                 "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "previousFinalTestsRetired": True, "sourceQualitySubstitutions": retired.get("sourceQualitySubstitutions", [])}
        atomic_json(path, saved)
    total = sum(item["bytes"] for item in files)
    if total > args.max_download_gib * 1024 ** 3:
        raise ValueError("Development extraction exceeds its bounded download budget.")
    print(json.dumps({"event": "spectral_plan", "blocks": len(files), "bytes": total, "people": dict(Counter(roles.values()))}), flush=True)
    progress_path = root / "feature-checksums.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for item in files:
        if item["path"] in progress:
            output = features / (Path(item["path"]).stem + ".jsonl")
            if not output.exists() or checksum(output) != progress[item["path"]]:
                raise ValueError("Previously extracted frequency observations changed.")
    pending = iter(item for item in files if item["path"] not in progress)
    import pyedflib
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        active = {}
        def enqueue():
            item = next(pending, None)
            if item is not None:
                source = cache / (Path(item["path"]).stem + ".edf")
                active[pool.submit(download, item, source)] = (item, source)
        for _ in range(args.download_workers):
            enqueue()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            future = next(iter(ready))
            item, source = active.pop(future)
            future.result()
            labels = cache / (item["recording"] + ".rml")
            data = labels.read_bytes() if labels.exists() else read_url(resolve(item["rml"]))
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != item["rmlGitOID"]:
                raise ValueError("Source annotation checksum mismatch.")
            labels.write_bytes(data)
            with pyedflib.EdfReader(str(source)) as edf:
                offset = (edf.getStartdatetime() - annotation_start(labels)).total_seconds()
                if offset != item["offsetSeconds"] or edf.getFileDuration() != item["durationSeconds"] or "Mic" not in edf.getSignalLabels():
                    raise ValueError("Full source disagrees with verified bedside channel/timing.")
            output = features / (Path(item["path"]).stem + ".jsonl")
            temporary = output.with_suffix(".partial")
            subprocess.run([sys.executable, str(Path(__file__).with_name("extract_spectral_psg.py")),
                            "--audio-edf", str(source), "--stages-rml", str(labels), "--subject", item["subject"],
                            "--audio-offset-seconds", str(offset), "--output", str(temporary)], check=True)
            temporary.replace(output)
            progress[item["path"]] = checksum(output)
            atomic_json(progress_path, progress)
            source.unlink()
            print(json.dumps({"event": "spectral_extracted", "complete": len(progress), "total": len(files)}), flush=True)
            enqueue()
    with (root / "epochs.jsonl").open("w") as output:
        for item in files:
            output.write((features / (Path(item["path"]).stem + ".jsonl")).read_text())
    print(json.dumps({"event": "spectral_complete", "blocks": len(files), "featureCount": len(SPECTRAL_NAMES)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--retired-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-download-gib", type=float, default=112)
    parser.add_argument("--download-workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    if not 0 < args.max_download_gib <= 128:
        parser.error("Bounded development download budget must not exceed 128 GiB.")
    run(args)
