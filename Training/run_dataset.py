"""Bounded, resumable private training run on pinned PSG-Audio bedside recordings.

Raw EDF files are checksum-verified, prefetched within a bounded queue and removed after extraction.
Only derived observations, provenance, evaluation and the experimental model are retained.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

from train import fit, load_rows, partition, probabilities

REPOSITORY = "Samuelsantos777/psg-audio-v3-unofficial-mirror"
REVISION = "cb22565b02c6161be30d791ad72847ec8bd0de5a"
SOURCE = "https://www.scidb.cn/en/detail?dataSetId=778740145531650048"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"


def read_url(url, limit=2_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": "SleepWakeKit-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Metadata exceeds its bounded size.")
    return data


class LinkedData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self.active = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False

    def handle_data(self, value):
        if self.active:
            self.parts.append(value)


def licence_evidence():
    parser = LinkedData()
    parser.feed(read_url(SOURCE).decode())
    for value in parser.parts:
        obj = json.loads(value)
        if obj.get("name") == "PSG-Audio":
            if obj.get("license") != LICENSE:
                raise ValueError("The original provider no longer declares the expected CC BY 4.0 licence.")
            return {"source": SOURCE, "name": obj["name"], "version": obj.get("version"),
                    "license": obj["license"], "metadataSHA256": hashlib.sha256(value.encode()).hexdigest()}
    raise ValueError("Could not verify the dataset licence in the original provider's metadata.")


def listing(path):
    encoded = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{encoded}?recursive=true&limit=1000"
    return json.loads(read_url(url))


def resolve(path):
    return f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/" + urllib.parse.quote(path, safe="/")


def select_subjects(rml_files, count):
    # All nights from a participant use the same identifier and split. One night per participant in this baseline.
    subjects = {}
    for item in sorted(rml_files, key=lambda item: item["path"]):
        recording = Path(item["path"]).stem
        subject = recording.split("-", 1)[0]
        subjects.setdefault(subject, (recording, item))
    ordered = sorted(subjects, key=lambda subject: hashlib.sha256(subject.encode()).hexdigest())
    return [(subject, *subjects[subject]) for subject in ordered[:count]]


def build_plan(subject_count, blocks):
    result = []
    for subject, recording, rml in select_subjects(listing("APNEA_RML"), subject_count):
        files = [item for item in listing("APNEA_EDF/" + recording) if item["type"] == "file" and item["path"].endswith(".edf")]
        files.sort(key=lambda item: item["path"])
        # Select temporal coverage before reading any labels. Never choose blocks by their test performance.
        indexes = sorted({min(len(files) - 1, int(len(files) * (i + 0.5) / blocks)) for i in range(blocks)})
        if not files:
            raise ValueError("No audio files for selected recording.")
        for index in indexes:
            item = files[index]
            if "lfs" not in item:
                raise ValueError("Missing immutable source checksum.")
            result.append({"subject": subject, "recording": recording, "rml": rml["path"],
                           "rmlGitOID": rml["oid"],
                           "path": item["path"], "bytes": item["size"], "sha256": item["lfs"]["oid"]})
    return result


def supplement_plan(plan, paths):
    """Explicit annotation-coverage additions, limited to already selected participants/nights."""
    result = list(plan)
    recordings = {item["recording"]: item for item in plan}
    for path in paths:
        if path in {item["path"] for item in result}:
            continue
        recording = Path(path).parent.name
        if recording not in recordings or str(Path(path).parent) != "APNEA_EDF/" + recording:
            raise ValueError("Coverage supplements must belong to an already selected recording.")
        matches = [item for item in listing("APNEA_EDF/" + recording)
                   if item["path"] == path and item["type"] == "file" and path.endswith(".edf")]
        if len(matches) != 1 or "lfs" not in matches[0]:
            raise ValueError("Supplement must be one immutable EDF source file.")
        source, original = matches[0], recordings[recording]
        result.append({**{key: original[key] for key in ("subject", "recording", "rml", "rmlGitOID")},
                       "path": path, "bytes": source["size"], "sha256": source["lfs"]["oid"]})
    return result


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def download(item, target):
    partial = target.with_suffix(".part")
    if target.exists() and target.stat().st_size == item["bytes"] and checksum(target) == item["sha256"]:
        return
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == item["bytes"]:
                if checksum(partial) == item["sha256"]:
                    partial.replace(target)
                    return
                partial.unlink()
                offset = 0
            if offset > item["bytes"]:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "SleepWakeKit-research/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            # Fresh redirect avoids stale cached CDN authorizations. The revision and content hash remain pinned.
            url = resolve(item["path"]) + "?download=true&request=" + uuid.uuid4().hex
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    partial.unlink(missing_ok=True)
                    raise ValueError("Server did not honor the bounded resume range.")
                if not offset and response.status != 200:
                    raise ValueError("Unexpected partial response for a full download.")
                with partial.open("ab" if offset else "wb") as output:
                    window_start, window_bytes = time.monotonic(), offset
                    while block := response.read(1024 * 1024):
                        if output.tell() + len(block) > item["bytes"]:
                            raise ValueError("Downloaded object exceeds its declared size.")
                        output.write(block)
                        elapsed = time.monotonic() - window_start
                        if elapsed >= 30:
                            if (output.tell() - window_bytes) / elapsed < 1024 * 1024:
                                raise TimeoutError("Retrying an unusually slow source transfer from its verified byte offset.")
                            window_start, window_bytes = time.monotonic(), output.tell()
            if partial.stat().st_size != item["bytes"] or checksum(partial) != item["sha256"]:
                partial.unlink(missing_ok=True)
                raise ValueError("Source checksum or length mismatch.")
            partial.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def annotation_start(path):
    root = ET.parse(path).getroot()
    values = [(node.text or "").strip() for node in root.iter()
              if node.tag.rsplit("}", 1)[-1] == "RecordingStart"]
    if len(values) != 1:
        raise ValueError("Ambiguous PSG recording start.")
    start = dt.datetime.fromisoformat(values[0])
    if start.tzinfo is not None:
        raise ValueError("Timezone conversion must be verified before using this recording.")
    return start


def prefetched_audio(plan, cache, features_dir, workers):
    """Yield verified sources while bounding both downloads and unprocessed raw files."""
    pending = iter((number, item) for number, item in enumerate(plan, 1)
                   if not (features_dir / (Path(item["path"]).stem + ".jsonl")).exists())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        def enqueue():
            pair = next(pending, None)
            if pair is not None:
                number, item = pair
                source = cache / (Path(item["path"]).stem + ".edf")
                active[pool.submit(download, item, source)] = (number, item, source)
        for _ in range(workers):
            enqueue()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            future = next(iter(ready))
            number, item, source = active.pop(future)
            future.result()
            yield number, item, source
            enqueue()


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "raw-cache"
    features_dir = root / "features"
    cache.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)
    evidence = licence_evidence()
    code = Path(__file__).parent
    profile = getattr(args, "feature_profile", "basic")
    if profile == "spectral":
        from run_spectral_corpus import feature_fingerprint
        extractor_fingerprint = feature_fingerprint()
    else:
        extractor_fingerprint = hashlib.sha256((code / "extract_psg.py").read_bytes()
                                               + (code / "features.py").read_bytes()).hexdigest()
    plan_path = root / "download-plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if (saved["revision"] != REVISION or saved["subjectsRequested"] != args.subjects
                or saved["blocksPerSubject"] != args.blocks or saved["extractorSHA256"] != extractor_fingerprint):
            raise ValueError("Use a new output directory for a different training plan.")
        plan = saved["files"]
    else:
        plan = build_plan(args.subjects, args.blocks)
        plan_path.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION,
                                       "subjectsRequested": args.subjects, "blocksPerSubject": args.blocks,
                                       "extractorSHA256": extractor_fingerprint,
                                       "licenceEvidence": evidence, "files": plan}, indent=2) + "\n")
    saved = json.loads(plan_path.read_text())
    if (saved.get("fittingForbidden") or profile == "spectral") and not args.extract_only:
        raise ValueError("This data plan is extraction-only; fitting a model here is prohibited.")
    requested = sorted(set(args.supplement))
    previous = saved.get("coverageSupplements", [])
    if previous and (previous != requested or saved.get("supplementReason") != args.supplement_reason):
        raise ValueError("Resume with the same documented coverage supplements.")
    if requested and not previous:
        if not args.supplement_reason:
            raise ValueError("Document the annotation-coverage reason before adding source blocks.")
        expanded = supplement_plan(plan, requested)
        if sum(item["bytes"] for item in expanded) > args.max_download_gib * 1024 ** 3:
            raise ValueError("Coverage supplements exceed the download limit.")
        (root / "initial-download-plan.json").write_text(plan_path.read_text())
        saved.update(files=expanded, coverageSupplements=requested, supplementReason=args.supplement_reason)
        plan_path.write_text(json.dumps(saved, indent=2) + "\n")
        plan = expanded
    total = sum(item["bytes"] for item in plan)
    if total > args.max_download_gib * 1024 ** 3:
        raise ValueError(f"Plan exceeds download limit: {total} bytes.")
    print(json.dumps({"event": "plan", "subjects": len({x["subject"] for x in plan}),
                      "blocks": len(plan), "bytes": total, "revision": REVISION}), flush=True)
    if args.prepare_only:
        return
    import pyedflib
    for number, item, source in prefetched_audio(plan, cache, features_dir, getattr(args, "download_workers", 1)):
        key = Path(item["path"]).stem
        output = features_dir / (key + ".jsonl")
        labels = cache / (item["recording"] + ".rml")
        if labels.exists():
            contents = labels.read_bytes()
            oid = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            if oid != item["rmlGitOID"]:
                labels.unlink()
        if not labels.exists():
            contents = read_url(resolve(item["rml"]))
            oid = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
            if oid != item["rmlGitOID"]:
                raise ValueError("Annotation checksum mismatch.")
            labels.write_bytes(contents)
        with pyedflib.EdfReader(str(source)) as edf:
            offset = (edf.getStartdatetime() - annotation_start(labels)).total_seconds()
            if not 0 <= offset < 24 * 3600 or "Mic" not in edf.getSignalLabels():
                raise ValueError("Unverified recording timing or missing canonical bedside Mic channel.")
            if "offsetSeconds" in item and (offset != item["offsetSeconds"] or edf.getFileDuration() != item["durationSeconds"]):
                raise ValueError("Full source timing differs from its verified header.")
        temporary = output.with_suffix(".partial")
        extractor = "extract_spectral_psg.py" if profile == "spectral" else "extract_psg.py"
        channel_args = [] if profile == "spectral" else ["--ambient-channel", "Mic"]
        subprocess.run([sys.executable, str(Path(__file__).with_name(extractor)),
                        "--audio-edf", str(source), "--stages-rml", str(labels),
                        "--subject", item["subject"], *channel_args,
                        "--audio-offset-seconds", str(offset), "--output", str(temporary)], check=True)
        temporary.replace(output)
        source.unlink()
        print(json.dumps({"event": "extracted", "block": number, "total": len(plan), "offsetSeconds": offset}), flush=True)
    combined = root / "epochs.jsonl"
    with combined.open("w") as output:
        for item in plan:
            output.write((features_dir / (Path(item["path"]).stem + ".jsonl")).read_text())
    if args.extract_only:
        print(json.dumps({"event": "extracted_only", "blocks": len(plan), "fittingPerformed": False}), flush=True)
        return
    rows = load_rows(combined)
    model, report = fit(rows, args.steps)
    model.update(modelID=args.model_id, dataSource=f"PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345; mirror {REVISION}")
    report.update(dataset=evidence, mirrorRevision=REVISION, downloadedBytes=total,
                  extractorSHA256=extractor_fingerprint,
                  trainingSteps=args.steps, modelID=args.model_id,
                  coverageSupplements=requested, supplementReason=args.supplement_reason,
                  sourceSha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  stageCounts={split: dict(Counter(row["stage"] for row in rows if partition(row["subject"]) == split))
                               for split in ["train", "test"]})
    (root / "model.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    (root / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # A small real held-out sample lets the native runtime independently verify exported inference.
    held_out = [row for row in rows if partition(row["subject"]) == "test"]
    sample = random.Random(37).sample(held_out, min(20, len(held_out)))
    for row in sample:
        values, _ = probabilities(row["features"], model["means"], model["scales"], model["weights"], model["biases"])
        row["expectedProbabilities"] = dict(zip(model["classes"], values))
    (root / "export-check.json").write_text(json.dumps(sample, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "trained", **report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--max-download-gib", type=float, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--model-id", default="psg-audio-baseline-v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true", help="Prepare observations without fitting or scoring a model.")
    parser.add_argument("--feature-profile", choices=("basic", "spectral"), default="basic")
    parser.add_argument("--download-workers", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--supplement", action="append", default=[], help="Explicit EDF block for annotation coverage, from an already selected night.")
    parser.add_argument("--supplement-reason", default="")
    arguments = parser.parse_args()
    if not (4 <= arguments.subjects <= 100 and 1 <= arguments.blocks <= 4 and 1 <= arguments.steps <= 1000
            and 0 < arguments.max_download_gib <= 128 and len(arguments.supplement) <= 4):
        parser.error("Invalid bounded training configuration.")
    run(arguments)
