"""Expand the private cohort, reserve new test people, and compare frozen models fairly."""
import argparse
import bisect
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import urllib.request
import uuid

from extract_psg import annotations
from features import STAGES
from run_dataset import (REVISION, REPOSITORY, annotation_start, build_plan, checksum,
                         download, licence_evidence, listing, read_url, resolve)
from train import evaluate, fit, load_rows, partition, probabilities

BASELINE_SHA256 = "bc20fa82ec7f086794300e0d72e77cde4be355a2df1ddeeca43b26db6014ae00"


def assignments(subjects, previous_subjects):
    result = {}
    for subject in subjects:
        if subject in previous_subjects:
            result[subject] = "validation" if partition(subject) == "test" else "train"
        else:
            bucket = int.from_bytes(hashlib.sha256(subject.encode()).digest()[:4], "big") % 10
            result[subject] = "test" if bucket < 2 else "validation" if bucket == 2 else "train"
    if any(result[s] == "test" for s in previous_subjects if s in result):
        raise ValueError("Previously observed people cannot enter the new final test set.")
    return result


def timing(header, start):
    """Resolve the EDF's two-digit year against the annotation date, including midnight/year rollover."""
    day, month, year = map(int, header[168:176].decode("ascii").split("."))
    hour, minute, second = map(int, header[176:184].decode("ascii").split("."))
    candidates = []
    for full_year in range(start.year - 1, start.year + 2):
        if full_year % 100 == year:
            try:
                offset = (dt.datetime(full_year, month, day, hour, minute, second) - start).total_seconds()
                if 0 <= offset < 86400:
                    candidates.append(offset)
            except ValueError:
                continue
    duration = int(header[236:244]) * float(header[244:252])
    if len(candidates) != 1 or not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ValueError("Unverified EDF/annotation timing.")
    return {"offsetSeconds": candidates[0], "durationSeconds": duration}


def header_timing(item, label_path, *, allow_empty=False):
    url = resolve(item["path"]) + "?request=" + uuid.uuid4().hex
    request = urllib.request.Request(url, headers={"Range": "bytes=0-255", "User-Agent": "SleepWakeKit-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get("Content-Range") != f"bytes 0-255/{item['bytes']}":
            raise ValueError("Source did not honor bounded EDF-header request.")
        data = response.read(257)
    if len(data) != 256:
        raise ValueError("Invalid EDF header length.")
    if not any(data):
        if allow_empty:
            return {**item, "unusableReason": "The pinned source has an all-zero EDF header."}
        raise ValueError("Empty EDF header.")
    return {**item, **timing(data, annotation_start(label_path))}


def scored_counts(item, stages):
    times = [start for start, _ in stages]
    result = Counter()
    for offset in range(0, int(item["durationSeconds"]) - 29, 30):
        start = item["offsetSeconds"] + offset
        index = bisect.bisect_right(times, start) - 1
        if index >= 0 and stages[index][1] and (index + 1 == len(times) or times[index + 1] >= start + 30):
            result[stages[index][1]] += 1
    return result


def atomic_json(path, value, *, compact=False):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=None if compact else 2,
                                    separators=(",", ":") if compact else None, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args, previous, fingerprint, root, cache, evidence):
    path = root / "cohort-plan.json"
    config = {"subjectsRequested": args.subjects, "blocksPerSubject": 2, "revision": REVISION,
              "extractorSHA256": fingerprint, "baselineSHA256": BASELINE_SHA256}
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(key) != value for key, value in config.items()):
            raise ValueError("Use a different directory for a different cohort or feature schema.")
        return saved
    plan = build_plan(args.subjects, 2)
    old_subjects = {item["subject"] for item in previous["files"]}
    people = {item["subject"] for item in plan}
    if not old_subjects < people:
        raise ValueError("The expanded cohort must contain all previous people and new people.")
    roles = assignments(people, old_subjects)
    recordings = {item["recording"]: item for item in plan}
    stages = {}
    for recording, item in recordings.items():
        label_path = cache / (recording + ".rml")
        data = read_url(resolve(item["rml"]))
        if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != item["rmlGitOID"]:
            raise ValueError("Annotation checksum mismatch.")
        label_path.write_bytes(data)
        stages[recording] = annotations(label_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        plan = list(pool.map(lambda item: header_timing(item, cache / (item["recording"] + ".rml")), plan))
    additions = []
    # Annotation coverage is fixed before fitting/prediction; never inspect test model scores to choose files.
    for role in ("train", "validation", "test"):
        counts = Counter()
        for item in plan:
            if roles[item["subject"]] == role:
                counts.update(scored_counts(item, stages[item["recording"]]))
        for stage in STAGES:
            if counts[stage]:
                continue
            found = False
            for recording, original in recordings.items():
                if roles[original["subject"]] != role or not any(label == stage for _, label in stages[recording]):
                    continue
                for source in listing("APNEA_EDF/" + recording):
                    if (source["type"] != "file" or not source["path"].endswith(".edf")
                            or source["path"] in {item["path"] for item in plan}):
                        continue
                    item = {**{key: original[key] for key in ("subject", "recording", "rml", "rmlGitOID")},
                            "path": source["path"], "bytes": source["size"], "sha256": source["lfs"]["oid"]}
                    item = header_timing(item, cache / (recording + ".rml"))
                    coverage = scored_counts(item, stages[recording])
                    if coverage[stage]:
                        plan.append(item)
                        counts.update(coverage)
                        additions.append({"path": item["path"], "split": role, "missingStage": stage})
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"The selected {role} people lack scored {stage} coverage.")
    saved = {**config, "repository": REPOSITORY, "licenceEvidence": evidence, "splits": roles,
             "coverageAdditionsBeforeFitting": additions, "files": plan,
             "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if sum(item["bytes"] for item in plan) > args.max_download_gib * 1024 ** 3:
        raise ValueError("Cohort exceeds its download limit.")
    atomic_json(path, saved)
    return saved


def run(args):
    root, previous_root = args.output.resolve(), args.previous.resolve()
    if root == previous_root:
        raise ValueError("Keep the frozen previous experiment in a separate directory.")
    root.mkdir(parents=True, exist_ok=True)
    cache, features_dir = root / "raw-cache", root / "features"
    cache.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)
    code = Path(__file__).parent
    fingerprint = hashlib.sha256((code / "extract_psg.py").read_bytes() + (code / "features.py").read_bytes()).hexdigest()
    previous = json.loads((previous_root / "download-plan.json").read_text())
    if previous["revision"] != REVISION or previous["extractorSHA256"] != fingerprint:
        raise ValueError("Previous features are not compatible with this pinned source/feature schema.")
    if checksum(previous_root / "model.json") != BASELINE_SHA256:
        raise ValueError("The frozen comparison checkpoint changed.")
    evidence = licence_evidence()
    saved = prepare(args, previous, fingerprint, root, cache, evidence)
    plan, roles = saved["files"], saved["splits"]
    if sum(item["bytes"] for item in plan) > args.max_download_gib * 1024 ** 3:
        raise ValueError("Cohort exceeds its download limit.")
    print(json.dumps({"event": "plan", "people": dict(Counter(roles.values())), "blocks": len(plan),
                      "declaredBytes": sum(x["bytes"] for x in plan),
                      "coverageAdditions": saved["coverageAdditionsBeforeFitting"]}), flush=True)
    if args.prepare_only:
        return
    progress_path = root / "feature-checksums.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    reusable = {item["path"]: item for item in previous["files"]}
    for item in plan:
        key = Path(item["path"]).stem + ".jsonl"
        output, old = features_dir / key, previous_root / "features" / key
        if item["path"] in progress:
            if not output.exists() or checksum(output) != progress[item["path"]]:
                raise ValueError("Derived feature cache changed; do not silently reuse it.")
        elif item["path"] in reusable and reusable[item["path"]]["sha256"] == item["sha256"] and old.exists():
            # Only reuse verified source/schema matches from the previous private experiment.
            if any(json.loads(line)["subject"] != item["subject"] for line in old.read_text().splitlines()):
                raise ValueError("Cached features belong to another participant.")
            shutil.copyfile(old, output)
            progress[item["path"]] = checksum(output)
    atomic_json(progress_path, progress)
    pending = iter([item for item in plan if item["path"] not in progress])
    print(json.dumps({"event": "reused", "blocks": len(progress), "remaining": len(plan) - len(progress)}), flush=True)
    import pyedflib
    import numpy as np
    # At most two independent source downloads and two raw files; feature extraction remains sequential.
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = {}
        def enqueue():
            item = next(pending, None)
            if item is not None:
                source = cache / (Path(item["path"]).stem + ".edf")
                active[pool.submit(download, item, source)] = (item, source)
        enqueue()
        enqueue()
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            future = next(iter(ready))
            item, source = active.pop(future)
            future.result()
            labels = cache / (item["recording"] + ".rml")
            if not labels.exists():
                data = read_url(resolve(item["rml"]))
                if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != item["rmlGitOID"]:
                    raise ValueError("Annotation checksum mismatch on resume.")
                labels.write_bytes(data)
            data = labels.read_bytes()
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != item["rmlGitOID"]:
                raise ValueError("Cached annotation content changed.")
            with pyedflib.EdfReader(str(source)) as edf:
                offset = (edf.getStartdatetime() - annotation_start(labels)).total_seconds()
                if offset != item["offsetSeconds"] or edf.getFileDuration() != item["durationSeconds"] or "Mic" not in edf.getSignalLabels():
                    raise ValueError("Full EDF did not match its preflight timing or bedside channel.")
            output = features_dir / (Path(item["path"]).stem + ".jsonl")
            temporary = output.with_suffix(".partial")
            subprocess.run([sys.executable, str(code / "extract_psg.py"), "--audio-edf", str(source),
                            "--stages-rml", str(labels), "--subject", item["subject"], "--ambient-channel", "Mic",
                            "--audio-offset-seconds", str(offset), "--output", str(temporary)], check=True)
            temporary.replace(output)
            progress[item["path"]] = checksum(output)
            atomic_json(progress_path, progress)
            source.unlink()
            print(json.dumps({"event": "extracted", "complete": len(progress), "total": len(plan)}), flush=True)
            enqueue()
    combined = root / "epochs.jsonl"
    with combined.open("w") as output:
        for item in plan:
            output.write((features_dir / (Path(item["path"]).stem + ".jsonl")).read_text())
    rows = load_rows(combined)
    if {row["subject"] for row in rows} != set(roles):
        raise ValueError("One or more planned people have no usable audio; disclose and revise the plan before fitting.")
    subsets = {role: [row for row in rows if roles[row["subject"]] == role] for role in ("train", "validation", "test")}
    for role, subset in subsets.items():
        if {row["stage"] for row in subset} != set(STAGES):
            raise ValueError(f"Usable audio lacks four-stage coverage in {role}.")
    model, report = fit(rows, steps=40, split_by_subject=roles)
    model.update(modelID="psg-audio-expanded-v2", dataSource=f"PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345; mirror {REVISION}")
    baseline = json.loads((previous_root / "model.json").read_text())
    test = evaluate(model, subsets["test"])
    report.update(modelID=model["modelID"], validation=evaluate(model, subsets["validation"]), freshTest=test,
                  frozenBaselineFreshTest=evaluate(baseline, subsets["test"]),
                  splitSubjects={role: sorted({row["subject"] for row in subset}) for role, subset in subsets.items()},
                  stageCounts={role: dict(Counter(row["stage"] for row in subset)) for role, subset in subsets.items()},
                  sourceSha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  trainingSteps=40, modelSelection="Fixed original features, optimizer, seed and 40 epochs; no test-based tuning.",
                  dataset=evidence, mirrorRevision=REVISION, extractorSHA256=fingerprint, baselineSHA256=BASELINE_SHA256,
                  environment={"python": sys.version.split()[0], "numpy": np.__version__, "pyedflib": pyedflib.__version__},
                  coverageAdditionsBeforeFitting=saved["coverageAdditionsBeforeFitting"])
    atomic_json(root / "model.json", model)
    atomic_json(root / "evaluation.json", report)
    sample = random.Random(37).sample(subsets["test"], min(20, len(subsets["test"])))
    for row in sample:
        values, _ = probabilities(row["features"], model["means"], model["scales"], model["weights"], model["biases"])
        row["expectedProbabilities"] = dict(zip(model["classes"], values))
    atomic_json(root / "export-check.json", sample)
    print(json.dumps({"event": "trained", "trainingPeople": len(report["splitSubjects"]["train"]),
                      "validationPeople": len(report["splitSubjects"]["validation"]), "testPeople": len(test["subjects"]),
                      "balancedAccuracy": test["balancedAccuracy"], "lightRecall": test["recallPerClass"][1],
                      "baselineBalancedAccuracyOnSamePeople": report["frozenBaselineFreshTest"]["balancedAccuracy"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=60)
    parser.add_argument("--max-download-gib", type=float, default=96)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 21 <= args.subjects <= 100 or not 0 < args.max_download_gib <= 128:
        parser.error("Cohort and download budget must stay within the documented bounds.")
    run(args)
