"""Prepare a fresh annotation-aligned test only after the candidate has been frozen."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

from expand_cohort import atomic_json, header_timing, scored_counts
from extract_psg import annotations
from features import STAGES
from improve_model import parameters_digest
from run_dataset import REVISION, REPOSITORY, licence_evidence, listing, read_url, resolve, run, select_subjects


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "raw-cache"
    cache.mkdir(exist_ok=True)
    selected = json.loads((args.selection / "selected-model.json").read_text())
    selection = json.loads((args.selection / "selection.json").read_text())
    if parameters_digest(selected) != selection["parametersSHA256"]:
        raise ValueError("Candidate parameters changed after validation selection.")
    known = set(selection["trainingSubjects"] + selection["validationSubjects"])
    if args.known_subjects:
        known.update(json.loads(args.known_subjects.read_text()))
    code = Path(__file__).parent
    fingerprint = hashlib.sha256((code / "extract_psg.py").read_bytes() + (code / "features.py").read_bytes()).hexdigest()
    plan_path = root / "download-plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if (saved["selectedParametersSHA256"] != selection["parametersSHA256"] or saved["subjectsRequested"] != args.subjects
                or saved["extractorSHA256"] != fingerprint or saved["revision"] != REVISION
                or set(saved["previouslyObservedSubjects"]) != known):
            raise ValueError("Resume only the identical frozen final-test experiment.")
    else:
        evidence = licence_evidence()
        people = [person for person in select_subjects(listing("APNEA_RML"), 1000) if person[0] not in known][:args.subjects]
        if len(people) != args.subjects:
            raise ValueError("Insufficient new participants for the frozen final test.")
        plan, stages, sources = [], {}, {}
        for subject, recording, rml in people:
            data = read_url(resolve(rml["path"]))
            if hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != rml["oid"]:
                raise ValueError("Test annotation checksum mismatch.")
            label_path = cache / (recording + ".rml")
            label_path.write_bytes(data)
            stages[recording] = annotations(label_path)
            files = sorted((item for item in listing("APNEA_EDF/" + recording)
                            if item["type"] == "file" and item["path"].endswith(".edf")), key=lambda item: item["path"])
            sources[recording] = [{"subject": subject, "recording": recording, "rml": rml["path"], "rmlGitOID": rml["oid"],
                                   "path": item["path"], "bytes": item["size"], "sha256": item["lfs"]["oid"]} for item in files]
            indexes = sorted({min(len(files) - 1, int(len(files) * fraction)) for fraction in (0.25, 0.75)})
            if not files:
                raise ValueError("New participant has no audio source files.")
            plan.extend(sources[recording][i] for i in indexes)
        with ThreadPoolExecutor(max_workers=4) as pool:
            plan = list(pool.map(lambda item: header_timing(item, cache / (item["recording"] + ".rml")), plan))
        short = [item for item in plan if item["durationSeconds"] < 30]
        plan = [item for item in plan if item["durationSeconds"] >= 30]
        substitutions = []
        for tail in short:
            options = sources[tail["recording"]]
            position = next(i for i, item in enumerate(options) if item["path"] == tail["path"])
            for index in sorted(range(len(options)), key=lambda i: (abs(i - position), i)):
                item = options[index]
                if item["path"] == tail["path"] or item["path"] in {old["path"] for old in plan}:
                    continue
                candidate = header_timing(item, cache / (item["recording"] + ".rml"))
                if candidate["durationSeconds"] >= 30:
                    plan.append(candidate)
                    substitutions.append({"excluded": tail["path"], "replacement": candidate["path"],
                                          "reason": "Source tail shorter than one 30-second observation."})
                    break
            else:
                raise ValueError("No usable replacement block in the selected participant's night.")
        counts = Counter()
        for item in plan:
            counts.update(scored_counts(item, stages[item["recording"]]))
        additions = []
        for stage in STAGES:
            if counts[stage]:
                continue
            for recording, files in sources.items():
                if not any(label == stage for _, label in stages[recording]):
                    continue
                for item in files:
                    if item["path"] in {old["path"] for old in plan}:
                        continue
                    candidate = header_timing(item, cache / (recording + ".rml"))
                    coverage = scored_counts(candidate, stages[recording])
                    if coverage[stage]:
                        plan.append(candidate)
                        counts.update(coverage)
                        additions.append({"path": item["path"], "missingStage": stage})
                        break
                if counts[stage]:
                    break
            if not counts[stage]:
                raise ValueError("Fresh cohort has no annotated " + stage + " coverage.")
        saved = {"repository": REPOSITORY, "revision": REVISION, "subjectsRequested": args.subjects, "blocksPerSubject": 2,
                 "extractorSHA256": fingerprint, "licenceEvidence": evidence, "files": plan,
                 "selectedParametersSHA256": selection["parametersSHA256"], "previouslyObservedSubjects": sorted(known),
                 "testSubjects": sorted(person[0] for person in people), "annotationCoverageAdditions": additions,
                 "sourceQualitySubstitutions": substitutions,
                 "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "fittingForbidden": True}
        if sum(item["bytes"] for item in plan) > args.max_download_gib * 1024 ** 3:
            raise ValueError("Fresh-test plan exceeds its download budget.")
        atomic_json(plan_path, saved)
    if set(saved["testSubjects"]) & known:
        raise ValueError("Previously observed participants leaked into final testing.")
    print(json.dumps({"event": "fresh_test_frozen", "people": len(saved["testSubjects"]), "blocks": len(saved["files"]),
                      "bytes": sum(item["bytes"] for item in saved["files"]), "coverageAdditions": saved["annotationCoverageAdditions"]}), flush=True)
    if not args.prepare_only:
        run(argparse.Namespace(output=root, subjects=args.subjects, blocks=2, max_download_gib=args.max_download_gib,
                               supplement=[], supplement_reason="", prepare_only=False, extract_only=True, steps=40, model_id="extract-only"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--max-download-gib", type=float, default=24)
    parser.add_argument("--known-subjects", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 8 <= args.subjects <= 30 or not 0 < args.max_download_gib <= 48:
        parser.error("Final-test cohort and download budget exceed the bounded protocol.")
    prepare(args)
