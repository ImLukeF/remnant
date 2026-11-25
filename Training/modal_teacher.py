"""Run extract_teacher.py's per-block extraction on Modal, many blocks at once, into one Modal volume.

    modal run Training/modal_teacher.py --inputs /path/to/teacher-inputs [--limit 1]

`inputs` holds the frozen plans copied from the VPS: v8/ and v11/ (each development-plan.json + labels/) and
known-subjects-after-test.json. Only the public PSG-Audio source is downloaded, hash-verified, and deleted after use.
Completed blocks already on the volume are skipped, so a rerun resumes.
"""
import json
from pathlib import Path
import subprocess

import modal

HERE = Path(__file__).resolve().parent
VOLUME = modal.Volume.from_name("sleep-teacher-v1", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg")
         .pip_install("numpy==2.5.5", "scipy==1.18.1", "pyedflib==0.1.42")
         .add_local_dir(HERE, "/root/training", ignore=["__pycache__", "*.pyc"]))
app = modal.App("sleep-teacher-extract", image=image)


@app.function(volumes={"/data": VOLUME}, cpu=2.0, memory=4096, timeout=1800, retries=2, max_containers=40)
def process(item, rml_bytes):
    import sys, tempfile
    sys.path.insert(0, "/root/training")
    from extract_teacher import extract_block
    from run_dataset import checksum, download
    stem = Path(item["path"]).stem
    with tempfile.TemporaryDirectory() as work:
        work = Path(work); source = work / f"{stem}.edf"; rml = work / f"{item['recording']}.rml"
        rml.write_bytes(rml_bytes); download(item, source)
        npz, audio = work / f"{stem}.npz", work / f"{stem}.opus"
        detail = extract_block(item, source, rml, npz, audio)
        detail.update(npzSHA256=checksum(npz), audioSHA256=checksum(audio), path=item["path"])
        for folder, file in (("blocks", npz), ("audio", audio)):
            Path("/data", folder).mkdir(parents=True, exist_ok=True)
            target = Path("/data", folder, file.name); partial = target.with_suffix(target.suffix + ".partial")
            partial.write_bytes(file.read_bytes()); partial.replace(target)
    VOLUME.commit()
    return detail


@app.function(volumes={"/data": VOLUME}, timeout=600)
def publish(files):
    for name, text in files.items():
        Path("/data", name).write_text(text)
    VOLUME.commit()


@app.function(volumes={"/data": VOLUME}, timeout=600)
def existing():
    import sys
    sys.path.insert(0, "/root/training")
    from extract_teacher import fingerprint
    done = {p.stem for p in Path("/data/blocks").glob("*.npz")} if Path("/data/blocks").exists() else set()
    done &= {p.name[:-5] for p in Path("/data/audio").glob("*.opus")} if Path("/data/audio").exists() else set()
    summary = json.loads(Path("/data/block-summary.json").read_text()) if Path("/data/block-summary.json").exists() else {}
    return sorted(done), summary, fingerprint()


@app.local_entrypoint()
def main(inputs: str, limit: int = 0):
    root = Path(inputs)
    known = set(json.loads((root / "known-subjects-after-test.json").read_text()))
    if len(known) != 230: raise ValueError("Expected the 230-person observed registry.")
    files, seen = [], set()
    for plan_root in (root / "v8", root / "v11"):
        for item in json.loads((plan_root / "development-plan.json").read_text())["files"]:
            if item["subject"] not in known: raise ValueError("A planned person is outside the observed registry.")
            if item["path"] not in seen:
                seen.add(item["path"]); files.append((item, plan_root / "labels" / f"{item['recording']}.rml"))
    if limit: files = files[:limit]
    done, summary, extractor = existing.remote()
    pending = [(i, r) for i, r in files if Path(i["path"]).stem not in done or i["path"] not in summary]
    print(json.dumps({"event": "teacher_modal", "total": len(files), "pending": len(pending)}), flush=True)
    failures = []
    for (item, _), result in zip(pending, process.starmap([(i, r.read_bytes()) for i, r in pending], return_exceptions=True)):
        if isinstance(result, Exception): failures.append({"path": item["path"], "error": repr(result)[:500]}); continue
        summary[item["path"]] = result
        if len(summary) % 25 == 0: print(json.dumps({"event": "teacher_block", "complete": len(summary), "total": len(files)}), flush=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    manifest = {"block-summary.json": json.dumps(summary, indent=1), "failures.json": json.dumps(failures, indent=1)}
    if not limit and not failures and all(i["path"] in summary for i, _ in files):
        manifest["complete.json"] = json.dumps({"extractorSHA256": extractor, "sourceSha": source, "blocks": len(files),
            "people": sorted({s["subject"] for s in summary.values()}), "epochs": sum(s["epochs"] for s in summary.values()),
            "scoredEpochs": sum(s["scoredEpochs"] for s in summary.values()),
            "usableEpochs": sum(s["usableEpochs"] for s in summary.values())}, indent=1)
    publish.remote(manifest)
    print(json.dumps({"event": "teacher_modal_done", "complete": len(summary), "failures": len(failures),
                      "finished": "complete.json" in manifest}), flush=True)
