"""Teacher corpus: bedside audio plus every time-aligned PSG signal, for multi-task GPU training.

Development people only (the 230 already observed); the untouched PSG-Audio people are never listed or read.
For every full 30-second epoch of each verified block, from audio sample zero:
- student inputs exactly as the phone computes them: spectral-v4 summary (58) and acoustic frames (240x25);
- teacher audio: the bedside Mic at 16 kHz, Opus 24 kb/s, one file per block, row k starts at k*30 s;
- labels: scored stage (-1 when unscored or crossing a transition), seconds of RML event groups,
  sensor measurements (breathing from chest/abdomen effort and airflow, leg EMG, body position, pulse,
  SpO2, snore sensor) and EEG relative band powers. Sensors are training targets only, never model inputs.
"""
import argparse
import bisect
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from acoustic_frames import extract_frames
from expand_cohort import atomic_json
from expanded_features import recording_duration
from extract_psg import annotations
from features import STAGES
from run_dataset import annotation_start, checksum, prefetched_audio
from spectral import extract_epoch

EVENTS = ("arousal", "snore", "apnea", "hypopnea", "desaturation", "legMovement")
APNEA = {"ObstructiveApnea", "CentralApnea", "MixedApnea"}
BREATHING = ("Effort THO", "Effort ABD", "Flow Patient")
SENSORS = (["thoRate", "thoRegularity", "thoAmplitude", "abdRate", "abdRegularity", "abdAmplitude",
            "flowRate", "flowRegularity", "flowAmplitude", "leg1LogRMS", "leg2LogRMS", "bodyPosition",
            "bodyChanged", "pulseMean", "pulseStd", "spo2Mean", "spo2Min", "snoreSensorLogRMS"])
EEG_BANDS = ((0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30))  # delta, theta, alpha, sigma, beta
AUDIO_RATE = 16000


def fingerprint():
    from expanded_features import fingerprint as summary
    return hashlib.sha256(summary().encode() + Path(__file__).read_bytes()
                          + inspect.getsource(extract_frames).encode()).hexdigest()


def git_oid(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def rml_events(path):
    result = []
    for node in ET.parse(path).getroot().iter():
        if node.tag.rsplit("}", 1)[-1] != "Event": continue
        kind, start, length = node.attrib.get("Type", ""), node.attrib.get("Start"), node.attrib.get("Duration")
        if start is None or length is None: continue
        group = ("arousal" if "Arousal" in kind else "snore" if kind == "Snore" else "apnea" if kind in APNEA
                 else "hypopnea" if kind == "Hypopnea" else "desaturation" if kind == "RelativeDesaturation"
                 else "legMovement" if kind == "LegMovement" else None)
        if group: result.append((EVENTS.index(group), float(start), float(start) + float(length)))
    return result


def breathing(signal, rate):
    """Dominant 6-42 breaths/minute rhythm of a 60-second window: rate, peak share of power, amplitude."""
    import numpy as np
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < rate * 40 or not np.isfinite(x).all() or x.std() == 0: return [np.nan] * 3
    x = (x - x.mean()) * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x, n=int(rate * 240))) ** 2  # zero-padded to 1/240 Hz bins
    f = np.fft.rfftfreq(int(rate * 240), 1 / rate)
    band, total = (f >= 0.1) & (f <= 0.7), (f >= 0.05) & (f <= 1.0)
    peak = f[band][np.argmax(power[band])]
    near = total & (np.abs(f - peak) <= 0.05)
    return [peak * 60, power[near].sum() / max(power[total].sum(), 1e-300), float(np.std(signal))]


def eeg_bands(signal, rate):
    import numpy as np
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < rate * 30 or not np.isfinite(x).all(): return [np.nan] * len(EEG_BANDS)
    power = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / rate)
    bands = np.asarray([power[(f >= lo) & (f < hi)].sum() for lo, hi in EEG_BANDS])
    return (10 * np.log10(np.maximum(bands / max(bands.sum(), 1e-300), 1e-12))).tolist()


def channels(edf):
    """First occurrence of each wanted label; duplicate labels (two 'Flow Patient') keep the first."""
    found = {}
    for index, label in enumerate(edf.getSignalLabels()):
        found.setdefault(label, index)
    return found


def extract_block(item, source, rml, output, audio_output):
    import numpy as np
    import pyedflib
    from scipy.signal import resample_poly
    if git_oid(rml.read_bytes()) != item["rmlGitOID"]: raise ValueError("Verified clinical annotations changed.")
    stages = annotations(rml); times = [s[0] for s in stages]; duration = recording_duration(rml); events = rml_events(rml)
    with pyedflib.EdfReader(str(source)) as edf:
        found = channels(edf)
        if "Mic" not in found: raise ValueError("No bedside Mic channel; contact audio is not accepted.")
        offset = (edf.getStartdatetime() - annotation_start(rml)).total_seconds()
        if offset != item["offsetSeconds"] or edf.getFileDuration() != item["durationSeconds"]:
            raise ValueError("Full source timing differs from its verified header.")
        mic = found["Mic"]; rate = edf.getSampleFrequency(mic)
        if rate != int(rate) or int(rate) % AUDIO_RATE: raise ValueError("Mic rate must be a multiple of 16 kHz.")
        rate = int(rate); samples = int(edf.getNSamples()[mic])
        minimum, maximum = edf.getDigitalMinimum(mic), edf.getDigitalMaximum(mic)
        center, scale = (maximum + minimum + 1) / 2, (maximum - minimum + 1) / 2
        if scale <= 0: raise ValueError("Invalid digital audio range.")
        signals = {}
        for label in (*BREATHING, "Leg 1", "Leg 2", "Body", "PulseRate", "SpO2", "Snore", "EEG C4-A1"):
            if label in found:
                signals[label] = (edf.readSignal(found[label]), edf.getSampleFrequency(found[label]))
        encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(AUDIO_RATE),
                                    "-ac", "1", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "24k", "-application", "audio",
                                    "-f", "ogg", "-y", str(audio_output)], stdin=subprocess.PIPE)
        rows = {k: [] for k in ("features", "frames", "start", "stage", "usable", "events", "sensors", "eeg")}
        try:
            for sample in range(0, samples - 30 * rate + 1, 30 * rate):
                start = sample / rate + offset
                if start + 30 > duration + 1e-6: break
                pcm = (edf.readSignal(mic, sample, 30 * rate, digital=True) - center) / scale
                features, usable = extract_epoch(pcm, rate)
                index = bisect.bisect_right(times, start) - 1
                stage = (-1 if start < 0 or index < 0 or stages[index][1] is None
                         or index + 1 < len(times) and times[index + 1] < start + 30 else STAGES.index(stages[index][1]))
                quiet = np.clip(resample_poly(pcm, 1, rate // AUDIO_RATE), -1, 32767 / 32768)
                encoder.stdin.write((np.rint(quiet * 32768)).astype("<i2").tobytes())
                seconds = [0.0] * len(EVENTS)
                for group, a, b in events:
                    seconds[group] += max(0.0, min(b, start + 30) - max(a, start))

                def window(label, before, after):
                    values, fs = signals[label]
                    a = max(0, int(round((start - offset - before) * fs))); b = min(len(values), int(round((start - offset + after) * fs)))
                    return values[a:b], fs
                sensor = []
                for label in BREATHING:
                    sensor += breathing(*window(label, 15, 45)) if label in signals else [np.nan] * 3
                for label in ("Leg 1", "Leg 2"):
                    sensor.append(float(np.log10(np.sqrt(np.mean(window(label, 0, 30)[0] ** 2)) + 1e-9)) if label in signals else np.nan)
                if "Body" in signals:
                    body = np.rint(window("Body", 0, 30)[0]); sensor += [float(np.median(body)), float(len(set(body.tolist())) > 1)]
                else: sensor += [np.nan, np.nan]
                for label in ("PulseRate", "SpO2"):
                    if label in signals:
                        values = window(label, 0, 30)[0]
                        sensor += [float(np.mean(values)), float(np.std(values) if label == "PulseRate" else np.min(values))]
                    else: sensor += [np.nan, np.nan]
                sensor.append(float(np.log10(np.sqrt(np.mean(window("Snore", 0, 30)[0] ** 2)) + 1e-9)) if "Snore" in signals else np.nan)
                rows["features"].append(features); rows["frames"].append(extract_frames(pcm, rate)); rows["start"].append(start)
                rows["stage"].append(stage); rows["usable"].append(usable); rows["events"].append(seconds); rows["sensors"].append(sensor)
                rows["eeg"].append(eeg_bands(*window("EEG C4-A1", 0, 30)) if "EEG C4-A1" in signals else [np.nan] * len(EEG_BANDS))
        finally:
            encoder.stdin.close()
            if encoder.wait() != 0: raise ValueError("Opus encoding failed.")
    n = len(rows["start"])
    if n == 0: raise ValueError("A planned block produced no epochs.")
    with output.open("wb") as file:
        np.savez_compressed(file, features=np.asarray(rows["features"], dtype=np.float32).reshape(n, 58),
            frames=np.asarray(rows["frames"], dtype=np.float32).reshape(n, 240, 25),
            startSeconds=np.asarray(rows["start"], dtype=np.float64), stage=np.asarray(rows["stage"], dtype=np.int8),
            usable=np.asarray(rows["usable"], dtype=bool), events=np.asarray(rows["events"], dtype=np.float32),
            eventNames=np.asarray(EVENTS), sensors=np.asarray(rows["sensors"], dtype=np.float32), sensorNames=np.asarray(SENSORS),
            eeg=np.asarray(rows["eeg"], dtype=np.float32), subject=np.asarray(item["subject"]), recording=np.asarray(item["recording"]),
            sourcePath=np.asarray(item["path"]), sourceSHA256=np.asarray(item["sha256"]), sampleRate=np.asarray(rate),
            audioFile=np.asarray(audio_output.name.replace(".partial", "")), audioRate=np.asarray(AUDIO_RATE),
            sensorChannels=np.asarray(sorted(signals)), extractorSHA256=np.asarray(fingerprint()))
    return {"subject": item["subject"], "epochs": n, "usableEpochs": int(sum(rows["usable"])),
            "scoredEpochs": int(sum(s >= 0 for s in rows["stage"])), "sensorChannels": sorted(signals)}


def plan_files(roots, known):
    files, seen = [], set()
    for root in roots:
        plan = json.loads((root / "development-plan.json").read_text())
        for item in plan["files"]:
            if item["subject"] not in known: raise ValueError("A planned person is outside the observed registry.")
            if item["path"] in seen: continue
            seen.add(item["path"]); files.append((item, root / "labels" / (item["recording"] + ".rml")))
    return files


def run(args):
    known = set(json.loads(args.known.read_text()))
    if len(known) != 230: raise ValueError("Expected the 230-person observed registry.")
    files = plan_files(args.plan, known)
    if args.limit_blocks: files = files[:args.limit_blocks]
    root = args.output.resolve(); (root / "blocks").mkdir(parents=True, exist_ok=True); (root / "audio").mkdir(exist_ok=True)
    cache = root / "raw-cache"; cache.mkdir(exist_ok=True)
    protocol = {"extractorSHA256": fingerprint(), "people": len({i["subject"] for i, _ in files}), "blocks": len(files),
                "sourceSha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    path = root / "extraction-protocol.json"
    if path.exists() and json.loads(path.read_text())["extractorSHA256"] != protocol["extractorSHA256"]:
        raise ValueError("Extraction math changed during the teacher run.")
    if not path.exists(): atomic_json(path, protocol)
    progress_path = root / "checksums.json"; progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    summary_path = root / "block-summary.json"; summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    pending = [(i, r) for i, r in files if i["path"] not in progress]
    selected, size = [], 0
    for item, rml in pending:
        if selected and size + item["bytes"] > args.batch_gib * 1024 ** 3: break
        selected.append((item, rml)); size += item["bytes"]
    labels = {i["path"]: r for i, r in selected}
    print(json.dumps({"event": "teacher_batch", "complete": len(progress), "total": len(files), "batchBlocks": len(selected)}), flush=True)
    names = lambda item: Path(item["path"]).stem
    with ProcessPoolExecutor(max_workers=args.extract_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {}

        def finish():
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                item, source = active.pop(future); detail = future.result(); stem = names(item)
                (root / "blocks" / f"{stem}.partial").replace(root / "blocks" / f"{stem}.npz")
                (root / "audio" / f"{stem}.opus.partial").replace(root / "audio" / f"{stem}.opus")
                detail["audioSHA256"] = checksum(root / "audio" / f"{stem}.opus"); summaries[item["path"]] = detail
                atomic_json(summary_path, summaries)
                progress[item["path"]] = checksum(root / "blocks" / f"{stem}.npz"); atomic_json(progress_path, progress)
                source.unlink()
                print(json.dumps({"event": "teacher_block", "complete": len(progress), "total": len(files), "epochs": detail["epochs"]}), flush=True)
        for _, item, source in prefetched_audio([i for i, _ in selected], cache, root / "blocks", args.download_workers):
            if shutil.disk_usage(root).free < 20 * 1024 ** 3: raise ValueError("Keep 20 GiB of free disk headroom.")
            stem = names(item)
            active[pool.submit(extract_block, item, source, labels[item["path"]], root / "blocks" / f"{stem}.partial",
                               root / "audio" / f"{stem}.opus.partial")] = (item, source)
            if len(active) >= args.extract_workers: finish()
        while active: finish()
    if len(progress) == len(files):
        complete = {"extractorSHA256": protocol["extractorSHA256"], "blocks": len(progress), "checksumsSHA256": checksum(progress_path),
                    "people": sorted({s["subject"] for s in summaries.values()}),
                    "epochs": sum(s["epochs"] for s in summaries.values()), "scoredEpochs": sum(s["scoredEpochs"] for s in summaries.values()),
                    "usableEpochs": sum(s["usableEpochs"] for s in summaries.values())}
        atomic_json(root / "complete.json", complete)
        print(json.dumps({"event": "teacher_complete", "people": len(complete["people"]), "hours": complete["epochs"] / 120}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, action="append", required=True, help="Frozen development plan root (with labels/).")
    p.add_argument("--known", type=Path, required=True, help="The 230-person observed registry.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--download-workers", type=int, choices=(1, 2, 3, 4), default=3)
    p.add_argument("--extract-workers", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--batch-gib", type=float, default=48)
    p.add_argument("--limit-blocks", type=int, default=0, help="Smoke test on the first N blocks.")
    args = p.parse_args()
    if not 0 < args.batch_gib <= 128: p.error("Keep each batch bounded.")
    run(args)
