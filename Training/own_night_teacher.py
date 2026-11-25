"""Convert one consented Parley night into the teacher-corpus block format, locally.

Uses the phone's own observations (spectral-v4 summary and acoustic frames, as computed live on the iPhone) and
encodes its saved 16 kHz audio to Opus like the PSG blocks. Nothing measured stages, events or body sensors at home,
so those are -1 / NaN, never zero or guessed. Context markers and the audio-only breathing estimate are kept separately.

    python3 own_night_teacher.py NIGHT_FOLDER OUTPUT_FOLDER --subject home-luke
"""
import argparse
import json
from pathlib import Path
import subprocess

from extract_teacher import EEG_BANDS, EVENTS, SENSORS


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("night", type=Path); p.add_argument("output", type=Path); p.add_argument("--subject", required=True)
    args = p.parse_args()
    night = json.loads((args.night / "night.json").read_text())
    rows = sorted((json.loads(f.read_text()) for f in (args.night / "observations").glob("*.json")), key=lambda o: o["epoch"]["start"])
    if len(rows) != len(night["observations"]): raise ValueError("Observation files and night.json disagree.")
    if night.get("audioSampleRate") != 16000 or not (args.night / "audio.caf").exists(): raise ValueError("Needs the saved 16 kHz audio.")
    t0 = night["started"]; n = len(rows)
    starts = np.asarray([o["epoch"]["start"] - t0 for o in rows])
    if abs(starts[0]) > 1e-3 or (np.abs(np.diff(starts) - 30) > 1e-3).any(): raise ValueError("Epochs must be contiguous from audio start.")
    frames = np.asarray([o["epoch"]["acousticFrames"] or [np.nan] * 6000 for o in rows], dtype=np.float32).reshape(n, 240, 25)
    labels = json.loads((args.night / "labels.json").read_text()) if (args.night / "labels.json").exists() else {}
    breathing = {e["epochID"]: e for e in labels.get("breathingEstimate", {}).get("epochs", [])}
    estimate = np.asarray([[breathing.get(o["id"], {}).get(k, np.nan) for k in ("breathsPerMinuteEstimate", "rhythmStrength")]
                           for o in rows], dtype=np.float32)
    markers = [(m["kind"], m["date"] - t0) for m in night.get("markers", [])]
    args.output.mkdir(parents=True, exist_ok=True); stem = f"{args.subject}-{night['id']}"
    audio = args.output / f"{stem}.opus"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.night / "audio.caf"), "-t", str(n * 30),
                    "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", "-application", "audio", "-f", "ogg", "-y", str(audio)], check=True)
    with (args.output / f"{stem}.npz").open("wb") as file:
        np.savez_compressed(file, features=np.asarray([o["epoch"]["features"] for o in rows], dtype=np.float32), frames=frames,
            startSeconds=starts, stage=np.full(n, -1, dtype=np.int8), usable=np.asarray([o["epoch"]["usable"] for o in rows]),
            events=np.full((n, len(EVENTS)), np.nan, dtype=np.float32), eventNames=np.asarray(EVENTS),
            sensors=np.full((n, len(SENSORS)), np.nan, dtype=np.float32), sensorNames=np.asarray(SENSORS),
            eeg=np.full((n, len(EEG_BANDS)), np.nan, dtype=np.float32), subject=np.asarray(args.subject),
            recording=np.asarray(night["id"]), sourcePath=np.asarray("parley-night"), sourceSHA256=np.asarray(""),
            sampleRate=np.asarray(night["inputSampleRate"]), audioFile=np.asarray(audio.name), audioRate=np.asarray(16000),
            sensorChannels=np.asarray([], dtype=str), extractorSHA256=np.asarray("phone-observations"),
            home=np.asarray(True), audioBreathingEstimate=estimate,
            markerKinds=np.asarray([k for k, _ in markers]), markerSeconds=np.asarray([s for _, s in markers]),
            expectedTVOffSeconds=np.asarray((night["expectedTVOff"] - t0) if night.get("expectedTVOff") else np.nan),
            device=np.asarray(night.get("deviceModel") or ""), modelID=np.asarray(night.get("modelID") or ""))
    print(json.dumps({"subject": args.subject, "epochs": n, "usable": int(sum(o["epoch"]["usable"] for o in rows)),
                      "framesMissing": int(sum(o["epoch"]["acousticFrames"] is None for o in rows)), "audio": audio.name}))


if __name__ == "__main__":
    main()
