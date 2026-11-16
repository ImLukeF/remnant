"""Extract spectral observations from the exact bedside Mic, retaining independently scored labels."""
import argparse
import bisect
import json
import math
from pathlib import Path
from extract_psg import annotations
from spectral import extract_epoch


def run(args):
    import pyedflib
    stages = annotations(args.stages_rml)
    times = [item[0] for item in stages]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pyedflib.EdfReader(str(args.audio_edf)) as edf, args.output.open("w") as output:
        channel = edf.getSignalLabels().index("Mic")
        rate = edf.getSampleFrequency(channel)
        if rate != int(rate) or not 8000 <= rate <= 192000:
            raise ValueError("Unsupported bedside audio sample rate.")
        rate = int(rate)
        minimum, maximum = edf.getDigitalMinimum(channel), edf.getDigitalMaximum(channel)
        center, scale = (maximum + minimum + 1) / 2, (maximum - minimum + 1) / 2
        for offset in range(0, int(edf.getNSamples()[channel]) - 30 * rate + 1, 30 * rate):
            start = offset / rate + args.audio_offset_seconds
            index = bisect.bisect_right(times, start) - 1
            if index < 0 or stages[index][1] is None or (index + 1 < len(times) and times[index + 1] < start + 30):
                continue
            pcm = (edf.readSignal(channel, offset, 30 * rate, digital=True) - center) / scale
            features, usable = extract_epoch(pcm, rate)
            output.write(json.dumps({"subject": args.subject, "stage": stages[index][1], "features": features,
                                     "usable": usable, "startSeconds": start, "sampleRate": rate}, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-edf", type=Path, required=True)
    parser.add_argument("--stages-rml", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--audio-offset-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.audio_offset_seconds):
        parser.error("Offset must be finite and independently verified.")
    run(args)
