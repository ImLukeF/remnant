"""Extract bedside audio epochs with aligned scored stages from a local PSG-Audio download.

Install pyedflib separately. Dataset access, consent and reuse terms are the caller's responsibility.
Never pass the tracheal/contact microphone to a model intended for phone bedside audio.
"""
import argparse
import bisect
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from features import extract_epoch

# These are the RML format's declared stage values, not free-text interpretation.
RML_STAGES = {"Wake": "awake", "NonREM1": "light", "NonREM2": "light", "NonREM3": "deep", "REM": "rem"}


def annotations(path):
    result = []
    root = ET.parse(path).getroot()
    scored = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "UserStaging"]
    if len(scored) != 1:
        raise ValueError("Provide exactly one UserStaging annotation sequence.")
    for node in scored[0].iter():
        if node.tag.rsplit("}", 1)[-1] == "Stage" and node.get("Type") in RML_STAGES:
            result.append((float(node.attrib["Start"]), RML_STAGES[node.attrib["Type"]]))
    result.sort()
    if not result or any(a[0] >= b[0] for a, b in zip(result, result[1:])):
        raise ValueError("Use one RML scored staging sequence, with distinct ascending stage times.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-edf", type=Path, required=True)
    parser.add_argument("--stages-rml", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--ambient-channel", required=True, help="Exact EDF bedside microphone label, not a contact sensor.")
    parser.add_argument("--audio-offset-seconds", type=float, required=True,
                        help="Explicit verified PSG time of audio sample zero.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyedflib
    import numpy as np
    stages = annotations(args.stages_rml)
    times = [item[0] for item in stages]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pyedflib.EdfReader(str(args.audio_edf)) as edf, args.output.open("w") as output:
        channel = edf.getSignalLabels().index(args.ambient_channel)
        rate = edf.getSampleFrequency(channel)
        if rate != int(rate):
            raise ValueError("Resample non-integral sample rates before extraction.")
        rate = int(rate)
        # EDF physical audio units differ by recorder. Normalize using the declared digital range.
        minimum, maximum = edf.getDigitalMinimum(channel), edf.getDigitalMaximum(channel)
        center = (maximum + minimum + 1) / 2
        scale = (maximum - minimum + 1) / 2
        for offset in range(0, int(edf.getNSamples()[channel]) - 30 * rate + 1, 30 * rate):
            start = offset / rate + args.audio_offset_seconds
            index = bisect.bisect_right(times, start) - 1
            if index < 0 or (index + 1 < len(times) and times[index + 1] < start + 30):
                continue  # Never assign one label across a scored stage transition.
            pcm = (edf.readSignal(channel, offset, 30 * rate, digital=True) - center) / scale
            # Vectorized equivalent of the dependency-free reference, for full-night recordings.
            valid = bool(np.isfinite(pcm).all() and (np.abs(pcm) <= 1).all())
            pcm = np.where(np.isfinite(pcm), pcm, 0).astype(np.float64)
            seconds = pcm.reshape(30, rate)
            levels = 10 * np.log10(np.maximum(np.mean(seconds * seconds, axis=1), 1e-12))
            crossed = np.concatenate(([False], (pcm[:-1] < 0) != (pcm[1:] < 0)))
            crossing_rate = crossed.reshape(30, rate).mean()
            clipped = float((np.abs(pcm) >= 0.999).mean())
            centered = levels - levels.mean()
            energy = float(np.sum(centered * centered))
            periodicity = max(float(np.sum(centered[lag:] * centered[:-lag])) / energy
                              for lag in range(2, 11)) if energy > 1e-9 else 0
            features = [float(levels.mean()), float(levels.std()), float(levels.min()), float(levels.max()),
                        float(crossing_rate), clipped, periodicity]
            usable = valid and features[0] > -119 and clipped < 0.01
            output.write(json.dumps({"subject": args.subject, "stage": stages[index][1],
                                     "features": features, "usable": usable, "startSeconds": start},
                                    allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
