"""Preserve within-epoch sound patterns while verifying the existing annotation-aligned observations."""
import argparse
import bisect
import json
import math
from pathlib import Path

from acoustic_frames import extract_frames
from spectral import extract_epoch


def run(args):
    import numpy as np
    import pyedflib
    expected = [json.loads(line) for line in args.previous_features.read_text().splitlines()] if args.previous_features else None
    basic, frames, times, labels, valid = [],[],[],[],[]
    from features import STAGES
    with pyedflib.EdfReader(str(args.audio_edf)) as edf:
        channel = edf.getSignalLabels().index("Mic")
        rate = edf.getSampleFrequency(channel)
        if rate != int(rate) or not 8000<=rate<=192000:
            raise ValueError("Unsupported bedside microphone sample rate.")
        rate = int(rate)
        minimum,maximum = edf.getDigitalMinimum(channel),edf.getDigitalMaximum(channel)
        center,scale = (maximum+minimum+1)/2,(maximum-minimum+1)/2
        if expected is None:
            from extract_psg import annotations
            stages=annotations(args.stages_rml); boundaries=[r[0] for r in stages]; work=[]
            for offset in range(0,int(edf.getNSamples()[channel])-30*rate+1,30*rate):
                start=offset/rate+args.audio_offset_seconds; position=bisect.bisect_right(boundaries,start)-1
                if position<0 or stages[position][1] is None or (position+1<len(stages) and boundaries[position+1]<start+30):
                    continue
                work.append({"subject":args.subject,"startSeconds":start,"stage":stages[position][1]})
        else: work=expected
        for row in work:
            if row["subject"]!=args.subject:
                raise ValueError("An acoustic block cannot change the verified participant identifier.")
            offset = (row["startSeconds"]-args.audio_offset_seconds)*rate
            if abs(offset-round(offset))>1e-6 or offset<0:
                raise ValueError("Existing clinical epoch is not on a verified audio boundary.")
            pcm = (edf.readSignal(channel,round(offset),30*rate,digital=True)-center)/scale
            measured,usable = extract_epoch(pcm,rate)
            if expected is not None and (usable != row["usable"] or not np.allclose(measured,row["features"],rtol=0,atol=2e-6)):
                raise ValueError("Reprocessed observations differ from the previously verified source.")
            basic.append(measured); frames.append(extract_frames(pcm,rate))
            times.append(row["startSeconds"]); labels.append(STAGES.index(row["stage"])); valid.append(usable)
    with args.output.open("wb") as output:
        np.savez_compressed(output,features=np.asarray(basic,dtype=np.float64).reshape(-1,58),
                            frames=np.asarray(frames,dtype=np.float32).reshape(-1,240,25),
                            startSeconds=np.asarray(times,dtype=np.float64),stage=np.asarray(labels,dtype=np.int8),
                            usable=np.asarray(valid,dtype=bool),subject=np.asarray(args.subject),sampleRate=np.asarray(rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-edf",type=Path,required=True)
    labels=parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--previous-features",type=Path)
    labels.add_argument("--stages-rml",type=Path)
    parser.add_argument("--audio-offset-seconds",type=float,required=True)
    parser.add_argument("--subject",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not math.isfinite(args.audio_offset_seconds): parser.error("A verified finite annotation offset is required.")
    run(args)
