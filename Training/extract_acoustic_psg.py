"""Preserve within-epoch sound patterns while verifying the existing annotation-aligned observations."""
import argparse
import json
from pathlib import Path

from acoustic_frames import extract_frames
from spectral import extract_epoch


def run(args):
    import numpy as np
    import pyedflib
    expected = [json.loads(line) for line in args.previous_features.read_text().splitlines()]
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
        for row in expected:
            offset = (row["startSeconds"]-args.audio_offset_seconds)*rate
            if abs(offset-round(offset))>1e-6 or offset<0:
                raise ValueError("Existing clinical epoch is not on a verified audio boundary.")
            pcm = (edf.readSignal(channel,round(offset),30*rate,digital=True)-center)/scale
            measured,usable = extract_epoch(pcm,rate)
            if usable != row["usable"] or not np.allclose(measured,row["features"],rtol=0,atol=2e-6):
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
    parser.add_argument("--previous-features",type=Path,required=True)
    parser.add_argument("--audio-offset-seconds",type=float,required=True)
    parser.add_argument("--subject",required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
