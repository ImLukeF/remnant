"""Extract clean and waveform-augmented bedside measurements within the scored recording."""
import bisect
import hashlib
import inspect
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from extract_psg import annotations
from features import STAGES
from run_dataset import annotation_start
from run_spectral_corpus import feature_fingerprint
from spectral import extract_epoch


def fingerprint():
    code=Path(__file__).parent
    return hashlib.sha256(feature_fingerprint().encode()+inspect.getsource(annotation_start).encode()
                          +b"".join((code/name).read_bytes() for name in ("expanded_features.py","room_noise.py"))).hexdigest()


def recording_duration(path):
    sessions=[]
    for node in ET.parse(path).getroot().iter():
        if node.tag.rsplit("}",1)[-1]!="Session": continue
        values={child.tag.rsplit("}",1)[-1]:child.text for child in node}
        if "RecordingStart" in values and "Duration" in values: sessions.append(float(values["Duration"]))
    if len(sessions)!=1 or not math.isfinite(sessions[0]) or not 0<sessions[0]<=86400:
        raise ValueError("Require exactly one verified recording duration; never extend the last stage beyond it.")
    return sessions[0]


def extract(item,source,labels,output,bank,role):
    import numpy as np
    import pyedflib
    stages=annotations(labels);times=[s[0] for s in stages];duration=recording_duration(labels)
    clean=[];noisy=[];starts=[];classes=[];usable=[];noisy_usable=[];snrs=[]
    with pyedflib.EdfReader(str(source)) as edf:
        if "Mic" not in edf.getSignalLabels(): raise ValueError("No bedside Mic channel; contact audio is not accepted.")
        offset=(edf.getStartdatetime()-annotation_start(labels)).total_seconds()
        if offset!=item["offsetSeconds"] or edf.getFileDuration()!=item["durationSeconds"]:
            raise ValueError("Full source timing differs from its verified header.")
        channel=edf.getSignalLabels().index("Mic");sample_rate=edf.getSampleFrequency(channel)
        if sample_rate!=int(sample_rate) or not 8000<=sample_rate<=192000: raise ValueError("Unsupported source microphone rate.")
        rate=int(sample_rate);samples=int(edf.getNSamples()[channel])
        if abs(samples/rate-edf.getFileDuration())>1e-5: raise ValueError("Microphone and EDF duration disagree.")
        minimum,maximum=edf.getDigitalMinimum(channel),edf.getDigitalMaximum(channel)
        center=(maximum+minimum+1)/2;scale=(maximum-minimum+1)/2
        if scale<=0: raise ValueError("Invalid digital audio range.")
        for sample in range(0,samples-30*rate+1,30*rate):
            start=sample/rate+offset;index=bisect.bisect_right(times,start)-1
            if (start+30>duration+1e-6 or index<0 or stages[index][1] is None
                    or index+1<len(times) and times[index+1]<start+30): continue
            pcm=(edf.readSignal(channel,sample,30*rate,digital=True)-center)/scale
            features,valid=extract_epoch(pcm,rate)
            if valid:
                augmented,snr=bank.mix(pcm,rate,item["subject"],start,role)
                augmented_features,augmented_valid=extract_epoch(augmented,rate)
            else:
                augmented_features=features;augmented_valid=False;snr=0
            clean.append(features);noisy.append(augmented_features);starts.append(start)
            classes.append(STAGES.index(stages[index][1]));usable.append(valid);noisy_usable.append(valid and augmented_valid);snrs.append(snr)
    # A binary handle keeps np.savez from silently appending an extension to atomic temporary files.
    with output.open("wb") as file:
        np.savez_compressed(file,features=np.asarray(clean,dtype=np.float64).reshape((-1,58)),
            augmentedFeatures=np.asarray(noisy,dtype=np.float64).reshape((-1,58)),startSeconds=np.asarray(starts,dtype=np.float64),
            stage=np.asarray(classes,dtype=np.int8),usable=np.asarray(usable,dtype=bool),augmentedUsable=np.asarray(noisy_usable,dtype=bool),
            noiseSNR=np.asarray(snrs,dtype=np.float32),subject=np.asarray(item["subject"]),sampleRate=np.asarray(rate),
            recordingDuration=np.asarray(duration),noiseRole=np.asarray(role),sourcePath=np.asarray(item["path"]),
            sourceSHA256=np.asarray(item["sha256"]),noiseManifestSHA256=np.asarray(bank.sha),extractorSHA256=np.asarray(fingerprint()))
    return {"scoredEpochs":len(starts),"usableEpochs":sum(usable),"augmentedUsableEpochs":sum(noisy_usable),"sampleRate":rate}
