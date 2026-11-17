"""Time-resolved Mel observations; shared geometry with the independently verified Swift spectrum."""
from spectral import geometry

FRAME_NAMES = [f"mel_relative_{i:02d}" for i in range(24)] + ["band_power_dbfs"]
FRAMES_PER_EPOCH = 240


def extract_frames(samples, rate):
    import numpy as np
    pcm = np.asarray(samples,dtype=np.float64)
    if len(pcm) != rate*30 or not np.isfinite(pcm).all():
        raise ValueError("Supply a finite 30-second normalized audio epoch.")
    size,indexes,window,filters = geometry(rate)
    windows = pcm[indexes]
    windows = (windows-windows.mean(axis=1,keepdims=True))*window
    power = np.abs(np.fft.rfft(windows))**2
    bands = power@filters.T
    totals = bands.sum(axis=1,keepdims=True)
    relative = 10*np.log10(np.maximum(bands/np.maximum(totals,1e-300),1e-12))
    level = 10*np.log10(np.maximum(totals/(size*size),1e-12))
    # Four decimal places are far below microphone resolution and stabilize cross-platform FFT rounding.
    return (np.rint(np.concatenate((relative,level),axis=1)*1e4)/1e4).astype(np.float32)
