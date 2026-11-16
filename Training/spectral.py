"""Gain-normalized bedside sound spectrum; independent reference for the Swift DSP."""
from features import FEATURE_NAMES
from functools import lru_cache

SPECTRAL_NAMES = FEATURE_NAMES + [f"mel_relative_{kind}_{i:02d}" for kind in ("mean", "std") for i in range(24)] + [
    "spectral_entropy_mean", "spectral_flux_mean", "spectral_flux_std"]


def fft_size(rate):
    size = 1
    while size * 2 <= rate // 8:
        size *= 2
    return size


@lru_cache(maxsize=4)
def geometry(rate):
    import numpy as np
    size = fft_size(rate)
    indexes = np.asarray([second * rate + frame * rate // 8 for second in range(30) for frame in range(8)])[:, None] + np.arange(size)
    mel_edges = np.linspace(2595 * np.log10(1 + 80 / 700), 2595 * np.log10(1 + 4000 / 700), 26)
    edges = 700 * (10 ** (mel_edges / 2595) - 1)
    frequencies = np.arange(size // 2 + 1) * rate / size
    filters = np.asarray([np.maximum(0, np.minimum((frequencies - edges[k]) / (edges[k + 1] - edges[k]),
                                                  (edges[k + 2] - frequencies) / (edges[k + 2] - edges[k + 1]))) for k in range(24)])
    return size, indexes, np.hanning(size), filters


def extract_epoch(samples, rate):
    import numpy as np
    if not 8000 <= rate <= 192000 or len(samples) != rate * 30:
        raise ValueError("Supply 30 seconds of normalized mono PCM at an integral supported rate.")
    pcm = np.asarray(samples, dtype=np.float64)
    valid = bool(np.isfinite(pcm).all() and (np.abs(pcm) <= 1).all())
    pcm = np.where(np.isfinite(pcm), pcm, 0)
    seconds = pcm.reshape(30, rate)
    levels = 10 * np.log10(np.maximum(np.mean(seconds * seconds, axis=1), 1e-12))
    crossings = np.concatenate(([False], (pcm[:-1] < 0) != (pcm[1:] < 0))).mean()
    clipped = float((np.abs(pcm) >= 0.999).mean())
    centered = levels - levels.mean()
    energy = float(np.sum(centered * centered))
    periodicity = max(float(np.sum(centered[lag:] * centered[:-lag])) / energy for lag in range(2, 11)) if energy > 1e-9 else 0
    basic = [float(levels.mean()), float(levels.std()), float(levels.min()), float(levels.max()), float(crossings), clipped, periodicity]
    size, indexes, window, filters = geometry(rate)
    frames = pcm[indexes]
    frames = (frames - frames.mean(axis=1, keepdims=True)) * window
    power = np.abs(np.fft.rfft(frames)) ** 2
    bands = power @ filters.T
    totals = bands.sum(axis=1, keepdims=True)
    normalized = bands / np.maximum(totals, 1e-300)
    relative = 10 * np.log10(np.maximum(normalized, 1e-12))
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-300)), axis=1) / np.log(24)
    flux = np.abs(np.diff(normalized, axis=0)).sum(axis=1)
    features = basic + relative.mean(axis=0).tolist() + relative.std(axis=0).tolist() + [float(entropy.mean()), float(flux.mean()), float(flux.std())]
    usable = valid and basic[0] > -119 and clipped < 0.01 and float(totals.mean()) / (size * size) > 1e-12
    return features, usable
