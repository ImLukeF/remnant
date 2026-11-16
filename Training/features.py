"""Reference feature schema. Raw audio is never part of the exported model."""
import math

FEATURE_NAMES = [
    "mean_dbfs", "std_dbfs", "min_dbfs", "max_dbfs",
    "mean_zero_crossing_rate", "clipped_fraction", "envelope_autocorrelation",
]
STAGES = ("awake", "light", "deep", "rem")


def extract_epoch(samples, sample_rate):
    if not 8000 <= sample_rate <= 192000 or len(samples) != sample_rate * 30:
        raise ValueError("Provide exactly 30 seconds of normalized mono PCM.")
    levels, crossings, clips = [], [], []
    previous = None
    valid = True
    for offset in range(0, len(samples), sample_rate):
        square_sum, crossed, clipped = 0.0, 0, 0
        for raw in samples[offset:offset + sample_rate]:
            value = float(raw) if math.isfinite(raw) else 0.0
            valid = valid and math.isfinite(raw) and abs(value) <= 1
            square_sum += value * value
            clipped += abs(value) >= 0.999
            if previous is not None:
                crossed += (previous < 0) != (value < 0)
            previous = value
        levels.append(10 * math.log10(max(square_sum / sample_rate, 1e-12)))
        crossings.append(crossed / sample_rate)
        clips.append(clipped / sample_rate)
    mean = sum(levels) / 30
    centered = [value - mean for value in levels]
    energy = sum(value * value for value in centered)
    periodicity = max(
        sum(a * b for a, b in zip(centered[lag:], centered)) / energy
        for lag in range(2, 11)
    ) if energy > 1e-9 else 0.0
    clipped_fraction = sum(clips) / 30
    return [
        mean, math.sqrt(energy / 30), min(levels), max(levels),
        sum(crossings) / 30, clipped_fraction, periodicity,
    ], valid and mean > -119 and clipped_fraction < 0.01
