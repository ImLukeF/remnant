"""Per-30-second sound features for one home night, lined up with the Watch's stages."""
import json
import sys

import numpy as np

SR, N, HOP, BANDS = 16000, 1024, 512, 24
FPS = SR / HOP
STAGES = {2: 0, 3: 1, 4: 2, 5: 3}  # awake, light, deep, REM


def read(path, bits, requantise):
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=4096)
    if bits == 16:
        return np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2"), 32768.0
    b = raw[: len(raw) // 3 * 3].reshape(-1, 3)
    out = np.empty(len(b), dtype=np.int32)
    for s in range(0, len(b), 1 << 24):
        c = b[s : s + (1 << 24)].astype(np.int32)
        v = c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16)
        out[s : s + len(c)] = np.where(v >= 1 << 23, v - (1 << 24), v)
    if requantise:  # what the old 16-bit saving would have kept
        return np.round(out / 256.0).astype(np.int32), 32768.0
    return out, float(1 << 23)


def mel_bank():
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = inv(np.linspace(mel(100), mel(7500), BANDS + 2))
    freqs = np.fft.rfftfreq(N, 1 / SR)
    bank = np.zeros((BANDS, len(freqs)))
    for i in range(BANDS):
        lo, mid, hi = edges[i : i + 3]
        bank[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank, freqs


def frames(audio, scale):
    bank, freqs = mel_bank()
    window = np.hanning(N)
    count = (len(audio) - N) // HOP + 1
    mels = np.empty((count, BANDS), dtype=np.float32)
    centroid = np.empty(count, dtype=np.float32)
    flat = np.empty(count, dtype=np.float32)
    hi_band = np.empty(count, dtype=np.float32)
    # 100..7500 Hz -> bins 7..480, 2000..7500 Hz -> bins 128..480 (exact match with Swift)
    voiced_slice = slice(7, 481)
    hi_slice = slice(128, 481)
    f_voiced = freqs[voiced_slice]
    for s in range(0, count, 4000):
        e = min(s + 4000, count)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N)
        x = audio[idx].astype(np.float64) / scale
        x -= x.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(x * window, axis=1)) ** 2 + 1e-20
        mels[s:e] = 10 * np.log10(power @ bank.T + 1e-20)
        voiced = power[:, voiced_slice]
        centroid[s:e] = (voiced * f_voiced).sum(1) / voiced.sum(1)
        flat[s:e] = np.exp(np.log(voiced).mean(1)) / voiced.mean(1)
        hi_band[s:e] = 10 * np.log10(power[:, hi_slice].mean(1) + 1e-20)
    return mels, centroid, flat, hi_band


def breathing(envelope):
    """Strength, period, harmonic peak stability, spectral RRV dispersion and entropy between 2 and 8 s."""
    step = 8
    n_trim = len(envelope) // step * step
    rate = FPS / step
    if n_trim // step <= int(8 * rate):
        return 0.0, 0.0, 0.0, 0.0, 0.0
    x = envelope[:n_trim].reshape(-1, step).mean(1)
    kernel = int(rate * 10) | 1
    x = x - np.convolve(np.pad(x, kernel // 2, mode="edge"), np.ones(kernel) / kernel, "valid")
    x = np.clip(x, -6, 6)
    if x.std() < 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    ac = ac / ac[0]
    lo, hi = int(2 * rate), int(8 * rate)
    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag])
    period = lag / rate
    peaks = [float(ac[k * lag]) for k in (1, 2, 3) if k * lag < len(ac)]
    peak_sd = float(np.std(peaks)) if len(peaks) >= 2 else 0.0
    # Spectral RRV in respiratory band (0.12 - 0.50 Hz = 7.2 - 30 breaths/min)
    m = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(m))) ** 2 + 1e-12
    sfreqs = np.fft.rfftfreq(m, 1.0 / rate)
    resp = spec[(sfreqs >= 0.12) & (sfreqs <= 0.50)]
    if len(resp) == 0:
        return strength, period, peak_sd, 1.0, 1.0
    total = float(resp.sum())
    pk = int(np.argmax(resp))
    h1 = float(resp[max(0, pk - 1) : min(len(resp), pk + 2)].sum())
    rrv_disp = float(1.0 - h1 / total)
    p = resp / total
    spec_ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))
    return strength, period, peak_sd, rrv_disp, spec_ent


def epoch_rows(mels, centroid, flat, hi_rel, starts):
    """One row of 71 features per half-minute. `mels` is 10th-pct noise-gated; `hi_rel` is relative to night median."""
    loud = 10 * np.log10((10 ** (mels[:, 2:16] / 10)).mean(1) + 1e-20)
    per = int(30 * FPS)
    rows = []
    for a in starts:
        m, lv, hr = mels[a : a + per], loud[a : a + per], hi_rel[a : a + per]
        s90, p90, sd90, rrv90, ent90 = breathing(loud[max(0, a - per) : a + 2 * per])
        s30, p30, sd30, rrv30, ent30 = breathing(lv)
        rows.append(np.concatenate([
            m.mean(0), m.std(0),
            [lv.mean(), lv.std(), np.percentile(lv, 95) - np.percentile(lv, 5),
             (lv > 10).mean(), (lv > 20).mean(), np.abs(np.diff(lv)).mean(),
             centroid[a : a + per].mean(), centroid[a : a + per].std(),
             np.log(flat[a : a + per]).mean(),
             s90, p90, sd90, rrv90, ent90, s30, p30, sd30, rrv30, ent30,
             hr.mean(), hr.std(), (hr > 6).mean(), (hr > 12).mean()],
        ]))
    return np.array(rows, dtype=np.float32)


def main(night, bits, requantise, out):
    meta = json.load(open(f"{night}/night.json"))
    audio, scale = read(f"{night}/night.caf", bits, requantise)
    mels, centroid, flat, hi_band = frames(audio, scale)
    mels = np.maximum(mels - np.percentile(mels, 10, axis=0), 0.0)  # adaptive spectral noise gate
    hi_rel = hi_band - np.median(hi_band)
    per = int(30 * FPS)
    kept = [o for o in meta["observations"] if 0 <= int((o["start"] - meta["started"]) * FPS) <= len(mels) - per]
    rows = epoch_rows(mels, centroid, flat, hi_rel, [int((o["start"] - meta["started"]) * FPS) for o in kept])
    labels, times = [], []
    for o in kept:
        mid = (o["start"] + o["end"]) / 2
        watch = [STAGES[h["category"]] for h in meta["healthReference"]
                 if h["start"] <= mid < h["end"] and h["category"] in STAGES]
        labels.append(watch[0] if watch else -1)
        times.append((o["start"] - meta["started"]) / 3600)
    names = ["awake", "light", "deep", "rem"]
    phone = [names.index(o["stage"]) if o["stage"] in names else -1 for o in kept]
    np.savez(out, x=rows, y=np.array(labels), hours=np.array(times), phone=np.array(phone))
    print(out, rows.shape, np.bincount(np.array(labels) + 1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4])
