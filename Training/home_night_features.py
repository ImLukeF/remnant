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
    edges = inv(np.linspace(mel(60), mel(8000), BANDS + 2))
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
    for s in range(0, count, 4000):
        e = min(s + 4000, count)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N)
        x = audio[idx].astype(np.float64) / scale
        x -= x.mean(axis=1, keepdims=True)
        power = np.abs(np.fft.rfft(x * window, axis=1)) ** 2 + 1e-20
        mels[s:e] = 10 * np.log10(power @ bank.T + 1e-20)
        voiced = power[:, 4:]
        centroid[s:e] = (voiced * freqs[4:]).sum(1) / voiced.sum(1)
        flat[s:e] = np.exp(np.log(voiced).mean(1)) / voiced.mean(1)
    return mels, centroid, flat


def breathing(envelope):
    """Strength and period of the strongest repeat between 2 and 8 seconds."""
    step = 8
    x = envelope[: len(envelope) // step * step].reshape(-1, step).mean(1)
    rate = FPS / step
    kernel = int(rate * 10) | 1
    x = x - np.convolve(np.pad(x, kernel // 2, mode="edge"), np.ones(kernel) / kernel, "valid")
    x = np.clip(x, -6, 6)
    if x.std() < 1e-6:
        return 0.0, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1 :]
    ac = ac / ac[0]
    lo, hi = int(2 * rate), int(8 * rate)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[lag]), lag / rate


def main(night, bits, requantise, out):
    meta = json.load(open(f"{night}/night.json"))
    audio, scale = read(f"{night}/night.caf", bits, requantise)
    mels, centroid, flat = frames(audio, scale)
    floor = np.median(mels, axis=0)
    mels = mels - floor  # everything relative to this night's own quiet level
    loud = 10 * np.log10((10 ** (mels[:, 4:16] / 10)).mean(1))
    rows, labels, times = [], [], []
    per = int(30 * FPS)
    for o in meta["observations"]:
        a = int((o["start"] - meta["started"]) * FPS)
        if a < 0 or a + per > len(mels):
            continue
        m, lv = mels[a : a + per], loud[a : a + per]
        wide = loud[max(0, a - per) : a + 2 * per]
        strength, period = breathing(wide)
        strength30, period30 = breathing(lv)
        rows.append(np.concatenate([
            m.mean(0), m.std(0),
            [lv.mean(), lv.std(), np.percentile(lv, 95) - np.percentile(lv, 5),
             (lv > 10).mean(), (lv > 20).mean(), np.abs(np.diff(lv)).mean(),
             centroid[a : a + per].mean(), centroid[a : a + per].std(),
             np.log(flat[a : a + per]).mean(), strength, period, strength30, period30],
        ]))
        mid = (o["start"] + o["end"]) / 2
        watch = [STAGES[h["category"]] for h in meta["healthReference"]
                 if h["start"] <= mid < h["end"] and h["category"] in STAGES]
        labels.append(watch[0] if watch else -1)
        times.append((o["start"] - meta["started"]) / 3600)
    phone = [["awake", "light", "deep", "rem"].index(o["stage"]) if o["stage"] in ("awake", "light", "deep", "rem") else -1
             for o in meta["observations"]][: len(rows)]
    np.savez(out, x=np.array(rows, dtype=np.float32), y=np.array(labels), hours=np.array(times), phone=np.array(phone))
    print(out, np.array(rows).shape, np.bincount(np.array(labels) + 1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4])
