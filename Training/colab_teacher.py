# %% [markdown]
# # Sleep teacher → phone student
# Runtime → Change runtime type → GPU (A100 best, L4 or T4 fine), then Runtime → Run all.
# It reads `My Drive/SleepWakeTeacher/data`, saves progress there, and resumes if the session stops.
#
# 1. Whisper-small turns each 30 s of bedside audio into an embedding (frozen, resumable).
# 2. A teacher learns stages plus arousal/snore/apnea events, breathing, movement, pulse, oxygen and EEG targets.
#    It sees the whole night (past and future); it never runs on the phone.
# 3. A student with the phone's exact conv-GRU format (schema 4) learns from the true stages and the teacher,
#    with random loudness changes so quiet phones are handled. Its JSON is saved for native checks.
# Validation uses 40 fixed people. The final independent test happens later on people not in this folder.

# %%
import hashlib, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch

LOCAL_TEST = os.environ.get("TEACHER_LOCAL_TEST") == "1"
if LOCAL_TEST:
    ROOT = Path(os.environ["TEACHER_ROOT"])
else:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/SleepWakeTeacher")
DATA, HOME, OUT = ROOT / "data", ROOT / "home", ROOT / "runs" / "v12"
EMB = OUT / "whisper"
for folder in (OUT, EMB): folder.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER = os.environ.get("TEACHER_WHISPER", "openai/whisper-small")
SEGMENTS = 10  # Whisper frames pooled to 10 x 3 s per epoch
STAGES = ["awake", "light", "deep", "rem"]
torch.manual_seed(12); np.random.seed(12)
print("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "| data", DATA)
complete = json.loads((DATA / "complete.json").read_text()) if (DATA / "complete.json").exists() else None
assert LOCAL_TEST or (complete and complete["blocks"] == 1226), "Copy of the 1,226-block corpus is not complete yet."

# %% Load every block (labels, phone features and frames) into memory.
def load_block(path):
    with np.load(path, allow_pickle=False) as b:
        return {k: b[k] for k in ("features", "frames", "startSeconds", "stage", "usable", "events", "sensors", "eeg")} | {
            "subject": str(b["subject"]), "audio": str(b["audioFile"]), "home": "home" in b.files}

block_paths = sorted((DATA / "blocks").glob("*.npz"))
assert LOCAL_TEST or (len(block_paths) == 1226 and len(list((DATA / "audio").glob("*.opus"))) == 1226), \
    f"Drive has {len(block_paths)} of 1,226 blocks so far; wait for Google Drive to finish uploading, then run again." 
if LOCAL_TEST: block_paths = block_paths[:int(os.environ.get("TEACHER_BLOCKS", "6"))]
blocks = [load_block(p) for p in block_paths]
home_paths = sorted(HOME.glob("*.npz")) if HOME.exists() else []
home = [load_block(p) for p in home_paths]
SENSOR_NAMES = list(np.load(block_paths[0])["sensorNames"]); EVENT_NAMES = list(np.load(block_paths[0])["eventNames"])
REGRESS = [i for i, n in enumerate(SENSOR_NAMES) if n != "bodyPosition"]

def concat(items, key): return np.concatenate([b[key] for b in items])
subject = np.concatenate([np.full(len(b["stage"]), b["subject"]) for b in blocks])
block_id = np.concatenate([np.full(len(b["stage"]), i) for i, b in enumerate(blocks)])
row_in_block = np.concatenate([np.arange(len(b["stage"])) for b in blocks])
stage = concat(blocks, "stage").astype(np.int64); usable = concat(blocks, "usable")
features = concat(blocks, "features").astype(np.float32); frames = concat(blocks, "frames").astype(np.float16)
events = (concat(blocks, "events") > 0).astype(np.float32)
people = sorted(set(subject))

def person_z(values):
    """Per-person z-scores: targets are changes within a night, not differences between people's sensors."""
    out = np.full_like(values, np.nan, dtype=np.float32)
    for p in people:
        m = subject == p; v = values[m]
        mu, sd = np.nanmean(v, 0), np.nanstd(v, 0)
        out[m] = (v - mu) / np.where(sd > 1e-6, sd, np.nan)
    return np.clip(out, -6, 6)

sensors = person_z(concat(blocks, "sensors")[:, REGRESS]); eeg = person_z(concat(blocks, "eeg"))
VALIDATION = sorted(people, key=lambda p: hashlib.sha256(f"teacher-v12:{p}".encode()).hexdigest())[:40 if not LOCAL_TEST else 2]
is_val = np.isin(subject, VALIDATION)
print(f"{len(people)} people, {len(stage)} epochs, {usable.sum()} usable, validation people {len(VALIDATION)}, home nights {len(home)}")

# %% Whisper embeddings (frozen), one file per block on Drive so a stopped session resumes.
def decode(path, rows):
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    audio = np.pad(audio, (0, max(0, rows * 480000 - len(audio))))[:rows * 480000]
    return audio.reshape(rows, 480000)

def embed_all(items, folder):
    from transformers import WhisperFeatureExtractor, WhisperModel
    pending = [b for b in items if not (folder / (Path(b["audio"]).stem + ".npy")).exists()]
    if not pending: return
    filters = torch.tensor(WhisperFeatureExtractor.from_pretrained(WHISPER).mel_filters, dtype=torch.float32, device=DEVICE).T
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    encoder = WhisperModel.from_pretrained(WHISPER, torch_dtype=dtype).encoder.to(DEVICE).eval()
    window = torch.hann_window(400, device=DEVICE); started = time.time()
    def decoded():  # at most 6 decoded hours in memory while the GPU works
        with ThreadPoolExecutor(4) as pool:
            queue = []
            for b in pending:
                queue.append((b, pool.submit(decode, (DATA / "audio" if not b["home"] else HOME) / b["audio"], len(b["stage"]))))
                if len(queue) >= 6: item, future = queue.pop(0); yield item, future.result()
            for item, future in queue: yield item, future.result()
    if True:
        for done, (b, audio) in enumerate(decoded(), 1):
            out = []
            for start in range(0, len(audio), 32):
                x = torch.from_numpy(audio[start:start + 32]).to(DEVICE)
                spec = torch.stft(x, 400, 160, window=window, return_complex=True)[..., :-1].abs() ** 2
                logmel = torch.clamp(filters @ spec, min=1e-10).log10()
                logmel = torch.maximum(logmel, logmel.amax(dim=(1, 2), keepdim=True) - 8.0)
                with torch.no_grad():
                    hidden = encoder(((logmel + 4.0) / 4.0).to(dtype)).last_hidden_state
                out.append(hidden.float().reshape(len(x), SEGMENTS, -1, hidden.shape[-1]).mean(2).cpu().numpy().astype(np.float16))
            np.save(folder / (Path(b["audio"]).stem + ".npy"), np.concatenate(out))
            if done % 25 == 0 or done == len(pending):
                print(f"whisper {done}/{len(pending)} blocks, {(time.time() - started) / 60:.1f} min", flush=True)
    del encoder; torch.cuda.empty_cache() if DEVICE == "cuda" else None

embed_all(blocks + home, EMB)
whisper = np.concatenate([np.load(EMB / (Path(b["audio"]).stem + ".npy")) for b in blocks])
assert whisper.shape[0] == len(stage)
print("whisper embeddings", whisper.shape)

# %% Teacher: whole-night context (40 epochs = 20 minutes), many targets.
feat_mu, feat_sd = features[~is_val].mean(0), features[~is_val].std(0) + 1e-6
features_n = np.clip((features - feat_mu) / feat_sd, -8, 8).astype(np.float32)
D = whisper.shape[-1]; WINDOW = 40

class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = torch.nn.Linear(D, 256); self.attend = torch.nn.Linear(256, 1)
        self.phone = torch.nn.Linear(58, 64); self.mix = torch.nn.Linear(320, 256)
        self.position = torch.nn.Parameter(torch.zeros(WINDOW, 256))
        layer = torch.nn.TransformerEncoderLayer(256, 4, 768, 0.1, batch_first=True, norm_first=True)
        self.body = torch.nn.TransformerEncoder(layer, 4)
        self.stage = torch.nn.Linear(256, 4); self.events = torch.nn.Linear(256, len(EVENT_NAMES))
        self.sensors = torch.nn.Linear(256, len(REGRESS)); self.eeg = torch.nn.Linear(256, 5)

    def forward(self, w, f, mask):
        a = torch.nn.functional.gelu(self.audio(w))                    # B,T,S,256
        a = (torch.softmax(self.attend(a), dim=2) * a).sum(2)            # B,T,256
        x = self.mix(torch.cat((a, torch.nn.functional.gelu(self.phone(f))), -1)) + self.position[: w.shape[1]]
        h = self.body(x, src_key_padding_mask=~mask)
        return self.stage(h), self.events(h), self.sensors(h), self.eeg(h)

def windows(index, length, stride):
    """Windows of consecutive epochs within one block (blocks are contiguous 30 s grids)."""
    out = []
    for b in np.unique(block_id[index]):
        rows = index[block_id[index] == b]
        for s in range(0, max(1, len(rows) - length + 1), stride): out.append(rows[s:s + length])
        if len(rows) > length and (len(rows) - length) % stride: out.append(rows[-length:])
    return out

def batch(rows_list):
    T = max(len(r) for r in rows_list); B = len(rows_list)
    idx = np.zeros((B, T), dtype=np.int64); mask = np.zeros((B, T), dtype=bool)
    for i, r in enumerate(rows_list): idx[i, :len(r)] = r; mask[i, :len(r)] = True
    t = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=DEVICE)
    return idx, torch.as_tensor(mask, device=DEVICE), t(whisper), t(features_n), t(stage, torch.int64), t(usable, torch.bool), t(events), t(sensors), t(eeg)

train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)
counts = np.bincount(stage[train_rows][(stage[train_rows] >= 0) & usable[train_rows]], minlength=4)
class_weight = torch.tensor((np.maximum(counts, 1) / counts.sum()) ** -0.5, dtype=torch.float32, device=DEVICE); class_weight /= class_weight.mean()

def masked_huber(pred, target, valid):
    valid = valid & torch.isfinite(target)
    return torch.nn.functional.huber_loss(pred[valid], target[valid]) if valid.any() else pred.sum() * 0

def scores(y, p):
    m = np.zeros((4, 4), dtype=int)
    for a, b in zip(y, p): m[a, b] += 1
    recall = np.diag(m) / np.maximum(m.sum(1), 1); f1 = 2 * np.diag(m) / np.maximum(m.sum(1) + m.sum(0), 1)
    return {"balancedAccuracy": float(recall.mean()), "macroF1": float(f1.mean()), "recall": recall.round(3).tolist(), "confusion": m.tolist()}

def teacher_probabilities(model, index):
    model.eval(); total = np.zeros((len(stage), 4)); seen = np.zeros(len(stage))
    with torch.no_grad():
        ws = windows(index, WINDOW, WINDOW // 2)
        for s in range(0, len(ws), 64):
            idx, mask, w, f, *_ = batch(ws[s:s + 64])
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                logits = model(w, f, mask)[0]
            p = torch.softmax(logits.float(), -1).cpu().numpy(); m = mask.cpu().numpy()
            np.add.at(total, idx[m], p[m]); np.add.at(seen, idx[m], 1)
    return total / np.maximum(seen, 1)[:, None]

def evaluate(probabilities, index):
    keep = index[(stage[index] >= 0) & usable[index]]
    return scores(stage[keep], probabilities[keep].argmax(1))

teacher = Teacher().to(DEVICE)
TEACHER_EPOCHS = 12 if not LOCAL_TEST else 1
opt = torch.optim.AdamW(teacher.parameters(), 3e-4, weight_decay=0.05)
train_windows = windows(train_rows, WINDOW, WINDOW // 2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=TEACHER_EPOCHS * math.ceil(len(train_windows) / 64))
scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda"); best = (-1, None); history = []
checkpoint = OUT / "teacher.pt"
if checkpoint.exists() and not LOCAL_TEST:
    saved = torch.load(checkpoint, map_location=DEVICE); teacher.load_state_dict(saved["model"]); best = (saved["score"], None)
    history = saved["history"]; print("loaded saved teacher", saved["score"])
else:
    for epoch in range(TEACHER_EPOCHS):
        teacher.train(); np.random.shuffle(train_windows); started = time.time()
        for s in range(0, len(train_windows), 64):
            _, mask, w, f, y, ok, ev, se, ee = batch(train_windows[s:s + 64])
            with torch.autocast(DEVICE, enabled=DEVICE == "cuda"):
                ls, le, lse, lee = teacher(w, f, mask)
            ls, le, lse, lee = ls.float(), le.float(), lse.float(), lee.float()
            valid = mask & ok & (y >= 0)
            loss = torch.nn.functional.cross_entropy(ls[valid], y[valid], weight=class_weight, label_smoothing=0.02) if valid.any() else ls.sum() * 0
            loss = loss + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(le[mask], ev[mask])
            loss = loss + 0.3 * masked_huber(lse, se, mask[..., None].expand_as(se)) + 0.3 * masked_huber(lee, ee, mask[..., None].expand_as(ee))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        result = evaluate(teacher_probabilities(teacher, val_rows), val_rows); history.append(result)
        print(f"teacher epoch {epoch + 1}: val balanced {result['balancedAccuracy']:.3f} macroF1 {result['macroF1']:.3f} recall {result['recall']} ({(time.time() - started) / 60:.1f} min)", flush=True)
        if result["macroF1"] > best[0]:
            best = (result["macroF1"], {k: v.detach().clone() for k, v in teacher.state_dict().items()})
            torch.save({"model": best[1], "score": best[0], "history": history}, checkpoint)
    teacher.load_state_dict(best[1])
soft = teacher_probabilities(teacher, np.arange(len(stage)))
teacher_val = evaluate(soft, val_rows); print("teacher validation", teacher_val)

# %% Student: the phone's exact conv-GRU (16 channels, GRU 64, last 20 epochs), loudness-augmented.
C, H, CONTEXT, TEMP = 16, 64, 20, 2.0
GAIN_LOW, GAIN_HIGH = -45.0, 6.0  # dB offsets; the phone recorded about 36 dB quieter than PSG-Audio
frame_mu = frames[train_rows].mean(axis=(0, 1), dtype=np.float64); frame_sd = frames[train_rows].std(axis=(0, 1), dtype=np.float64) + 1e-6

def gained(x, gain):
    x = x.clone(); x[..., 24] = torch.clamp(x[..., 24] + gain.view(-1, *([1] * (x.dim() - 2))), min=-120.0); return x

FRAME_MU, FRAME_SD = (torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for v in (frame_mu, frame_sd))
def normalized(x): return torch.clamp((x - FRAME_MU) / FRAME_SD, -8, 8)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(25, C, 5, padding=2); self.conv2 = torch.nn.Conv1d(C, C, 17, padding=8)
        self.drop = torch.nn.Dropout(0.15); self.stage = torch.nn.Linear(2 * C, 4); self.aux = torch.nn.Linear(2 * C, len(EVENT_NAMES) + len(REGRESS))
    def embedding(self, x):
        x = torch.relu(self.conv1(x.transpose(1, 2))); x = torch.nn.functional.avg_pool1d(x, 2); x = torch.relu(self.conv2(x))
        return torch.cat((x.mean(2), x.amax(2)), 1)
    def forward(self, x):
        e = self.drop(self.embedding(x)); return self.stage(e), self.aux(e)

def distill(logits, y, ok, target):
    loss = torch.nn.functional.cross_entropy(logits[ok], y[ok], weight=class_weight, label_smoothing=0.02) if ok.any() else logits.sum() * 0
    kl = torch.nn.functional.kl_div(torch.log_softmax(logits / TEMP, -1), target, reduction="batchmean") * TEMP * TEMP
    return loss + kl

def soft_target(index): return torch.as_tensor(soft[index] ** (1 / TEMP) / (soft[index] ** (1 / TEMP)).sum(1, keepdims=True), dtype=torch.float32, device=DEVICE)

encoder = Encoder().to(DEVICE); opt = torch.optim.AdamW(encoder.parameters(), 2e-3, weight_decay=1e-4)
student_rows = train_rows[usable[train_rows]]
for epoch in range(8 if not LOCAL_TEST else 1):
    encoder.train(); order = np.random.permutation(student_rows)
    for s in range(0, len(order), 512):
        i = order[s:s + 512]; x = torch.as_tensor(frames[i], dtype=torch.float32, device=DEVICE)
        gain = torch.empty(len(i), device=DEVICE).uniform_(GAIN_LOW, GAIN_HIGH) * (torch.rand(len(i), device=DEVICE) < 0.7)
        logits, aux = encoder(normalized(gained(x, gain)))
        y = torch.as_tensor(stage[i], device=DEVICE); ok = y >= 0
        targets = torch.as_tensor(np.concatenate((events[i], sensors[i]), 1), device=DEVICE)
        aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux[:, :len(EVENT_NAMES)], targets[:, :len(EVENT_NAMES)])
        aux_loss = aux_loss + masked_huber(aux[:, len(EVENT_NAMES):], targets[:, len(EVENT_NAMES):], torch.ones_like(targets[:, len(EVENT_NAMES):], dtype=torch.bool))
        loss = distill(logits, y, ok, soft_target(i)) + 0.3 * aux_loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0); opt.step()
    print(f"student encoder epoch {epoch + 1} done", flush=True)
encoder.eval()

def embed_frames(gain_db=0.0):
    out = np.zeros((len(stage), 2 * C), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(stage), 2048):
            x = torch.as_tensor(frames[s:s + 2048], dtype=torch.float32, device=DEVICE)
            out[s:s + 2048] = encoder.embedding(normalized(gained(x, torch.full((len(x),), gain_db, device=DEVICE)))).cpu().numpy()
    return out

clean = embed_frames(0.0)
emb_mu, emb_sd = clean[student_rows].mean(0), clean[student_rows].std(0) + 1e-6
quiet_gains = [-36.0, -20.0]
variants = {0.0: clean} | {g: embed_frames(g) for g in quiet_gains}
emb_n = {g: np.clip((v - emb_mu) / emb_sd, -8, 8).astype(np.float32) for g, v in variants.items()}

def contexts(index):
    """For each usable target, its trailing usable epochs (<= 20) in the same block; unusable epochs reset context."""
    result = []
    for t in index:
        start = t
        while start > 0 and t - start + 1 < CONTEXT and block_id[start - 1] == block_id[t] and usable[start - 1]: start -= 1
        result.append(np.arange(start, t + 1))
    return result

class Recurrent(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.gru = torch.nn.GRU(2 * C, H, batch_first=True); self.drop = torch.nn.Dropout(0.15); self.output = torch.nn.Linear(H, 4)
    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        return self.output(self.drop(self.gru(packed)[1][-1]))

def padded(ctx, source):
    L = max(len(c) for c in ctx); x = np.zeros((len(ctx), L, 2 * C), dtype=np.float32)
    for k, c in enumerate(ctx): x[k, :len(c)] = source[c]
    return torch.as_tensor(x, device=DEVICE), torch.as_tensor([len(c) for c in ctx])

train_ctx = contexts(student_rows); val_targets = val_rows[usable[val_rows] & (stage[val_rows] >= 0)]; val_ctx = contexts(val_targets)
recurrent = Recurrent().to(DEVICE); opt = torch.optim.AdamW(recurrent.parameters(), 2e-3, weight_decay=1e-4); best = (-1, None)

def student_predict(ctx, source):
    recurrent.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(ctx), 4096):
            x, n = padded(ctx[s:s + 4096], source); out.append(torch.softmax(recurrent(x, n), -1).cpu().numpy())
    return np.concatenate(out)

for epoch in range(10 if not LOCAL_TEST else 1):
    recurrent.train(); order = np.random.permutation(len(student_rows))
    for s in range(0, len(order), 512):
        k = order[s:s + 512]; i = student_rows[k]
        source = emb_n[[0.0, *quiet_gains][np.random.randint(3)] if np.random.rand() < 0.5 else 0.0]
        x, n = padded([train_ctx[j] for j in k], source); logits = recurrent(x, n)
        y = torch.as_tensor(stage[i], device=DEVICE); loss = distill(logits, y, y >= 0, soft_target(i))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(recurrent.parameters(), 1.0); opt.step()
    result = scores(stage[val_targets], student_predict(val_ctx, emb_n[0.0]).argmax(1))
    print(f"student GRU epoch {epoch + 1}: val balanced {result['balancedAccuracy']:.3f} macroF1 {result['macroF1']:.3f} recall {result['recall']}", flush=True)
    if result["macroF1"] > best[0]: best = (result["macroF1"], {k: v.detach().clone() for k, v in recurrent.state_dict().items()})
recurrent.load_state_dict(best[1])
student_val = {f"{g:+.0f} dB": scores(stage[val_targets], student_predict(val_ctx, emb_n[g]).argmax(1)) for g in emb_n}
print("student validation by loudness offset", {k: (round(v["balancedAccuracy"], 3), round(v["macroF1"], 3)) for k, v in student_val.items()})

# %% Export the phone model (schema 4) and a report; the Mac checks it natively before any use.
def R(v): return [R(x) for x in v] if isinstance(v, list) else float(f"{v:.7g}")  # 7 significant digits fit the phone's 512 KB limit
L = lambda t: R(t.detach().double().cpu().tolist())
model = {"schemaVersion": 4, "modelKind": "conv-gru", "modelID": "psg-audio-distilled-v12",
         "dataSource": "PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345", "featureProfile": "mel-sequence-v6",
         "rawFeatureNames": [f"mel_relative_{i:02d}" for i in range(24)] + ["band_power_dbfs"],
         "featureNames": [f"embedding_{kind}_{i}" for kind in ("mean", "max") for i in range(C)],
         "contextEpochs": CONTEXT, "classes": STAGES, "heldOutSubjects": VALIDATION,
         "balancedAccuracy": student_val["+0 dB"]["balancedAccuracy"],
         "encoder": {"channels": C, "means": R(frame_mu.astype(float).tolist()), "scales": R(frame_sd.astype(float).tolist()), "normalizedClip": 8,
                     "weights1": L(encoder.conv1.weight), "biases1": L(encoder.conv1.bias), "weights2": L(encoder.conv2.weight),
                     "biases2": L(encoder.conv2.bias), "embeddingMeans": R(emb_mu.astype(float).tolist()),
                     "embeddingScales": R(emb_sd.astype(float).tolist()), "embeddingClip": 8},
         "recurrent": {"hiddenSize": H, "inputWeights": L(recurrent.gru.weight_ih_l0), "recurrentWeights": L(recurrent.gru.weight_hh_l0),
                       "inputBiases": L(recurrent.gru.bias_ih_l0), "recurrentBiases": L(recurrent.gru.bias_hh_l0),
                       "outputWeights": L(recurrent.output.weight), "outputBiases": L(recurrent.output.bias)}}
(OUT / "student-model.json").write_text(json.dumps(model, separators=(",", ":")))
assert (OUT / "student-model.json").stat().st_size <= 512_000, "Phone model exceeds the 512 KB import limit."

torch.save({"encoder": encoder.state_dict(), "recurrent": recurrent.state_dict()}, OUT / "student.pt")
check = []  # Chronological rows from two validation people, with this notebook's float32 outputs, for the Mac's native check.
for person in VALIDATION[:2]:
    rows = np.flatnonzero(subject == person)[:120]
    rows = rows[block_id[rows] == block_id[rows[0]]]
    probs = student_predict(contexts(rows[usable[rows]]), emb_n[0.0]); k = 0
    for r in rows:
        item = {"subject": person, "startSeconds": float(r - rows[0]) * 30, "features": features[r].astype(float).tolist(),
                "usable": bool(usable[r]), "acousticFrames": frames[r].astype(np.float32).ravel().tolist()}
        if usable[r]: item["torchProbabilities"] = dict(zip(STAGES, probs[k].astype(float).tolist())); k += 1
        check.append(item)
(OUT / "export-check.json").write_text(json.dumps(check))
np.save(OUT / "teacher-probabilities.npy", soft.astype(np.float32))
home_report = []
for b in home:
    hf = torch.as_tensor(b["frames"], dtype=torch.float32, device=DEVICE)
    with torch.no_grad(): he = encoder.embedding(normalized(hf)).cpu().numpy()
    hn = np.clip((he - emb_mu) / emb_sd, -8, 8).astype(np.float32); ctx = []
    for t in range(len(hn)):
        s = t
        while s > 0 and t - s + 1 < CONTEXT and b["usable"][s - 1]: s -= 1
        ctx.append(np.arange(s, t + 1))
    p = student_predict(ctx, hn); home_report.append({"subject": b["subject"], "epochs": len(hn),
        "share": dict(zip(STAGES, np.bincount(p.argmax(1), minlength=4).round(0).tolist()))})
report = {"whisper": WHISPER, "people": len(people), "validationPeople": VALIDATION, "teacherValidation": teacher_val,
          "teacherHistory": history, "studentValidation": student_val, "home": home_report,
          "modelBytes": (OUT / "student-model.json").stat().st_size, "note": "Validation only; not an independent test."}
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: report[k] for k in ("teacherValidation", "studentValidation", "home", "modelBytes")}, indent=1)[:4000])
print("Saved to", OUT)
