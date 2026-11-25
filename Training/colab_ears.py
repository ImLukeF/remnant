# %% [markdown]
# # Which ear hears sleep best?
# Runtime → Change runtime type → GPU (G4, H100 or A100), then Runtime → Run all.
# It reads `My Drive/SleepWakeTeacher/data`, saves each ear's output there as it goes, and resumes if the session stops.
#
# Four frozen pretrained audio models ("ears") each turn every 30 s of bedside audio into numbers:
# Dasheng 1.2B (general sound), HeAR (Google, health sounds), OPERA-CT (breathing sounds) and Gemma 4's audio encoder.
# The same small sleep model is then trained on top of each ear, plus one on all four together, and scored on the
# same 40 validation people as the Whisper-small run (v12). Scores: sleep stages and snoring. Validation only.

# %%
import hashlib, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOCAL_TEST = os.environ.get("TEACHER_LOCAL_TEST") == "1"
if "TEACHER_ROOT" in os.environ:  # Modal or a local test, not Colab
    ROOT = Path(os.environ["TEACHER_ROOT"]); WORK = Path(os.environ.get("EARS_WORK", "/tmp/ears"))
else:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/SleepWakeTeacher"); WORK = Path("/content/ears")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5.5", "dasheng==0.0.9",
                    "efficientnet_pytorch", "torchlibrosa", "pytorch_lightning"], check=True)
token_file = ROOT / "hf_token.txt"
if token_file.exists(): os.environ["HF_TOKEN"] = token_file.read_text().strip()
WORK.mkdir(parents=True, exist_ok=True)
for repo, commit in (("Google-Health/hear", "36bbb90879545a2f1106668048b1a12a61ff2a0d"),
                     ("evelyn0414/OPERA", "3622310e667afb8aa40169050b4dd45de75946a2")):
    target = WORK / repo.split("/")[1]
    if not target.exists():
        subprocess.run(["git", "clone", "-q", f"https://github.com/{repo}", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", commit], check=True)

import numpy as np
import torch
DATA, HOME, OUT = ROOT / "data", ROOT / "home", ROOT / "runs" / "ears-v1"
WHISPER_DIR = ROOT / "runs" / "v12" / "whisper"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGES = ["awake", "light", "deep", "rem"]
torch.manual_seed(12); np.random.seed(12)
print("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "| data", DATA)
complete = json.loads((DATA / "complete.json").read_text()) if (DATA / "complete.json").exists() else None
assert LOCAL_TEST or (complete and complete["blocks"] == 1226), "The 1,226-block corpus is not complete on Drive."

# %% Labels only (stages, snore and other events). No phone features: this compares ears alone.
def load_block(path):
    with np.load(path, allow_pickle=False) as b:
        return {"stage": b["stage"].astype(np.int64), "usable": b["usable"].astype(bool), "events": b["events"],
                "eventNames": list(b["eventNames"]), "subject": str(b["subject"]), "audio": str(b["audioFile"]),
                "home": "home" in b.files}

block_paths = sorted((DATA / "blocks").glob("*.npz"))
assert LOCAL_TEST or len(block_paths) == 1226
blocks = [load_block(p) for p in block_paths]
home = [load_block(p) for p in sorted(HOME.glob("*.npz"))] if HOME.exists() else []
EVENT_NAMES = blocks[0]["eventNames"]; SNORE = EVENT_NAMES.index("snore")
subject = np.concatenate([np.full(len(b["stage"]), b["subject"]) for b in blocks])
block_id = np.concatenate([np.full(len(b["stage"]), i) for i, b in enumerate(blocks)])
stage = np.concatenate([b["stage"] for b in blocks]); usable = np.concatenate([b["usable"] for b in blocks])
events = (np.concatenate([b["events"] for b in blocks]) > 0).astype(np.float32)
people = sorted(set(subject))
VALIDATION = sorted(people, key=lambda p: hashlib.sha256(f"teacher-v12:{p}".encode()).hexdigest())[:40 if not LOCAL_TEST else 1]
is_val = np.isin(subject, VALIDATION)
print(f"{len(people)} people, {len(stage)} epochs, {usable.sum()} usable, validation people {len(VALIDATION)}, home nights {len(home)}")

# %% The ears. Each maps a batch of 30 s epochs (n, 480000) at 16 kHz to (n, segments, dims).
def pooled(hidden, segments):  # (n, frames, d) -> (n, segments, d), mean over equal time slices
    return torch.nn.functional.adaptive_avg_pool1d(hidden.float().transpose(1, 2), segments).transpose(1, 2)

def dasheng_ear():
    from dasheng import dasheng_12B
    model = dasheng_12B().to(DEVICE).eval()
    def run(x):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            return pooled(model(x), 10)
    return run, 16

def hear_ear():
    import importlib
    sys.path.insert(0, str(WORK))
    preprocess = importlib.import_module("hear.python.data_processing.audio_utils").preprocess_audio
    from transformers import AutoModel
    model = AutoModel.from_pretrained("google/hear-pytorch", revision="f791cd42437c3e268c8ac84707e3508900f65f1a").to(DEVICE).eval()
    def run(x):
        n = len(x); clips = x.reshape(n * 15, 32000)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(preprocess(clips), return_dict=True).pooler_output
        return out.float().reshape(n, 15, -1)
    return run, 16

def opera_ear():
    import librosa
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(WORK / "OPERA"))
    from src.model.models_cola import Cola
    model = Cola(encoder="htsat")
    ckpt = torch.load(hf_hub_download("evelyn0414/OPERA", "encoder-operaCT.ckpt", revision="d8de4322870b596f0a6ff6ea907b9a6996cd243a"),
                      map_location="cpu", weights_only=False)
    missing, _ = model.load_state_dict(ckpt["state_dict"], strict=False)
    assert not [k for k in missing if k.startswith("encoder.")], "OPERA encoder weights did not load."
    model = model.to(DEVICE).eval()
    mel = torch.tensor(librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64, fmin=50, fmax=8000), device=DEVICE)
    window = torch.hann_window(1024, device=DEVICE)
    def run(x):  # OPERA's own preprocessing (librosa mel, dB re max, top 80 dB, min-max), on 7.5 s pieces
        n = len(x); clips = x.reshape(n * 4, 120000)
        spec = torch.stft(clips, 1024, 512, window=window, center=True, pad_mode="constant", return_complex=True).abs() ** 2
        db = 10 * torch.log10(torch.clamp(mel @ spec, min=1e-10))
        db = db - db.amax(dim=(1, 2), keepdim=True); db = torch.maximum(db, torch.full_like(db, -80.0))
        lo, hi = db.amin(dim=(1, 2), keepdim=True), db.amax(dim=(1, 2), keepdim=True)
        x = torch.where(hi > lo, (db - lo) / (hi - lo), db).transpose(1, 2)  # (clips, frames, 64)
        return model.extract_feature(x, 768).float().reshape(n, 4, -1)
    return run, 32

def gemma_ear():
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, Gemma4AudioFeatureExtractor, Gemma4AudioModel
    from transformers.utils import logging as hf_logging
    repo, rev = "google/gemma-4-E2B-it", "4adf15748de9e1a0097b6274cee9c15836c41c37"
    config = AutoConfig.from_pretrained(repo, revision=rev)
    hf_logging.set_verbosity_error()  # the text and vision weights are skipped on purpose; only the audio encoder loads
    model, info = Gemma4AudioModel.from_pretrained(repo, revision=rev, config=config.audio_config, dtype=torch.bfloat16,
                                                   output_loading_info=True)
    assert not info["missing_keys"], f"Gemma audio weights missing: {sorted(info['missing_keys'])[:5]}"
    model = model.to(DEVICE).eval()
    settings = json.loads(Path(hf_hub_download(repo, "processor_config.json", revision=rev)).read_text())["feature_extractor"]
    extractor = Gemma4AudioFeatureExtractor(**{k: v for k, v in settings.items() if k != "feature_extractor_type"})
    def run(x):
        f = extractor(list(x.cpu().numpy()), sampling_rate=16000, return_tensors="pt", padding="longest", max_length=480000)
        out = model(f["input_features"].to(DEVICE, torch.bfloat16), f["input_features_mask"].to(DEVICE), return_dict=True)
        return pooled(out.last_hidden_state, 10)
    return run, 16

EARS = {"dasheng": dasheng_ear, "hear": hear_ear, "opera": opera_ear, "gemma": gemma_ear}

# %% Run each ear over every block (and the home night), one file per block so a stopped session resumes.
def decode(path, rows):
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    return np.pad(audio, (0, max(0, rows * 480000 - len(audio))))[:rows * 480000].reshape(rows, 480000)

def source(b): return (HOME if b["home"] else DATA / "audio") / b["audio"]

def embed(name, items):
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    pending = [b for b in items if not (folder / (Path(b["audio"]).stem + ".npy")).exists()]
    if not pending: print(name, "already done"); return
    run, size = EARS[name](); started = time.time()
    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(decode, source(b), len(b["stage"])) for b in pending[:6]]
        for done, b in enumerate(pending, 1):
            audio = futures.pop(0).result()
            if done + 5 < len(pending): futures.append(pool.submit(decode, source(pending[done + 5]), len(pending[done + 5]["stage"])))
            out = []
            with torch.no_grad():
                for s in range(0, len(audio), size):
                    out.append(run(torch.from_numpy(audio[s:s + size]).to(DEVICE)).cpu().numpy().astype(np.float16))
            result = np.concatenate(out); assert np.isfinite(result).all(), f"{name} gave non-finite output"
            partial = folder / (Path(b["audio"]).stem + ".partial.npy"); np.save(partial, result)
            partial.replace(folder / (Path(b["audio"]).stem + ".npy"))
            if done % 50 == 0 or done == len(pending):
                print(f"{name} {done}/{len(pending)} blocks, {(time.time() - started) / 60:.1f} min", flush=True)
    del run; torch.cuda.empty_cache() if DEVICE == "cuda" else None

ONLY, SHARD = os.environ.get("EARS_ONLY"), os.environ.get("EARS_SHARD")  # Modal splits the ears across GPUs
todo = blocks + home
if SHARD: index, count = map(int, SHARD.split("/")); todo = todo[index::count]
for name in EARS:
    if not ONLY or name in ONLY.split(","): embed(name, todo)
if ONLY: sys.exit(0)

def load_ear(name, items):
    folder = WHISPER_DIR if name == "whisper" else OUT / name
    return np.concatenate([np.load(folder / (Path(b["audio"]).stem + ".npy")) for b in items])

available = ["whisper"] if all((WHISPER_DIR / (Path(b["audio"]).stem + ".npy")).exists() for b in blocks) else []
ear_data = {name: load_ear(name, blocks) for name in available + list(EARS)}
for name, value in ear_data.items(): assert value.shape[0] == len(stage); print(name, value.shape)

# %% The same small sleep model on top of each ear: attention over segments, a transformer over 20 minutes.
WINDOW = 40

class Probe(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.audio = torch.nn.ModuleList(torch.nn.Linear(d, 256) for d in dims)
        self.attend = torch.nn.ModuleList(torch.nn.Linear(256, 1) for _ in dims)
        self.mix = torch.nn.Linear(256 * len(dims), 256); self.position = torch.nn.Parameter(torch.zeros(WINDOW, 256))
        layer = torch.nn.TransformerEncoderLayer(256, 4, 768, 0.1, batch_first=True, norm_first=True)
        self.body = torch.nn.TransformerEncoder(layer, 4)
        self.stage = torch.nn.Linear(256, 4); self.events = torch.nn.Linear(256, len(EVENT_NAMES))

    def forward(self, inputs, mask):
        parts = []
        for x, audio, attend in zip(inputs, self.audio, self.attend):
            a = torch.nn.functional.gelu(audio(x)); parts.append((torch.softmax(attend(a), dim=2) * a).sum(2))
        h = self.body(self.mix(torch.cat(parts, -1)) + self.position[: mask.shape[1]], src_key_padding_mask=~mask)
        return self.stage(h), self.events(h)

def windows(index, length, stride):
    out = []
    for b in np.unique(block_id[index]):
        rows = index[block_id[index] == b]
        for s in range(0, max(1, len(rows) - length + 1), stride): out.append(rows[s:s + length])
        if len(rows) > length and (len(rows) - length) % stride: out.append(rows[-length:])
    return out

def scores(y, p):
    m = np.zeros((4, 4), dtype=int); np.add.at(m, (y, p), 1)
    recall = np.diag(m) / np.maximum(m.sum(1), 1); f1 = 2 * np.diag(m) / np.maximum(m.sum(1) + m.sum(0), 1)
    return {"balancedAccuracy": round(float(recall.mean()), 4), "macroF1": round(float(f1.mean()), 4),
            "recall": dict(zip(STAGES, recall.round(3).tolist())), "confusion": m.tolist()}

def snore_scores(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y.min() == y.max(): return {"auc": None, "averagePrecision": None, "share": float(y.mean())}
    return {"auc": round(float(roc_auc_score(y, p)), 4), "averagePrecision": round(float(average_precision_score(y, p)), 4),
            "share": round(float(y.mean()), 4)}

def train_probe(names, epochs=10 if not LOCAL_TEST else 1):
    arrays = [ear_data[n] for n in names]
    mu = [a[~is_val].mean((0, 1), dtype=np.float64).astype(np.float32) for a in arrays]
    sd = [a[~is_val].std((0, 1), dtype=np.float64).astype(np.float32) + 1e-4 for a in arrays]
    def batch(rows_list):
        T = max(len(r) for r in rows_list); idx = np.zeros((len(rows_list), T), dtype=np.int64); mask = np.zeros_like(idx, dtype=bool)
        for i, r in enumerate(rows_list): idx[i, :len(r)] = r; mask[i, :len(r)] = True
        xs = [torch.as_tensor(np.clip((a[idx].astype(np.float32) - m) / s, -8, 8), device=DEVICE) for a, m, s in zip(arrays, mu, sd)]
        return idx, torch.as_tensor(mask, device=DEVICE), xs
    train_rows, val_rows = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    counts = np.bincount(stage[train_rows][(stage[train_rows] >= 0) & usable[train_rows]], minlength=4)
    weight = torch.tensor((np.maximum(counts, 1) / counts.sum()) ** -0.5, dtype=torch.float32, device=DEVICE); weight /= weight.mean()
    model = Probe([a.shape[-1] for a in arrays]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    train_windows = windows(train_rows, WINDOW, WINDOW // 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=epochs * math.ceil(len(train_windows) / 64))
    y_all, e_all = torch.as_tensor(stage, device=DEVICE), torch.as_tensor(events, device=DEVICE)
    ok_all = torch.as_tensor(usable, device=DEVICE)

    def predict(index):
        model.eval(); p, sn, seen = np.zeros((len(stage), 4)), np.zeros(len(stage)), np.zeros(len(stage))
        with torch.no_grad():
            ws = windows(index, WINDOW, WINDOW // 2)
            for s in range(0, len(ws), 64):
                idx, mask, xs = batch(ws[s:s + 64])
                with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    ls, le = model(xs, mask)
                m = mask.cpu().numpy()
                np.add.at(p, idx[m], torch.softmax(ls.float(), -1).cpu().numpy()[m])
                np.add.at(sn, idx[m], torch.sigmoid(le.float()[..., SNORE]).cpu().numpy()[m]); np.add.at(seen, idx[m], 1)
        return p / np.maximum(seen, 1)[:, None], sn / np.maximum(seen, 1)

    best, history = None, []
    for epoch in range(epochs):
        model.train(); np.random.shuffle(train_windows); started = time.time()
        for s in range(0, len(train_windows), 64):
            idx, mask, xs = batch(train_windows[s:s + 64]); t = torch.as_tensor(idx, device=DEVICE)
            y, ok, ev = y_all[t], ok_all[t], e_all[t]
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                ls, le = model(xs, mask)
            valid = mask & ok & (y >= 0)
            loss = torch.nn.functional.cross_entropy(ls.float()[valid], y[valid], weight=weight, label_smoothing=0.02) if valid.any() else ls.sum() * 0
            loss = loss + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(le.float()[mask], ev[mask])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        p, sn = predict(val_rows)
        keep = val_rows[(stage[val_rows] >= 0) & usable[val_rows]]
        result = scores(stage[keep], p[keep].argmax(1)) | {"snore": snore_scores(events[val_rows, SNORE], sn[val_rows])}
        history.append(result)
        print(f"  {'+'.join(names)} epoch {epoch + 1}: balanced {result['balancedAccuracy']:.3f} macroF1 {result['macroF1']:.3f} "
              f"recall {list(result['recall'].values())} snore AUC {result['snore']['auc']} ({(time.time() - started) / 60:.1f} min)", flush=True)
        if best is None or result["balancedAccuracy"] > best[0]["balancedAccuracy"]:
            best = (result, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch + 1)
    model.load_state_dict(best[1])
    return best[0] | {"bestEpoch": best[2], "history": history}, model, (mu, sd)

results = {}
for names in [[n] for n in ear_data] + [list(EARS)]:
    key = "+".join(names) if len(names) > 1 else names[0]
    print("probe", key, flush=True); results[key] = train_probe(names)[0]

# %% Summary, saved to Drive.
print(f"\n{'ear':<28}{'balanced':>9}{'macroF1':>9}{'awake':>7}{'light':>7}{'deep':>7}{'rem':>7}{'snoreAUC':>10}")
for key, r in sorted(results.items(), key=lambda kv: -kv[1]["balancedAccuracy"]):
    rec = r["recall"]
    print(f"{key:<28}{r['balancedAccuracy']:>9.3f}{r['macroF1']:>9.3f}{rec['awake']:>7.2f}{rec['light']:>7.2f}{rec['deep']:>7.2f}{rec['rem']:>7.2f}{str(r['snore']['auc']):>10}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.json").write_text(json.dumps({"validationPeople": VALIDATION, "results": results,
    "note": "Validation only; same 40 people as v12. Not an independent test."}, indent=1))
print("saved", OUT / "report.json")
