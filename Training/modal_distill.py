"""Distil the four big ears into one small one: Dasheng base (86M) learns to reproduce Dasheng 1.2B, HeAR, OPERA and
Gemma 4 audio on the PSG-Audio training people, plus stages and snoring directly. Then the same sleep probe used in the
ears comparison is trained on top of it, scored on the same 40 validation people (also with 8-bit weights), and run on
one home night.

    modal run modal_distill.py --steps 150             # short timing run (about 1.2 s a step on an L4)
    modal run --detach modal_distill.py                # full run on Modal: train, embed, probe
    modal run modal_distill.py --stage probe --night <name> --night-epochs <30 s epochs in the night>

The 40 validation people are never used to train the student. Teacher outputs come from the ears-v1 cache.
Results land in /root/runs/distill-v1/ on the `sleep-ears` volume.
"""
import json
import os
import modal
from modal_ears import image, volume

app = modal.App("sleep-distill", image=image.add_local_python_source("modal_ears"))
TEACHERS = ["dasheng", "hear", "opera", "gemma"]
ROOT, RUN = "/vol/root", "/vol/root/runs/distill-v1"
CKPT, SEGMENTS, EPOCH_SAMPLES = f"{RUN}/student.pt", 10, 480000


def blocks_and_split():
    import hashlib
    from pathlib import Path
    import numpy as np
    blocks = []
    for path in sorted(Path(f"{ROOT}/data/blocks").glob("*.npz")):
        with np.load(path, allow_pickle=False) as b:
            blocks.append({"stage": b["stage"].astype(np.int64), "usable": b["usable"].astype(bool),
                           "events": (b["events"] > 0).astype(np.float32), "eventNames": list(b["eventNames"]),
                           "subject": str(b["subject"]), "stem": Path(str(b["audioFile"])).stem,
                           "audio": f"{ROOT}/data/audio/{b['audioFile']}"})
    assert len(blocks) == 1226, len(blocks)
    people = sorted({b["subject"] for b in blocks})
    validation = set(sorted(people, key=lambda p: hashlib.sha256(f"teacher-v12:{p}".encode()).hexdigest())[:40])
    return blocks, validation


def decode(path, rows):
    import subprocess
    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         check=True, capture_output=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    return np.pad(audio, (0, max(0, rows * EPOCH_SAMPLES - len(audio))))[:rows * EPOCH_SAMPLES].reshape(rows, EPOCH_SAMPLES)


def make_student(n_events, dims):
    import torch
    from dasheng import dasheng_base

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ear = dasheng_base()
            self.copy = torch.nn.ModuleList(torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, d)) for d in dims)
            self.stage, self.events = torch.nn.Linear(768, 4), torch.nn.Linear(768, n_events)

        def frames(self, x):  # (n, 480000) -> (n, 750, 768), one row per 40 ms
            return self.ear(x)

    return Student()


def pooled(h, segments):
    import torch
    return torch.nn.functional.adaptive_avg_pool1d(h.float().transpose(1, 2), segments).transpose(1, 2)


def freeze_norms(model):  # keep Dasheng's input BatchNorm on its pretrained statistics
    import torch
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.eval()


def fake_int8(model):
    """Round every Linear/Conv weight to 8-bit (per output channel), as the phone build would store them."""
    import torch
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                w = m.weight.data; flat = w.reshape(w.shape[0], -1)
                scale = flat.abs().amax(1).clamp(min=1e-8) / 127
                m.weight.data = (torch.round(flat / scale[:, None]).clamp(-127, 127) * scale[:, None]).reshape(w.shape)
    return model


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=12, memory=65536, timeout=8 * 3600)
def train(passes: int = 4, max_steps: int = 0, batch: int = 32):
    import math, random, threading, time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    torch.manual_seed(13); random.seed(13); np.random.seed(13)
    os.makedirs(RUN, exist_ok=True)
    stop = threading.Event()
    threading.Thread(target=lambda: [volume.commit() for _ in iter(lambda: stop.wait(300), True)], daemon=True).start()

    blocks, validation = blocks_and_split()
    train_blocks = [b for b in blocks if b["subject"] not in validation]
    teacher = lambda name, b: np.load(f"{ROOT}/runs/ears-v1/{name}/{b['stem']}.npy")
    sample = [np.concatenate([np.asarray(teacher(t, b), dtype=np.float32) for b in train_blocks[::12]]) for t in TEACHERS]
    shapes = [s.shape[1:] for s in sample]                                       # (segments, dims) per teacher
    mu = [torch.tensor(s.mean((0, 1)), device="cuda") for s in sample]
    sd = [torch.tensor(s.std((0, 1)) + 1e-4, device="cuda") for s in sample]
    del sample
    rows_total = sum(len(b["stage"]) for b in train_blocks)
    steps_per_pass = rows_total // batch
    total = max_steps or passes * steps_per_pass
    print(f"{len(train_blocks)} training blocks, {rows_total} epochs ({rows_total / 120:.0f} h), teachers {dict(zip(TEACHERS, shapes))}, "
          f"{total} steps", flush=True)

    stage_counts = np.bincount(np.concatenate([b["stage"][(b["stage"] >= 0) & b["usable"]] for b in train_blocks]), minlength=4)
    weight = torch.tensor((stage_counts / stage_counts.sum()) ** -0.5, dtype=torch.float32, device="cuda"); weight /= weight.mean()
    model = make_student(len(blocks[0]["eventNames"]), [s[1] for s in shapes]).cuda()
    print(f"student {sum(p.numel() for p in model.ear.parameters()) / 1e6:.1f}M parameters", flush=True)
    heads = [p for n, p in model.named_parameters() if not n.startswith("ear.")]
    opt = torch.optim.AdamW([{"params": model.ear.parameters(), "lr": 5e-5}, {"params": heads, "lr": 1e-3}], weight_decay=0.01)
    warm = min(200, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step = 0
    if os.path.exists(CKPT) and not max_steps:
        state = torch.load(CKPT, map_location="cuda", weights_only=False)
        if state.get("total") == total and state["step"] < total:
            model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"]); step = state["step"]
            print("resumed at step", step, flush=True)

    def load(b):
        return decode(b["audio"], len(b["stage"])), [teacher(t, b) for t in TEACHERS], b

    def batches():
        while True:
            order = train_blocks[:]; random.shuffle(order)
            with ThreadPoolExecutor(10) as pool:
                pending = deque(pool.submit(load, b) for b in order[:12]); nxt = 12
                while pending:
                    group = []
                    for _ in range(6):
                        if not pending: break
                        group.append(pending.popleft().result())
                        if nxt < len(order): pending.append(pool.submit(load, order[nxt])); nxt += 1
                    rows = [(g, i) for g, (a, _, _) in enumerate(group) for i in range(len(a))]; random.shuffle(rows)
                    for s in range(0, len(rows) - batch + 1, batch):
                        pick = rows[s:s + batch]
                        yield (np.stack([group[g][0][i] for g, i in pick]),
                               [np.stack([group[g][1][t][i] for g, i in pick]) for t in range(len(TEACHERS))],
                               np.array([group[g][2]["stage"][i] for g, i in pick]),
                               np.array([group[g][2]["usable"][i] for g, i in pick]),
                               np.stack([group[g][2]["events"][i] for g, i in pick]))

    model.train(); freeze_norms(model)
    started, seen, log = time.time(), 0, {"copy": 0.0, "stage": 0.0, "events": 0.0}
    for audio, targets, y, ok, ev in batches():
        if step >= total: break
        x = torch.from_numpy(audio).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.frames(x)
            copy_loss = 0
            for t, ((segs, _), head) in enumerate(zip(shapes, model.copy)):
                target = ((torch.from_numpy(targets[t]).cuda().float() - mu[t]) / sd[t]).clamp(-8, 8)
                pred = head(pooled(h, segs)).float()
                copy_loss = copy_loss + torch.nn.functional.smooth_l1_loss(pred, target) \
                    + (1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1)).mean()
            copy_loss = copy_loss / len(TEACHERS)
            mean = h.float().mean(1)
            yt, okt = torch.from_numpy(y).cuda(), torch.from_numpy(ok).cuda() & (torch.from_numpy(y).cuda() >= 0)
            stage_loss = torch.nn.functional.cross_entropy(model.stage(mean)[okt].float(), yt[okt], weight=weight) if okt.any() else mean.sum() * 0
            event_loss = torch.nn.functional.binary_cross_entropy_with_logits(model.events(mean).float(), torch.from_numpy(ev).cuda())
            loss = copy_loss + 0.3 * (stage_loss + event_loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1; seen += 1
        for k, v in (("copy", copy_loss), ("stage", stage_loss), ("events", event_loss)): log[k] += float(v)
        if step % 50 == 0 or step == total:
            rate = (time.time() - started) / seen
            print(f"step {step}/{total} copy {log['copy'] / 50:.3f} stage {log['stage'] / 50:.3f} events {log['events'] / 50:.3f} "
                  f"| {rate:.2f} s/step, {(total - step) * rate / 3600:.2f} h left", flush=True)
            log = dict.fromkeys(log, 0.0)
        if step % 1000 == 0 and not max_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "total": total,
                        "shapes": shapes}, CKPT)
    if not max_steps:
        torch.save({"model": model.state_dict(), "step": step, "total": total, "shapes": shapes}, CKPT)
    stop.set(); volume.commit()
    return {"steps": step, "secondsPerStep": round((time.time() - started) / max(seen, 1), 3)}


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=12, memory=65536, timeout=4 * 3600)
def embed():
    """Write the student's output for every block (and the int8 version for the validation people) beside the other ears."""
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    import torch
    volume.reload()
    blocks, validation = blocks_and_split()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    full = make_student(len(blocks[0]["eventNames"]), [s[1] for s in state["shapes"]]); full.load_state_dict(state["model"])
    small = make_student(len(blocks[0]["eventNames"]), [s[1] for s in state["shapes"]]); small.load_state_dict(state["model"])
    variants = [("student", full.cuda().eval(), blocks), ("student-int8", fake_int8(small).cuda().eval(), [b for b in blocks if b["subject"] in validation])]
    for name, model, items in variants:
        folder = f"{ROOT}/runs/ears-v1/{name}"; os.makedirs(folder, exist_ok=True)
        todo = [b for b in items if not os.path.exists(f"{folder}/{b['stem']}.npy")]
        with ThreadPoolExecutor(10) as pool:
            for done, (audio, b) in enumerate(pool.map(lambda b: (decode(b["audio"], len(b["stage"])), b), todo), 1):
                out = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for s in range(0, len(audio), 32):
                        out.append(pooled(model.frames(torch.from_numpy(audio[s:s + 32]).cuda()), SEGMENTS).cpu().numpy().astype(np.float16))
                np.save(f"{folder}/{b['stem']}.npy", np.concatenate(out))
                if done % 100 == 0 or done == len(todo): print(name, done, "/", len(todo), flush=True); volume.commit()
    volume.commit()


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=4, memory=131072, timeout=2 * 3600)
def probe(night: str = "", night_epochs: int = 0):
    """The ears-comparison probe on the student (full and 8-bit), then the home night with the full student."""
    import numpy as np
    import torch
    volume.reload()
    cells = open("/app/colab_ears.py").read().split("\n# %%")
    embed_cell, probe_cell = cells[4], cells[5]
    scope = {"__name__": "distill"}
    for code in (cells[1], cells[2], embed_cell.split("def embed")[0],
                 "\n" + embed_cell[embed_cell.index("def load_ear"):embed_cell.index("available =")],
                 probe_cell.split("results = {}")[0]):
        exec("#" + code, scope)
    blocks, is_val = scope["blocks"], scope["is_val"]
    full = scope["load_ear"]("student", blocks)
    int8 = full.copy(); int8[is_val] = np.concatenate([np.load(f"{ROOT}/runs/ears-v1/student-int8/{Path_stem(b)}.npy")
                                                      for b in blocks if b["subject"] in scope["VALIDATION"]])
    report = {}
    for name, data in (("student", full), ("student-int8", int8)):
        scope["ear_data"] = {name: data}
        result, model, (mu, sd) = scope["train_probe"]([name])
        report[name] = {k: v for k, v in result.items() if k != "history"}
        if name == "student": kept = (model, mu, sd)
    if night:
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        student = make_student(len(blocks[0]["eventNames"]), [s[1] for s in state["shapes"]]); student.load_state_dict(state["model"])
        student = student.cuda().eval(); model, mu, sd = kept; WINDOW = scope["WINDOW"]
        audio = decode(f"{ROOT}/nights/{night}.opus", night_epochs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = np.concatenate([pooled(student.frames(torch.from_numpy(audio[s:s + 32]).cuda()), SEGMENTS).cpu().numpy()
                                for s in range(0, len(audio), 32)])
        x = np.clip((x.astype(np.float32) - mu[0]) / sd[0], -8, 8)
        n = len(x); p, sn, seen = np.zeros((n, 4)), np.zeros(n), np.zeros(n)
        starts = list(range(0, max(1, n - WINDOW + 1), WINDOW // 2)) + ([n - WINDOW] if n > WINDOW else [])
        model.eval()
        with torch.no_grad():
            for s in starts:
                rows = np.arange(s, min(n, s + WINDOW)); mask = torch.ones(1, len(rows), dtype=torch.bool, device="cuda")
                ls, le = model([torch.as_tensor(x[rows][None], device="cuda")], mask)
                p[rows] += torch.softmax(ls.float(), -1)[0].cpu().numpy()
                sn[rows] += torch.sigmoid(le.float()[0, :, scope["SNORE"]]).cpu().numpy(); seen[rows] += 1
        report["night"] = {"name": night, "stages": (p / seen[:, None]).round(4).tolist(), "snore": (sn / seen).round(4).tolist()}
    with open(f"{RUN}/report.json", "w") as f: json.dump(report, f)
    volume.commit()
    return json.dumps(report)


def Path_stem(b):
    from pathlib import Path
    return Path(b["audio"]).stem


@app.function(timeout=14 * 3600)
def pipeline(passes: int, night: str, night_epochs: int):
    print(train.remote(passes=passes))
    embed.remote()
    return probe.remote(night, night_epochs)


@app.local_entrypoint()
def main(stage: str = "all", steps: int = 0, passes: int = 4, night: str = "2026-11-25", night_epochs: int = 728):
    if steps: print(train.remote(max_steps=steps))
    elif stage == "all": print(pipeline.remote(passes, night, night_epochs))
    elif stage == "embed": embed.remote()
    elif stage == "probe": print(probe.remote(night, night_epochs))
