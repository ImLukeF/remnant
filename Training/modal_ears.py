"""Run colab_ears.py on Modal: each ear split across L4 GPUs (cheapest per block), then the probe and report.

Data lives on the Modal volume `sleep-ears` under /root (same layout as Drive's SleepWakeTeacher folder).
    HF_TOKEN_FILE=<path> modal run --detach modal_ears.py
The report lands at /root/runs/ears-v1/report.json on the volume.
"""
import os
import threading
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
EARS, SHARDS = ["dasheng", "hear", "opera", "gemma"], 2
volume = modal.Volume.from_name("sleep-ears")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "git", "curl")
         .pip_install("torch", "torchaudio", "transformers>=5.5", "dasheng==0.0.9", "efficientnet_pytorch", "torchlibrosa",
                      "pytorch_lightning", "librosa", "scikit-learn", "huggingface_hub", "numpy", "scipy", "einops")
         .env({"HF_HOME": "/vol/cache/hf", "TORCH_HOME": "/vol/cache/torch", "TEACHER_ROOT": "/vol/root",
               "EARS_WORK": "/tmp/ears", "PYTHONUNBUFFERED": "1"})
         .add_local_file(os.path.join(HERE, "colab_ears.py"), "/app/colab_ears.py"))
token = open(os.environ["HF_TOKEN_FILE"]).read().strip() if modal.is_local() and os.environ.get("HF_TOKEN_FILE") else ""
app = modal.App("sleep-ears", image=image, secrets=[modal.Secret.from_dict({"HF_TOKEN": token})])


def run_script(env):
    """Run the notebook script, committing the volume every 2 minutes so finished blocks survive a stop."""
    import subprocess, sys
    stop = threading.Event()
    def keep_committing():
        while not stop.wait(120): volume.commit()
    threading.Thread(target=keep_committing, daemon=True).start()
    code = subprocess.run([sys.executable, "/app/colab_ears.py"], env=dict(os.environ, **env)).returncode
    stop.set(); volume.commit()
    if code: raise RuntimeError(f"colab_ears.py exited {code} with {env}")


@app.function(volumes={"/vol": volume}, cpu=4, memory=65536, timeout=7200)
def prep():
    """Download every ear's weights into the volume cache once, so the GPU jobs don't each fetch them."""
    import hashlib, subprocess
    # Zenodo drops long downloads and torch.hub keeps the cut-off file, so fetch Dasheng with resume and check it.
    ckpt, size = "/vol/cache/torch/hub/checkpoints/dasheng_12b.pt", 4536407290
    url = "https://zenodo.org/records/11511780/files/dasheng_12b.pt?download=1"
    if os.path.exists(ckpt) and os.path.getsize(ckpt) != size: os.remove(ckpt)
    if not os.path.exists(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True); part = ckpt + ".part"
        for _ in range(30):
            if os.path.exists(part) and os.path.getsize(part) >= size: break
            subprocess.run(["curl", "-L", "-sS", "--retry", "20", "--retry-all-errors", "--retry-delay", "5", "-C", "-", "-o", part, url])
        digest = hashlib.md5()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""): digest.update(chunk)
        assert digest.hexdigest() == "8ee2396e5cb5a273778ee447c9e634c6", "Dasheng checkpoint failed its checksum."
        os.replace(part, ckpt); volume.commit(); print("Dasheng checkpoint verified", flush=True)
    source = open("/app/colab_ears.py").read().split("\n# %%")
    scope = {"__name__": "prep"}
    exec("#" + source[1], scope); exec("#" + source[3], scope)  # setup, then the ears
    for name, make in scope["EARS"].items():
        make(); print("ready", name, flush=True)
    volume.commit()


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=4, memory=32768, timeout=6 * 3600)
def embed(ear: str, shard: int):
    run_script({"EARS_ONLY": ear, "EARS_SHARD": f"{shard}/{SHARDS}"})


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=4, memory=131072, timeout=4 * 3600)
def probe():
    volume.reload()
    run_script({})
    return open("/vol/root/runs/ears-v1/report.json").read()


@app.function(timeout=12 * 3600)
def pipeline():
    prep.remote()
    list(embed.starmap([(ear, shard) for ear in EARS for shard in range(SHARDS)]))
    return probe.remote()


@app.local_entrypoint()
def main():
    print(pipeline.remote())
