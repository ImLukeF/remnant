"""Train the Dasheng sleep probe on the cached PSG-Audio embeddings, then score one home night with it.

    modal volume put sleep-ears night.opus /root/nights/<name>.opus
    modal run modal_night.py --night <name> --epochs <30 s epochs in the night>
Prints JSON: validation scores plus per-30 s stage probabilities and snore probability for the night.
"""
import json
import modal
from modal_ears import image, volume

app = modal.App("sleep-night", image=image.add_local_python_source("modal_ears"))


@app.function(volumes={"/vol": volume}, gpu="L4", cpu=4, memory=65536, timeout=2 * 3600)
def score(night: str, epochs_in_night: int):
    import os
    cells = open("/app/colab_ears.py").read().split("\n# %%")
    embed_cell, probe_cell = cells[4], cells[5]
    scope = {"__name__": "night"}
    for code in (cells[1], cells[2], cells[3],
                 embed_cell.split("ONLY, SHARD")[0],                                  # decode, source, embed
                 "\n" + embed_cell[embed_cell.index("def load_ear"):embed_cell.index("available =")],
                 probe_cell.split("results = {}")[0]):                                # Probe, windows, train_probe
        exec("#" + code, scope)
    np, torch, DEVICE, WINDOW = scope["np"], scope["torch"], scope["DEVICE"], scope["WINDOW"]
    scope["ear_data"] = {"dasheng": scope["load_ear"]("dasheng", scope["blocks"])}
    result, model, (mu, sd) = scope["train_probe"](["dasheng"])
    os.makedirs("/vol/root/runs/ears-v1/models", exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, "/vol/root/runs/ears-v1/models/dasheng-probe.pt")

    run, size = scope["EARS"]["dasheng"]()
    audio = scope["decode"](f"/vol/root/nights/{night}.opus", epochs_in_night)
    with torch.no_grad():
        x = np.concatenate([run(torch.from_numpy(audio[s:s + size]).to(DEVICE)).cpu().numpy() for s in range(0, len(audio), size)])
    x = np.clip((x.astype(np.float32) - mu[0]) / sd[0], -8, 8)
    n = len(x); p, sn, seen = np.zeros((n, 4)), np.zeros(n), np.zeros(n)
    starts = list(range(0, max(1, n - WINDOW + 1), WINDOW // 2)) + ([n - WINDOW] if n > WINDOW else [])
    model.eval()
    with torch.no_grad():
        for s in starts:
            rows = np.arange(s, min(n, s + WINDOW))
            xs = [torch.as_tensor(x[rows][None], device=DEVICE)]; mask = torch.ones(1, len(rows), dtype=torch.bool, device=DEVICE)
            ls, le = model(xs, mask)
            p[rows] += torch.softmax(ls.float(), -1)[0].cpu().numpy()
            sn[rows] += torch.sigmoid(le.float()[0, :, scope["SNORE"]]).cpu().numpy(); seen[rows] += 1
    volume.commit()
    return json.dumps({"validation": {k: v for k, v in result.items() if k != "history"},
                       "stages": (p / seen[:, None]).round(4).tolist(), "snore": (sn / seen).round(4).tolist()})


@app.local_entrypoint()
def main(night: str, epochs: int):
    print(score.remote(night, epochs))
