"""The weekly home run on the VPS: bring down Luke's nights, fine-tune, report, and hand back a model that passed.

Reads REMY_URL and VPS_TOKEN from the environment (the VPS keeps them in /etc/remy-runner.env). Keeps everything under
--root: nights/ (one JSON per night, as the phone sent it), runs/<time>/ (each run's report and candidate model).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLED = HERE.parent / "Research" / "relative-v11" / "model.json"  # the model the Parley app ships (sleep-alpha-model.json)


def call(path, method="GET", body=None, content_type="application/json"):
    request = urllib.request.Request(os.environ["REMY_URL"].rstrip("/") + path, data=body, method=method,
        headers={"Authorization": "Bearer " + os.environ["VPS_TOKEN"], "Content-Type": content_type, "User-Agent": "sleep-home-week"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response: return response.status, response.read()
    except urllib.error.HTTPError as error: return error.code, error.read()


def bring_down(nights):
    """Every night the server holds, fetching only the new or changed ones; nights deleted on the phone go here too."""
    status, body = call("/vps/sleep/nights")
    if status != 200: raise RuntimeError(f"Couldn't list nights: {status} {body[:200]!r}")
    held = json.loads(body)["nights"]; index_path = nights.parent / "nights-index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    for night in held:
        stamp = f"{night['size']} {night['uploaded']}"
        if index.get(night["id"]) == stamp and (nights / f"{night['id']}.json").exists(): continue
        status, data = call("/vps/file?key=" + urllib.parse.quote(night["key"]))
        if status != 200: raise RuntimeError(f"Couldn't fetch night {night['id']}: {status}")
        (nights / f"{night['id']}.json").write_bytes(data); index[night["id"]] = stamp
    for gone in set(index) - {n["id"] for n in held}:
        (nights / f"{gone}.json").unlink(missing_ok=True); index.pop(gone)
    index_path.write_text(json.dumps(index, indent=2))
    return len(held)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/srv/ai-team/artifacts/sleep-home"))
    p.add_argument("--development", type=Path, default=Path("/srv/ai-team/artifacts/sleep-wake-expanded-v8"))
    p.add_argument("--previous", type=Path, default=Path("/srv/ai-team/artifacts/sleep-wake-relative-v11"))
    args = p.parse_args()
    nights = args.root / "nights"; nights.mkdir(parents=True, exist_ok=True)
    bring_down(nights)
    run = args.root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    status, current = call("/vps/sleep/model")
    baseline = args.root / "current-model.json"
    baseline.write_bytes(current if status == 200 else BUNDLED.read_bytes())
    trained = subprocess.run([sys.executable, str(HERE / "train_home.py"), "--development", str(args.development),
        "--previous", str(args.previous), "--baseline", str(baseline), "--nights", str(nights), "--output", str(run)], cwd=HERE)
    report_path = run / "report.json"
    if trained.returncode or not report_path.exists():
        report = {"status": "failed", "summary": "This week's training didn't finish; it'll try again next week.", "kept": False}
    else:
        report = json.loads(report_path.read_text())
    if report.get("kept"):
        status, body = call("/vps/sleep/model", "PUT", (run / "candidate-model.json").read_bytes())
        if status != 200:
            report.update(kept=False, status="failed", summary=f"Kept, but couldn't hand the model back ({status}).")
    brief = {key: report.get(key) for key in ("status", "summary", "kept", "nights", "trainingNights", "heldBackNights",
                                                 "modelID", "baselineModelID", "home", "wakeRatings", "seconds")}
    if "lab" in report:
        brief["lab"] = {when: {view: {m: report["lab"][when][view][m] for m in ("macroF1", "balancedAccuracy")}
                               for view in ("clean", "noisy")} for when in ("before", "after")}
    status, body = call("/vps/sleep/runs", "POST", json.dumps(brief).encode())
    print(json.dumps({"run": str(run), "status": report.get("status"), "summary": report.get("summary"), "posted": status}), flush=True)
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
