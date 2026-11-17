"""Validate cached acoustic blocks without retaining public raw audio or loading pickle objects."""
import hashlib
import json
from pathlib import Path

from acoustic_frames import FRAME_NAMES
from features import STAGES
from run_acoustic_corpus import fingerprint


def load(root,require_complete=True):
    import numpy as np
    plan_path=root/("cohort-plan.json" if (root/"cohort-plan.json").exists() else "download-plan.json")
    plan=json.loads(plan_path.read_text()); checks=json.loads((root/"feature-checksums.json").read_text())
    if plan["extractorSHA256"]!=fingerprint(): raise ValueError("Acoustic feature code changed after extraction.")
    if require_complete and (not (root/"complete.json").exists() or len(checks)!=len(plan["files"])):
        raise ValueError("Acoustic corpus is incomplete.")
    rows,parts,shas=[],[],{}
    for item in plan["files"]:
        path=root/"features"/(Path(item["path"]).stem+".npz")
        if not path.exists(): raise ValueError("Missing acoustic block.")
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=checks[item["path"]]: raise ValueError("Acoustic observations changed after extraction.")
        shas[item["path"]]=sha
        with np.load(path,allow_pickle=False) as block:
            n=len(block["usable"])
            if (block["frames"].shape!=(n,240,25) or block["features"].shape!=(n,58)
                    or block["startSeconds"].shape!=(n,) or block["stage"].shape!=(n,)
                    or str(block["subject"].item())!=item["subject"]
                    or not np.isfinite(block["frames"]).all() or not np.isfinite(block["features"]).all()
                    or not np.isfinite(block["startSeconds"]).all()
                    or any(int(x) not in range(4) for x in block["stage"])):
                raise ValueError("Invalid acoustic block schema or participant provenance.")
            usable=block["usable"].astype(bool)
            frames=block["frames"][usable]
            for index in np.flatnonzero(usable):
                rows.append({"subject":item["subject"],"stage":STAGES[int(block["stage"][index])],
                             "startSeconds":float(block["startSeconds"][index]),"usable":True,
                             "features":block["features"][index].tolist()})
            parts.append(frames)
    frames=np.concatenate(parts) if parts else np.empty((0,240,25),dtype=np.float32)
    order=sorted(range(len(rows)),key=lambda i:(rows[i]["subject"],rows[i]["startSeconds"]))
    rows=[rows[i] for i in order]; frames=frames[order]
    if len({(r["subject"],r["startSeconds"]) for r in rows})!=len(rows):
        raise ValueError("Overlapping acoustic epochs would duplicate clinical observations.")
    return rows,frames,{"planSHA256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),"extractorSHA256":plan["extractorSHA256"],
                        "fileSHA256":shas,"people":sorted({r["subject"] for r in rows}),"frameNames":FRAME_NAMES}
