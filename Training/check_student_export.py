"""Check a Colab student export: independent NumPy reference vs the notebook's PyTorch outputs, then write
verify_native.swift's input (expected probabilities from the reference, float64).

    python3 check_student_export.py student-model.json export-check.json native-input.json
"""
import json
import sys

from acoustic_model import predict_rows
from features import STAGES


def main():
    import numpy as np
    model = json.loads(open(sys.argv[1]).read()); rows = json.loads(open(sys.argv[2]).read())
    if model.get("schemaVersion") != 4 or len(open(sys.argv[1], "rb").read()) > 512_000: raise ValueError("Not an importable schema 4 model.")
    output, torch_gap, compared = [], 0.0, 0
    for person in dict.fromkeys(r["subject"] for r in rows):
        mine = [r for r in rows if r["subject"] == person]
        frames = np.asarray([r["acousticFrames"] for r in mine], dtype=np.float64).reshape(-1, 240, 25)
        reference = predict_rows(model, mine, frames)
        for row, probabilities in zip(mine, reference):
            item = {k: row[k] for k in ("subject", "startSeconds", "features", "usable", "acousticFrames")}
            if row["usable"]:
                expected = dict(zip(STAGES, probabilities)); item["expectedProbabilities"] = expected
                torch_gap = max(torch_gap, max(abs(expected[s] - row["torchProbabilities"][s]) for s in STAGES)); compared += 1
            output.append(item)
    open(sys.argv[3], "w").write(json.dumps(output))
    print(json.dumps({"rows": len(output), "usableCompared": compared, "maxTorchVsReference": torch_gap}))
    if torch_gap > 1e-3: raise SystemExit("PyTorch training outputs and the exported model disagree.")


if __name__ == "__main__":
    main()
