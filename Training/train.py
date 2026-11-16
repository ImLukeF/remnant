"""Train and evaluate an experimental causal baseline on subject-separated epochs.

Input JSONL rows: subject, stage, features, usable.
No test subject contributes to normalization, fitting or model selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from features import FEATURE_NAMES, STAGES


def partition(subject):
    bucket = int.from_bytes(hashlib.sha256(subject.encode()).digest()[:4], "big") % 10
    return "test" if bucket < 2 else "train"


def probabilities(features, means, scales, weights, biases):
    normalized = [(x - m) / s for x, m, s in zip(features, means, scales)]
    logits = [b + sum(w * x for w, x in zip(row, normalized)) for row, b in zip(weights, biases)]
    maximum = max(logits)
    values = [math.exp(x - maximum) for x in logits]
    total = sum(values)
    return [value / total for value in values], normalized


def fit(rows, steps=40):
    train = [row for row in rows if partition(row["subject"]) == "train"]
    test = [row for row in rows if partition(row["subject"]) == "test"]
    for name, subset in (("train", train), ("test", test)):
        if {row["stage"] for row in subset} != set(STAGES):
            raise ValueError(f"{name} needs independent subjects with all four sleep stages.")
    count = len(FEATURE_NAMES)
    means = [sum(row["features"][i] for row in train) / len(train) for i in range(count)]
    scales = [max(math.sqrt(sum((row["features"][i] - means[i]) ** 2 for row in train) / len(train)), 1e-6)
              for i in range(count)]
    weights = [[0.0] * count for _ in STAGES]
    biases = [0.0] * len(STAGES)
    counts = {stage: sum(row["stage"] == stage for row in train) for stage in STAGES}
    rng = random.Random(7)
    for _ in range(steps):
        rng.shuffle(train)
        for row in train:
            predicted, features = probabilities(row["features"], means, scales, weights, biases)
            expected = STAGES.index(row["stage"])
            balance = len(train) / (4 * counts[row["stage"]])
            rate = 0.01 * min(balance, 10)
            for k in range(4):
                error = predicted[k] - (k == expected)
                for i in range(count):
                    weights[k][i] -= rate * (error * features[i] + 0.001 * weights[k][i])
                biases[k] -= rate * error
    confusion = [[0] * 4 for _ in STAGES]
    for row in test:
        values, _ = probabilities(row["features"], means, scales, weights, biases)
        confusion[STAGES.index(row["stage"])][max(range(4), key=values.__getitem__)] += 1
    recalls = [confusion[k][k] / sum(confusion[k]) for k in range(4)]
    return {
        "schemaVersion": 1, "featureNames": FEATURE_NAMES, "classes": STAGES,
        "means": means, "scales": scales, "weights": weights, "biases": biases,
        "heldOutSubjects": sorted({row["subject"] for row in test}),
        "balancedAccuracy": sum(recalls) / 4,
    }, {
        "status": "experimental", "confusionMatrix": confusion, "classOrder": STAGES,
        "recallPerClass": recalls, "trainingEpochs": len(train), "testEpochs": len(test),
        "trainingSubjects": len({row["subject"] for row in train}),
        "testSubjects": len({row["subject"] for row in test}),
        "usesFutureAudio": False, "approvedForAlarmTiming": False,
    }


def load_rows(path):
    rows = []
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            features = row["features"]
            if not isinstance(row.get("subject"), str) or not row["subject"]:
                raise ValueError("Every epoch needs a nonempty, stable subject identifier.")
            if row.get("stage") not in STAGES or len(features) != len(FEATURE_NAMES):
                raise ValueError("Wrong sleep label or feature schema.")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in features):
                raise ValueError("Non-finite features.")
            if row.get("usable") is True:
                rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epochs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-source", required=True, help="Dataset name, version and licence reference.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be between 1 and 1000")
    model, report = fit(load_rows(args.epochs), args.steps)
    model.update(modelID=args.model_id, dataSource=args.data_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
