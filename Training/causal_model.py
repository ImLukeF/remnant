"""Dependency-free causal context and inference for portable research models."""
import math
import struct
import json
from pathlib import Path
from features import FEATURE_NAMES, STAGES


def load_observations(path, raw_names=FEATURE_NAMES):
    result = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if (not isinstance(row.get("subject"), str) or not row["subject"] or row.get("stage") not in STAGES
                or len(row["features"]) != len(raw_names) or not math.isfinite(row["startSeconds"])
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in row["features"])):
            raise ValueError("Invalid declared observation schema.")
        if row.get("usable") is True:
            result.append(row)
    return result


def names(history, raw_names=FEATURE_NAMES):
    if history == 1:
        return list(raw_names)
    return list(raw_names) + [prefix + name for prefix in ("past_mean_", "past_std_", "past_delta_") for name in raw_names]


class Context:
    def __init__(self, history, dimensions=7):
        if history not in (1, 10):
            raise ValueError("Unsupported causal history.")
        self.limit = history
        self.dimensions = dimensions
        self.values = []
        self.last_start = None

    def append(self, start, values):
        if len(values) != self.dimensions or not all(math.isfinite(x) for x in [start, *values]):
            raise ValueError("Invalid observation.")
        values = list(values)
        if self.dimensions > 7:
            values[7:] = [round(value * 1e6) / 1e6 for value in values[7:]]
        if self.last_start is not None and abs(start - self.last_start - 30) > 1:
            self.values.clear()
        self.last_start = start
        self.values.append(list(values))
        self.values = self.values[-self.limit:]
        if self.limit == 1:
            return list(values)
        means = [sum(row[k] for row in self.values) / len(self.values) for k in range(self.dimensions)]
        std = [math.sqrt(sum((row[k] - means[k]) ** 2 for row in self.values) / len(self.values)) for k in range(self.dimensions)]
        delta = [values[k] - self.values[0][k] for k in range(self.dimensions)]
        return list(values) + means + std + delta


def transform(rows, history, raw_names=FEATURE_NAMES):
    result, person, context = [], None, None
    for row in sorted(rows, key=lambda row: (row["subject"], row["startSeconds"])):
        if person != row["subject"]:
            person, context = row["subject"], Context(history, len(raw_names))
        result.append({**row, "modelFeatures": context.append(row["startSeconds"], row["features"][:len(raw_names)])})
    return result


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def predict(model, values):
    if model["modelKind"] == "linear":
        normalized = [(x - mean) / scale for x, mean, scale in zip(values, model["means"], model["scales"])]
        logits = [bias + sum(w * x for w, x in zip(weights, normalized)) for weights, bias in zip(model["weights"], model["biases"])]
    else:
        inputs = list(map(f32, values))
        logits = list(map(f32, model["biases"]))
        for tree in model["trees"]:
            index = 0
            while tree["left"][index] != -1:
                index = tree["left"][index] if inputs[tree["features"][index]] < f32(tree["values"][index]) else tree["right"][index]
            k = tree["classIndex"]
            logits[k] = f32(logits[k] + f32(tree["values"][index]))
    maximum = max(logits)
    values = [math.exp(x - maximum) for x in logits]
    return [value / sum(values) for value in values]
