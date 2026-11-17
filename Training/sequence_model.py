"""Dependency-free, bounded causal GRU inference. No future observations or stage rules."""
import math
from spectral import SPECTRAL_NAMES


def encoded(values):
    if len(values) != len(SPECTRAL_NAMES) or not all(math.isfinite(x) for x in values):
        raise ValueError("A complete finite spectral observation is required.")
    if not all(math.isfinite(value*1e6) for value in values[7:]):
        raise ValueError("Spectral input encoding overflow.")
    return list(values[:7]) + [round(value * 1e6) / 1e6 for value in values[7:]]


def normalize(model, values):
    raw = encoded(values)
    result = [(x - m) / s for x, m, s in zip(raw, model["means"], model["scales"])]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Normalization overflow.")
    limit = model["normalizedClip"]
    return [max(-limit, min(limit, x)) for x in result]


def sequence_indexes(rows, history):
    """Rows must be chronological per person; gaps break windows rather than being filled."""
    if history not in (1, 10, 20):
        raise ValueError("Unsupported bounded sequence length.")
    windows, recent, person, last = [], [], None, None
    for index, row in enumerate(rows):
        if person != row["subject"] or last is None or abs(row["startSeconds"] - last - 30) > 1:
            recent = []
        person, last = row["subject"], row["startSeconds"]
        recent.append(index)
        recent = recent[-history:]
        windows.append(list(recent))
    return windows


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def probabilities(model, frames):
    size = model["hiddenSize"]
    hidden = [0.0] * size
    for frame in frames:
        incoming = [bias + sum(w * x for w, x in zip(row, frame))
                    for row, bias in zip(model["inputWeights"], model["inputBiases"])]
        recurrent = [bias + sum(w * x for w, x in zip(row, hidden))
                     for row, bias in zip(model["recurrentWeights"], model["recurrentBiases"])]
        if not all(math.isfinite(x) for x in incoming + recurrent):
            raise ValueError("Recurrent affine overflow.")
        reset = [sigmoid(incoming[k] + recurrent[k]) for k in range(size)]
        update = [sigmoid(incoming[size+k] + recurrent[size+k]) for k in range(size)]
        new = [math.tanh(incoming[2*size+k] + reset[k] * recurrent[2*size+k]) for k in range(size)]
        hidden = [(1-update[k]) * new[k] + update[k] * hidden[k] for k in range(size)]
    logits = [bias + sum(w * x for w, x in zip(row, hidden))
              for row, bias in zip(model["outputWeights"], model["outputBiases"])]
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("Recurrent output overflow.")
    values = [math.exp(x-max(logits)) for x in logits]
    return [x/sum(values) for x in values]


class SequencePredictor:
    def __init__(self, model):
        self.model = model
        self.frames = []
        self.last = None

    def reset(self):
        self.frames.clear()
        self.last = None

    def append(self, row):
        if not row["usable"]:
            self.reset()
            return None
        start = row["startSeconds"]
        if not math.isfinite(start):
            self.reset()
            raise ValueError("Invalid observation time.")
        if self.last is not None and abs(start - self.last - 30) > 1:
            self.reset()
        self.last = start
        self.frames.append(normalize(self.model, row["features"]))
        self.frames = self.frames[-self.model["contextEpochs"]:]
        return probabilities(self.model, self.frames)
