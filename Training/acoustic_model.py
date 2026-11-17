"""NumPy convolutional encoding plus independent past-only recurrent inference; PyTorch is not required."""
import math

from sequence_model import probabilities


def convolution(values, weights, biases):
    import numpy as np
    weights=np.asarray(weights,dtype=np.float64); kernel=weights.shape[2]
    padded=np.pad(values,((0,0),(kernel//2,kernel//2),(0,0)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,kernel,axis=1)
    result=np.einsum('ntck,ock->nto',windows,weights,optimize=True)+np.asarray(biases)
    if not np.isfinite(result).all(): raise ValueError("Acoustic convolution overflow.")
    return np.maximum(result,0)


def encode(encoder, frames):
    import numpy as np
    values=np.asarray(frames,dtype=np.float64)
    if values.ndim!=3 or values.shape[1:]!=(240,25) or not np.isfinite(values).all():
        raise ValueError("A finite 240x25 acoustic epoch is required.")
    values=(values-np.asarray(encoder["means"]))/np.asarray(encoder["scales"])
    if not np.isfinite(values).all(): raise ValueError("Acoustic normalization overflow.")
    values=np.clip(values,-encoder["normalizedClip"],encoder["normalizedClip"])
    first=convolution(values,encoder["weights1"],encoder["biases1"])
    pooled=(first[:,0::2]+first[:,1::2])/2
    second=convolution(pooled,encoder["weights2"],encoder["biases2"])
    return np.concatenate((second.mean(axis=1),second.max(axis=1)),axis=1)


class EmbeddingPredictor:
    def __init__(self,model):
        self.model=model
        self.frames=[]; self.last=None; self.subject=None

    def reset(self):
        self.frames.clear(); self.last=None; self.subject=None

    def append(self,row,embedding):
        if not row["usable"]:
            self.reset(); return None
        if not math.isfinite(row["startSeconds"]):
            self.reset(); raise ValueError("Invalid acoustic observation time.")
        if self.subject!=row["subject"] or (self.last is not None and abs(row["startSeconds"]-self.last-30)>1): self.reset()
        try:
            encoder=self.model["encoder"]
            if len(embedding)!=len(encoder["embeddingMeans"]): raise ValueError("Invalid acoustic embedding dimensions.")
            normalized=[(x-m)/s for x,m,s in zip(embedding,encoder["embeddingMeans"],encoder["embeddingScales"])]
            if not all(math.isfinite(x) for x in normalized): raise ValueError("Acoustic embedding overflow.")
            limit=encoder["embeddingClip"]
            self.frames.append([max(-limit,min(limit,x)) for x in normalized]); self.frames=self.frames[-self.model["contextEpochs"]:]
            self.last=row["startSeconds"]; self.subject=row["subject"]
            return probabilities(self.model["recurrent"],self.frames)
        except (ValueError,OverflowError):
            self.reset(); raise


def predict_rows(model,rows,frames):
    import numpy as np
    embeddings=[]
    for start in range(0,len(rows),64): embeddings.extend(encode(model["encoder"],frames[start:start+64]).tolist())
    predictor=EmbeddingPredictor(model)
    return [predictor.append(row,embedding) for row,embedding in zip(rows,embeddings)]
