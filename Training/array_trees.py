"""Bounded vectorized tree inference preserving the native Float32 split and margin rules."""


def probabilities(model,features):
    import numpy as np
    inputs=np.asarray(features,dtype=np.float32)
    if inputs.ndim!=2 or not np.isfinite(inputs).all(): raise ValueError("Finite matrix inputs are required.")
    biases=np.asarray(model["biases"],dtype=np.float32)
    if biases.shape!=(4,) or not np.isfinite(biases).all(): raise ValueError("Four finite class margins are required.")
    margins=np.broadcast_to(biases,(len(inputs),4)).copy()
    for tree in model["trees"]:
        left=np.asarray(tree["left"],dtype=np.int32);right=np.asarray(tree["right"],dtype=np.int32)
        indexes=np.asarray(tree["features"],dtype=np.int32);values=np.asarray(tree["values"],dtype=np.float32)
        if not len(left) or any(len(x)!=len(left) for x in (right,indexes,values)) or not np.isfinite(values).all():
            raise ValueError("Invalid numerical tree arrays.")
        positions=np.zeros(len(inputs),dtype=np.int32)
        for _ in range(33):
            active=np.flatnonzero(left[positions]!=-1)
            if not len(active): break
            nodes=positions[active]
            if ((indexes[nodes]<0).any() or (indexes[nodes]>=inputs.shape[1]).any()
                    or (left[nodes]<0).any() or (right[nodes]<0).any()
                    or (left[nodes]>=len(left)).any() or (right[nodes]>=len(left)).any()):
                raise ValueError("Invalid numerical tree edges.")
            positions[active]=np.where(inputs[active,indexes[nodes]]<values[nodes],left[nodes],right[nodes])
        else: raise ValueError("Numerical tree exceeds the native depth bound.")
        k=tree["classIndex"]
        if k not in range(4): raise ValueError("Invalid stage index.")
        # Both operands and each accumulation remain Float32, matching XGBoost/native inference.
        margins[:,k]+=values[positions]
    if not np.isfinite(margins).all(): raise ValueError("Non-finite class margins.")
    logits=margins.astype(np.float64);values=np.exp(logits-logits.max(axis=1,keepdims=True))
    return values/values.sum(axis=1,keepdims=True)
