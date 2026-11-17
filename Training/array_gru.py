"""Batched causal GRU equations, independent of PyTorch and matching native spectral encoding."""
from sequence_model import encoded


def layout(arrays,history):
    import numpy as np
    if history not in (1,10,20): raise ValueError("Unsupported causal history.")
    indexes=np.flatnonzero(arrays["usable"]&arrays["augmentedUsable"])
    windows=np.zeros((len(indexes),history),dtype=np.int64);lengths=np.empty(len(indexes),dtype=np.int64)
    person=None;last=None;recent=[]
    for row,index in enumerate(indexes):
        current=str(arrays["subject"][index]);start=float(arrays["startSeconds"][index])
        if current!=person or last is None or abs(start-last-30)>1: recent=[]
        recent=(recent+[row])[-history:];windows[row,:len(recent)]=recent;lengths[row]=len(recent)
        person,last=current,start
    return {"indexes":indexes,"windows":windows,"lengths":lengths,"people":arrays["subject"][indexes],
        "labels":arrays["stage"][indexes],"starts":arrays["startSeconds"][indexes]}


def encode_matrix(values):
    import numpy as np
    return np.asarray([encoded(row.tolist()) for row in values],dtype=np.float64)


def normalized(values,means,scales):
    import numpy as np
    values=(values-np.asarray(means))/np.asarray(scales)
    if not np.isfinite(values).all(): raise ValueError("Invalid normalized recurrent observations.")
    return np.clip(values,-8,8)


def batch(values,windows,lengths,indexes):
    import numpy as np
    result=values[windows[indexes]].copy()
    result[np.arange(result.shape[1])[None,:]>=lengths[indexes,None]]=0
    return result


def forward(model,values,lengths):
    import numpy as np
    from scipy.special import expit
    values=np.asarray(values,dtype=np.float64);lengths=np.asarray(lengths)
    size=model["hiddenSize"];hidden=np.zeros((len(values),size),dtype=np.float64)
    wi=np.asarray(model["inputWeights"]);wh=np.asarray(model["recurrentWeights"])
    bi=np.asarray(model["inputBiases"]);bh=np.asarray(model["recurrentBiases"])
    if (values.ndim!=3 or not np.isfinite(values).all() or lengths.shape!=(len(values),)
            or (lengths<1).any() or (lengths>values.shape[1]).any()): raise ValueError("Invalid bounded recurrent batch.")
    for step in range(values.shape[1]):
        incoming=values[:,step]@wi.T+bi;recurrent=hidden@wh.T+bh
        reset=expit(incoming[:,:size]+recurrent[:,:size]);update=expit(incoming[:,size:2*size]+recurrent[:,size:2*size])
        new=np.tanh(incoming[:,2*size:]+reset*recurrent[:,2*size:])
        proposed=(1-update)*new+update*hidden
        hidden=np.where((step<lengths)[:,None],proposed,hidden)
    logits=hidden@np.asarray(model["outputWeights"]).T+np.asarray(model["outputBiases"])
    values=np.exp(logits-logits.max(axis=1,keepdims=True));values/=values.sum(axis=1,keepdims=True)
    if not np.isfinite(values).all(): raise ValueError("Invalid recurrent probabilities.")
    return values


def predict(model,arrays,noisy=False):
    import numpy as np
    # Final comparisons must cover exactly the same usable intervals as the tree references.
    if not np.array_equal(arrays["usable"],arrays["augmentedUsable"]):
        raise ValueError("Paired recurrent evaluation requires identical clean/noisy usable intervals.")
    data=layout(arrays,model["contextEpochs"])
    raw=arrays["augmentedFeatures" if noisy else "features"][data["indexes"]]
    values=normalized(encode_matrix(raw),model["means"],model["scales"])
    result=[]
    for start in range(0,len(values),512):
        indexes=np.arange(start,min(start+512,len(values)))
        result.append(forward(model,batch(values,data["windows"],data["lengths"],indexes),data["lengths"][indexes]))
    return np.concatenate(result),data["indexes"]
