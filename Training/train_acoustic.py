"""Small convolutional acoustic encoder and bounded past-only GRU, selected by participant folds."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from acoustic_data import load
from acoustic_frames import FRAME_NAMES
from acoustic_model import encode, EmbeddingPredictor
from expand_cohort import atomic_json
from features import STAGES
from improve_model import metrics, parameters_digest
from sequence_model import sequence_indexes
from train_sequence import make_network, torch_probabilities, fold_for


SETTINGS=[{"history":10,"hidden":16,"classPower":1.0},
          {"history":20,"hidden":16,"classPower":1.0},
          {"history":20,"hidden":32,"classPower":1.0},
          {"history":20,"hidden":16,"classPower":0.75}]


def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def make_encoder(channels=8):
    import torch
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=torch.nn.Conv1d(25,channels,5,padding=2)
            self.conv2=torch.nn.Conv1d(channels,channels,17,padding=8)
            self.dropout=torch.nn.Dropout(0.15)
            self.output=torch.nn.Linear(2*channels,4)

        def embedding(self,values):
            x=torch.relu(self.conv1(values.transpose(1,2)))
            x=torch.nn.functional.avg_pool1d(x,2)
            x=torch.relu(self.conv2(x))
            return torch.cat((x.mean(dim=2),x.amax(dim=2)),dim=1)

        def forward(self,values): return self.output(self.dropout(self.embedding(values)))
    return Encoder().double()


def export_encoder(network,means,scales):
    return {"channels":network.conv1.out_channels,"means":means.tolist(),"scales":scales.tolist(),"normalizedClip":8,
            "weights1":network.conv1.weight.detach().tolist(),"biases1":network.conv1.bias.detach().tolist(),
            "weights2":network.conv2.weight.detach().tolist(),"biases2":network.conv2.bias.detach().tolist()}


def import_encoder(model):
    import torch
    network=make_encoder(model["channels"])
    with torch.no_grad():
        for name,values in (("conv1.weight",model["weights1"]),("conv1.bias",model["biases1"]),
                            ("conv2.weight",model["weights2"]),("conv2.bias",model["biases2"])):
            dict(network.named_parameters())[name].copy_(torch.tensor(values,dtype=torch.float64))
    network.eval()
    return network


def labels_and_weights(rows,power):
    import numpy as np
    from collections import Counter
    labels=np.asarray([STAGES.index(r["stage"]) for r in rows])
    counts=np.bincount(labels,minlength=4); people=Counter(r["subject"] for r in rows)
    if any(counts==0): raise ValueError("Every fitting cohort must contain all four scored stages.")
    weights=np.asarray([(len(rows)/(4*counts[k]))**power*len(rows)/(len(people)*people[r["subject"]]) for r,k in zip(rows,labels)])
    weights/=weights.mean()
    return labels,weights


def checkpoint_key(score,reference):
    return (all(x>0 for x in score["recallPerClass"]),
            min(score["macroF1"]/reference["macroF1"],score["balancedAccuracy"]/reference["balancedAccuracy"]),
            score["macroF1"],score["balancedAccuracy"])


def encoder_predictions(network,values,embeddings=False):
    import numpy as np
    import torch
    network.eval(); output=[]
    with torch.no_grad():
        for start in range(0,len(values),128):
            x=torch.from_numpy(values[start:start+128])
            result=network.embedding(x) if embeddings else torch.softmax(network(x),dim=1)
            output.append(result.numpy())
    result=np.concatenate(output)
    if not np.isfinite(result).all(): raise ValueError("Non-finite acoustic encoder output.")
    return result


def fit_encoder(rows,frames,validation,validation_frames,reference,seed,max_epochs,patience,fixed_epochs=None):
    import numpy as np
    import torch
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    x=np.asarray(frames,dtype=np.float64)
    means=x.mean(axis=(0,1)); scales=np.maximum(x.std(axis=(0,1)),1e-6)
    x=(x-means)/scales
    vx=np.clip((np.asarray(validation_frames,dtype=np.float64)-means)/scales,-8,8) if validation else None
    labels,weights=labels_and_weights(rows,1.0)
    network=make_encoder(); optimizer=torch.optim.AdamW(network.parameters(),lr=0.001,weight_decay=0.01)
    objective=torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.02)
    best,best_key,best_epoch,stale=None,None,0,0; history=[]
    for epoch in range(1,(fixed_epochs or max_epochs)+1):
        started=time.monotonic(); network.train()
        for indexes in np.array_split(rng.permutation(len(rows)),(len(rows)+127)//128):
            batch=x[indexes].copy()
            batch[:,:,24]+=rng.uniform(-6,6,len(indexes))[:,None]/scales[24]
            np.clip(batch,-8,8,out=batch)
            optimizer.zero_grad(set_to_none=True)
            losses=objective(network(torch.from_numpy(batch)),torch.from_numpy(labels[indexes]))
            loss=(losses*torch.from_numpy(weights[indexes])).mean()
            if not torch.isfinite(loss): raise ValueError("Non-finite encoder fitting objective.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1,error_if_nonfinite=True); optimizer.step()
        entry={"epoch":epoch,"seconds":time.monotonic()-started}
        if validation:
            score=metrics(validation,encoder_predictions(network,vx).tolist()); key=checkpoint_key(score,reference)
            entry.update(macroF1=score["macroF1"],balancedAccuracy=score["balancedAccuracy"],recalls=score["recallPerClass"])
            if best_key is None or key>best_key:
                best,best_key,best_epoch,stale=copy.deepcopy(network.state_dict()),key,epoch,0
            else: stale+=1
        history.append(entry); print(json.dumps({"event":"encoder_epoch",**entry}),flush=True)
        if validation and epoch>=8 and stale>=patience: break
    if validation: network.load_state_dict(best)
    else: best_epoch=fixed_epochs
    network.eval(); model=export_encoder(network,means,scales)
    probe=np.asarray(validation_frames[:8] if validation else frames[:8],dtype=np.float64)
    expected=encoder_predictions(network,np.clip((probe-means)/scales,-8,8),embeddings=True)
    difference=float(np.max(np.abs(encode(model,probe)-expected)))
    if difference>2e-10: raise ValueError(f"Independent convolution export differs from PyTorch: {difference}")
    return model,{"bestEpoch":best_epoch,"epochsRun":len(history),"history":history,"portableMaximumDifference":difference}


def encode_all(encoder,frames):
    import numpy as np
    result=[]
    for start in range(0,len(frames),64): result.append(encode(encoder,frames[start:start+64]))
    return np.concatenate(result)


def vector_arrays(rows,values,history,means,scales):
    import numpy as np
    values=np.clip((values-means)/scales,-8,8); indexes=sequence_indexes(rows,history)
    x=np.zeros((len(rows),history,values.shape[1]),dtype=np.float64)
    for i,window in enumerate(indexes): x[i,:len(window)]=values[window]
    return x,np.asarray([len(window) for window in indexes])


def recurrent_parameters(network):
    return {"hiddenSize":network.gru.hidden_size,"inputWeights":network.gru.weight_ih_l0.detach().tolist(),
            "recurrentWeights":network.gru.weight_hh_l0.detach().tolist(),"inputBiases":network.gru.bias_ih_l0.detach().tolist(),
            "recurrentBiases":network.gru.bias_hh_l0.detach().tolist(),"outputWeights":network.output.weight.detach().tolist(),
            "outputBiases":network.output.bias.detach().tolist()}


def assemble_model(encoder,recurrent,settings,held_out):
    return {"schemaVersion":4,"modelKind":"conv-gru","modelID":"psg-audio-acoustic-v6",
            "dataSource":"PSG-Audio V3; CC BY 4.0; DOI 10.11922/sciencedb.00345","featureProfile":"mel-sequence-v6",
            "rawFeatureNames":FRAME_NAMES,"featureNames":[f"embedding_{kind}_{i}" for kind in ("mean","max") for i in range(encoder["channels"])],
            "contextEpochs":settings["history"],"classes":STAGES,"heldOutSubjects":held_out,
            "balancedAccuracy":0,"encoder":encoder,"recurrent":recurrent}


def fit_recurrent(rows,embeddings,validation,validation_embeddings,encoder,settings,reference,seed,max_epochs,patience,fixed_epochs=None):
    import numpy as np
    import torch
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    means=embeddings.mean(axis=0); scales=np.maximum(embeddings.std(axis=0),1e-6)
    x,lengths=vector_arrays(rows,embeddings,settings["history"],means,scales)
    vx,vl=vector_arrays(validation,validation_embeddings,settings["history"],means,scales) if validation else (None,None)
    labels,weights=labels_and_weights(rows,settings["classPower"])
    network=make_network(settings["hidden"],embeddings.shape[1])
    optimizer=torch.optim.AdamW(network.parameters(),lr=0.001,weight_decay=0.01)
    objective=torch.nn.CrossEntropyLoss(reduction="none",label_smoothing=0.02)
    best,best_key,best_epoch,stale=None,None,0,0; history=[]
    for epoch in range(1,(fixed_epochs or max_epochs)+1):
        started=time.monotonic(); network.train()
        for indexes in np.array_split(rng.permutation(len(rows)),(len(rows)+255)//256):
            optimizer.zero_grad(set_to_none=True)
            loss=(objective(network(torch.from_numpy(x[indexes]),torch.from_numpy(lengths[indexes])),torch.from_numpy(labels[indexes]))
                  *torch.from_numpy(weights[indexes])).mean()
            if not torch.isfinite(loss): raise ValueError("Non-finite acoustic sequence objective.")
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1,error_if_nonfinite=True); optimizer.step()
        entry={"epoch":epoch,"seconds":time.monotonic()-started}
        if validation:
            score=metrics(validation,torch_probabilities(network,vx,vl).tolist()); key=checkpoint_key(score,reference)
            entry.update(macroF1=score["macroF1"],balancedAccuracy=score["balancedAccuracy"],recalls=score["recallPerClass"])
            if best_key is None or key>best_key:
                best,best_key,best_epoch,stale=copy.deepcopy(network.state_dict()),key,epoch,0
            else: stale+=1
        history.append(entry); print(json.dumps({"event":"acoustic_sequence_epoch",**entry}),flush=True)
        if validation and epoch>=8 and stale>=patience: break
    if validation: network.load_state_dict(best)
    else: best_epoch=fixed_epochs
    network.eval(); encoded=copy.deepcopy(encoder)
    encoded.update(embeddingMeans=means.tolist(),embeddingScales=scales.tolist(),embeddingClip=8)
    model=assemble_model(encoded,recurrent_parameters(network),settings,sorted({r["subject"] for r in validation}))
    score=metrics(validation,torch_probabilities(network,vx,vl).tolist()) if validation else None
    check_rows=validation[:128] if validation else rows[:128]
    check_values=validation_embeddings[:128] if validation else embeddings[:128]
    ax,al=vector_arrays(check_rows,check_values,settings["history"],means,scales)
    expected=torch_probabilities(network,ax,al); predictor=EmbeddingPredictor(model)
    actual=[predictor.append(row,value) for row,value in zip(check_rows,check_values)]
    difference=float(np.max(np.abs(expected-np.asarray(actual))))
    if difference>2e-10: raise ValueError("Independent acoustic recurrent export differs from PyTorch.")
    return model,score,{"bestEpoch":best_epoch,"epochsRun":len(history),"history":history,"portableMaximumDifference":difference}


def run(args):
    import numpy as np
    import torch
    if torch.version.cuda is not None: raise ValueError("Use the bounded CPU-only research environment.")
    torch.set_num_threads(3); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    if (root/"selection.json").exists() or (root/"development-review.json").exists():
        raise ValueError("This development experiment is already complete.")
    rows,frames,manifest=load(args.development)
    known=json.loads(args.registry.read_text())
    if set(manifest["people"])!=set(known): raise ValueError("The acoustic fitting cohort must match the known-participant registry.")
    folds={p:fold_for(p) for p in known}
    prior=json.loads((args.baseline_cache/"protocol.json").read_text())
    if prior["folds"]!=folds or prior["development"]["people"]!=manifest["people"]:
        raise ValueError("Paired baseline folds use different participants.")
    baseline=[]
    for fold in range(3):
        report=json.loads((args.baseline_cache/f"baseline-fold-{fold}.json").read_text())
        validation=[r for r in rows if folds[r["subject"]]==fold]
        if (report["validation"]["subjects"]!=sorted({r["subject"] for r in validation})
                or report["validation"]["epochs"]!=len(validation)
                or any(sum(report["validation"]["confusionMatrix"][k])!=sum(r["stage"]==s for r in validation) for k,s in enumerate(STAGES))):
            raise ValueError("Acoustic observations differ from the paired baseline evaluation cohort.")
        baseline.append(report)
    protocol={"experiment":"acoustic-conv-gru-v6","development":manifest,"folds":folds,"configurations":SETTINGS,
              "previouslyObservedSubjects":known,"developmentExtractorSHA256":manifest["extractorSHA256"],
              "encoder":{"channels":8,"kernels":[5,17],"pool":2,"embedding":"mean and max","gainAugmentationDB":6},
              "maximumEncoderEpochs":args.encoder_epochs,"maximumSequenceEpochs":args.sequence_epochs,"patience":args.patience,
              "checkpointMetric":"Minimum relative macro F1 and balanced recall against paired baseline; every stage recognized.",
              "selectionMetric":"Both mean paired development metrics must improve, then maximize mean macro F1.",
              "refitEpochs":"Fixed median best encoder and sequence epochs over participant folds.","seed":7,
              "baselineProtocolSHA256":file_sha(args.baseline_cache/"protocol.json"),
              "baselineReportsSHA256":{str(f):file_sha(args.baseline_cache/f"baseline-fold-{f}.json") for f in range(3)},
              "torchVersion":torch.__version__,"numpyVersion":np.__version__,"usesFutureAudio":False,
              "sourceSha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
    if (root/"protocol.json").exists() and json.loads((root/"protocol.json").read_text())!=protocol:
        raise ValueError("Resume only the identical frozen acoustic development experiment.")
    atomic_json(root/"protocol.json",protocol); candidates=[[] for _ in SETTINGS]; encoder_reports=[]
    for fold in range(3):
        train_indexes=np.asarray([i for i,r in enumerate(rows) if folds[r["subject"]]!=fold]); validation_indexes=np.asarray([i for i,r in enumerate(rows) if folds[r["subject"]]==fold])
        train=[rows[i] for i in train_indexes]; validation=[rows[i] for i in validation_indexes]
        path=root/f"encoder-fold-{fold}.json"; report_path=root/f"encoder-fit-{fold}.json"
        if path.exists() and report_path.exists():
            encoder=json.loads(path.read_text()); detail=json.loads(report_path.read_text())
            if file_sha(path)!=detail["modelFileSHA256"]: raise ValueError("Cached acoustic encoder changed.")
        else:
            print(json.dumps({"event":"encoder_fit","fold":fold,"trainPeople":len({r["subject"] for r in train})}),flush=True)
            encoder,detail=fit_encoder(train,frames[train_indexes],validation,frames[validation_indexes],baseline[fold]["validation"],7+fold,args.encoder_epochs,args.patience)
            atomic_json(path,encoder,compact=True); detail["modelFileSHA256"]=file_sha(path); atomic_json(report_path,detail)
        encoder_reports.append(detail)
        embeddings=encode_all(encoder,frames)
        for index,settings in enumerate(SETTINGS):
            report_path=root/f"candidate-{index}-fold-{fold}.json"; model_path=root/f"weights-{index}-fold-{fold}.json"
            if report_path.exists() and model_path.exists():
                report=json.loads(report_path.read_text())
                if report["modelFileSHA256"]!=file_sha(model_path): raise ValueError("Cached acoustic sequence checkpoint changed.")
            else:
                print(json.dumps({"event":"sequence_fit","candidate":index,"fold":fold,"settings":settings}),flush=True)
                model,score,fit=fit_recurrent(train,embeddings[train_indexes],validation,embeddings[validation_indexes],encoder,settings,
                                            baseline[fold]["validation"],7+fold,args.sequence_epochs,args.patience)
                atomic_json(model_path,model,compact=True)
                report={"fold":fold,"validation":score,"fit":fit,"modelFileSHA256":file_sha(model_path)}
                atomic_json(report_path,report)
            candidates[index].append(report)
    summaries=[{"id":i,"settings":settings,"folds":reports,
                "meanMacroF1":float(np.mean([r["validation"]["macroF1"] for r in reports])),
                "meanBalancedAccuracy":float(np.mean([r["validation"]["balancedAccuracy"] for r in reports])),
                "everyFoldRecognizesEveryStage":all(all(x>0 for x in r["validation"]["recallPerClass"]) for r in reports)}
               for i,(settings,reports) in enumerate(zip(SETTINGS,candidates))]
    atomic_json(root/"candidates.json",summaries)
    baseline_f1=float(np.mean([r["validation"]["macroF1"] for r in baseline])); baseline_accuracy=float(np.mean([r["validation"]["balancedAccuracy"] for r in baseline]))
    eligible=[x for x in summaries if x["everyFoldRecognizesEveryStage"] and x["meanMacroF1"]>baseline_f1 and x["meanBalancedAccuracy"]>baseline_accuracy]
    if not eligible:
        atomic_json(root/"development-review.json",{"status":"not_improved","freshTestUntouched":True,"baselineMeanMacroF1":baseline_f1,
                    "baselineMeanBalancedAccuracy":baseline_accuracy,"candidates":[{k:r[k] for k in ("id","meanMacroF1","meanBalancedAccuracy")} for r in summaries]})
        print(json.dumps({"event":"not_improved","freshTestUntouched":True}),flush=True); return
    winner=max(eligible,key=lambda r:(r["meanMacroF1"],r["meanBalancedAccuracy"],-r["id"]))
    encoder_epochs=int(np.median([r["bestEpoch"] for r in encoder_reports])); sequence_epochs=int(np.median([r["fit"]["bestEpoch"] for r in winner["folds"]]))
    encoder,encoder_fit=fit_encoder(rows,frames,[],frames[:0],None,7,args.encoder_epochs,args.patience,fixed_epochs=encoder_epochs)
    embeddings=encode_all(encoder,frames)
    model,_,sequence_fit=fit_recurrent(rows,embeddings,[],embeddings[:0],encoder,winner["settings"],None,7,args.sequence_epochs,args.patience,fixed_epochs=sequence_epochs)
    atomic_json(root/"selected-model.json",model,compact=True)
    if (root/"selected-model.json").stat().st_size>508000: raise ValueError("Final acoustic model exceeds the native import budget.")
    selection={**winner,"parametersSHA256":parameters_digest(model),"finalTestRead":False,"trainingSubjects":manifest["people"],"validationSubjects":[],
               "previouslyObservedSubjects":known,"baselineMeanMacroF1":baseline_f1,"baselineMeanBalancedAccuracy":baseline_accuracy,
               "finalRefit":{"encoderEpochs":encoder_epochs,"sequenceEpochs":sequence_epochs,"encoderFit":encoder_fit,"sequenceFit":sequence_fit}}
    atomic_json(root/"selection.json",selection)
    print(json.dumps({"event":"selected","id":winner["id"],"macroF1":winner["meanMacroF1"],"balancedAccuracy":winner["meanBalancedAccuracy"],
                      "parametersSHA256":selection["parametersSHA256"]}),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development",type=Path,required=True); p.add_argument("--baseline-cache",type=Path,required=True)
    p.add_argument("--registry",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--encoder-epochs",type=int,default=30); p.add_argument("--sequence-epochs",type=int,default=40); p.add_argument("--patience",type=int,default=6)
    args=p.parse_args()
    if not 8<=args.encoder_epochs<=40 or not 8<=args.sequence_epochs<=60 or not 3<=args.patience<=10: p.error("Bound the acoustic CPU experiment.")
    run(args)
