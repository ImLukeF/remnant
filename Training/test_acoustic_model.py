"""Independent neural math and fitting-isolation proofs, using synthetic observations only."""
from pathlib import Path
import sys
import unittest


def synthetic_frames(seed):
    import numpy as np
    return ((((np.arange(240)[:,None]+seed)*(np.arange(25)[None,:]+3))%37-18)/16).astype(np.float32)


def reference():
    import numpy as np
    import torch
    from train_acoustic import make_encoder,export_encoder,recurrent_parameters,assemble_model,encoder_predictions,vector_arrays
    from train_sequence import make_network,torch_probabilities
    from features import STAGES
    torch.set_num_threads(2); torch.manual_seed(7)
    encoder=make_encoder(2); encoder.eval()
    recurrent=make_network(2,4); recurrent.eval()
    means=np.linspace(-0.125,0.125,25); scales=np.linspace(0.5,1.5,25)
    params=export_encoder(encoder,means,scales)
    em=np.asarray([-0.1,0,0.1,0.2]); es=np.asarray([0.5,0.75,1.25,2.0])
    params.update(embeddingMeans=em.tolist(),embeddingScales=es.tolist(),embeddingClip=8)
    model=assemble_model(params,recurrent_parameters(recurrent),{"history":10},["synthetic-reference"])
    model.update(modelID="synthetic-acoustic-reference",dataSource="independent synthetic PyTorch math reference")
    rows=[{"subject":"synthetic-reference","startSeconds":t,"seed":i+1,"usable":i!=3,"stage":"light"}
          for i,t in enumerate((0,30,60,90,120,180,210))]
    valid=[r for r in rows if r["usable"]]
    frames=np.asarray([synthetic_frames(r["seed"]) for r in valid])
    embeddings=encoder_predictions(encoder,np.clip((frames.astype(np.float64)-means)/scales,-8,8),embeddings=True)
    x,lengths=vector_arrays(valid,embeddings,10,em,es)
    expected=iter(torch_probabilities(recurrent,x,lengths).tolist())
    for row in rows: row["expectedProbabilities"]=dict(zip(STAGES,next(expected))) if row["usable"] else None
    return {"model":model,"rows":rows,"validEmbeddings":embeddings.tolist()}


class AcousticModelContracts(unittest.TestCase):
    def test_independent_convolution_and_recurrence_match_pytorch_with_gaps(self):
        import numpy as np
        from acoustic_model import encode,EmbeddingPredictor
        fixture=reference(); model=fixture["model"]; predictor=EmbeddingPredictor(model)
        valid=[r for r in fixture["rows"] if r["usable"]]
        embeddings=encode(model["encoder"],np.asarray([synthetic_frames(r["seed"]) for r in valid]))
        np.testing.assert_allclose(embeddings,fixture["validEmbeddings"],rtol=0,atol=2e-10)
        values=iter(embeddings)
        for row in fixture["rows"]:
            actual=predictor.append(row,next(values) if row["usable"] else [])
            if row["usable"]: np.testing.assert_allclose(actual,list(row["expectedProbabilities"].values()),rtol=0,atol=2e-10)
            else: self.assertIsNone(actual)

    def test_encoder_fitting_cannot_learn_validation_values(self):
        import numpy as np
        from train_acoustic import fit_encoder
        from features import STAGES
        rows=[{"subject":"fitting","stage":stage,"startSeconds":30*i,"usable":True} for i,stage in enumerate(STAGES*2)]
        frames=np.asarray([synthetic_frames(i+1) for i in range(len(rows))])
        validation=[{**r,"subject":"validation"} for r in rows]
        prior={"macroF1":0.2,"balancedAccuracy":0.3}
        a,_=fit_encoder(rows,frames,validation,frames,prior,7,1,1,fixed_epochs=1)
        b,_=fit_encoder(rows,frames,validation,frames+1000,prior,7,1,1,fixed_epochs=1)
        for key in ("means","scales","weights1","weights2","biases1","biases2"): self.assertEqual(a[key],b[key])


if __name__ == "__main__":
    output=None
    if "--write-reference" in sys.argv:
        index=sys.argv.index("--write-reference"); output=Path(sys.argv[index+1]); del sys.argv[index:index+2]
    result=unittest.main(exit=False)
    if output and result.result.wasSuccessful():
        from expand_cohort import atomic_json
        output.parent.mkdir(parents=True,exist_ok=True); atomic_json(output,reference(),compact=True)
    sys.exit(not result.result.wasSuccessful())
