"""Focused recurrent inference and subject-isolation contracts; no accuracy claims from synthetic data."""
import math
from pathlib import Path
import sys
import unittest

from sequence_model import sequence_indexes, encoded, normalize, SequencePredictor
from features import STAGES


def reference():
    import numpy as np
    import torch
    from train_sequence import arrays, export, make_network, torch_probabilities
    torch.set_num_threads(2)
    torch.manual_seed(7)
    network = make_network(2)
    network.eval()
    config = {"history": 10, "hidden": 2, "gainDB": 0}
    model = export(network, config, np.zeros(58), np.ones(58), ["synthetic-reference"])
    model.update(modelID="synthetic-gru-reference", dataSource="independent synthetic PyTorch math reference")
    rows = [{"subject": "synthetic-reference", "startSeconds": start, "usable": index != 3,
             "features": [(index+1)*(k%5-2)/16+0.1234567 for k in range(58)]}
            for index, start in enumerate((0, 30, 60, 90, 120, 180, 210))]
    valid = [r for r in rows if r["usable"]]
    values, lengths, _ = arrays([{**r, "stage": "light"} for r in valid], 10, np.zeros(58), np.ones(58))
    result = iter(torch_probabilities(network, values, lengths).tolist())
    for row in rows:
        row["expectedProbabilities"] = dict(zip(STAGES, next(result))) if row["usable"] else None
    return {"model": model, "rows": rows}


class SequenceContracts(unittest.TestCase):
    def test_windows_preserve_order_and_reset_at_gaps_and_participants(self):
        rows = [{"subject": p, "startSeconds": t} for p,t in (("a",0),("a",30),("a",60),("a",120),("b",150))]
        self.assertEqual(sequence_indexes(rows,10), [[0],[0,1],[0,1,2],[3],[4]])
        self.assertEqual(sequence_indexes(rows,1), [[0],[1],[2],[3],[4]])
        self.assertEqual(sequence_indexes(rows[:3],10), sequence_indexes(rows,10)[:3])

    def test_normalization_retains_original_features_and_rejects_overflow(self):
        values = [0.1234567]*58
        self.assertEqual(encoded(values)[:7],values[:7])
        self.assertEqual(encoded(values)[7:],[0.123457]*51)
        with self.assertRaises(ValueError): encoded([1e308]*58)
        with self.assertRaises(ValueError): normalize({"means":[0]*58,"scales":[1e-308]*58,"normalizedClip":8},[1e100]*58)

    def test_portable_gru_matches_independent_torch_and_clears_unusable_history(self):
        fixture = reference()
        predictor = SequencePredictor(fixture["model"])
        for row in fixture["rows"]:
            actual = predictor.append(row)
            if not row["usable"]:
                self.assertIsNone(actual)
                self.assertEqual(predictor.frames,[])
            else:
                self.assertLess(max(abs(x-y) for x,y in zip(actual,row["expectedProbabilities"].values())),2e-10)

    def test_validation_values_cannot_affect_training_normalization_or_weights(self):
        from train_sequence import fit
        rows = [{"subject": "train", "stage": stage, "startSeconds": i*120+k*30,"usable":True,
                 "features":[(k+1)/8+(j%3)/16 for j in range(58)]}
                for i in range(4) for k,stage in enumerate(STAGES)]
        validation = [{**r,"subject":"validation"} for r in rows]
        settings = {"history":10,"hidden":2,"gainDB":0}
        a,_,_ = fit(rows,validation,settings,7,1,1,fixed_epochs=1)
        changed = [{**r,"features":[x+10000 for x in r["features"]]} for r in validation]
        b,_,_ = fit(rows,changed,settings,7,1,1,fixed_epochs=1)
        for field in ("means","scales","inputWeights","recurrentWeights","inputBiases","recurrentBiases","outputWeights","outputBiases"):
            self.assertEqual(a[field],b[field])


if __name__ == "__main__":
    if "--write-reference" in sys.argv:
        index = sys.argv.index("--write-reference")
        target = Path(sys.argv[index+1])
        del sys.argv[index:index+2]
        from expand_cohort import atomic_json
        target.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(target,reference(),compact=True)
    unittest.main()
