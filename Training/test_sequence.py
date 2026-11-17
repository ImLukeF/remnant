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
    def test_batched_paired_gru_matches_torch_and_never_reads_future_or_other_people(self):
        import numpy as np
        from array_gru import layout,encode_matrix,normalized,batch,forward,predict
        from train_sequence import make_network,export,torch_probabilities
        import torch
        torch.set_num_threads(2);torch.manual_seed(7)
        network=make_network(2);network.eval()
        raw=np.arange(9*58,dtype=np.float64).reshape(9,58)/600
        data={"features":raw,"augmentedFeatures":raw+0.1,"usable":np.ones(9,dtype=bool),"augmentedUsable":np.ones(9,dtype=bool),
              "subject":np.asarray(["a"]*7+["b"]*2),"startSeconds":np.asarray([0,30,60,90,120,180,210,240,270]),"stage":np.arange(9)%4}
        data["usable"][3]=False;data["augmentedUsable"][3]=False
        windows=layout(data,10)
        self.assertEqual(windows["lengths"].tolist(),[1,2,3,1,1,2,1,2])
        model=export(network,{"history":10,"hidden":2},np.zeros(58),np.ones(58),[])
        x=batch(normalized(encode_matrix(raw[windows["indexes"]]),model["means"],model["scales"]),windows["windows"],windows["lengths"],np.arange(8))
        expected=torch_probabilities(network,x,windows["lengths"])
        actual,indexes=predict(model,data)
        self.assertLess(float(np.abs(expected-actual).max()),2e-10)
        self.assertLess(float(np.abs(forward(model,x,windows["lengths"])-expected).max()),2e-10)
        changed={**data,"features":raw.copy()};changed["features"][5:]+=500
        self.assertTrue(np.array_equal(predict(model,changed)[0][:4],actual[:4]))
        changed={**data,"features":raw.copy()};changed["features"][:7]-=500
        self.assertTrue(np.array_equal(predict(model,changed)[0][-2:],actual[-2:]))
        from sequence_model import SequencePredictor
        scalar=SequencePredictor(model);last=None
        for position,original in enumerate(indexes):
            person=data["subject"][original]
            if person!=last: scalar=SequencePredictor(model)
            result=scalar.append({"usable":True,"startSeconds":float(data["startSeconds"][original]),"features":raw[original].tolist()})
            self.assertLess(max(abs(a-b) for a,b in zip(result,actual[position])),2e-10);last=person

    def test_paired_consistency_is_symmetric_finite_and_excludes_checkpoint_people_from_fit(self):
        import numpy as np
        import torch
        from train_consistency import consistency,fit
        from train_consistency import checkpoint_key
        torch.set_num_threads(2)
        reference={"clean":{"macroF1":0.3,"balancedAccuracy":0.45},"noisy":{"macroF1":0.08,"balancedAccuracy":0.25}}
        a_score={"clean":{"macroF1":0.35,"balancedAccuracy":0.5,"recallPerClass":[0.5]*4},
                 "noisy":{"macroF1":0.22,"balancedAccuracy":0.4,"recallPerClass":[0.4]*4}}
        b_score={"clean":{"macroF1":0.26,"balancedAccuracy":0.44,"recallPerClass":[0.44]*4},
                 "noisy":{"macroF1":0.25,"balancedAccuracy":0.4,"recallPerClass":[0.4]*4}}
        self.assertGreater(checkpoint_key(a_score,reference),checkpoint_key(b_score,reference))
        self.assertLess(checkpoint_key(a_score),checkpoint_key(b_score))
        a=torch.tensor([[2.0,-1,0,1]],dtype=torch.float64,requires_grad=True)
        b=torch.tensor([[-1.0,2,1,0]],dtype=torch.float64,requires_grad=True)
        self.assertLess(abs(float(consistency(a,a).detach())),1e-12)
        self.assertTrue(torch.allclose(consistency(a,b),consistency(b,a)))
        loss=consistency(a,b).sum();self.assertGreater(float(loss.detach()),0);loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.isfinite(b.grad).all())
        raw=np.arange(32*58,dtype=np.float64).reshape(32,58)/2000
        arrays={"features":raw,"augmentedFeatures":raw+np.sin(raw),"usable":np.ones(32,dtype=bool),"augmentedUsable":np.ones(32,dtype=bool),
                "subject":np.asarray(["fit"]*16+["checkpoint"]*16),"startSeconds":np.tile(np.arange(16)*30,2),"stage":np.tile(np.arange(16)%4,2)}
        recipe={"history":10,"hidden":2,"consistency":0.25,"noiseShare":0.25,"classPower":1.0}
        first,_,_,report=fit(arrays,["fit"],["checkpoint"],recipe,60,maximum_epochs=1)
        changed={**arrays,"features":raw.copy(),"augmentedFeatures":arrays["augmentedFeatures"].copy(),"stage":arrays["stage"].copy()}
        changed["features"][16:]+=10000;changed["augmentedFeatures"][16:]-=10000;changed["stage"][16:]=(changed["stage"][16:]+1)%4
        second,_,_,_=fit(changed,["fit"],["checkpoint"],recipe,60,maximum_epochs=1)
        for field in ("means","scales","inputWeights","recurrentWeights","inputBiases","recurrentBiases","outputWeights","outputBiases"):
            self.assertEqual(first[field],second[field])
        self.assertEqual(report["trainingPeople"],1);self.assertEqual(report["calibrationPeople"],1)
        self.assertLess(report["exportMaximumDifference"],2e-10)
        with self.assertRaises(ValueError): fit(arrays,["fit"],["fit"],recipe,60,maximum_epochs=1)

    def test_consistency_refit_requires_frozen_admission_and_keeps_final_people_unopened(self):
        import hashlib
        import json
        import tempfile
        import numpy as np
        from train_consistency import frozen_refit,RECIPES
        from expanded_contracts import final_admission
        from expand_cohort import atomic_json
        from run_dataset import checksum
        people=[f"synthetic-{n}" for n in range(199)];excluded={"synthetic-silent":{"reason":"synthetic wholly silent fixture"}}
        final=[f"unopened-{n}" for n in range(30)]
        features=np.arange(796*58,dtype=np.float64).reshape(796,58)/50000
        arrays={"features":features,"augmentedFeatures":features+0.1,"subject":np.repeat(people,4),"stage":np.tile(np.arange(4),199),
            "startSeconds":np.tile(np.arange(4)*30,199),"usable":np.ones(796,dtype=bool),"augmentedUsable":np.ones(796,dtype=bool)}
        good={name:{"macroF1":0.5,"balancedAccuracy":0.5,"recallPerClass":[0.5]*4} for name in ("clean","noisy")}
        baseline={name:{"macroF1":0.3,"balancedAccuracy":0.3,"recallPerClass":[0.3]*4} for name in ("clean","noisy")}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weights=root/"recipe-0-weights.json";atomic_json(weights,{"synthetic":"checkpoint checksum fixture"})
            report={"validation":good,"modelFileSHA256":checksum(weights),"fit":{"bestEpoch":1}}
            protocol={"development":{"synthetic":True},"baselineModelSHA256":"synthetic-baseline","recipes":RECIPES,
                "refitSubjects":people,"reservedTestSubjects":final,"qualityExcludedSubjects":excluded,"previouslyObservedSubjects":people+list(excluded)}
            summary={"status":"trained_not_improved","chosenRecipe":"recipe-0","validation":{"recipe-0":report},"baselineValidation":baseline,
                "qualityExcludedSubjects":excluded,"finalTestRead":False}
            atomic_json(root/"protocol.json",protocol);atomic_json(root/"training-summary.json",summary);atomic_json(root/"recipe-0-report.json",report)
            with self.assertRaises(ValueError): frozen_refit(root,arrays,protocol["development"],"synthetic-baseline",60)
            summary["status"]="admitted_pending_refit";atomic_json(root/"training-summary.json",summary)
            atomic_json(weights,{"synthetic":"changed"})
            with self.assertRaises(ValueError): frozen_refit(root,arrays,protocol["development"],"synthetic-baseline",60)
            atomic_json(weights,{"synthetic":"checkpoint checksum fixture"})
            frozen_refit(root,arrays,protocol["development"],"synthetic-baseline",60)
            result=json.loads((root/"training-summary.json").read_text())
            self.assertEqual(result["finalRefitPeople"],199);self.assertFalse(result["finalTestRead"])
            reservation={"roles":{p:"train" for p in people+list(excluded)},"reservedTestSubjects":final}
            self.assertEqual(final_admission(root,reservation,list(reservation["roles"])),result["parametersSHA256"])
            self.assertEqual(len(json.loads((root/"export-check.json").read_text())),8)
            with self.assertRaises(ValueError): frozen_refit(root,arrays,protocol["development"],"synthetic-baseline",60)

    def test_final_registry_keeps_previously_tested_people_outside_this_fitting_corpus(self):
        from evaluation_contracts import participant_registry
        selected={"trainingSubjects":["fit"],"validationSubjects":["validate"],"previouslyObservedSubjects":["fit","validate"]}
        plan={"previouslyObservedSubjects":["fit","validate","earlier-test"],"testSubjects":["new"],"fittingForbidden":True}
        self.assertEqual(participant_registry(selected,plan),["earlier-test","fit","new","validate"])
        for change in ({"testSubjects":["earlier-test"]},{"previouslyObservedSubjects":["fit"]},{"fittingForbidden":False}):
            with self.assertRaises(ValueError): participant_registry(selected,plan|change)

    def test_new_test_requires_both_development_metrics_and_all_stage_coverage(self):
        from prepare_sequence_test import admit
        good = {"meanMacroF1":0.4,"baselineMeanMacroF1":0.3,"meanBalancedAccuracy":0.5,
                "baselineMeanBalancedAccuracy":0.45,"everyFoldRecognizesEveryStage":True}
        admit(good)
        for change in ({"meanMacroF1":0.3},{"meanBalancedAccuracy":0.4},{"everyFoldRecognizesEveryStage":False},{"meanMacroF1":float("nan")}):
            with self.assertRaises(ValueError): admit(good|change)

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
