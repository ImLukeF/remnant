"""Synthetic waveform and EDF proofs for expanded-night training; no clinical performance claim."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def write_bank(root):
    import numpy as np
    root.mkdir(exist_ok=True);clips=[]
    for i,(role,kind) in enumerate((r,k) for r in ("train","validation","test") for k in ("speech","music","noise")):
        path=root/f"synthetic-{i}.npy"
        values=(0.1*np.sin(np.arange(60*16000)*2*np.pi*(100+17*i)/16000)).astype(np.float32)
        np.save(path,values)
        clips.append({"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "sourcePath":f"synthetic-only/{i}.wav","licence":"CC0-1.0","role":role,"kind":kind})
    (root/"bank.json").write_text(json.dumps({"approvedForAugmentation":True,"clips":clips}))


class FullNightContracts(unittest.TestCase):
    def test_bias_calibration_learns_on_separate_people_and_preserves_the_original_trees(self):
        import copy
        import numpy as np
        from array_trees import probabilities
        from train_correction import calibrate,adjusted
        model={"modelKind":"trees","biases":[0,0,2,0],"trees":[{"classIndex":0,"left":[-1],"right":[-1],"features":[-1],"values":[0]}]}
        original=copy.deepcopy(model);people=[f"synthetic-{i}" for i in range(8)]
        labels=np.tile([0,1,1,1,1,1,1,1,2,3],8)
        view={"inputs":np.zeros((80,1)),"labels":labels,"people":np.repeat(people,10)}
        bias,report=calibrate(model,{"clean":view,"noisy":view},people)
        corrected=adjusted(model,bias)
        self.assertEqual(model,original)
        self.assertEqual(corrected["trees"],original["trees"])
        self.assertLess(report["objective"],report["beforeObjective"])
        self.assertGreater(probabilities(corrected,[[0]])[0,1],probabilities(model,[[0]])[0,1])
        self.assertFalse(report["usesValidationPeople"])
        with self.assertRaises(ValueError): calibrate(model,{"clean":view,"noisy":view},people+["absent"])

    def test_controller_upgrades_only_a_stopped_worker_with_the_same_fingerprint(self):
        from argparse import Namespace
        from orchestrate_expansion import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state_path=root/"controller-state.json"
            old="a"*40;new="b"*40;fingerprint="c"*64
            original={"featureSource":old,"jobs":[{"status":"cancelled","reason":"planned scheduling upgrade"}],"completedPhases":[]}
            (root/"extraction-protocol.json").write_text(json.dumps({"extractorSHA256":fingerprint}))
            # Completed development data avoids dispatching external jobs; the transition
            # still has to reject active ownership and a changed measurement fingerprint.
            (root/"development-complete.json").write_text("{}")
            args=Namespace(root=root,noise_bank=root,feature_source=new,previous_feature_source=old,
                expected_extractor=fingerprint,extract_workers=3,idle_wait=1,initial_job=None,max_batches=1)
            for state,expected,previous in (({**original,"activeJob":"still-running"},fingerprint,old),
                    (original,"d"*64,old),(original,fingerprint,"e"*40)):
                state_path.write_text(json.dumps(state));args.expected_extractor=expected;args.previous_feature_source=previous
                with self.assertRaises(ValueError): run(args)
                self.assertEqual(json.loads(state_path.read_text()),state)
            state_path.write_text(json.dumps(original));args.expected_extractor=fingerprint;args.previous_feature_source=old
            run(args);state=json.loads(state_path.read_text())
            self.assertEqual(state["status"],"development_ready")
            self.assertEqual(state["featureSource"],new)
            self.assertEqual(state["sourceUpgrades"][0]["extractorSHA256"],fingerprint)
            self.assertEqual(state["jobs"],original["jobs"])
            self.assertEqual(state["workerSettings"],{"extractWorkers":3,"blasThreads":2,"idleWait":1})

    def test_short_idle_wait_preserves_measurements_bit_for_bit(self):
        import os
        import subprocess
        import sys
        code='''import hashlib,sys
import numpy as np
from run_expanded_corpus import extraction_bank
from spectral import extract_epoch
bank=extraction_bank(sys.argv[1]);digest=hashlib.sha256()
for rate in (8000,48000):
 for i,scale in enumerate((0,1e-7,1e-4,0.02,0.1,0.3)):
  pcm=np.random.default_rng(i).normal(0,scale,30*rate)
  clean,valid=extract_epoch(pcm,rate)
  for role in ("train","validation"):
   mixed,snr=bank.mix(pcm,rate,"synthetic-only",i*30,role)
   noisy,usable=extract_epoch(mixed,rate)
   digest.update(np.asarray(clean+noisy+[valid,usable,snr],dtype=np.float64).tobytes())
print(digest.hexdigest())'''
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_bank(root)
            # Thread-count changes can alter rounding, so retain two BLAS threads.
            # Only their idle wait changes; exact measurements must remain unchanged.
            base={k:v for k,v in os.environ.items() if k!="OPENBLAS_THREAD_TIMEOUT"}
            base.update(OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2")
            results=[subprocess.check_output([sys.executable,"-c",code,str(root)],text=True,
                env={**base,**({"OPENBLAS_THREAD_TIMEOUT":str(wait)} if wait is not None else {})},timeout=120).strip()
                for wait in (None,1)]
            self.assertEqual(results[0],results[1])

    def test_parallel_extraction_preserves_every_array_and_complete_checkpoint(self):
        import datetime
        import shutil
        from argparse import Namespace
        from unittest.mock import patch
        import numpy as np
        import pyedflib
        from run_dataset import checksum
        from run_expanded_corpus import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_bank(root/"noise")
            rate=48000;start=datetime.datetime(2020,1,1)
            tone=np.where(np.arange(30*rate)//8%2==0,2000,-2000).astype(np.int32)
            signal=np.concatenate((tone,np.zeros(30*rate,dtype=np.int32),tone*2,tone*3))
            original=root/"original.edf"
            with pyedflib.EdfWriter(str(original),2,file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                writer.setStartdatetime(start)
                for channel,name in enumerate(("Tracheal","Mic")):
                    writer.setSignalHeader(channel,{"label":name,"dimension":"a.u.","sample_frequency":rate,
                        "physical_min":-1,"physical_max":1,"digital_min":-32768,"digital_max":32767,"transducer":"","prefilter":""})
                writer.writeSamples([np.full(len(signal),16000,dtype=np.int32),signal],digital=True)
            labels=(f'<Root><Session><RecordingStart>{start.isoformat()}</RecordingStart><Duration>120</Duration></Session>'
                    '<UserStaging><Stage Type="NonREM2" Start="0"/><Stage Type="REM" Start="30"/>'
                    '<Stage Type="NonREM3" Start="60"/><Stage Type="Wake" Start="90"/></UserStaging></Root>').encode()
            oid=hashlib.sha1(f"blob {len(labels)}\0".encode()+labels).hexdigest()
            roles={"synthetic1":"train","synthetic2":"train","synthetic3":"validation"}
            items=[{"subject":p,"recording":p,"path":p+".edf","sha256":checksum(original),"rmlGitOID":oid,
                    "offsetSeconds":0,"durationSeconds":120,"bytes":original.stat().st_size} for p in roles]
            # Only external transport is replaced. Extraction, worker processes, DSP and
            # checkpoint publication run through the production owner on real synthetic EDFs.
            def sources(selected,cache,features,workers):
                for number,item in enumerate(selected):
                    path=cache/(item["subject"]+".edf");shutil.copyfile(original,path)
                    self.assertEqual(checksum(path),item["sha256"])
                    yield number,item,path
            outputs=[]
            for workers in (1,3):
                output=root/str(workers);output.mkdir();(output/"labels").mkdir()
                for p in roles: (output/"labels"/(p+".rml")).write_bytes(labels)
                (output/"development-plan.json").write_text(json.dumps({"roles":roles,"files":items,"reservedTestSubjects":["unopened"]}))
                args=Namespace(output=output,noise_bank=root/"noise",max_download_gib=1,batch_gib=1,download_workers=2,extract_workers=workers)
                with patch("run_expanded_corpus.prefetched_audio",sources): run(args)
                checks=json.loads((output/"feature-checksums.json").read_text())
                complete=json.loads((output/"development-complete.json").read_text())
                self.assertEqual(set(checks),{i["path"] for i in items})
                self.assertEqual(complete["blocks"],3)
                self.assertEqual(complete["usableEpochs"],9)
                self.assertEqual(set(complete["people"]),set(roles))
                for item in items:
                    self.assertEqual(checksum(output/"features"/(item["subject"]+".npz")),checks[item["path"]])
                self.assertFalse(list((output/"raw-cache").glob("*.edf")))
                # A completed run must reuse all checkpoints without fetching the sources again.
                with patch("run_expanded_corpus.prefetched_audio",sources): run(args)
                self.assertEqual(json.loads((output/"feature-checksums.json").read_text()),checks)
                outputs.append(output)
            for p in roles:
                with np.load(outputs[0]/"features"/(p+".npz"),allow_pickle=False) as serial, np.load(outputs[1]/"features"/(p+".npz"),allow_pickle=False) as parallel:
                    self.assertEqual(serial.files,parallel.files)
                    for key in serial.files: np.testing.assert_array_equal(serial[key],parallel[key],err_msg=key)

    def test_expanded_training_refits_all_200_people_without_reading_reserved_people(self):
        import numpy as np
        from argparse import Namespace
        from causal_model import names
        from expanded_features import fingerprint
        from expanded_contracts import final_admission
        from features import STAGES
        from run_dataset import checksum
        from spectral import SPECTRAL_NAMES
        from train_expanded import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/"development";models=root/"models"
            data.mkdir();(data/"features").mkdir();(data/"labels").mkdir()
            save=lambda path,value:path.write_text(json.dumps(value))
            people=[f"fitting{i:03d}" for i in range(200)]
            known=people[:84];reserved=[f"untouched-{i:03d}" for i in range(30)]
            roles={p:"train" if i<160 else "validation" for i,p in enumerate(people)}
            save(root/"original-registry.json",known)
            reservation={"roles":roles,"reservedTestSubjects":reserved,"knownRegistrySHA256":checksum(root/"original-registry.json")}
            save(data/"participants.json",reservation)
            files=[{"path":p+".edf","subject":p,"sha256":"a"*64} for p in people]
            save(data/"development-plan.json",{"roles":roles,"reservedTestSubjects":reserved,"files":files})
            inputs=np.zeros((4,58));inputs[:,0]=np.arange(4)
            noisy=inputs.copy();noisy[:,1]=0.1
            checks={}
            for item in files:
                person=item["subject"];path=data/"features"/(person+".npz")
                (data/"labels"/(person+".rml")).write_text(
                    f'<Root><Patient><PatientID>{person}</PatientID><PatientDBID>db-{person}</PatientDBID></Patient></Root>')
                np.savez_compressed(path,features=inputs,augmentedFeatures=noisy,startSeconds=np.arange(4)*30.,
                    stage=np.arange(4,dtype=np.int8),usable=np.ones(4,dtype=bool),augmentedUsable=np.ones(4,dtype=bool),
                    subject=np.asarray(person),sourcePath=np.asarray(item["path"]),sourceSHA256=np.asarray(item["sha256"]),
                    noiseRole=np.asarray(roles[person]),noiseManifestSHA256=np.asarray("synthetic-bank"),
                    extractorSHA256=np.asarray(fingerprint()),recordingDuration=np.asarray(120.))
                checks[item["path"]]=checksum(path)
            save(data/"feature-checksums.json",checks)
            protocol={"extractorSHA256":fingerprint(),"noiseManifestSHA256":"synthetic-bank",
                      "developmentPlanSHA256":checksum(data/"development-plan.json")}
            save(data/"extraction-protocol.json",protocol)
            save(data/"development-complete.json",{**protocol,"featureChecksumsSHA256":checksum(data/"feature-checksums.json"),"usableEpochs":800})
            baseline={"schemaVersion":2,"modelKind":"trees","modelID":"synthetic-only","dataSource":"synthetic only",
                "featureProfile":"spectral-v4","rawFeatureNames":SPECTRAL_NAMES,"featureNames":names(10,SPECTRAL_NAMES),
                "contextEpochs":10,"classes":STAGES,"heldOutSubjects":[],"balancedAccuracy":0,
                "biases":[1,0,0,0],"trees":[{"classIndex":0,"left":[-1],"right":[-1],"features":[-1],"values":[0]}]}
            save(root/"baseline.json",baseline)
            run(Namespace(output=models,development=data,original_registry=root/"original-registry.json",baseline=root/"baseline.json"))
            summary=json.loads((models/"training-summary.json").read_text())
            frozen=json.loads((models/"protocol.json").read_text())
            self.assertEqual(summary["status"],"admitted")
            self.assertEqual(summary["refit"]["fit"]["trainingPeople"],200)
            self.assertEqual(frozen["trainingSubjects"],people[:160])
            self.assertEqual(frozen["validationSubjects"],people[160:])
            self.assertEqual(frozen["reservedTestSubjects"],reserved)
            self.assertFalse(summary["finalTestRead"])
            self.assertFalse(summary["approvedForAlarmTiming"])
            self.assertEqual(final_admission(models,reservation,people),summary["parametersSHA256"])
            self.assertEqual(json.loads((models/"selected-model.json").read_text())["heldOutSubjects"],[])
            self.assertEqual({p.stem for p in (data/"labels").glob("*.rml")},set(people))
            # A separate synthetic corpus exercises the real silence-only recovery.
            import shutil
            from corpus_quality import amend,complete
            amended=root/"silent-development";shutil.copytree(data,amended);bad=people[0]
            path=amended/"features"/(bad+".npz")
            with np.load(path,allow_pickle=False) as old: values={k:old[k].copy() for k in old.files}
            silent=np.zeros((4,58));silent[:,[0,2,3]]=-120
            values.update(features=silent,augmentedFeatures=silent.copy(),usable=np.zeros(4,dtype=bool),augmentedUsable=np.zeros(4,dtype=bool))
            np.savez_compressed(path,**values);checks[bad+".edf"]=checksum(path)
            save(amended/"feature-checksums.json",checks)
            summaries={p+".edf":{"subject":p,"role":roles[p],"usableEpochs":0 if p==bad else 4,
                "cleanStageCounts":[0]*4 if p==bad else [1]*4} for p in people}
            save(amended/"block-summary.json",summaries)
            plan=json.loads((amended/"development-plan.json").read_text())
            with self.assertRaises(ValueError): complete(amended,plan,checks,summaries,protocol)
            with self.assertRaises(ValueError): amend(amended,models)
            models2=root/"models-199"
            amend(amended,models2)
            self.assertEqual(json.loads((amended/"development-complete.json").read_text())["usablePeople"],199)
            with self.assertRaises(ValueError): amend(amended,models2)
            run(Namespace(output=models2,development=amended,original_registry=root/"original-registry.json",baseline=root/"baseline.json"))
            report=json.loads((models2/"training-summary.json").read_text());selection=json.loads((models2/"selection.json").read_text())
            self.assertEqual((report["fittingPeople"],report["validationPeople"],report["finalRefitPeople"]),(159,40,199))
            self.assertEqual(report["refit"]["fit"]["trainingPeople"],199)
            self.assertNotIn("validation",report["refit"])
            self.assertEqual(report["fullNight84Validation"]["fit"]["trainingPeople"],83)
            self.assertNotIn(bad,selection["trainingSubjects"])
            self.assertIn(bad,selection["previouslyObservedSubjects"])
            self.assertEqual(final_admission(models2,reservation,people),report["parametersSHA256"])
            self.assertFalse(report["finalTestRead"])
            from train_correction import run as correct
            corrected=root/"correction"
            correct(Namespace(output=corrected,development=amended,previous=models2,baseline=root/"baseline.json",max_seconds=1100))
            corrected_summary=json.loads((corrected/"training-summary.json").read_text())
            corrected_protocol=json.loads((corrected/"protocol.json").read_text())
            self.assertEqual(corrected_summary["status"],"admitted")
            self.assertEqual(corrected_summary["refit"]["fit"]["trainingPeople"],199)
            self.assertEqual(len(corrected_summary["validation"]),6)
            self.assertEqual(len(corrected_protocol["innerFittingSubjects"]),135)
            self.assertEqual(len(corrected_protocol["calibrationSubjects"]),24)
            self.assertFalse(set(corrected_protocol["calibrationSubjects"])&set(corrected_protocol["validationSubjects"]))
            self.assertFalse(set(corrected_protocol["innerFittingSubjects"])&set(corrected_protocol["calibrationSubjects"]))
            self.assertEqual(corrected_protocol["reservedTestSubjects"],reserved)
            self.assertTrue(all(not x["calibration"]["usesValidationPeople"] for x in corrected_summary["validation"].values() if x["calibration"]))
            for checkpoint in corrected_summary["validation"].values():
                self.assertAlmostEqual(checkpoint["fit"]["noiseWeightShare"],checkpoint["fit"]["actualNoiseWeightShare"],places=12)
            self.assertEqual(final_admission(corrected,reservation,people),corrected_summary["parametersSHA256"])
            self.assertFalse(corrected_summary["finalTestRead"])
            with self.assertRaises(ValueError): correct(Namespace(output=corrected,development=amended,previous=models2,baseline=root/"baseline.json",max_seconds=1100))

    def test_frozen_expanded_evaluation_keeps_controls_weights_and_participant_exclusions(self):
        import numpy as np
        from argparse import Namespace
        from causal_model import names
        from evaluate_expanded import run
        from expanded_features import fingerprint
        from features import STAGES
        from improve_model import parameters_digest
        from run_dataset import checksum
        from spectral import SPECTRAL_NAMES
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model_root=root/"models";test=root/"test"
            model_root.mkdir();test.mkdir();(test/"features").mkdir()
            save=lambda path,value:path.write_text(json.dumps(value))
            people=[f"new-{i:03d}" for i in range(30)];known=[f"fitting-{i:03d}" for i in range(200)]
            model={"schemaVersion":2,"modelKind":"trees","modelID":"synthetic-only","dataSource":"synthetic only",
                   "featureProfile":"spectral-v4","rawFeatureNames":SPECTRAL_NAMES,"featureNames":names(10,SPECTRAL_NAMES),
                   "contextEpochs":10,"classes":STAGES,"heldOutSubjects":[],"balancedAccuracy":0,
                   "biases":[1,0,0,0],"trees":[{"classIndex":0,"left":[-1],"right":[-1],"features":[-1],"values":[0]}]}
            digest=parameters_digest(model);save(model_root/"selected-model.json",model)
            save(model_root/"selection.json",{"parametersSHA256":digest,"trainingSubjects":known[:199]})
            save(root/"incumbent.json",{**model,"biases":[0,1,0,0]})
            reference=model_root/"full-night-original-84-weights.json";save(reference,{**model,"biases":[0,0,1,0]})
            save(model_root/"full-night-original-84-report.json",{"modelFileSHA256":checksum(reference),"fit":{"trainingPeople":83}})
            save(model_root/"protocol.json",{"previouslyObservedSubjects":known,"reservedTestSubjects":people,"refitSubjects":known[:199],
                "baselineModelSHA256":checksum(root/"incumbent.json"),"development":{"extractorSHA256":fingerprint(),"noiseManifestSHA256":"synthetic-bank"}})
            files=[{"path":person+".edf","subject":person,"sha256":"a"*64} for person in people]
            plan={"phase":"final","fittingForbidden":True,"roles":{p:"test" for p in people},"reservedTestSubjects":[],
                  "previouslyObservedSubjects":known,"selectedParametersSHA256":digest,"files":files}
            save(test/"development-plan.json",plan);checks={}
            for item in files:
                path=test/"features"/(item["subject"]+".npz")
                np.savez_compressed(path,features=np.zeros((4,58)),augmentedFeatures=np.ones((4,58)),startSeconds=np.arange(4)*30.,
                    stage=np.arange(4,dtype=np.int8),usable=np.ones(4,dtype=bool),augmentedUsable=np.ones(4,dtype=bool),
                    subject=np.asarray(item["subject"]),sourcePath=np.asarray(item["path"]),sourceSHA256=np.asarray(item["sha256"]),
                    noiseRole=np.asarray("test"),recordingDuration=np.asarray(120),noiseManifestSHA256=np.asarray("synthetic-bank"),extractorSHA256=np.asarray(fingerprint()))
                checks[item["path"]]=checksum(path)
            save(test/"feature-checksums.json",checks)
            save(test/"extraction-protocol.json",{"extractorSHA256":fingerprint(),"noiseManifestSHA256":"synthetic-bank","developmentPlanSHA256":checksum(test/"development-plan.json")})
            save(test/"development-complete.json",{"extractorSHA256":fingerprint(),"noiseManifestSHA256":"synthetic-bank",
                "featureChecksumsSHA256":checksum(test/"feature-checksums.json"),"usableEpochs":120})
            args=Namespace(selection=model_root,test=test,baseline=root/"incumbent.json")
            save(model_root/"selected-model.json",{**model,"biases":[0,0,0,1]})
            with self.assertRaises(ValueError): run(args)
            save(model_root/"selected-model.json",model);run(args)
            report=json.loads((model_root/"evaluation.json").read_text())
            self.assertEqual(report["trainingPeople"],199)
            self.assertEqual(report["originalControlTrainingPeople"],83)
            self.assertEqual(report["results"]["expanded199"]["clean"]["confusionMatrix"],[[30,0,0,0]]*4)
            self.assertEqual(report["results"]["incumbent72"]["clean"]["confusionMatrix"],[[0,30,0,0]]*4)
            self.assertEqual(report["results"]["fullNight83"]["clean"]["confusionMatrix"],[[0,0,30,0]]*4)
            self.assertEqual(parameters_digest(json.loads((model_root/"model.json").read_text())),digest)
            self.assertEqual(len(json.loads((model_root/"known-subjects-after-test.json").read_text())),230)
            self.assertEqual(report["pairedParticipantIntervals"]["clean"]["validResamples"],1000)
            with self.assertRaises(ValueError): run(args)

    def test_distinct_recording_names_cannot_hide_a_repeated_clinical_participant(self):
        from expanded_contracts import participant_identities
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,patient in (("001","same-patient"),("002","same-patient")):
                (root/(name+"-night.rml")).write_text(f'<Root><Patient><PatientID>{patient}</PatientID><PatientDBID>{name}</PatientDBID></Patient></Root>')
            with self.assertRaises(ValueError): participant_identities(root,{"001","002"})
            (root/"002-night.rml").write_text('<Root><Patient><PatientID>different</PatientID><PatientDBID>002</PatientDBID></Patient></Root>')
            self.assertEqual(set(participant_identities(root,{"001","002"})),{"001","002"})

    def test_fresh_participants_require_gains_in_both_clean_and_noisy_validation(self):
        import copy
        from expanded_contracts import admitted
        baseline={name:{"macroF1":0.3,"balancedAccuracy":0.4,"recallPerClass":[0.4]*4} for name in ("clean","noisy")}
        improved={name:{"macroF1":0.35,"balancedAccuracy":0.45,"recallPerClass":[0.45]*4} for name in ("clean","noisy")}
        self.assertTrue(admitted(improved,baseline))
        for view,metric in (("noisy","balancedAccuracy"),("noisy","macroF1"),("clean","macroF1")):
            regressed=copy.deepcopy(improved);regressed[view][metric]=0.2
            self.assertFalse(admitted(regressed,baseline))
        collapsed=copy.deepcopy(improved);collapsed["clean"]["recallPerClass"][3]=0
        self.assertFalse(admitted(collapsed,baseline))
        nonfinite=copy.deepcopy(improved);nonfinite["noisy"]["balancedAccuracy"]=float("inf")
        self.assertFalse(admitted(nonfinite,baseline))

    def test_expanded_tree_fitting_uses_training_people_and_not_the_export_probe(self):
        import numpy as np
        from array_trees import probabilities
        from train_expanded import fit_arrays
        labels=np.tile(np.repeat(np.arange(4,dtype=np.int8),20),8)
        people=np.repeat(np.arange(8).astype(str),80)
        inputs=np.column_stack((labels.astype(np.float32),np.repeat(np.arange(8),80)/100)).astype(np.float32)
        a,report=fit_arrays(inputs,labels,people,inputs[:32],rounds=8)
        b,_=fit_arrays(inputs,labels,people,inputs[:32]+1000,rounds=8)
        self.assertEqual(a,b)
        self.assertEqual(report["trainingPeople"],8)
        self.assertGreater(float((probabilities(a,inputs).argmax(axis=1)==labels).mean()),0.75)

    def test_batched_tree_inference_matches_independent_scalar_float32_contract(self):
        import numpy as np
        from array_trees import probabilities
        from causal_model import predict
        leaf=lambda stage,value:{"classIndex":stage,"left":[-1],"right":[-1],"features":[-1],"values":[value]}
        model={"modelKind":"trees","biases":[0.2,-0.1,0,0.3],"trees":[
            {"classIndex":0,"left":[1,-1,-1],"right":[2,-1,-1],"features":[0,-1,-1],"values":[0.5,-0.25,0.75]},
            leaf(2,1e8),leaf(2,1),leaf(2,-1e8)]+[leaf(1,-0.1)]*50}
        values=[[float(np.nextafter(np.float32(0.5),np.float32(0))),0],[0.5-1e-12,0],[0.5,0],[0.5+1e-6,0]]
        np.testing.assert_allclose(probabilities(model,values),[predict(model,row) for row in values],rtol=0,atol=2e-15)
        with self.assertRaises(ValueError): probabilities(model,[[float("inf"),0]])

    def test_final_measurements_are_refused_before_opening_any_observations(self):
        from expanded_data import load
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"development-plan.json").write_text(json.dumps({"phase":"final","fittingForbidden":True,"roles":{"reserved":"test"}}))
            with self.assertRaises(ValueError): load(root)

    def test_waveform_augmentation_has_declared_snr_and_independent_noise_roles(self):
        import numpy as np
        from room_noise import NoiseBank
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_bank(root);bank=NoiseBank(root)
            signal=0.005*np.sin(np.arange(30*8000)*2*np.pi*233/8000)
            augmented,snr=bank.mix(signal,8000,"synthetic",0,"train")
            repeated,_=bank.mix(signal,8000,"synthetic",0,"train")
            independent,_=bank.mix(signal,8000,"synthetic",0,"validation")
            np.testing.assert_array_equal(augmented,repeated)
            self.assertFalse(np.array_equal(augmented,independent))
            measured=10*np.log10(np.mean(signal**2)/np.mean((augmented-signal)**2))
            self.assertAlmostEqual(float(measured),snr,places=9)
            self.assertLessEqual(float(np.max(np.abs(augmented))),0.99)

    def test_noise_content_cannot_cross_roles_or_change_after_licence_audit(self):
        import numpy as np
        from room_noise import NoiseBank
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_bank(root);path=root/"bank.json";original=path.read_text()
            manifest=json.loads(original);manifest["clips"][3]["sha256"]=manifest["clips"][0]["sha256"]
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): NoiseBank(root)
            path.write_text(original);bank=NoiseBank(root)
            np.save(root/"synthetic-0.npy",np.ones(60*16000,dtype=np.float32))
            with self.assertRaises(ValueError): bank.samples(0)

    def test_bedside_source_stops_at_true_recording_end_and_silence_stays_unusable(self):
        import datetime
        import numpy as np
        import pyedflib
        from expanded_features import extract
        from room_noise import NoiseBank
        rate=8000;first=np.where(np.arange(30*rate)//8%2==0,4000,-4000).astype(np.int32)
        mic=np.concatenate((first,np.zeros(30*rate,dtype=np.int32),first*2))
        start=datetime.datetime(2020,1,1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_bank(root/"noise");bank=NoiseBank(root/"noise")
            source=root/"source.edf";labels=root/"labels.rml";output=root/"observations.partial"
            with pyedflib.EdfWriter(str(source),2,file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                writer.setStartdatetime(start)
                for channel,name in enumerate(("Tracheal","Mic")):
                    writer.setSignalHeader(channel,{"label":name,"dimension":"a.u.","sample_frequency":rate,
                        "physical_min":-1,"physical_max":1,"digital_min":-32768,"digital_max":32767,"transducer":"","prefilter":""})
                writer.writeSamples([np.full(len(mic),16000,dtype=np.int32),mic],digital=True)
            labels.write_text(f'<Root><Session><RecordingStart>{start.isoformat()}</RecordingStart><Duration>60</Duration></Session>'
                              '<UserStaging><Stage Type="NonREM2" Start="0"/><Stage Type="REM" Start="30"/></UserStaging></Root>')
            source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            extract({"subject":"synthetic","offsetSeconds":0,"durationSeconds":90,"path":"source.edf","sha256":source_hash},source,labels,output,bank,"train")
            with np.load(output,allow_pickle=False) as observations:
                self.assertEqual(observations["startSeconds"].tolist(),[0,30])
                self.assertEqual(observations["stage"].tolist(),[1,3])
                self.assertEqual(observations["usable"].tolist(),[True,False])
                self.assertEqual(observations["augmentedUsable"].tolist(),[True,False])
                self.assertAlmostEqual(float(observations["features"][0,0]),20*np.log10(4000/32768),places=9)
                self.assertEqual(observations["features"].shape,(2,58))
                self.assertEqual(observations["noiseRole"].item(),"train")
                self.assertEqual(observations["sourceSHA256"].item(),source_hash)


if __name__=="__main__": unittest.main()
