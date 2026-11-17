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
