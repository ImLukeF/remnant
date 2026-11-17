"""Focused optional integration proof. Uses only synthetic audio and annotations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pyedflib
from features import extract_epoch


class EDFContracts(unittest.TestCase):
    def test_frame_cache_preserves_verified_bedside_audio_and_rejects_changed_observations(self):
        from argparse import Namespace
        from spectral import extract_epoch as spectral_reference
        from extract_acoustic_psg import run
        rate=8000; index=np.arange(rate*30)
        amplitude=(1+index//rate%4)*2000
        mic=np.where(index//4%2==0,amplitude,-amplitude).astype(np.int32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"source.edf"; previous=root/"previous.jsonl"; output=root/"frames.npz"
            with pyedflib.EdfWriter(str(source),2,file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                for channel,name in enumerate(("Tracheal","Mic")):
                    writer.setSignalHeader(channel,{"label":name,"dimension":"a.u.","sample_frequency":rate,
                        "physical_min":-1,"physical_max":1,"digital_min":-32768,"digital_max":32767,"transducer":"","prefilter":""})
                writer.writeSamples([np.full(rate*30,8192,dtype=np.int32),mic],digital=True)
            features,usable=spectral_reference(mic/32768,rate)
            row={"subject":"synthetic-only","startSeconds":0,"stage":"light","features":features,"usable":usable}
            previous.write_text(json.dumps(row)+"\n")
            args=Namespace(audio_edf=source,previous_features=previous,audio_offset_seconds=0,subject="synthetic-only",output=output)
            run(args)
            with np.load(output,allow_pickle=False) as result:
                self.assertEqual(result["frames"].shape,(1,240,25))
                self.assertEqual(result["features"].shape,(1,58))
                self.assertEqual(result["stage"].tolist(),[1])
                self.assertEqual(result["startSeconds"].tolist(),[0])
                self.assertEqual(result["usable"].tolist(),[True])
                self.assertGreater(np.ptp(result["frames"][0,:,-1]),8)
                verified_frames=result["frames"].copy()
            labels=root/"labels.rml"
            labels.write_text('<Root><UserStaging><Stage Type="NonREM2" Start="0"/></UserStaging></Root>')
            fresh=Namespace(audio_edf=source,previous_features=None,stages_rml=labels,audio_offset_seconds=0,
                            subject="synthetic-only",output=root/"from-independent-labels.npz")
            run(fresh)
            with np.load(fresh.output,allow_pickle=False) as result:
                np.testing.assert_array_equal(result["frames"],verified_frames)
                self.assertEqual(result["stage"].tolist(),[1])
            row["features"][0]+=10; previous.write_text(json.dumps(row)+"\n")
            args.output=root/"must-not-exist.npz"
            with self.assertRaises(ValueError): run(args)
            self.assertFalse(args.output.exists())

    def test_bedside_extraction_preserves_timing_labels_and_reference_features(self):
        rate = 8000
        index = np.arange(rate * 60)
        amplitude = (index // rate % 5 + 1) * 2048
        pcm = np.where(index // 8 % 2 == 0, amplitude, -amplitude).astype(np.int32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, output = root / "audio.edf", root / "stages.rml", root / "epochs.jsonl"
            with pyedflib.EdfWriter(str(source), 1, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
                writer.setSignalHeader(0, {
                    "label": "Microphone", "dimension": "a.u.", "sample_frequency": rate,
                    "physical_min": -1, "physical_max": 1, "digital_min": -32768,
                    "digital_max": 32767, "transducer": "", "prefilter": "",
                })
                writer.writeSamples([pcm], digital=True)
            labels.write_text('<Root><UserStaging><Stage Type="Wake" Start="0"/>'
                              '<Stage Type="NonREM2" Start="30"/></UserStaging></Root>')
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("extract_psg.py")),
                "--audio-edf", str(source), "--stages-rml", str(labels),
                "--subject", "synthetic-only", "--ambient-channel", "Microphone",
                "--audio-offset-seconds", "0", "--output", str(output),
            ], check=True)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["stage"] for row in rows], ["awake", "light"])
            self.assertEqual([row["startSeconds"] for row in rows], [0, 30])
            expected, usable = extract_epoch(pcm[:rate * 30] / 32768, rate)
            self.assertEqual(rows[0]["usable"], usable)
            for actual, reference in zip(rows[0]["features"], expected):
                self.assertAlmostEqual(actual, reference, places=9)


if __name__ == "__main__":
    unittest.main()
