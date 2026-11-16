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
