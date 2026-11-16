import unittest
from pathlib import Path
import tempfile
from features import extract_epoch
from train import fit, partition


class TrainingContracts(unittest.TestCase):
    def test_annotation_tracks_cannot_be_silently_combined(self):
        from extract_psg import annotations
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.rml"
            path.write_text('<Root xmlns="urn:rml"><UserStaging><Stage Type="Wake" Start="0"/>'
                            '<Stage Type="NonREM2" Start="30"/></UserStaging></Root>')
            self.assertEqual(annotations(path), [(0, "awake"), (30, "light")])
            path.write_text('<Root><UserStaging><Stage Type="Wake" Start="0"/></UserStaging>'
                            '<UserStaging><Stage Type="REM" Start="30"/></UserStaging></Root>')
            with self.assertRaises(ValueError):
                annotations(path)

    def test_feature_reference_for_known_pcm(self):
        features, usable = extract_epoch([0.5, -0.5] * (8000 * 15), 8000)
        self.assertAlmostEqual(features[0], -6.020599913, places=8)
        self.assertLess(features[1], 1e-10)
        self.assertGreater(features[4], 0.999)
        self.assertTrue(usable)
        self.assertFalse(extract_epoch([0.0] * (8000 * 30), 8000)[1])

    def test_normalization_and_fitting_exclude_held_out_subjects(self):
        from features import STAGES
        names = [str(i) for i in range(100)]
        train_subject = next(name for name in names if partition(name) == "train")
        test_subject = next(name for name in names if partition(name) == "test")
        rows = [{"subject": subject, "stage": stage, "features": [offset + i] * 7, "usable": True}
                for subject, offset in ((train_subject, 0), (test_subject, 1000))
                for i, stage in enumerate(STAGES)]
        model, report = fit(rows, steps=1)
        self.assertEqual(model["means"], [1.5] * 7)
        self.assertEqual(model["heldOutSubjects"], [test_subject])
        self.assertEqual(report["testEpochs"], 4)
        self.assertFalse(report["usesFutureAudio"])
        self.assertFalse(report["approvedForAlarmTiming"])
        with self.assertRaises(ValueError):
            fit(rows[:-1], steps=1)


if __name__ == "__main__":
    unittest.main()
