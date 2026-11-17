import unittest
from pathlib import Path
import tempfile
from features import extract_epoch
from train import fit, partition


class TrainingContracts(unittest.TestCase):
    def test_frozen_spectral_evaluation_compares_incumbent_once_and_keeps_all_exclusions(self):
        import json
        from argparse import Namespace
        from causal_model import names
        from evaluate_final import run
        from features import STAGES
        from improve_model import parameters_digest
        from spectral import SPECTRAL_NAMES
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); selected=root/"selected"; test=root/"test"
            selected.mkdir(); test.mkdir(); (test/"features").mkdir()
            def save(path,value): path.write_text(json.dumps(value))
            model={"schemaVersion":2,"modelKind":"trees","modelID":"synthetic-evaluation", "dataSource":"synthetic only",
                   "featureProfile":"spectral-v4","rawFeatureNames":SPECTRAL_NAMES,"featureNames":names(10,SPECTRAL_NAMES),
                   "contextEpochs":10,"classes":STAGES,"heldOutSubjects":[],"balancedAccuracy":0,
                   "biases":[0,0,0,0],"trees":[{"classIndex":0,"left":[-1],"right":[-1],"features":[-1],"values":[0]}]}
            parameters=parameters_digest(model)
            save(selected/"selected-model.json",model)
            save(selected/"selection.json",{"id":0,"parametersSHA256":parameters,"trainingSubjects":["training"],
                "validationSubjects":[],"previouslyObservedSubjects":["training"],"validation":{"kind":"development only"}})
            save(selected/"protocol.json",{"developmentExtractorSHA256":"synthetic-extractor"})
            save(test/"download-plan.json",{"extractorSHA256":"synthetic-extractor","selectedParametersSHA256":parameters,
                "testSubjects":["new-person"],"previouslyObservedSubjects":["training","older-outside-fitting"],
                "fittingForbidden":True,"files":[{"path":"synthetic.edf"}]})
            rows=[{"subject":"new-person","startSeconds":i*30,"stage":stage,"features":[float(i)]*58,"usable":True}
                  for i,stage in enumerate(STAGES)]
            data="".join(json.dumps(row)+"\n" for row in rows)
            (test/"features/synthetic.jsonl").write_text(data); (test/"epochs.jsonl").write_text(data)
            incumbent={**model,"biases":[0,1,0,0]}; save(root/"incumbent.json",incumbent)
            args=Namespace(selection=selected,test=test,baseline14=None,baseline40=None,baseline_causal=None,baseline_spectral=root/"incumbent.json")
            save(selected/"selected-model.json",{**model,"biases":[1,0,0,0]})
            with self.assertRaises(ValueError): run(args)
            self.assertFalse((selected/"evaluation.json").exists())
            save(selected/"selected-model.json",model)
            run(args)
            report=json.loads((selected/"evaluation.json").read_text())
            self.assertEqual(report["freshTest"]["confusionMatrix"],[[1,0,0,0]]*4)
            self.assertEqual(report["comparisons"]["spectralV4"]["confusionMatrix"],[[0,1,0,0]]*4)
            self.assertEqual(parameters_digest(json.loads((selected/"model.json").read_text())),parameters)
            self.assertEqual(json.loads((selected/"known-subjects-after-test.json").read_text()),["new-person","older-outside-fitting","training"])
            with self.assertRaises(ValueError): run(args)

    def test_selection_rejects_a_model_that_never_recognizes_one_stage(self):
        from improve_model import choose_candidate, metrics
        from features import STAGES
        rows = [{"stage": stage, "subject": "validation"} for stage in STAGES for _ in range(25)]
        collapsed = metrics(rows, [[0, 1, 0, 0]] * len(rows))
        weak = [[1 if k == (STAGES.index(row["stage"]) if i % 25 == 0 else (STAGES.index(row["stage"]) + 1) % 4) else 0
                 for k in range(4)] for i, row in enumerate(rows)]
        complete = metrics(rows, weak)
        self.assertGreater(collapsed["macroF1"], complete["macroF1"])
        candidates = [{"id": i, "fitsNativeImporter": True, "validation": score}
                      for i, score in enumerate((collapsed, complete))]
        self.assertEqual(choose_candidate(candidates, "macroF1")["id"], 1)
        with self.assertRaises(ValueError):
            choose_candidate(candidates[:1], "macroF1")

    def test_final_tree_refit_uses_the_frozen_round_count(self):
        try:
            import xgboost
        except ModuleNotFoundError:
            self.skipTest("Install the optional CPU research dependencies for the refit check.")
        from improve_model import boosted
        from features import STAGES
        rows = [{"stage": stage, "modelFeatures": [float(k), float(i % 3)] + [0.0] * 5}
                for i in range(40) for k, stage in enumerate(STAGES)]
        model, detail = boosted(rows, rows[::3], 1, depth=2, balance=0.5, fixed_rounds=3)
        self.assertEqual(detail["rounds"], 3)
        self.assertEqual(len(model["trees"]), 12)
        self.assertLess(detail["nativeExportMaximumDifference"], 2e-6)

    def test_prefetch_bounds_unprocessed_audio_and_resumes_completed_features(self):
        from unittest.mock import patch
        from run_dataset import prefetched_audio
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, features = root / "raw", root / "features"
            raw.mkdir(); features.mkdir()
            plan = [{"path": f"block-{n}.edf"} for n in range(6)]
            (features / "block-1.jsonl").write_text("already extracted")
            def fetch(item, destination):
                destination.write_text(item["path"])
            completed = []
            with patch("run_dataset.download", side_effect=fetch):
                for number, item, source in prefetched_audio(plan, raw, features, 2):
                    self.assertLessEqual(len(list(raw.glob("*.edf"))), 2)
                    self.assertEqual(source.read_text(), item["path"])
                    completed.append(number)
                    source.unlink()
            self.assertEqual(sorted(completed), [1, 3, 4, 5, 6])
            self.assertEqual(list(raw.iterdir()), [])

    def test_causal_context_uses_bounded_past_and_resets_at_gaps_and_people(self):
        import math
        from causal_model import Context, transform
        context = Context(10)
        for i in range(12):
            result = context.append(i * 30, [float(i + 1)] * 7)
        self.assertEqual(result[:7], [12] * 7)
        self.assertEqual(result[7:14], [7.5] * 7)
        self.assertEqual(result[14:21], [math.sqrt(8.25)] * 7)
        self.assertEqual(result[21:], [9] * 7)
        self.assertEqual(context.append(1000, [20] * 7), [20] * 14 + [0] * 14)
        rows = [{"subject": "one", "startSeconds": i * 30, "features": [float(i)] * 7} for i in range(4)]
        before = transform(rows, 10)
        changed = transform(rows[:3] + [{**rows[3], "features": [9000.0] * 7}], 10)
        self.assertEqual(before[:3], changed[:3])
        other = transform(rows + [{"subject": "two", "startSeconds": 120, "features": [50.0] * 7}], 10)
        self.assertEqual(other[-1]["modelFeatures"], [50.0] * 14 + [0] * 14)
        spectral = Context(1, 58).append(0, [0.1234567] * 58)
        self.assertEqual(spectral[:7], [0.1234567] * 7)
        self.assertEqual(spectral[7:], [0.123457] * 51)

    def test_portable_tree_respects_strict_float32_split_and_class_margins(self):
        from causal_model import predict
        model = {"modelKind": "trees", "biases": [0, 0, 0, 0], "trees": [
            {"classIndex": 1, "left": [1, -1, -1], "right": [2, -1, -1],
             "features": [0, -1, -1], "values": [1, 2, -2]}]}
        left, equal = predict(model, [0.5] * 7), predict(model, [1] * 7)
        self.assertGreater(left[1], 0.7)
        self.assertLess(equal[1], 0.05)
        self.assertAlmostEqual(sum(left), 1)

    def test_downloader_recovers_completed_and_slow_partial_files(self):
        import hashlib
        import io
        from unittest.mock import patch
        from run_dataset import download
        payload = b"abcdef"
        item = {"path": "fixture.edf", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        class Response(io.BytesIO):
            def __init__(self, data, status, headers):
                super().__init__(data)
                self.status, self.headers = status, headers
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "fixture.edf"
            partial = target.with_suffix(".part")
            partial.write_bytes(payload)
            with patch("run_dataset.urllib.request.urlopen", side_effect=AssertionError("Completed verified file must not be downloaded again")):
                download(item, target)
            self.assertEqual(target.read_bytes(), payload)
            target.unlink()
            requests = []
            def respond(request, **kwargs):
                requests.append(request)
                return (Response(payload[:3], 200, {}) if len(requests) == 1 else
                        Response(payload[3:], 206, {"Content-Range": "bytes 3-5/6"}))
            with patch("run_dataset.urllib.request.urlopen", side_effect=respond), patch("run_dataset.time.sleep"), patch("run_dataset.time.monotonic", side_effect=[0, 31, 32, 33]):
                download(item, target)
            self.assertEqual(requests[1].get_header("Range"), "bytes=3-")
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_expanded_cohort_keeps_previous_test_people_out_of_final_test(self):
        from expand_cohort import assignments
        names = [str(i) for i in range(100)]
        old_test = next(name for name in names if partition(name) == "test")
        old_train = next(name for name in names if partition(name) == "train")
        new_test = next(name for name in names if partition(name) == "test" and name != old_test)
        roles = assignments([old_test, old_train, new_test], {old_test, old_train})
        self.assertEqual(roles, {old_test: "validation", old_train: "train", new_test: "test"})

    def test_preflight_timing_resolves_year_rollover_and_rejects_unaligned_dates(self):
        import datetime as dt
        from expand_cohort import timing
        header = bytearray(b" " * 256)
        header[168:176], header[176:184] = b"01.01.20", b"00.00.00"
        header[236:244], header[244:252] = b"3600    ", b"1       "
        start = dt.datetime(2019, 12, 31, 23, 30)
        self.assertEqual(timing(header, start), {"offsetSeconds": 1800, "durationSeconds": 3600})
        header[236:244] = b"14      "
        self.assertEqual(timing(header, start)["durationSeconds"], 14)
        header[168:176] = b"02.01.20"
        with self.assertRaises(ValueError):
            timing(header, start)

    def test_repeated_nights_do_not_create_new_training_subjects(self):
        from run_dataset import select_subjects
        files = [{"path": "APNEA_RML/00000001-night-a.rml"},
                 {"path": "APNEA_RML/00000001-night-b.rml"},
                 {"path": "APNEA_RML/00000002-night-a.rml"}]
        selected = select_subjects(files, 10)
        self.assertEqual({item[0] for item in selected}, {"00000001", "00000002"})
        self.assertEqual(len(selected), 2)

    def test_coverage_supplements_cannot_introduce_new_participants(self):
        from run_dataset import supplement_plan
        plan = [{"subject": "one", "recording": "one-night", "path": "APNEA_EDF/one-night/first.edf"}]
        self.assertEqual(supplement_plan(plan, [plan[0]["path"]]), plan)
        with self.assertRaises(ValueError):
            supplement_plan(plan, ["APNEA_EDF/different-night/first.edf"])

    def test_annotation_tracks_cannot_be_silently_combined(self):
        from extract_psg import annotations
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.rml"
            path.write_text('<Root xmlns="urn:rml"><UserStaging><Stage Type="Wake" Start="0"/>'
                            '<Stage Type="NonREM2" Start="30"/></UserStaging></Root>')
            self.assertEqual(annotations(path), [(0, "awake"), (30, "light")])
            path.write_text('<Root><UserStaging><Stage Type="Wake" Start="0"/>'
                            '<Stage Type="Unscored" Start="30"/><Stage Type="REM" Start="60"/></UserStaging></Root>')
            self.assertEqual(annotations(path), [(0, "awake"), (30, None), (60, "rem")])
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
        import json
        from evaluate_final import compare_legacy
        serialized = json.loads(json.dumps(model))
        compared = compare_legacy(serialized, [row for row in rows if row["subject"] == test_subject])
        self.assertEqual(compared["confusionMatrix"], report["confusionMatrix"])
        serialized["classes"].reverse()
        with self.assertRaises(ValueError):
            compare_legacy(serialized, rows)
        with self.assertRaises(ValueError):
            fit(rows[:-1], steps=1)

    def test_explicit_validation_and_test_people_cannot_affect_fitting(self):
        from features import STAGES
        roles = {"training-person": "train", "development-person": "validation", "new-person": "test"}
        rows = [{"subject": subject, "stage": stage, "features": [offset + i] * 7, "usable": True}
                for subject, offset in (("training-person", 0), ("development-person", 100), ("new-person", 1000))
                for i, stage in enumerate(STAGES)]
        first, _ = fit(rows, steps=2, split_by_subject=roles)
        changed = [{**row, "features": [x + 9000 for x in row["features"]]} if roles[row["subject"]] != "train" else row for row in rows]
        second, _ = fit(changed, steps=2, split_by_subject=roles)
        for key in ("means", "scales", "weights", "biases"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(first["heldOutSubjects"], ["new-person"])
        with self.assertRaises(ValueError):
            fit(rows, split_by_subject={"training-person": "train"})


if __name__ == "__main__":
    unittest.main()
