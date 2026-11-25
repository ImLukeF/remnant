"""Home-night labels and the fine-tune's starting point; no accuracy claims from synthetic data."""
import unittest

from train_home import APPLE_EPOCH, home_night, network_from


def night(epochs, markers, feedback):
    return {"night": {"id": "0f8fad5b-d9cb-469f-a165-70867728950e", "markers": markers, "feedback": feedback},
            "epochs": [{"start": s, "end": s + 30, "usable": True, "features": [0.0] * 58} for s in epochs]}


class HomeLabels(unittest.TestCase):
    def test_awake_times_and_the_likely_asleep_window(self):
        start = 1_000_000.0  # Swift seconds; 10 hours of 30-second epochs.
        epochs = [start + 30 * i for i in range(1200)]
        at = lambda hours: start + hours * 3600
        body = night(epochs, [{"date": at(0.5), "kind": "tryingToSleep"}, {"date": at(4), "kind": "awake"}],
                     {"fellAsleepAt": at(1), "wokeAt": at(9), "wakeDifficulty": 4})
        n = home_night(body)
        row = lambda hours: int(hours * 120)
        self.assertEqual(n["id"], "0F8FAD5B-D9CB-469F-A165-70867728950E")
        self.assertAlmostEqual(n["started"], start + APPLE_EPOCH)
        self.assertTrue(n["awake"][row(0.2)])            # before "Trying to sleep"
        self.assertFalse(n["awake"][row(0.95)])          # just before falling asleep is unsure, not awake
        self.assertTrue(n["awake"][row(4)])              # the "I'm awake" tap
        self.assertFalse(n["awake"][row(6)])
        self.assertTrue(n["awake"][row(9.1)])            # after waking
        self.assertTrue(n["asleep"][row(6)])
        self.assertFalse(n["asleep"][row(4.1)])          # near a time Luke was awake
        self.assertFalse(n["asleep"][row(1.1)])          # the first 15 minutes after falling asleep
        self.assertTrue(n["beforeWaking"][row(8.95)])
        self.assertFalse(n["beforeWaking"][row(8.5)])
        self.assertFalse((n["awake"] & n["asleep"]).any())

    def test_unusable_epochs_carry_no_labels(self):
        body = night([0.0, 30.0], [{"date": 15.0, "kind": "awake"}], {})
        body["epochs"][0]["features"] = [float("nan")] * 58
        n = home_night(body)
        self.assertFalse(n["usable"][0]); self.assertFalse(n["awake"][0]); self.assertTrue(n["awake"][1])


class StartingPoint(unittest.TestCase):
    def test_fine_tune_starts_from_the_current_model_exactly(self):
        import numpy as np
        import torch
        from train_sequence import export, make_network
        torch.manual_seed(3)
        model = export(make_network(4), {"history": 10, "hidden": 4}, np.zeros(58), np.ones(58), [])
        again = export(network_from(model), {"history": 10, "hidden": 4}, np.zeros(58), np.ones(58), [])
        for key in ("inputWeights", "recurrentWeights", "inputBiases", "recurrentBiases", "outputWeights", "outputBiases"):
            self.assertEqual(model[key], again[key])


if __name__ == "__main__":
    unittest.main()
