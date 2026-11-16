"""Focused signal contracts; synthetic input provides no clinical accuracy evidence."""
import unittest
import numpy as np
from spectral import extract_epoch, SPECTRAL_NAMES
from features import extract_epoch as basic_reference


class SpectralContracts(unittest.TestCase):
    def test_relative_spectrum_ignores_gain_and_dc_and_matches_basic_measurements(self):
        rate = 8000
        wave = np.where(np.arange(rate * 30) // 4 % 2 == 0, 0.125, -0.125)
        first, usable = extract_epoch(wave, rate)
        amplified, _ = extract_epoch(wave * 2, rate)
        offset, _ = extract_epoch(wave + 0.25, rate)
        self.assertTrue(usable)
        self.assertEqual(len(first), len(SPECTRAL_NAMES))
        np.testing.assert_allclose(first[7:], amplified[7:], atol=1e-9)
        np.testing.assert_allclose(first[7:], offset[7:], atol=1e-9)
        basic, _ = basic_reference(wave, rate)
        np.testing.assert_allclose(first[:7], basic, atol=1e-9)
        self.assertFalse(extract_epoch(np.full(rate * 30, 0.25), rate)[1])

    def test_reference_fixture_for_streaming_swift_parity(self):
        # Exact binary fractions and a deterministic integer generator avoid waveform generation differences.
        rate, state = 8000, 7
        samples = []
        for i in range(rate * 30):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            value = ((state >> 16) - 32768) / 262144 + (0.125 if i // 4 % 2 == 0 else -0.125)
            samples.append(value)
        features, usable = extract_epoch(samples, rate)
        self.assertTrue(usable)
        self.assertTrue(np.isfinite(features).all())
        if __name__ == "__main__":
            import json
            print(json.dumps({"syntheticSwiftReference": features}), flush=True)


if __name__ == "__main__":
    unittest.main()
