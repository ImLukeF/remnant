"""Focused signal contracts; synthetic input provides no clinical accuracy evidence."""
import unittest
import numpy as np
import sys
from spectral import extract_epoch, SPECTRAL_NAMES
from features import extract_epoch as basic_reference


def synthetic_samples(rate):
    state, samples = 7, []
    for i in range(rate*30):
        state = (1664525*state+1013904223)&0xFFFFFFFF
        samples.append(((state>>16)-32768)/262144+(0.125 if i//(rate//2000)%2==0 else -0.125))
    return samples


class SpectralContracts(unittest.TestCase):
    def test_acoustic_frames_preserve_level_timing_and_relative_gain_invariance(self):
        from acoustic_frames import extract_frames
        rate=8000; index=np.arange(rate*30)
        wave=np.where(index//4%2==0,0.125,-0.125)*(1+index//rate%3)/3
        first=extract_frames(wave,rate); louder=extract_frames(wave*2,rate); dc=extract_frames(wave+0.25,rate)
        self.assertEqual(first.shape,(240,25))
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(np.ptp(first[:,-1]),8)
        np.testing.assert_allclose(first[:,:24],louder[:,:24],atol=1e-4)
        np.testing.assert_allclose(first,dc,atol=1e-4)
        np.testing.assert_allclose(louder[:,-1]-first[:,-1],20*np.log10(2),atol=2e-4)

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
        references = {}
        for rate in (8000, 48000):
            samples = synthetic_samples(rate)
            features, usable = extract_epoch(samples, rate)
            self.assertTrue(usable)
            self.assertTrue(np.isfinite(features).all())
            references[str(rate)] = features
        if __name__ == "__main__":
            import json
            print(json.dumps({"syntheticSwiftReferences": references}), flush=True)


if __name__ == "__main__":
    reference_path=None
    if "--acoustic-reference" in sys.argv:
        index=sys.argv.index("--acoustic-reference")
        reference_path=sys.argv[index+1]; del sys.argv[index:index+2]
    result=unittest.main(exit=False)
    if result.result.wasSuccessful() and reference_path:
        import json
        from pathlib import Path
        from acoustic_frames import extract_frames
        indices=[0,1,7,8,119,120,238,239]
        reference={str(rate):{"indices":indices,"frames":extract_frames(synthetic_samples(rate),rate)[indices].tolist()} for rate in (8000,48000)}
        Path(reference_path).write_text(json.dumps(reference,separators=(",",":"))+"\n")
    sys.exit(not result.result.wasSuccessful())
