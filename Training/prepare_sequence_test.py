"""Spend a new test cohort only after the recurrent candidate beats the paired development baseline."""
import argparse
import json
import math
from pathlib import Path

from expand_cohort import atomic_json
from prepare_final_test import prepare


def admit(selection):
    if (not all(math.isfinite(selection[key]) for key in ("meanMacroF1","baselineMeanMacroF1","meanBalancedAccuracy","baselineMeanBalancedAccuracy"))
            or selection["meanMacroF1"] <= selection["baselineMeanMacroF1"]
            or selection["meanBalancedAccuracy"] <= selection["baselineMeanBalancedAccuracy"]
            or not selection["everyFoldRecognizesEveryStage"]):
        raise ValueError("The recurrent candidate has not improved both paired development metrics; keep the next final cohort untouched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=12)
    parser.add_argument("--max-download-gib", type=float, default=24)
    parser.add_argument("--known-subjects", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download-workers", type=int, choices=(1,2,3,4), default=4)
    args = parser.parse_args()
    if not 8 <= args.subjects <= 30 or not 0 < args.max_download_gib <= 48:
        parser.error("Bound the independent test cohort and its download size.")
    selection = json.loads((args.selection/"selection.json").read_text())
    admit(selection)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/"development-admission.json", {"macroF1Improved":True,"balancedAccuracyImproved":True,
        "parametersSHA256":selection["parametersSHA256"],"testPredictionsRead":False})
    prepare(args)
