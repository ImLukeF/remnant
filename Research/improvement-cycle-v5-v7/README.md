Private improvement cycle, 17 November 2026.

24 new configurations were tested using three fixed participant folds across all 84 previously observed people: 4 summary GRUs, 6 reweighted GRUs, 4 convolutional-audio GRUs and 10 spectral-tree variants. None improved both paired mean macro F1 and balanced accuracy while recognizing all stages in every fold. The existing spectral-v4 model was retained, and the next independent cohort remains untouched.

The paired development benchmark is macro F1 0.3062 and balanced accuracy 42.64%. The highest new balanced accuracy was 42.16%, with macro F1 0.2742. The highest macro F1 was 0.3631, with balanced accuracy 39.13%. comparison.html shows every configuration, not just its best metric. These development scores use a different cohort and protocol from the prior 12-person spectral-v4 final test; they must not be compared directly with its 49.0% score.

This work added a verified 240x25 time-resolved acoustic representation, a 157.15-hour reusable clinical cache, causal GRU/CNN native inference, independent training/export math checks and stricter test-participant provenance. The data cache stays private on the VPS; raw recordings were deleted. No live phone audio was collected.

Focused Python/Swift contracts and the iOS audio-library cross-compile pass. No real-phone overnight, TV-noise, battery or native-alarm hardware acceptance was performed. Model-derived light-sleep waking remains disabled. Both repositories remain private; no MIT/public release has occurred.

Per-experiment reports and exact-source receipts are in ../sequence-v5, ../sequence-balanced-v5, ../acoustic-v6 and ../spectral-cv-v7. Existing model and independent-test evidence remain in ../spectral-v4.
