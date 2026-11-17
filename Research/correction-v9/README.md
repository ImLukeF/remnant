# Private cached correction diagnostic

Completed 17 November 2026 at source c169b1bd7f6e739159149e82482008209fda7748.
Actual fitting used 135 people, bias calibration used 24 separate inner people, and the same
40 reserved validation people measured the frozen checkpoints. All recordings/features came
from the verified expanded-v8 cache; no new clinical audio was downloaded.

Three fixed class/noise weight mixtures were each measured raw and with regularized bias
calibration. Every variant missed REM under both conditions and failed unchanged admission.
Some aggregate F1 scores increased by labelling too much deep/REM audio as light. These
checkpoints are unsuitable for stage-driven waking. No refit, final-test read, app launch,
microphone session, model promotion or alarm approval occurred.

The real run completed in 308.7 seconds. training-receipt.json and proof-receipt.json verify
exact pushed source, exit zero and clean post-run source. Synthetic contract tests establish
implementation behavior, not clinical accuracy. verified-result.json records independent
confusion-matrix metric recomputation and model-file checksum verification.

JSON model parameters, reports and the frozen protocol are private research artifacts.
Raw audio, clinical annotations and derived clinical feature arrays remain outside Git.
Both repositories remain private; the incumbent model remains unchanged. The next bounded
experiment tests past-only paired-noise consistency training rather than changing these results.
