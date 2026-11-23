# Private causal paired-noise experiment

Completed 17 November 2026 in 315.9 seconds, source 79128488b0a9b01fcb47172bc4e452ab8d5ad0d0.
Actual fitting used 135 people, checkpoint selection 24 separate inner people, and frozen
checkpoints were measured on the same 40 validation people. Both fixed GRU recipes recognized
all four stages under clean and added-noise conditions. Both lost clean macro F1 and failed
the unchanged paired admission rule. No refit, final-test read or alarm approval occurred.

Recipe 0 used paired Jensen-Shannon consistency; recipe 1 was its zero-consistency control.
Each selected epoch 2 on minimum absolute clean/noisy inner macro F1. This favored added-noise
F1 even while later epochs improved clean inner scores. Relative four-metric checkpoint
selection and a lower supervised noise fraction are tested separately as v11; v10 is frozen.

verified-result.json records independent confusion-matrix metric recomputation, model hashes
and exact-source receipt verification. The independent batched GRU matches PyTorch within
2e-10; synthetic proofs are not clinical accuracy. Private JSON weights and aggregate reports
are retained here. Raw audio, clinical labels and derived observations remain outside Git.
The incumbent is unchanged. No app, microphone or phone use.

The separate fixed-wake-signal review found zero qualifying adjacent epoch pairs across all
40 validation people for both new GRUs and the incumbent, under both clean and added noise.
The actual app rule requires two contiguous light predictions each at least .85 confidence.
These are class-weighted model outputs, without probability calibration. Consequently even
better aggregate stage scores would not produce useful early wake attempts with this rule.
The threshold is not lowered or enabled. A separate diagnostic fits light-probability
calibration only on the 24 inner people and measures it on the 40 validation people.

Separate person-balanced monotonic light-probability calibration used only the 24 inner
people. On the 40 reused validation people it produced 1,691/1,790 clean candidate pairs
across 23/24 people, with 90.1%/90.7% current-epoch light labels. Added noise fell to
83.9%/83.3%, compared with 83.75% light prevalence across available validation epochs.
Noise Brier error was worse than an inner-prior constant predictor. Clean candidates still
included 143 deep/REM epochs per model. This diagnostic does not demonstrate useful
stage-aware waking and is not integrated or approved. Calibration cannot transfer to a
newly refitted backbone without separate validation. All 30 final people remain untouched.
