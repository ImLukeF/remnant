# Private spectral v4 experiment

Recorded 2026-11-16 UTC. Experimental; not approved for alarm timing. Keep private.

The 72-person model improved balanced accuracy and macro F1 on a new 12-person cohort.
This establishes an improvement on this experiment, not reliable phone sleep staging or waking.

| Model | Final fitting people | Balanced accuracy | Macro F1 | Light recall | Ordinary accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 14 | 28.4% | 0.158 | 14.4% | 17.6% |
| More people | 40 | 42.8% | 0.154 | 2.7% | 13.5% |
| Causal v3 | 40 | 35.6% | 0.142 | 6.1% | 13.8% |
| Spectral v4 | 72 | 49.0% | 0.312 | 41.3% | 42.6% |

All four models were scored on the same 2,819 usable 30-second observations from 12 new people.
Balanced accuracy is mean recall across awake, light (N1/N2), deep (N3), and REM; always predicting
one stage gives 25%. Light accounts for 81.3% of these observations, so ordinary accuracy
alone is misleading: always predicting light gives 81.3% ordinary accuracy and
0.224 macro F1. No model output was used to select these participants or blocks.

The new model's light precision is 83.6%, and light recall is 41.3%.
Its stage recalls are awake 62.4%, light 41.3%, deep 57.1%, REM 35.2%.
It incorrectly called 40 of 105 deep intervals and 113 of 253 REM intervals light sleep.
These errors are too frequent for model-driven waking. Predictions are uncalibrated research outputs.

## Training and isolation

- 145 checksum-verified source blocks yielded 16,039 usable epochs, about 134 hours, from 72 people.
- Hyperparameters used 40 fitting / 32 validation people. Twelve causal candidates combined
  58 signal features with either current-only or ten-epoch trailing context.
- A first selection based on macro F1 ignored REM on validation and was rejected **before**
  any fresh test was prepared. `development-review/` preserves that decision and its original protocol.
- The revised rule requires an importable model with positive validation recall for every stage,
  then maximizes macro F1 with balanced accuracy and light recall as tie-breakers.
- Candidate 9 selected depth-2 boosted trees, balance power 1, and 160 rounds. Those fixed settings
  were refitted from scratch on all 72 development people, without further early stopping.
  Reported validation metrics belong to the 40-person selection checkpoint, not this final refit.
- Final refit: 640 numerical trees, 10 trailing epochs, 58 raw / 232 contextual inputs.
- Its inference parameters were frozen before selecting or downloading the new 12-person test.
  Testing performed no fitting. Parameters remained unchanged; only test metadata was added.
- All 84 observed people are now excluded from any future fresh final test. Do not tune against
  this cohort. The known-subject registry includes every observed participant.

Frozen parameters SHA256: `b122ccc5531ccb17f5c0f5bdee69cad37d11b83c634dc7a0af831290208304d5`.
Final JSON file SHA256: `01ca9302ce94a703266da2fee3c61c6b9703b9e7422f22eb2cccf7c69d44ccb6`.

## Provenance and checks

Data: [PSG-Audio V3](https://doi.org/10.11922/sciencedb.00345), provider metadata CC BY 4.0,
with [dataset paper](https://doi.org/10.1038/s41597-021-00977-w).
The ambient `Mic` channel was used, never the tracheal microphone. The mirror revision,
source hashes, annotation timing, original licence evidence and fixed participant plans are retained
in the manifests. Dataset terms are separate from the planned future MIT code release.
Raw audio, RML annotations, per-epoch clinical features and the native verification input are not
committed here. Derived observations remain in the private research workspace for later experiments.

- Exact selection/refit source: `99514ec6433854a06375ed6c4c0cbfeb1b18af40`.
- Exact final-evaluation source: `c551b884f743163baab5133dbc443d3e185f2355`; the intervening change added the handoff record only.
- Source receipts record passing jobs, clean source and exact dependency arguments.
- Independently implemented Python/Swift spectral features agree at 8 and 48 kHz on deterministic
  synthetic audio, across streaming buffer boundaries. Synthetic tests establish math, not clinical accuracy.
- XGBoost and the portable export were compared during fitting. Swift matched the final portable
  model on 120 chronological fresh-test observations: maximum probability difference
  `1.11e-16`.
- Focused continuity, malformed model, context isolation, bounded prefetch and frozen-round refit
  contracts passed. The unsigned iOS Simulator build passed; native sources have not changed since.

## Limits

Only 12 new clinical participants were tested, with two temporal recording blocks per person.
Deep and REM observations each came from five people. These are usable, stably scored intervals;
transition-crossing and invalid-audio windows are excluded. These results are not whole-night or
prospective home validation. This experiment does not isolate the effects of richer features,
causal context and the larger final fitting cohort.

No real-phone overnight, TV-noise, battery, latency, wearable-reference or alarm hardware trial has
been performed. No microphone session was started. Parley requires explicit start, keeps observations
on the phone, and does not let any imported model control alarm timing. Both code repositories remain
private; MIT publication is deferred until a working version is established.
