# Private causal model v3

Experimental research only. The app requires an explicit model import. Smart waking remains disabled;
model metadata never grants alarm permission. Public release and MIT licensing remain deferred.

## Result: failed independent generalization

The selected candidate did not improve the four-stage result on 12 wholly new people:

| Metric on the same new people | Original 14-person model | Expanded 40-person model | Causal v3 |
| --- | ---: | ---: | ---: |
| Balanced accuracy | 39.0% | 38.1% | 28.7% |
| Light-sleep recall | 18.3% | 4.6% | 20.8% |

The v3 checkpoint remains a failed research experiment. Its small light-recall gain does not repair the
overall stage result, and no phone or early-wake claim follows. Its final test contained 2,705 usable epochs.
Both development and final Swift exports matched Python probabilities across 120 contiguous observations
to within `5.56e-17`; an unsigned iPhone simulator build passed without launching or capturing audio.
The weak result is therefore not explained by the tested JSON/runtime conversion.

These 12 test people are now retired into development for the next representation experiment. The
next final test must exclude every participant in `known-subjects-after-test.json` (72 people).

## Development protocol

The previous 60 participants are now development data: the existing 40 training people fit parameters,
and the other 20 select candidates. This explicitly retires the earlier nine-person test into development.
No person from those 60 enters the new final test. The final cohort has 12 new participants selected by
hashed identifier, one recording per person and two temporal blocks per recording.

The bounded search declares 12 candidates before fitting: a converged regularized multinomial linear
model (L-BFGS-B, penalties 0.01/0.1), or numerical boosted trees (depth 2/4 and class-balance powers
0.5/1.0), each using either the current observation or ten trailing observations. Tree training is capped
at 300 rounds and stops after 40 rounds without improved validation balanced accuracy. Seed is 7.
Selection ranks validation balanced accuracy, then light-sleep recall, then the lower candidate index.
No final-test files are read by the selection program.

A five-minute causal tree model won development selection: candidate 9, depth 2, full class balancing,
six boosting rounds and 24 trees. The parameters were frozen before final-test preparation. The raw
seven measurements are unchanged. Context adds their trailing means, standard deviations and changes
from the oldest retained observation. Context includes only the current and preceding nine usable epochs,
resets on gaps/participant changes, and does not include scored stages or future audio.

Both older models and the selected model are scored on exactly the same new final-test observations.
The final model's inference-parameter hash must match the hash saved at selection; only held-out IDs
and accuracy metadata can be updated during evaluation. Final evaluation has no fitting step and refuses
to overwrite a recorded result. `known-subjects-after-test.json` records all people already used, for
excluding them from any later independent test.

## Source and implementation

Data: PSG-Audio V3, Georgia Korompili et al., https://doi.org/10.11922/sciencedb.00345,
https://www.scidb.cn/en/detail?dataSetId=778740145531650048, under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/. Full creator attribution is retained in
[the original checkpoint](../baseline-v1/README.md). The unofficial mirror is pinned to
`cb22565b02c6161be30d791ad72847ec8bd0de5a`; checksums verify mirror content, not an independent
byte comparison to the original provider. No author endorsement is implied.

Changes to source: select people/time blocks, verify EDF/RML timestamps, use only ambient `Mic`, merge
N1/N2, skip unknown labels/transitions, filter unusable audio, derive seven measurements, then add causal
context and fit independently written classifiers. One initially selected source tail was only 14 seconds
long; before testing it was replaced with a full verified block from the same person. The manifest records
that quality substitution. No recording, source clinical annotation or per-epoch observation is committed here.

Optional training dependencies are NumPy 2.5.5, SciPy 1.18.1 and XGBoost CPU 3.4.1. Training uses the
libraries' public APIs. Tree parameters are read from XGBoost's saved-model JSON and converted to our own
format; no third-party implementation or pretrained model is copied. The portable Python interpreter is
checked against the training library on all development observations. Swift uses the same Float32 split
and accumulation semantics and is checked with contiguous sequences, including context resets.
References: [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html),
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

## Limits

A small clinical-audio cohort does not establish phone accuracy. Context and flexible models can still
learn room/gain artifacts. Epochs from the same person are correlated, so many epochs do not equal many
independent people. Phone placement/processing, TV, battery, lock-screen capture and live latency remain
unvalidated. Softmax outputs have not been calibrated as confidence estimates. Prospective phone testing
and another untouched cohort are needed before a final deployment claim or a later tuned model's evaluation.
