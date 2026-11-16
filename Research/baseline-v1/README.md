# Private audio baseline v1

Research checkpoint only. This is not bundled into the SDK or Parley and is not approved for alarm timing.
Its softmax outputs have not been calibrated as confidence estimates. Public release and licensing are deferred.

The baseline uses seven measurements of the bedside `Mic` channel, 30-second epochs and a four-class linear
classifier (awake, combined N1/N2 light sleep, N3 deep sleep, REM). It has no future audio context. It is a
measurement of what this simple representation can learn, not a claim that volume reveals sleep stages.

## Result

This baseline is too weak for smart waking: **35.3% balanced accuracy** and **20.1% ordinary accuracy**
on the selected held-out observations. It detected just **10.2% of light-sleep epochs**. Keep it disabled
for alarm decisions. These results do not establish phone or overnight accuracy.

| Scored stage | Test epochs | Recall | Test participants with this stage |
| --- | ---: | ---: | ---: |
| Awake | 88 | 36.4% | 6 |
| Light | 1,151 | 10.2% | 6 |
| Deep | 293 | 41.6% | 4 |
| REM | 113 | 53.1% | 2 |

Training used 3,073 epochs from 14 participants; testing used 1,645 epochs from six different participants.
The Swift runtime matched Python probabilities on 20 held-out epochs within `1.67e-16`.
That checks export correctness, not sleep accuracy. See `verification.json` for artifact hashes and checks.

## Data and attribution

PSG-Audio V3, Georgia Korompili, Anastasia Amfilochiou, Lampros Kokkalas, Stelios A. Mitilineos,
Nicolas-Alexander Tatlas, Marios Kouvaras, Emannouil Kastanakis, Chrysoula Maniou and Stelios M. Potirakis.
Original dataset: https://doi.org/10.11922/sciencedb.00345
Original provider: https://www.scidb.cn/en/detail?dataSetId=778740145531650048
Paper: https://doi.org/10.1038/s41597-021-00977-w
Dataset licence: https://creativecommons.org/licenses/by/4.0/
The original provider's structured metadata was checked directly for this licence before downloading.
No endorsement by the dataset authors is implied.

Files came from the unofficial mirror `Samuelsantos777/psg-audio-v3-unofficial-mirror`, pinned to
`cb22565b02c6161be30d791ad72847ec8bd0de5a`. Every downloaded EDF was checked against that mirror's
SHA-256; annotations were checked against their Git blob identifiers. These checks establish pinned mirror
content, not an independent byte-for-byte comparison with the original provider's download.

Changes to the data: select 20 people deterministically by hashed identifier, one night per person and two
preselected temporal blocks per night; use only the ambient microphone; align audio starts using EDF/RML
timestamps; merge N1/N2; skip unscored intervals and epochs crossing transitions; extract measurements and
fit the classifier. No source recordings or per-epoch observations are committed here.

**Coverage adjustment:** the initial 40-block sample had no REM in its test epochs, so the first run
stopped before fitting or prediction. The RML annotations showed REM elsewhere in two of those same
six test participants' nights. Both REM-containing blocks were added explicitly, retaining the original
training sample, people, model and hyperparameters. The final plan has 42 blocks. The initial plan,
failed coverage-check receipt, amended plan and successful exact-source training receipt are retained.
This label-based coverage adjustment must be disclosed alongside these results; it does not represent
a randomly sampled full-night distribution or a fresh external validation cohort.

## Evaluation boundaries

The fixed subject hash split assigns 14 people to training and six to testing. All normalization and fitting
use training people only; the test cohort is not used to choose features, hyperparameters or the checkpoint.
This run uses the predefined 40-epoch training procedure with seed 7 and inverse-frequency class weights.

See `evaluation.json` for the confusion matrix and per-class recall. Balanced accuracy is the mean recall
across all four classes; an always-one-class predictor scores 25%, even when that class is common.
Epochs from the same night are correlated, so thousands of epochs do not equal thousands of independent
participants. The small test cohort does not establish deployment accuracy.

This is clinical bedside recording data, not prospective iPhone data. Phone gain and processing, placement,
room acoustics, TV during the first hour, timing latency, interruptions, background reliability and battery
use remain unvalidated. The iPhone must keep its independently scheduled wake attempts even when audio
capture or the model fails. Further model selection needs a separate validation cohort before another
untouched test evaluation; do not tune repeatedly against this six-person test set.
