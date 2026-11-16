# Private expanded-cohort experiment

Research only. Public release and licensing remain deferred. Neither checkpoint is approved for alarm timing.

## Result

More training people did **not** improve this fixed model on the shared nine-person test set:

| Metric on the same new test people | Original: 14 training people | Expanded: 40 training people |
| --- | ---: | ---: |
| Balanced accuracy | 38.7% | 31.3% |
| Ordinary accuracy | 26.8% | 10.4% |
| Light-sleep recall | 21.0% | 3.5% |

Balanced accuracy declined by 7.4 percentage points. This is one small test cohort, not proof that more
data generally hurts. It shows that adding people alone did not repair this particular representation
and training procedure. Neither candidate meets the intended light-sleep waking goal. Keep both as
research checkpoints; do not promote the expanded checkpoint to app alarm control.

The expanded model used 8,631 training epochs, 2,666 validation epochs and 2,037 test epochs, each 30 seconds.
Quality filtering rejected 331, 15 and 22 epochs respectively. All 60 planned people retained usable audio.
The test set contains awake observations from seven people, light from nine, deep from five and REM from four.
Validation REM coverage comes from only one person and 26 epochs; it is especially weak evidence.

The JSON export matched Swift probabilities on 20 held-out epochs to within `1.12e-16`. This confirms
runtime export parity, not sleep-stage accuracy. Receipts bind training to committed source
`c3aabf0a7b0ccf02aff53983551f9925e2b31df6`; `verification.json` records consistency, isolation and hashes.
The feature schema and original optimizer are unchanged. A slow download required one source-only
retry/resume fix; the frozen cohort and model procedure were preserved.

## Protocol

This experiment asks whether more people improve the original seven-feature linear model. It keeps the
features, class weighting, optimizer, seed 7 and 40 training epochs fixed. There is no hyperparameter search,
checkpoint selection or tuning against the final test set.

The deterministic cohort expands from 20 to 60 people, one night per person and two temporal audio blocks
per night. The original 14 training people remain in training; 26 new people bring training to 40. The six
previously inspected test people become development/validation people, joined by five new people. The final
nine test people are all new to both models. All nights from one person share a single assignment.

The original frozen checkpoint and the expanded checkpoint are evaluated on the **same nine fresh test
people**. The original result on the earlier six-person test set is a different cohort and is not a fair
before/after comparator for this experiment. Only training people contribute to normalization and fitting.

Before fitting, the pipeline verifies the original dataset licence, pinned source hashes, EDF/RML start
timestamps and scored coverage for every split. One additional block supplies missing REM coverage in the
validation split. This is an explicit annotation-based coverage adjustment, made before any prediction or
fitting. The plan contains 121 blocks; 41 compatible previously processed blocks are reused and 80 are new.
The sample is not a full-night natural-prevalence study.

The source is PSG-Audio V3, Georgia Korompili et al., DOI https://doi.org/10.11922/sciencedb.00345,
https://www.scidb.cn/en/detail?dataSetId=778740145531650048, under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/. Full creator attribution is retained in
[the original checkpoint](../baseline-v1/README.md). Source files come from the unofficial mirror
`Samuelsantos777/psg-audio-v3-unofficial-mirror` at `cb22565b02c6161be30d791ad72847ec8bd0de5a`.
Mirror hashes pin the downloaded content; they are not an independent byte comparison with the original
provider. Authors do not endorse this model.

Changes to the source data: select participants/time blocks, use only the ambient `Mic` channel, align to
scored stages, merge N1/N2, skip unknown labels and transitions, normalize the declared digital range,
extract seven measurements per 30-second epoch and fit a classifier. No raw recordings, source clinical
annotations or per-epoch observations are committed here. Derived feature files are retained privately on
the VPS with checksums. EDF downloads are removed after extraction.

## Limits

Neither more data nor a correct model export establishes useful phone accuracy. This remains a simple
clinical bedside-audio model, with correlated epochs from a small test cohort. Per-class participant counts
matter; many epochs from one person do not substitute for independent people. Phone placement, microphone
gain/processing, TV, room noise, capture gaps, battery and prospective overnight behavior remain untested.
Softmax outputs are uncalibrated. Keep independently scheduled wake attempts and keep smart waking disabled.
Further model selection must use development data and a new untouched test cohort for a final claim.
This nine-person final test has now been used. Replaying `expand_cohort.py` reproduces this experiment;
increasing its subject count does not automatically create a new independent final test. Future experiments
must explicitly reserve people who have never been evaluated before.
