# Private relative-checkpoint GRU experiment

The frozen 135-person backbone selected epoch 4 using 24 separate inner people and
improved all four admission metrics on the same 40 development validation people.
It uses class power .9, supervised noise share .1 and paired consistency .25. The
second declared recipe was not run after the first passed admission.

Clean macro F1 .31505 / balanced recall .44367 versus incumbent .27693 / .44187;
added-noise F1 .20584 / balanced recall .37065 versus .07638 / .25805. Every stage
was recognized, but noisy REM recall is only .035 and the clean balanced-recall gain
is .18 percentage points. Repeated development validation is not independent proof.

The fixed four-epoch refit on all 199 usable people passed at source
d15545bdc01cb7c71396d3a1d1295a154fc030b4 in job 5c1c9fc3493348b788e0d58c1ea86c2e.
The untested refit has no held-out accuracy; final preparation is running. The exact original
30 final-test people remain conditional on frozen refit completion and final_admission.
The bounded final controller attaches the existing refit job; it never starts another
training worker. final-pipeline-request.json pins every source and bounded command.

Separate light-probability calibration used only the inner 24 people. Clean candidates
contained 341 deep/REM epochs among 4,061 candidate pairs; noisy candidates contained
120 among 1,225. Calibration does not establish useful stage-aware waking, cannot
transfer unchecked to a different backbone and is not integrated or approved.

All raw audio, annotations and derived clinical observations stay outside Git. These
private files contain parameters, protocols and aggregate evidence. Neither repository
is public or MIT-released. No app, microphone, phone or alarm approval occurred.

The native verifier compiled successfully. Importing the unscored refit was rejected
by the existing nonempty held-out metadata guard. That guard and the model were not
changed to force a pass. Native parity will use model.json with genuine final-test
metadata after once-only evaluation. Private export-check observations remain outside Git.
