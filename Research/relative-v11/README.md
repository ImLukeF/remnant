# Completed private v11 independent test

The model was refitted on 199 usable people for four fixed epochs, then scored once
on 30 untouched people: 169 verified source blocks, 17,754 usable epochs, 147.95 hours.
All five exact-source training/preparation/extraction/evaluation jobs passed. The final
evaluation source was d15545bdc01cb7c71396d3a1d1295a154fc030b4. Final evaluation completed
17 November 2026 at 11:14:18 UTC. Every final point metric improved over the incumbent;
this research result does not establish reliable phone sleep staging or wake timing.

| Model | Quiet balanced recall | Quiet macro F1 | Added-noise balanced recall | Added-noise macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Previous 72-person model | .42097 | .27429 | .25950 | .09118 |
| Original full-night control, 83 usable people | .47888 | .32846 | .25566 | .06092 |
| New 199-person causal GRU | .49089 | .34724 | .36910 | .30280 |

All models used the same final intervals. The experiment changes architecture and training
as well as people/nights, so the result does not isolate the effect of adding people.
Paired participant-bootstrap 95% change intervals versus the incumbent: quiet balanced
recall [-.00815, .14061], quiet F1 [.01494, .12415], added-noise balanced recall
[.07982, .15047], added-noise F1 [.17873, .24224]. The quiet balanced-recall improvement
is uncertain. Controlled added noise is synthetic, not a real-bedroom TV trial.

The smart alarm remains unapproved. Deep recall is .174 quiet / .136 noisy. Among genuine
deep/REM epochs, .4417 quiet / .6270 noisy were labelled light. These are classification
errors, not measured alarm-firing rates. Only 16 of the 30 people contributed deep sleep;
22 contributed REM. No actual phone overnight, battery, room or wake-comfort validation
occurred. The previous model remains unchanged; no model-driven alarm was enabled.

Selection fitted 135 people, chose epoch 4 on 24 separate inner people, then passed the
unchanged paired admission on 40 reused development-validation people. Fixed class power
.9, supervised noise share .1 and paired consistency .25 were refitted on all 199.
All 230 observed/reserved people remain retired from future independent tests. Never
retune against or rescore this final cohort. The previous inner calibration diagnostic
was not integrated and cannot transfer unchecked to this refitted backbone.

The inference digest remained frozen through final scoring:
19b1438e0534ec62d34e2b0c361062ecf5b76577c30fe24e5a662aa5311665c5.
Final model file SHA256:
e25b45a7265017efeef88c716222467ece9b67a599adba9e0e9a8bdf311643b6.

Native Swift verification matched 120 chronological observations from two final people,
480 probability comparisons, maximum difference 4.44e-16. It imported the genuine final
held-out metadata; the earlier unscored-refit rejection was preserved. This numerical
proof is not clinical or hardware validation. No app or microphone was launched.

Raw recordings, annotations, NPZ/epoch observations and export-check inputs remain outside
Git. Local private observations and the disposable verifier were removed. The controller
and workers exited; the empty raw-cache directory was removed. Useful private feature
and public-source noise caches remain on the VPS. The completion check was disabled.
No PR, deployment or phone use was involved.
