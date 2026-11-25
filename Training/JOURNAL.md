# Training journal

A running record of what we tried, what it cost and how it scored. Newest entries at the bottom.
Times are Melbourne time. Costs are actual Modal charges from `modal billing report`, which counts in UTC days.

## Spend so far

| Date | Run | What it was | Cost (USD) |
| --- | --- | --- | ---: |
| 24–25 Nov | `ears-smoke-test` | Checking the four ears load and run | 0.36 |
| 24–25 Nov | `sleep-teacher-extract` | Preparing the PSG-Audio blocks | 2.26 |
| 24–25 Nov | `sleep-ears` | Four ears over every block, then the probe on each | 17.53 |
| 25 Nov | `sleep-night` | Dasheng probe on home night 1 | 0.34 |
| 25 Nov | `sleep-distill` | Distilling the four ears into one 85M model, incl. timing test | 11.47 |
| | | **Total** | **31.96** |

Earlier work (the v11 phone model and the v12 Whisper teacher) ran on the VPS and Colab and wasn't costed here.

## 24–25 Nov: which ear hears sleep best?

The same small sleep probe on four frozen audio models, scored on 40 PSG-Audio validation people
(best epoch by balanced accuracy, so this is validation, not an independent test).
Chance is 0.25 balanced accuracy.

| Ear | Balanced accuracy | Snore AUC |
| --- | ---: | ---: |
| Dasheng 1.2B | 0.546 | 0.891 |
| All four together | 0.546 | 0.913 |
| Whisper-small (v12 teacher) | 0.522 | – |
| HeAR | 0.512 | 0.896 |
| Gemma 4 audio encoder | 0.488 | 0.901 |
| OPERA-CT | 0.438 | 0.850 |

Takeaway: combining all four adds nothing for stages and a little for snoring.

## 25 Nov: home night 1 against an Apple Watch

The first real-bedroom night (about 6 hours) with an Apple Watch as the reference.
Share of each Watch stage that the model also called that stage:

| Model | Awake | Light | Deep | REM |
| --- | ---: | ---: | ---: | ---: |
| Phone model (v11, 9k numbers) | 45% | 65% | 35% | 0% |
| Dasheng 1.2B + probe | 40% | 100% | 0% | 0% |

Takeaway: both were trained only on sleep-lab audio (mostly people with severe apnoea) and don't carry over
to a quiet bedroom. Dasheng called almost the whole night light sleep. The phone model's 35% deep came
mostly from calling about 100 minutes deep when the Watch said 32. No single sound feature
separated REM on this night (best AUC about 0.6).

## 25 Nov: distilling the four ears into one small one

Plan: Dasheng base (86M) learns to reproduce all four big ears' outputs on the training people,
plus stages and snoring directly. The 40 validation people stay out. Then the same probe is scored on it,
with full and 8-bit weights, and on home night 1. Script: `modal_distill.py`.

- **Timing test (25 Nov, 1:45 pm):** 150 steps on one L4. Data loads (1,014 training blocks, 916 hours,
  40 validation people left out). The student has 85.4M parameters. The copy loss fell from 0.86 to 0.56.
  It settles at about 1.2 s a step, or about 70 minutes per pass over the training audio.
- **Full run started (25 Nov, 2:01 pm):** 4 passes (about 13,700 steps, roughly 4.6 hours), then embedding
  every block, the probe, and home night 1. Code at commit `b2f7885`.
- **Finished (25 Nov, about 7:45 pm):** 13,732 steps at 1.14 s each, then every block embedded and scored.
  Cost $11.47 in total.

| Model | Size | Balanced accuracy | Deep recall | REM recall | Snore AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Dasheng 1.2B (teacher) | 1.2B | 0.546 | – | – | 0.891 |
| All four together (teachers) | – | 0.546 | – | – | 0.913 |
| Student | 85M | 0.490 | 0.07 | 0.30 | 0.906 |
| Student, 8-bit weights | 85M | 0.484 | 0.10 | 0.25 | 0.853* |

\*The 8-bit probe's best stage epoch was its first, before snoring had trained; by its last epoch snore AUC was 0.908,
the same as the full student. So 8-bit costs almost nothing.

Home night 1 with the student: awake 43%, light 99%, deep 0%, REM 0%. It called the whole night light or
awake, the same failure as Dasheng 1.2B.

Takeaway: the student kept the snoring skill of all four ears (0.906, better than Dasheng 1.2B alone) at about
a fourteenth of the size, and survives 8-bit. It lost some staging (0.49 against 0.546), mostly deep sleep.
It doesn't fix the bedroom problem; that still needs Watch-labelled home nights.
