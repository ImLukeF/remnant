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
| 26 Nov | `sleep-home-check` | Home-night sound clues tested on every lab recording | 0.76 |
| | | **Total** | **32.72** |

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

## 25 Nov: saving the night's audio in more detail

Home night 1 was saved at 16-bit. A quiet bedroom sat at about -75 dBFS, which is only about 6 steps of
loudness out of 32,768, so most of the breathing detail was rounded away when the file was saved.
The app now saves at 24-bit. This changes the saved recording we train on, not the stages shown on the phone.

## 26 Nov: home night 2 against an Apple Watch

The first night saved at 24-bit (about 5.9 hours, 704 half-minutes). Same quiet room (about -74 dBFS), but it now
has about 1,580 steps of loudness to work with instead of 6. A close microphone on the bedhead recorded the
same night for comparison; its battery ran out before morning.

Share of each Watch stage that the phone model also called that stage:

| Model | Awake | Light | Deep | REM |
| --- | ---: | ---: | ---: | ---: |
| Phone model (v11), night 1 | 45% | 65% | 35% | 0% |
| Phone model (v11), night 2 | 19% | 44% | 71% | 0% |

Takeaway: balanced accuracy on night 2 was 0.34, close to chance. The phone model never calls REM, and it called
about 154 minutes deep when the Watch said 49, so the higher deep score is mostly over-calling again.
Two nights now agree: the lab-trained model doesn't carry over to this bedroom.

## 26 Nov: learning from the two home nights

Lined the Watch's stages up with the audio half-minute by half-minute, pulled simple sound features
(loudness, movement, breathing rhythm, tone), then trained on one night and tested on the other.
Ran on a laptop, so no cost. Scripts: `home_night_features.py`, `home_night_train.py`.

| Trained on → tested on | REM AUC | Balanced accuracy | REM found | Phone model's balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Night 1 → night 2 | 0.70 | 0.41 | 2% | 0.34 |
| Night 2 → night 1 | 0.76 | 0.51 | 46% | 0.36 |

Takeaway: this is the first time anything has picked up REM in this bedroom. What gives it away is a long
quiet, still stretch: fewer loud moments and steadier sound over about 20 minutes, the same on both nights.
REM AUC is 0.5 for a guess, and time of night alone scored below 0.5, so it isn't just the clock.
Sliding the Watch's stages out of line beat the real score 7% of the time one way and never the other.

Cautions: two nights from one person, so this is a lead, not a result. Trained on night 1, it ranks REM well
but rarely commits to calling it, so the cut-off doesn't carry over yet. Night 1 was saved at 16-bit;
re-saving night 2 at 16-bit gave almost the same scores, so these coarse features don't depend on the fix.
The Watch also leans on stillness, so part of this is matching how the Watch decides.

## 26 Nov: does a close microphone hear more than the phone?

A wireless clip-on mic on the bedhead recorded the same night as the phone (5 hours overlap, 604 half-minutes).
The two were lined up by matching sounds: the same offset fit every hour to within 0.1 s.

| | Phone | Bedhead mic |
| --- | ---: | ---: |
| Typical level | -75.8 dBFS | -80.6 dBFS |
| Quiet floor to loud moments | 15.2 dB | 13.0 dB |
| Half-minutes with a clear breathing rhythm | 34% | 33% |
| REM AUC, model trained on phone night 1 | 0.75 | 0.60 |

Takeaway: the close mic heard no more than the phone. Its loudness followed the phone's almost exactly
(0.89 match per second), breathing rhythm showed up just as often, and sounds stood out from the background
slightly less. The lower REM score on the mic is at least partly because the model was trained on phone audio.
So the phone's microphone and placement aren't what's holding this back. Couldn't confirm whether the mic's
noise cancelling was off; nothing in the audio suggests it was on.

## 26 Nov: do the home-night clues hold in the sleep lab?

The same simple sound features, pulled from all 230 lab people (129,706 half-minutes) and scored on the
40 validation people. Script: `modal_home_check.py`, code at commit `1c87d97`. Cost $0.76.

| Trained on → tested on | REM AUC | Deep AUC | Awake AUC | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Lab → lab, simple model | 0.66 | 0.74 | 0.88 | 0.52 |
| Lab → lab, trees | 0.72 | 0.71 | 0.90 | 0.46 |
| Home nights → lab | 0.41 | 0.59 | 0.73 | 0.32 |
| Lab → home night 1, simple | 0.43 | 0.71 | 0.78 | 0.41 |
| Lab → home night 2, simple | 0.58 | 0.65 | 0.80 | 0.42 |

Takeaway, in three parts:

- The home REM clue only half holds. In the lab REM also has fewer loud moments over 20 minutes, but weakly
  (single-clue AUC about 0.37 to 0.42 against 0.25 to 0.30 at home), and the "steadier sound" part doesn't show at all.
- REM doesn't carry over either way. A model trained on the two home nights does worse than a guess on lab REM, and a
  lab-trained one is at or near a guess on home REM. Most lab people have severe apnoea, which is worst in REM, so
  their REM is noisy where a healthy sleeper's is quiet.
- Awake and deep do carry over. Trained only on the lab, this simple model scores 0.41 and 0.42 on the home nights,
  against 0.36 and 0.34 for the phone model. In the lab it reaches 0.52, close to the 1.2B model's 0.546 and above
  the 85M student's 0.49. The likely reason is that every feature is measured against that night's own quiet level.

Caution: these features look 20 minutes ahead and need the whole night to set the quiet level, so they suit a
morning summary, not live tracking.
