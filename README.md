# SleepWakeKit

Experimental, private-by-default sleep research and coordinated wake plans for Swift apps.
Private development. An MIT release is planned once a working version is ready.
No account, API key, paid SDK or network service is required.

## What works

- Streaming mono PCM into bounded-memory 30-second observations. Waveforms are not stored.
- An iOS microphone source with explicit permission/start, lock-screen background audio support,
  interruption handling, and no automatic restart after capture is interrupted.
- A portable linear classifier format, an independently implemented Python feature reference,
  labelled-data extraction and subject-separated baseline training/evaluation.
- Wake plans with precomputed safety attempts, a stop/finish distinction, and a conservative
  early-wake decision engine. Hosts remain responsible for scheduling real system alarms.

## What is experimental

**No trained or validated sleep-stage model is bundled.** Sound level is not a sleep stage.
The observations are signal measurements. The trainable baseline is a research starting point;
its simple features may be insufficient to separate light, deep and REM sleep reliably.
Quiet rooms, TV sound, other sleepers, microphone placement and device processing all need evaluation.

An imported model does not become trusted by declaring itself accurate. The wake decision engine's
separate modelApprovedForAlarmTiming argument defaults to false. Hosts must validate a particular
model and explicitly approve it before allowing early waking. Parley's research preview does not
approve imported models for alarm timing.

## Swift integration

Add this directory as a Swift package, or use its standalone Git repository once published.
The SleepWakeKit product contains the platform-independent engine.
The SleepWakeAudio product adds the iOS microphone source.

Create a WakePlan(upBy:attemptCount:spacing:earlyAllowance:), then schedule **every**
plan.attempts entry using the operating system before starting audio observation.
Persist the occurrence and actual scheduled IDs. A failed partial schedule must not be shown as armed.

Feed normalized mono float PCM to AudioFeatureExtractor. Sample rates must be integral, between
8 kHz and 192 kHz. Reinitialize the extractor at discontinuities; do not join samples across gaps.
Use the same processing and sample scaling in training and deployment.

On iOS, retain an OvernightAudio instance, explicitly call start, and consume its event stream.
Add a microphone usage explanation and the audio background mode to the host app. Call stop
when done. Use the canStart callback to check your app's other microphone owners after permission
has been granted. Calls, route changes and audio interruptions end capture; the host must show
that interruption rather than silently assume uninterrupted sleep.

WakeSession.stopped records a stopped attempt without removing backups. Its finish method returns IDs
for the host to cancel. Persist intent and reconcile failed cancellations; calling a reducer
alone does not cancel any system alarm. An early-wake suggestion is issued only once and never
replaces or postpones the safety sequence.

## Train an owned model

Use independently scored bedroom audio, with permission and appropriate dataset rights.
PSG-Audio is one research starting point: https://doi.org/10.1038/s41597-021-00977-w
and https://doi.org/10.11922/sciencedb.00345. Dataset terms are separate from the planned code licence.
No dataset, recordings or external model weights are included or downloaded automatically.

The explicit `Training/run_dataset.py` command verifies the original provider's CC BY 4.0
metadata, pins a source mirror revision, checks source hashes and derives offsets from EDF/RML
timestamps. Its default run selects 20 people and two temporal blocks per person before reading
labels, caps planned downloads at 32 GiB, and deletes each EDF after extraction. Run it only in a
private research workspace; derived features and annotation caches are also sensitive research data.

    python3 Training/run_dataset.py --output /private/research/baseline-v1

The optional Training/extract_psg.py tool requires pyedflib. Supply an audio EDF, one RML
UserStaging sequence, a stable subject identifier, the exact **ambient** microphone channel label,
and an explicitly verified audio/PSG offset. It reads bounded 30-second windows, normalizes using
the EDF digital range, and skips epochs crossing an annotated stage transition.

Run python3 Training/extract_psg.py --help for its required arguments. Combine the resulting
JSONL rows in a private local file. Do not use a tracheal/contact microphone to claim bedside-phone accuracy.

Train with:

    python3 Training/train.py data/epochs.jsonl --output models/baseline.json \
      --model-id baseline-v1 --data-source "dataset/version/licence"

The SHA-256 subject split is fixed before fitting. Normalization and all optimization use training
subjects only. The held-out set must contain all four stages. The tool exports a compatible JSON
model and a separate evaluation report with the confusion matrix, per-class recall and balanced
accuracy. It never marks a model as approved for alarm timing. Do not tune parameters repeatedly
against this test set; use a separate validation cohort for model selection.

Check live epoch end timestamps against callback receipt times. Offline accuracy does not establish
live latency or accuracy. TV-noise testing, held-out phone/room evaluation and prospective overnight
trials remain necessary. Consumer Watch stages can be a comparison, not clinical ground truth.

## Focused checks

    swift test --filter SleepWakeKitTests --jobs 2
    cd Training && python3 -m unittest -v test_training

These checks prove feature math, buffer-boundary invariance, input validation, subject isolation and
wake-policy contracts. Synthetic fixtures do not demonstrate sleep-stage accuracy. iOS microphone,
background operation, battery use and native alarm actions require real-device verification.

## Privacy and reuse

The package performs no network requests and does not write recordings. The host chooses whether
to retain derived observations and must provide deletion and explicit recording controls.
The core, Apple audio source and training tools were independently written; no proprietary SDK,
research repository code or third-party model was copied. Contributions should include provenance
and subject-separated results for any proposed model or dataset.
