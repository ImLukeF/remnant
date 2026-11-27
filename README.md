# SleepWakeKit

SleepWakeKit is an experimental Swift package for on-device bedroom-audio measurement and wake-plan logic. It turns
normalized mono PCM into 30-second observations, imports causal research classifiers, stores nights, and can reread a
saved night with a separate whole-night model. `Training/` contains the Python tools used to extract data, train models
and evaluate experiments.

The Swift products make no network requests and need no account, API key, paid SDK or third-party Swift package. The
optional Python tools are separate from package runtime: some download explicitly named public datasets or model
weights, and some use Colab, Modal or a VPS, only when run.

**Still in development. No included or imported model is clinically validated or approved for alarm timing.**
Sound level is not a sleep stage. Licensed under [Apache 2.0](LICENSE).

## What is included

- `SleepWakeKit`: streaming features, portable model import/inference, wake plans, night storage/ZIP export and the
  built-in morning model.
- `SleepWakeAudio`: the iOS microphone source, optional local recording and the finished-night reader.
- `Training/`: dependency-free baseline tools plus optional NumPy/SciPy/XGBoost/PyTorch research pipelines.
- `Research/`: experiment protocols, reports, receipts, model parameters and rejected candidates. Raw recordings,
  clinical annotations and derived clinical epoch arrays are not included, so these artifacts do not make the
  experiments reproducible from this repository alone.

There is no standalone app, server, hosted service, account tier or pricing in this repository. A host application is
responsible for permissions, audio sessions, system alarms, storage policy and the user interface.

## Requirements and build

- Swift tools 6.0.
- Package platforms: iOS 18 and macOS 15 (`Package.swift`). The `OvernightAudio` microphone API is iOS-only; the core
  engine and storage/archive code also build on macOS.
- No external Swift dependencies.

```sh
swift build
swift test
```

This is a library package, not an executable application; there is no `swift run` entry point. The Python baseline
trainer uses the standard library. Extraction needs NumPy and pyedflib; model selection needs SciPy and XGBoost; neural
research needs the pinned CPU-only PyTorch environment described below.

## Swift integration

### Wake plans and alarms

Create a `WakePlan(upBy:attemptCount:spacing:earlyAllowance:)`. Defaults are three attempts, five-minute spacing and a
30-minute early-wake allowance. Valid ranges are 1–12 attempts, 60–1,800 seconds between attempts and 0–7,200 seconds
of early allowance.

Schedule **every** `plan.attempts` entry with the operating system before starting audio observation. Persist the
occurrence and actual scheduled IDs. A failed partial schedule must not be shown as armed. The package computes plans;
it does not schedule, cancel or ring system alarms. Persist cancellation intent and reconcile failed cancellations;
mutating a session or reducer alone does not cancel an alarm.

`WakeSession.stopped(_:)` records a stopped attempt without removing backups. `finish(at:)` returns the plan and early
attempt IDs for the host to cancel and is idempotent. `WakeCompletion.actions(for:now:)` distinguishes stopping today's
occurrence from cancelling a one-off alarm or skipping a repeating alarm's next occurrence.

### Audio observations

Feed finite, normalized mono float PCM in `[-1, 1]` to `AudioFeatureExtractor`. Sample rates must be integral and between
8 kHz and 192 kHz. Reinitialize the extractor at discontinuities; do not join samples across gaps. Use the same sample
scaling and processing in training and deployment.

The basic profile emits seven measurements per 30-second epoch. `AudioFeatureExtractor(sampleRate:start:spectral: true)`
or `OvernightAudio(spectral: true)` enables the experimental `spectral-v4` profile: 58 values comprising the original
seven, gain-normalized Mel-band means/spreads, entropy and spectral change. It uses eight short frames per second and
retains at most one second of waveform. DC-only audio is unusable. Live extraction discards waveforms; local recording
is a separate host-controlled output.

`acousticFrames: true` additionally retains 240 frames of 24 relative Mel bands plus band power per epoch. These 6,000
Float32 values preserve within-epoch timing without retaining waveforms; they are rounded to four decimal places.
Models that need this profile reject epochs without it. Non-finite, out-of-range, silent/DC or heavily clipped audio
cannot become a usable observation; the iOS worker also ends capture on timeline discontinuities.

### Model import and early waking

Use `SleepModel(data:)` to import caller-supplied JSON, then retain one `makePredictor()` result per observation stream.
Feed epochs in chronological order with the mutating `predict` method and call `reset()` at session boundaries. Gaps or
unusable epochs clear context; unusable epochs return no stage probabilities. The importer accepts schemas 1–4 and
rejects payloads over 512,000 bytes, unsupported schemas, non-finite parameters and invalid tree graphs.

- Schema 1 is the legacy seven-feature linear model.
- Schema 2 supports causal linear/tree models with the current epoch alone or a window of up to ten epochs including
  the current one.
- Schema 3 adds a bounded GRU over windows of up to 1, 10 or 20 epochs including the current one.
- Schema 4 encodes each completed epoch with temporal convolutions, then applies a bounded GRU to the embeddings. It
  requires the acoustic-frame profile.

An epoch is classified only after its 30 seconds complete. Only current and past observations are read. Additional
spectral inputs are rounded to six decimal places before context summaries, matching the Python reference. The
recurrent equations follow the
[PyTorch 2.10 GRU convention](https://docs.pytorch.org/docs/2.10/generated/torch.nn.GRU.html), and convolutions follow
its [cross-correlation convention](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Conv1d.html). Native inference
is independently implemented in Swift with no PyTorch runtime dependency.

Imported metadata never grants alarm permission. `WakeSession.consider(_:now:modelApprovedForAlarmTiming:)` defaults to
`false`. Even when a host explicitly approves a validated model, an early attempt requires two contiguous usable light
predictions, each with at least 0.85 confidence, ending no more than 60 seconds before `now`, inside the allowed window
and before the first safety attempt. It returns one suggestion two seconds later; it never replaces or postpones the
safety sequence. Model softmax outputs are not calibrated confidence estimates.

### iOS capture

Retain an `OvernightAudio` instance, explicitly call `start`, and consume its event stream. Add a microphone usage
explanation and the audio background mode to the host app. Use `canStart` to check the app's other microphone owners
after permission is granted. The tap copies PCM; conversion, file writing and feature extraction run on a worker, and a
queue/timeline failure ends capture.

`stop()` stops input synchronously for microphone handoff. `await finish()` also drains queued PCM, writes final epochs
and closes any recording before export. Interruption notifications and route/session changes away from the built-in
microphone, measurement `playAndRecord` mode or expected sample rate end capture. The host must surface the
interruption; capture never restarts automatically. Calls, route changes, lock-screen operation, battery use and real
alarm actions still require device testing.

### Night storage and the morning reread

`SleepNightStore` keeps separate nights, atomic per-epoch observation files, markers, personal feedback and optional
host-supplied Health references with their original source. It does not request HealthKit access or read Health data.
The store expects 58-feature spectral observations and optionally 6,000 acoustic-frame values, with at most 1,440
observations per night. Pass `modelData` to retain the exact imported model bytes; without it, only the declared model
ID is stored. Model attribution can be stored separately.

ZIP export streams files and requires a separate `includeAudio` decision. Interrupted manifests can recover observations
written before a process kill. Night JSON declares its date encoding
(`secondsSince2001-01-01T00:00:00Z`). Predictions and personal feedback are not clinical labels.

After a recorded night ends, `MorningReading.read(audio:night:model:)` can reread its 16-kHz mono audio with
`MorningModel.shipped` (`morning-v1`) and save a second stage sequence in `SleepNight.Morning`. It returns `nil` when
there are fewer than 60 saved/covered epochs or the audio format is incompatible. Unlike live inference, this pass
measures each epoch against the whole night's quiet level and uses windows up to 20 minutes either side, so it is not
causal and is not bounded like the live 30-second stream. Its runtime and memory use depend on night length and device.

The built-in morning coefficients are experimental. [Training/JOURNAL.md](Training/JOURNAL.md) records that the shipped
model was trained on lab participants plus both available home nights; scores on those two nights are therefore not
independent validation. A new home night and prospective overnight evaluation are still needed.

## Train an owned model

Use independently scored bedroom audio with permission and appropriate dataset rights. PSG-Audio is one research
starting point: <https://doi.org/10.1038/s41597-021-00977-w> and <https://doi.org/10.11922/sciencedb.00345>. Dataset terms
are separate from the code licence. No dataset, recording or external pretrained-model weights are vendored or
downloaded automatically.

### Baseline extraction and training

`Training/run_dataset.py` verifies the provider's declared CC BY 4.0 metadata, pins the mirror revision, checks source
hashes and derives offsets from EDF/RML timestamps. Its initial default selects 20 people and two temporal blocks per
person before reading labels, caps planned downloads at 32 GiB, and removes each EDF after successful extraction.

```sh
python3 Training/run_dataset.py --output /private/research/baseline-v1
```

The initial temporal plan may lack a stage in a split; extraction/training then stops rather than fitting an invalid
comparison. The archived baseline required documented `--supplement` blocks from already selected participants and a
`--supplement-reason` before fitting. Inspect `download-plan.json` and disclose any annotation-based coverage amendment.
Run only in a private research workspace: derived features and annotation caches are sensitive too.

`Training/extract_psg.py` needs NumPy and pyedflib. Supply an audio EDF, one RML `UserStaging` sequence, a stable
subject ID, the exact **ambient** microphone channel label and an independently verified audio/PSG offset. It reads
bounded 30-second windows, normalizes by the EDF digital range and skips epochs crossing a stage transition. Run
`python3 Training/extract_psg.py --help` for required arguments. Do not use a tracheal/contact microphone to claim
bedside-phone accuracy.

Combine the resulting JSONL rows in a private local file, then train:

```sh
python3 Training/train.py data/epochs.jsonl --output models/baseline.json \
  --model-id baseline-v1 --data-source "dataset/version/licence"
```

The deterministic SHA-256 subject split is fixed before fitting. Normalization and optimization use training subjects
only; both training and held-out groups need all four stages. The tool exports compatible JSON and a separate
`*.evaluation.json` report with the confusion matrix, per-class recall and balanced accuracy. It never approves a model
for alarm timing. Do not tune repeatedly against the held-out set; use a separate validation cohort for selection.

### Research pipelines

- `Training/run_spectral_corpus.py` re-extracts verified development recordings with the spectral profile.
  `Training/improve_model.py --profile spectral` compares current-only and ten-epoch causal context using regularized
  linear and boosted-tree candidates. Selection maximizes validation macro F1, subject to the native import limit and
  positive validation recall for every stage; balanced accuracy and light recall break ties. Normalization and fitting
  use training participants only, and tree early stopping uses validation only. `--refit-development` retrains the
  fixed settings on all development people without further early stopping; its validation score still belongs to the
  selection checkpoint.
- `Training/prepare_final_test.py` reserves new people only after parameters are frozen, and
  `Training/evaluate_final.py` scores once without fitting. Previously tested people are retired into development and
  excluded from later fresh tests, including people outside a particular fitting corpus. Evaluators check preprocessing
  fingerprints and compare older models on identical usable epochs. `--baseline-spectral` enables the incumbent
  spectral comparison.
- `Training/train_sequence.py` evaluates bounded GRUs over cached summaries using three fixed participant folds.
  `Training/run_acoustic_corpus.py` prepares a hash-verified time-resolved cache and checks the original summary
  observations before accepting each block. `Training/train_acoustic.py` fits the per-epoch encoder on training people,
  freezes it, then trains the sequence classifier on its embeddings. Both trainers require improvement in paired
  development macro F1 and balanced accuracy before producing a selected candidate; the sequence admission tool
  enforces the same rule again before reserving a cohort. Final fitting uses fixed median best epoch counts from
  development folds. `Training/prepare_sequence_test.py` enforces that admission;
  `Training/evaluate_sequence.py` or `Training/evaluate_acoustic.py` scores a frozen model once, and the acoustic
  evaluator compares the previous summary model on the same recording intervals.
- `Training/train_spectral_cv.py` checks fixed tree depths, round counts and class weights on the same participant
  folds and applies the same two-metric admission rule.
- `Training/full_night_cohort.py` freezes the expanded protocol before reading labels: by default 160 fitting people
  (including every previously observed person), 40 new validation people and 30 untouched final-test people.
  `Training/run_expanded_corpus.py` extracts verified bedside `Mic` measurements and controlled MUSAN mixtures. Noise
  roles use unique source paths and file hashes; this is not a perceptual content-disjointness proof or real-TV
  validation. `Training/train_expanded.py` compares clean and augmented recipes, then refits the chosen recipe on all
  usable development people. Both macro F1 and balanced accuracy must improve under both conditions, with every stage
  recognized, before the reserved final cohort may be read.
- `Training/evaluate_expanded.py` compares the frozen model, original-84 full-night control and incumbent on the same
  30 new people, including paired participant-bootstrap uncertainty, without fitting.
- `Training/orchestrate_expansion.py` runs bounded exact-source VPS jobs and stops on failures. Its current job requests
  target `ImLukeF/parley` at `Packages/SleepWakeKit/Training`, so it is not directly runnable from a standalone Remnant
  checkout without adapting those operational paths.

Corpus-based scripts require explicit private workspace/corpus paths; the initial downloader creates its own private
workspace. None runs during app startup. The pinned Linux CPU environment is `Training/requirements-research-linux.txt`;
the neural environment is `Training/requirements-sequence-linux.txt` (Linux x86-64, CPython 3.12). The spectral reference
needs NumPy, extraction also needs pyedflib, and model selection needs SciPy and XGBoost. The optional refit unit check
skips when XGBoost is absent, keeping the original standard-library checks usable without the research environment.

### Weekly home fine-tuning

`Training/train_home.py` starts from a `spectral-v4` GRU baseline and uses explicitly consented home nights. A fixed
quarter of nights is held back by ID. A candidate is kept only if it catches at least as many marked-awake epochs while
calling no more than five percentage points extra of the likely-asleep window awake, and the clean/noisy lab check does
not regress by more than 0.005 in macro F1 or balanced accuracy while retaining positive recall for every stage.
Reserved final-test people are never read, and `approvedForAlarmTiming` remains false.

`Training/home_week.py` and `Training/vps/` automate the weekly run. The runner reads `REMY_URL` and `VPS_TOKEN`, downloads
the server-held night JSON files, and, when a candidate is kept, sends it to `/vps/sleep/model`. The supplied systemd timer
runs Sundays at 13:00 Australia/Melbourne. These are network/storage operations, not package runtime behavior.

`Training/extract_teacher.py`, `Training/own_night_teacher.py`, `Training/colab_teacher.py`, `Training/colab_ears.py` and
the `Training/modal_*.py` scripts provide optional teacher/student and pretrained-audio experiments. They operate on
explicit private inputs and may require Colab/Modal/VPS credentials or gated-model access. Review each command before
giving it network access or research data.

Compile `Training/verify_native.swift` with the core Swift sources to compare a final `model.json` with its private
`export-check.json`:

```sh
swiftc Sources/SleepWakeKit/*.swift Training/verify_native.swift -o /tmp/verify-native
/tmp/verify-native model.json export-check.json
```

It checks chronological per-person inference within `2e-10` and prints a compact receipt without starting an app or
audio session.

## Evidence and limitations

The research artifacts preserve accepted and rejected results; they are not product endorsements. The seven-feature
baseline is a research starting point and may not separate light, deep and REM reliably. The archived v11 clinical test
improved its lab metrics but still had substantial deep/REM errors and no real-phone overnight validation. Controlled
noise is not a bedroom TV trial. Offline accuracy does not establish live latency, battery life, wake comfort or
alarm-firing safety. Consumer Watch stages can be a comparison, not clinical ground truth.

Check live epoch-end timestamps against callback receipt times. Quiet rooms, TV sound, other sleepers, microphone
placement and device processing all need evaluation. Real-device microphone, background operation and native alarm
acceptance remain necessary. No research route in this repository enables model-driven alarms.

## Focused checks

```sh
swift test --filter 'acousticModelMatchesIndependentEncoderAndRecurrentReference|acousticFramesMatchIndependentReferenceAcrossStreamingBuffers' --jobs 2
swift test --filter 'morningModelMatchesItsTrainingMathsOnTheSameSound|nightAudioWriterClosesAReadableLosslessResampledRecording' --jobs 2
cd Training && python3 -m unittest -v test_sequence.SequenceContracts.test_final_registry_keeps_previously_tested_people_outside_this_fitting_corpus
```

These focused examples cover feature/inference math, buffer-boundary invariance, the morning reference, recording format
and participant exclusions. The package also has independent model-validation, context, archive and wake-policy
contracts. Synthetic fixtures do not demonstrate sleep-stage accuracy.

## Privacy and reuse

The Swift package performs no network requests. Observation is disabled until a host explicitly starts it, and waveform
recording is disabled unless the host supplies a private local output URL after consent. Pass `audioURL` to
`OvernightAudio.start` only for explicitly consented local audio. The worker writes lossless mono 24-bit / 16-kHz CAF
with bounded conversion buffers, capped at twelve hours: about 2.1 GB (1.9 GiB) uncompressed. Hosts must provide
deletion, free-space checks, retention controls and explicit export controls. Never start a microphone session or upload
data automatically.

Night files use atomic writes, are excluded from backup on Darwin, and use iOS data-protection classes where available.
ZIP export uses stored, uncompressed entries and streams file data. The host remains responsible for access control,
retention and any network transfer.

No external dataset, recording or pretrained-model weights are vendored. Optional research scripts name their external
sources and licences; their outputs and dataset terms remain separate from the Apache-licensed code. Contributions
should include provenance and subject-separated results for any proposed model or dataset.

## Licence

[Apache 2.0](LICENSE).
