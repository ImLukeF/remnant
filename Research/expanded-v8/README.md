# Expanded private full-night experiment

This directory records the frozen inputs and execution requests for the active experiment.
It does not contain completed training results yet. Both repositories remain private.

The development split contains 160 fitting people and 40 new validation people. All 84 people
observed in previous experiments are in fitting. After selecting between fixed clean and
noise-augmented recipes, the chosen recipe is refitted on all 200 development people.
Another 30 new people remain untouched unless validation admission passes.

The pinned source plan contains 1,057 verified source blocks: 632.856 GiB and 956.081 available
hours before quality filtering. Every block uses the ambient Mic channel, exact EDF/RML timing
and verified source hashes. Empty or ambiguous source headers are excluded without guessed times.
No recordings, clinical annotations or derived clinical feature arrays are committed here.

The 48-clip public noise bank separates fitting/validation/test content. Only audited compatible
source licences are used. Its attributions are retained in noise-licence-audit.json and noise-bank.json.
Controlled speech/music/ambient mixtures do not establish performance with a real bedroom TV.

training-request.json pins the exact pushed source for fitting, final preparation, extraction and
once-only evaluation. The VPS controller completes bounded extraction jobs, then runs those phases.
It stops on failures. Final testing is conditional on improved macro F1 AND balanced accuracy in
both clean and controlled-noise validation, with positive recall for every stage. Otherwise the
refitted 200-person weights remain experimental and the reserved final cohort is not opened.

The incumbent remains unchanged and model-driven waking stays disabled. Focused synthetic receipts
verify fitting/export and final-evaluation safeguards; they make no clinical accuracy claim.

## Verified processing upgrade

Luke requested faster preparation on 17 November. The same experiment resumes from 211 verified
blocks with three independent extraction processes; only the parent publishes atomic checkpoints.
The existing per-job 3-core/8-GiB limit is retained. Two BLAS numerical threads remain fixed;
OPENBLAS_THREAD_TIMEOUT=1 reduces idle CPU waiting. No signal code, participant split, fitting
settings or admission rule changed. The extractor fingerprint remains
48239b36ff58c8604e42dfe58e51b81e7e048b4ee1b98021cbc7c11dfb235a23.

The parallel owner proof checks equality of every serial/parallel NPZ array, complete publication
and checkpoint reuse using real synthetic 48-kHz EDFs. The idle-wait proof checks byte equality
across varied 8/48-kHz synthetic measurements. A proposed change to one BLAS thread failed strict
byte equality and was rejected before clinical use; its failed receipt/log are retained here.
The replacement proof preserves the numerical thread count and checks only the selected idle
wait setting. These synthetic proofs do not measure clinical sleep-stage accuracy.

The previous controller/worker were verified stopped before reusing their output/lane. The
cancelled serial job remains in history as cancelled. speed-upgrade.json records checkpoint
preservation and the runtime transition; training-request.json pins the updated final extractor.

## Pre-fitting silence-only amendment

All 1,057 planned blocks were processed, producing 932.933 usable hours. The completion gate
stopped because one training participant's 304 scored intervals were entirely silent. No model
had been fitted and no fresh-test participant had been opened. The failed receipt is preserved.

quality-amendment.json records an explicit pre-fitting exclusion tied to the unchanged source plan
and every cached feature checksum. The audited policy accepts wholly silent training observations
only; it cannot exclude validation/final participants, other quality failures or missing labels.
The actual audit checked every cache hash. All 200 observed people remain excluded from future
independent tests. No waveform processing, parameter choice or admission metric was changed.

Actual fitting now uses 159 people with the same 40-person validation group, followed by a refit
on 199 usable people. The original full-night control uses 83 usable people from its 84-person
reservation. The same 30 fresh people remain conditional on validation admission. Reports name
actual training counts. The synthetic recovery/evaluator/checkpoint proofs are retained alongside
the actual quality-audit receipt; their synthetic accuracy is not a clinical result.
