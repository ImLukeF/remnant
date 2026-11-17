# Expanded private full-night experiment

This completed experiment fitted both fixed recipes on 159 people and refitted the chosen recipe
on all 199 usable development people, using 932.933 usable hours. Neither recipe passed validation
admission. The reserved 30-person final test remains unopened. Both repositories remain private.

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


## Completed result

Training job 6f1ef869a481408392006ae77b163b66 passed at exact source
3d05739349d298c9a11fed00e7efe482c48260df, finishing at 08:35:30 UTC. All candidate, control and
199-person refit weights, reports and the frozen protocol are in models/. The original-cohort
control used 83 usable people. The 40-person validation checkpoints used 159 fitting people; their
scores must not be attributed to the later 199-person refit.

Clean validation (balanced recall / macro F1): incumbent .44187 / .27693, clean candidate .49156 /
.27228, noise-trained candidate .47367 / .25775. Added-noise validation: incumbent .25805 / .07638,
clean candidate .27380 / .08540, noise-trained candidate .41453 / .20757. Noise augmentation helped
under synthetic interference, but both candidates lost clean macro F1. The clean candidate also
failed to recognize REM under added noise. Neither met all frozen admission requirements.

The completed 199-person clean refit remains experimental and unadmitted. It has no independent
final-test score; its metadata balancedAccuracy=0 is an unmeasured placeholder. The 30 reserved
people were never opened, no final model was imported or deployed, and stage-driven waking stays
disabled. The incumbent is unchanged. The verified-result.json file records independent metric,
parameter, file-hash and participant-split checks. observed-subjects.json retires all 200 observed
people, including the silent exclusion, from future independent tests.

The controller and workers exited, the empty raw-cache directory was removed, and the completion
check was disabled. Useful verified clinical feature/annotation and public-noise caches remain
private on the VPS, outside Git. No new native or phone validation was performed on this
unadmitted refit. This experiment is complete; do not rerun selection against these results.
