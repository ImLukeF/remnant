# Changelog

All notable changes to SleepWakeKit are documented here.

## [Unreleased]

### Added

- Experimental `SleepWakeKit` and `SleepWakeAudio` Swift products for bounded 30-second audio observations, wake plans
  with backup attempts and an iOS microphone source.
- Spectral and time-resolved acoustic feature profiles, plus native causal linear, tree, GRU and acoustic-convolution
  model imports with past-only context and reset behavior.
- `SleepNightStore` with atomic observations, model and marker history, optional Health references, interrupted-export
  recovery and explicitly consented local lossless audio capture.
- Subject-separated PSG-Audio training and evaluation tools, native/Python parity checks, expanded and noisy-cohort
  experiments, and research artifacts that preserve both accepted and rejected results.
- Weekly home-night fine-tuning with held-back nights and lab regression gates, plus optional teacher/student extraction
  and pretrained-audio experiments for Colab, Modal and the research VPS
  ([#3](https://github.com/ImLukeF/remnant/pull/3)).

### Changed

- Relicensed the package under Apache 2.0 ([#2](https://github.com/ImLukeF/remnant/pull/2)).
- Expanded the integration, model-import, training, privacy and focused-check documentation
  ([#1](https://github.com/ImLukeF/remnant/pull/1)).

### Security

- Audio recording remains opt-in and separate from observation data, with explicit host-controlled export and no
  automatic upload.
- Research admission rules preserve participant exclusions and keep reserved final-test cohorts out of fitting and
  home fine-tuning; no research tool approves a model for alarm timing.
