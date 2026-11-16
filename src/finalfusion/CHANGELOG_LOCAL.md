# Local Change Record

This repository tracks source-only checkpoints for lightweight rollback.
Datasets, model checkpoints, caches, logs, and inference outputs are excluded.

## Baseline

- 2026-09-08: Established the baseline after restoring the previous MEF loss
  (`MSE + SSIM + SAM + chroma residual + range penalty`).

## Changes

- 2026-09-08: Restored MEF loss v2 after ISP A/B testing showed that most of
  the observed edge color shift came from rendering. Loss v2 uses confidence-
  weighted Charbonnier, dual-source structure, detail, chroma residual, and
  tolerant range terms.
- 2026-09-10: Added DatasetMEF scenes 0326-0442 without changing any previous
  assignment. The 117 additions are stratified by their real BGGR/RGGB CFA as
  93 train, 12 validation, and 12 test scenes. MEF discovery now ignores
  unrelated top-level download folders, split loading rejects unassigned
  scenes, and AutoDL cache reuse checks scene coverage plus the training-split
  fingerprint before training.
- 2026-09-15: Added U2Fusion-style inference-only hard source chromaticity for
  MEF. It preserves the network's packed brightness while selecting each final
  pixel's complete RAW chromaticity from one SNR-valid source, with saturated
  over-exposure fallback and edge-protected mask cleanup. Existing direct
  network outputs can be corrected into a distinct A/B output directory;
  no model checkpoint or training data is modified.

For every later code change, use one focused commit and record the reason and
verification in its commit message.
