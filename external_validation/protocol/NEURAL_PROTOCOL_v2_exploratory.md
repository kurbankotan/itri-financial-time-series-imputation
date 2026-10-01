# ITRI external validation: protocol amendment v2 (EXPLORATORY), 2026-09-29

This amendment was written **after** the complete v1 results (classical
comparators, BRITS, SAITS; 5,508 conditions) had been inspected. Every analysis
under this amendment is therefore exploratory. It does not replace the
prespecified v1 primary result (ITRI `w=1`, `p=5` versus the frozen comparator
set) and must be reported in a separately labeled section.

## What v1 showed and why the amendment exists

- Under v1, BRITS and SAITS reached feature-macro NRMSE of 1.20 and 1.15
  (asset-level means), against 0.27 for linear interpolation. Their error was
  almost flat across gap lengths (1-day gaps were not filled better than
  20-day gaps), rose with the chronological distance between training and test
  periods (fold A 1.6-1.7, fold B 0.75-0.80, fold C 1.05-1.09), and correlated
  with the test-period displacement from the training distribution
  (Spearman 0.96 between the mean absolute training-scaled z-score of test
  prices and neural NRMSE; 0.23 for linear interpolation). KNN (0.97) and the
  chained-equation imputer (0.75) showed the same dependence.
- The v1 feature scaling (training-observed mean and standard deviation) is
  inherited from the classical benchmark. It is level-dependent: any imputer
  whose output depends on the absolute scaled level fails when test prices
  leave the training range. Temporal interpolators are level-invariant.
- A CPU pilot (fold A; THD, EWZ, EWK, EWG, EWT, EDEN; blocks 1, 5, 20; rate
  0.20; seed 11) showed that per-window instance normalization alone reduces
  SAITS test NRMSE from 1.81 to 0.27 at the unchanged 30-epoch budget and to
  0.24 with a 200-epoch budget. Pilot numbers are not reported as results.

## Amendment (the only changes relative to v1)

1. **Per-window instance normalization.** Every 64-day window given to BRITS
   or SAITS (training, validation, test) is standardized per feature by the
   mean and standard deviation of that window's *observed* cells, computed
   after the v1 training-period scaling. Predictions are de-standardized with
   the same window statistics before overlapping windows are averaged and
   observed cells are restored. Hidden cells never enter the statistics. If a
   window has fewer than two observed cells for a feature, or a standard
   deviation below 1e-6, the identity transform is used and the fallback is
   counted. This is the reversible instance normalization of Kim et al.
   (ICLR 2022) applied without learnable affine parameters.
2. **Two training budgets.** Variant V1 keeps the v1 budget exactly (30
   epochs, patience 5) so that the normalization effect is isolated. Variant
   V2 raises the budget to 200 epochs with patience 20. Architecture, window
   length, stride, batch size, optimizer defaults, seeds and mask sets are
   unchanged.
3. **Scope.** SAITS is rerun on the full 5,508-condition grid for V1 and V2.
   BRITS is rerun for V1 on the full grid when compute allows, otherwise on a
   labeled subset; the subset is stated with the results.
4. **Reporting.** Results are reported beside the v1 results with the
   exploratory label. The v1 numbers are not removed or replaced. No further
   architecture or hyperparameter search is performed after inspecting the
   amended scores. The same asset-level unit of analysis applies; no
   inferential test is added.
5. **Provenance.** Each run writes its own fingerprint (runner code, manifest,
   configuration, method, variant), library versions and hardware. Bitwise
   equality across hardware is not claimed; the pilot reproduced the archived
   v1 BRITS score for one condition to three decimals on CPU, and SAITS to
   within 0.05-0.20 NRMSE, which is the expected nondeterminism.

## Interpretation rules fixed before the amended results were seen

- If amended SAITS or BRITS beat ITRI on the external panel, the manuscript
  reports this as the primary finding of the amendment; ITRI's claim is
  restricted to the regimes where it retains an advantage (short gaps,
  single-feature dependence, negligible cost).
- If they do not, the manuscript still reports the level-shift mechanism and
  the recovery, without claiming ITRI superiority over properly normalized
  deep imputers in general.

## Status (2026-09-30, after execution)

- SAITS V1 (30 epochs, per-window normalization): completed on the full
  5,508-condition grid on a two-core CPU (PyPOTS 1.5, torch 2.14, median
  6.2 s per condition, zero normalization fallbacks).
- SAITS V2 (200 epochs, patience 20): completed on the full 5,508-condition
  grid on an NVIDIA A100-SXM4-40GB in Colab (PyPOTS 1.5, torch 2.11+cu128,
  median 5.6 s per condition, zero fallbacks). Fingerprint 989b0a6f961f...
- BRITS V1 (30 epochs): completed on the labeled subset fold A x mask seed 11
  (34 assets x 6 block lengths x 3 rates = 612 conditions) on the same A100
  (median 32 s per condition). The full grid was not run because the
  recurrent implementation does not accelerate on the GPU. Fingerprint
  3b5963449731...
- All amended shards were verified (SHA-256, fingerprint, row counts) and use
  exactly the frozen train/validation/test masks (test mask seeds and masked
  cell counts identical to the classical run for every row).
- Cross-hardware check: four pieces (EWC and EWZ, fold A, seed 11; BRITS V1
  and SAITS V2) were computed on both the CPU and the A100. BRITS agreed to
  within 2.1e-5 per feature row; SAITS V2 to within 0.039 per condition, with
  piece means within 0.0033. Bitwise equality across hardware is not claimed.
- Reported in the manuscript as Tables 18-21 and Figures 5(b) and 7. The CPU
  pilot values that motivated the amendment are excluded from all tables.
