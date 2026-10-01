# ITRI external validation: BRITS/SAITS extension, frozen 2026-09-23

This implementation plan was fixed **after** the classical results were
inspected but **before** BRITS/SAITS external results were generated. It is
not a blinded preregistration. Do not adjust architecture, epochs, windows,
or exclusions in response to held-out neural scores without labeling a new
exploratory analysis.

## Data and evaluation

- Use the same audited private 34-ETF snapshot, OHLCV features, chronological
  A/B/C folds, six block lengths, three missing rates, three mask seeds,
  edge guard, and strict mask generator as the completed classical run.
- For every asset × fold × block × rate × seed, fit each model separately on
  the corresponding masked training partition. No test-period value or mask
  is used in optimization or epoch selection. Validation-period ground truth
  is used only for checkpoint selection; test ground truth is used only to
  calculate final metrics.
- Compute each feature's mean and standard deviation from initially observed
  **training** cells for the matching condition, exactly as in the classical
  benchmark. Apply that scale to train, validation and test. No future asset
  data are pooled across chronological folds.
- Training, validation and test windows have 64 trading days and stride 16.
  Add a final window ending at the partition end so every day is covered.
  Average overlapping test-window estimates cellwise, then restore all
  originally observed test cells before scoring. No window crosses a
  train/validation/test partition boundary.
- BRITS and SAITS each receive 30 maximum epochs, early-stopping patience 5,
  batch size 32, and validation MSE as supplied by PyPOTS 1.5. BRITS has
  recurrent hidden size 64. SAITS uses two layers, model width 64, four heads,
  key/value width 16, feed-forward width 128 and dropout 0.1. SAITS retains
  its library-default internal 20% masked-imputation-training augmentation;
  that self-supervised augmentation is distinct from the fixed test mask.
- Initialization/training seeds are stable hashes of manifest, asset, fold,
  condition and method. Python, NumPy and Torch are seeded. CuDNN
  deterministic mode is enabled, but bitwise equality across GPU hardware,
  drivers, Python and Torch versions is **not** claimed. Exact versions and
  hardware must be reported with results.
- Each test condition yields the same five raw MAE/RMSE and training-standardized
  MAE/RMSE measurements as the classical code. The primary endpoint is the
  macro mean of five feature-wise standardized RMSE values. Conditions and
  folds are averaged within each asset; 5,508 masks are not independent
  market replicates. Regional/shared-shock dependence remains a limitation.
- Validation diagnostics are recorded separately for all seven methods. The
  same condition-specific masked training input is used to fit each classical
  validation comparator; the neural validation scores are calculated from
  the validation period used for epoch selection. Therefore validation scores
  are descriptive model-selection diagnostics, **not** a second held-out test.
- Runtime includes neural fitting, validation imputation and test imputation
  for a condition, while
  classical runtime uses each method's existing implementation. These costs
  are reported descriptively, not claimed as perfectly hardware-matched.

## Computation and provenance

- The grid requires 34 × 3 × 6 × 3 × 3 × 2 = **11,016 neural fits**. A Colab
  session may disconnect before completion. Result shards are checkpointed
  after every condition to Google Drive and verified by SHA-256 and a
  code/environment fingerprint on resumption. Re-run the notebook after a
  disconnection; do not report a partial result directory.
- The full-run marker is emitted only after all 5,508 condition shards and
  55,080 neural test-feature rows and 192,780 validation-feature rows are
  present and verified. The combined test audit must then reconcile neural
  rows with 137,700 classical rows to 192,780 test-feature rows.
- The public result archive contains metrics, protocol, versions, code and
  input hashes, but not vendor OHLCV or model weights. Keep the private
  snapshot outside the public repository unless its license permits release.
- A one-condition/one-epoch pilot checks code execution only. Pilot metrics
  are explicitly not publishable and must not enter manuscript tables.

Official implementation/API references:

- BRITS: https://docs.pypots.com/en/latest/_modules/pypots/imputation/brits/model.html
- SAITS: https://docs.pypots.com/en/latest/_modules/pypots/imputation/saits/model.html
- PyPOTS package/version: https://pypi.org/project/pypots/
