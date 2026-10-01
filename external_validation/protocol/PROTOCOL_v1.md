# ITRI external-validation protocol v1 (locked before new scores)

Locked: 2026-09-23. The asset universe is `asset_universe_v1.json` (34
US-listed country-exposure ETFs; no result-based substitutions). This protocol
is distinct from the archived BIST 30/S&P 500 study. A run that deviates from
this file must receive a new version and be labeled exploratory.

## Scientific question and scope

Does the previously selected ITRI configuration (`w=1`, `p=5`) retain useful
reconstruction accuracy on additional country-exposure ETF time series and
chronologically distinct test periods? This is **offline imputation**, which
may use observations after the missing day. It is not a prospective trading
or causal study. The ETFs are not independent draws from a population of
markets, and their USD prices embed exchange-rate and US-trading-hour effects.

## Input and eligibility

- Request daily, dividend/split-adjusted OHLCV from Yahoo Finance through a
  pinned `yfinance` version for 2015-01-01 (inclusive) to 2026-01-01
  (exclusive). Record retrieval UTC time, library versions, provider response,
  raw CSV SHA-256, cleaned CSV SHA-256, and all exclusions. Do not publicly
  redistribute raw vendor data without a rights check.
- Eligibility is determined **without running ITRI**: at least 95% of the
  expected annual US exchange trading sessions from 2015–2025 in each year;
  all five OHLCV columns present; strictly increasing unique dates; positive
  finite OHLC prices, nonnegative finite volume; no more than five consecutive
  absent ETF trading sessions. These thresholds are intentionally strict. A
  failed symbol is excluded and logged, with no replacement.
- Do not fill native missing rows from Yahoo with unreported interpolation.
  The actual observed-session intersection within each ETF is retained. All
  engineered features, if used, are computed from past and present rows only;
  warm-up rows are removed before masking.

## Features, partition and preprocessing

- **Primary panel:** adjusted open, high, low, close, volume (5 features).
  This avoids giving cross-feature methods deterministic technical indicators
  computed from hidden raw prices. The historic 14-feature experiment remains
  a separate, labeled development result. A 14-feature external analysis is
  secondary and must acknowledge algebraic relationships among features.
- Three expanding-origin folds, per ETF, with nonoverlapping tests:
  A: train 2015–2018, validation 2019, test 2020–2021;
  B: train 2015–2020, validation 2021, test 2022–2023;
  C: train 2015–2022, validation 2023, test 2024–2025.
  Train/validation/test are not randomly shuffled. The 2020–2021 pandemic
  period and subsequent periods are retained, not selected post hoc.
- Fit each feature's location/scale from the **observed cells of the masked
  training partition only**, then apply to validation/test. Report any zero
  scale and a prespecified deterministic fallback. Training-hidden truth may
  be used as a supervision target by methods designed for supervised masking,
  but not to calculate scaling or input features for a masked case.
- ITRI remains exactly `w=1`, `p=5` for the primary external test. These
  settings were chosen on the old two-index development study. No new ETF test
  result may alter them. If a genuine algorithmic extension is developed,
  tune it only on the original two indices or an explicitly separate
  development subset and freeze it before the 34-ETF external test.

## Missingness and comparator fairness

- Main artificial-missingness mechanism: `strict_block_mask`, independently
  within each feature. Missing-cell fractions are 0.10, 0.20, 0.30 of eligible
  cells. Requested maximal contiguous run lengths are 1, 2, 3, 5, 10, 20
  trading days, and fixed mask seeds are 11, 23, 37. Edge guard = 20 rows.
  A single final shorter run may occur when the target count is not divisible
  by run length. Log the *actual* run-length histogram; assert no run exceeds
  the requested length. The same train/validation/test masks and truth are
  given to every comparator at a matching asset/fold/condition.
  Implementation clarification (2026-09-23, before complete external results):
  reserve one full run in each feature before sampling the remaining runs.
  This keeps the prespecified five-feature macro endpoint defined in every
  condition while retaining the exact total masked-cell count, fixed seeds,
  requested maximal run length, and at most one final shorter run. The first
  incomplete Colab attempt stopped on an unmasked feature and yielded no
  complete external-result set; its partial work must not be reported.
- The historic mask generator is analyzed only as a separately labeled
  sensitivity study because adjacent sampled segments can make actual runs
  longer than their nominal length.
- Minimum comparators: linear interpolation and PCHIP (deterministic temporal);
  KNN and chained/iterative imputation (multivariate statistical); BRITS and
  SAITS (dedicated learned temporal). Model implementation, context length,
  augmentation, training epochs, stopping, seed, and hyperparameter budget
  must be frozen **before** their external test scores are inspected.
  Methods are compared under both equal available information and clearly
  reported computational budgets; impossible perfect equality is not claimed.
- Add observed-cell preservation and no-future-data checks. A bidirectional
  offline imputer may use future values *within its own test segment*, but may
  not use another asset's labels or future periods for weight fitting.

## Metrics and unit of analysis

- Compute errors only on artificially hidden test cells. Predeclare primary
  endpoint as macro-average normalized RMSE across five features, where each
  feature RMSE is divided by its **training-observed** standard deviation.
  This avoids treating high-volume features as numerically dominant and keeps
  the denominator independent of the test mask. Also report standardized MAE,
  raw-feature RMSE/MAE, actual hidden-cell counts, run-length strata, and wall
  time/hardware. Near-zero training standard deviation is an eligibility/QC
  failure, not a silent denominator clamp.
- Aggregate three mask seeds within each asset × outer test period ×
  rate × run-length condition. Compare paired methods on those units. For
  the main summary, first average conditions within each asset × period,
  then show **each asset's** three period scores/differences and a distribution
  across the 34 assets. Do not treat 34 × 3 × 18 × 3 rows as independent market
  replicates. Shared shocks can correlate different ETFs; show region-wise
  and leave-one-region-out sensitivity. Any inferential p-value requires a
  dependence-aware rationale; otherwise use effect sizes and descriptive
  uncertainty, not pseudo-precise significance.
- Report all prespecified symbols including QC exclusions. No selective
  removal for poor ITRI performance. Include negative results and failure
  cases. Distinguish original-development results from new external results.

## Reproducibility and publication gate

- A Colab `Run all` release must download inputs (subject to source terms),
  verify/cache hashes, run QC and all methods, checkpoint progress, verify
  row counts/invariants, then create and download a result ZIP automatically.
  It must never silently fall back to a different data date/provider or
  silently omit failed runs. Publish a pinned environment, source commit,
  protocol version, manifest, raw-input hashes, exclusion log, row-level
  output, aggregation scripts, and figure-generation scripts.
- The first-stage acquisition notebook downloads a public-safe QC receipt
  **and a separate private data snapshot**. Keep the private ZIP for exact
  reruns; never add it to the public repository without confirming the data
  vendor's redistribution terms.
- Manuscript updates happen only after complete outputs and table-level
  numerical reconciliation. A broad benchmark alone does not establish
  algorithmic novelty. Journal-specific scope, subscription-route fees,
  current SCIE inclusion, and JCR Q1 must be checked at submission. Acceptance
  cannot be guaranteed.
