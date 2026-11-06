# CFRI simple holdout v1 — locked new-asset protocol

**Status:** Internally specified new-asset holdout. The first Colab run stores the exact source bundle, its SHA-256 digest and a UTC timestamp in the private Drive root before any target data are acquired. GitHub publication may follow the completed run. A later GitHub commit does **not** provide an independently verifiable pre-run timestamp, so this study must not be described as publicly preregistered or as an independent future-period confirmation. A code or analysis change after the local freeze requires a new version and an explicit amendment.

## 1. Question and study boundary

This study tests the simple algorithm chosen after inspecting CFRI v2 results: the original fixed CFRI price/volume rule, a joint OHLC consistency projection, and a peer-return bridge only for an asset-specific whole-row outage. The interaction model and its validation gate are excluded. The study evaluates generalization to **new target assets**. It reuses the January–September 2026 calendar period, so it is not an independent future-period replication. The fixed peer pool consists of the 34 country ETFs used in the earlier CFRI study; their presence also limits the scope of independence.

The target list is fixed, in this order, with no substitutions: `XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, SPY, QQQ, IWM, EFA, EEM, IBB, IYR`. None is among the earlier 34 targets. The peer pool is fixed to `EWC, EWZ, ECH, EWW, EWO, EWK, EDEN, EFNL, EWQ, EWG, EIRL, EWI, EWN, ENOR, EPOL, EWP, EWD, EWL, EWU, EWA, MCHI, EWH, INDA, EIDO, EWJ, EWM, ENZL, EPHE, EWS, EWY, EWT, THD, EIS, EZA`. A target never enters its own peer set. No target or peer is added after inspecting data or scores.

## 2. Data, partitions, and eligibility

- Download daily adjusted OHLCV through Yahoo Finance/yfinance with `auto_adjust=True` and `actions=False`, from **2015-01-01 inclusive** to **2026-10-01 exclusive**. Freeze package versions in the run manifest. Store the downloaded raw tables privately; export only metadata and SHA-256 hashes.
- Train: **2015-01-01 through 2024-12-31**. Validation: **2025-01-01 through 2025-12-31**. Test: **2026-01-01 through 2026-09-30**. Only actual XNYS trading sessions within these intervals are used. The test labels are read only by the frozen scoring step.
- Before running any method, apply the same score-blind QC to every target and peer: unique increasing dates within the requested interval; all five OHLCV columns; finite positive OHLC prices and finite nonnegative volume; `high >= max(open, low, close)` and `low <= min(open, high, close)` within relative tolerance `1e-6`; no dates outside the XNYS calendar; at least 95% coverage in each year 2015–2025 and in the 2026 period through September; at most five consecutive missing XNYS sessions. A successfully downloaded asset that fails QC is logged and excluded without replacement. Provider errors receive three download attempts and halt the data stage if unresolved; rerun the data stage before any method. Do not select assets using imputation scores.
- Eligible targets form the scoring cohort. Eligible peers form the available peer pool. If fewer than **12** of the 16 targets pass QC or fewer than five peers pass QC, stop after the QC receipt and report the run as **inconclusive**; no primary method comparison is made. Keep and report all eligible targets even if a method performs badly. If a peer fails QC, log it and use the remaining eligible peers; if fewer than five peers support an individual gap, the bridge follows its frozen no-peer fallback.
- Every method in a condition must use the identical downloaded target snapshot, train/validation/test partition and hidden-cell mask. Save raw-file and normalized-frame SHA-256 values, QC decisions, mask SHA-256 values and masked-cell counts. Verify them on resume and before analysis.

## 3. Hidden-data scenarios and condition grid

Use the frozen `cfri_core.condition_mask` generator, with 20 observed edge rows, block lengths `(1, 2, 3, 5, 10, 20)`, hidden rates `(0.10, 0.20, 0.30)`, and repetitions/seeds `(11, 23, 37)`. Derive seeds deterministically from ticker, fold/study ID, partition, block, rate, repetition and mask type; freeze that derivation in code before acquisition. Generate masks separately for training, validation and test. Apply the same masks to every method. Skip a grid cell only when the frozen `cfri_core.feasible` predicate says it is impossible for at least one partition; report the skipped cells. Under full 2015–September 2026 coverage, the expected count is **51 conditions per target per mask type** (816 per type if all 16 targets qualify).

Two mask types are evaluated separately:

1. **Feature:** hide OHLCV fields independently in the target ETF. No peer information is used in the primary feature comparison.
2. **Whole row:** hide all five target fields on the same dates. In the **target-specific outage** scenario, peer closes remain observed on those dates. In the **market-wide outage** stress scenario, the peer closes are hidden synchronously before peer returns are computed. Never use a hidden target value as an input or training label.

This is offline imputation: the algorithm may use observed target values on both sides of a gap. The study does not measure real-time forecasting.

## 4. Frozen methods and information sets

**CFRI base.** Use the original fixed CFRI rule: price ridge alpha `10`, price residual gain `0.75`, volume `log_local_0.50`; on a whole-row gap, where same-day target OHLC features are all absent, use its frozen log-linear price fallback. Apply `joint_ohlc_project(weights=None)` only to missing OHLC cells, preserving observed cells. Do not fit or select the v2 interaction model, power, or gain gate.

**Peer bridge.** For target-specific whole-row gaps only, fit the frozen `v2_peer.py` return bridge on masked training target closes and the eligible fixed peer pool (`alpha=10`, at least five available peers). Select its gain from `(0, 0.25, 0.5, 0.75, 1.0)` using **2025 validation labels only**. Apply its endpoint-anchored adjustment to hidden target OHLC rows, followed by the same joint projection. If support is insufficient, use the base estimate. For the market-wide stress scenario, hiding peer closes synchronously must make the adjusted result equal the no-peer result; any deviation is a failed run invariant, not a favorable result.

**SAITS V2.** Use the frozen normalized SAITS V2 recipe: 64-row windows, stride 16, batch size 32; two layers, model dimension 64, four heads, key/value dimensions 16, feed-forward dimension 128, dropout 0.1; at most 200 epochs, patience 20; training-observed scaling and the same per-window instance normalization and fallback as the earlier V2 run. Set deterministic model seeds from the condition key. The target-only SAITS receives the target OHLCV and no peer inputs. Restore observed cells and apply the identical joint OHLC projection for comparison.

**SAITS with peer bridge.** For the equal-information whole-row comparison, apply the **same frozen peer model and peer pool** to the SAITS target-only predictions. Select a SAITS-specific bridge gain from the same five-value grid on SAITS **validation predictions and 2025 validation labels only**; then apply it to SAITS test predictions and joint-project. Training labels, peer dates, support rules and test masks must match those used for CFRI. If this comparator cannot be completed for every eligible target and condition, H3 is not evaluated; do not substitute target-only SAITS and call it equal-information evidence. Report any failures and the attempted condition counts.

## 5. Endpoints and prespecified decisions

For each feature in each condition, compute RMSE on **hidden test cells only**, divided by that feature's population standard deviation (`ddof=0`) computed from **observed training cells only**. The condition NRMSE is the arithmetic mean of the five normalized feature RMSEs. Average conditions equally within each target, then average eligible targets equally. Lower is better. Report price NRMSE (mean of open, high, low and close), volume NRMSE, normalized MAE, and OHLC violation share separately. Do not pool hidden cells or treat the repeated masks as independent observations.

Let `N` be the number of QC-eligible targets (`N >= 12`). A primary hypothesis succeeds **only if both** its equal-target mean NRMSE is lower and the proposed method has lower mean NRMSE in at least `ceil(0.75 × N)` targets. Ties are not wins. Calculate differences as proposed method minus comparator. Report every hypothesis regardless of outcome; do not change the endpoint, thresholds, target cohort or comparator after seeing test scores.

| Hypothesis | Mask/scenario | Proposed method | Comparator |
|---|---|---|---|
| H1 | Feature | Original fixed CFRI + joint projection | Target-only SAITS V2 + same joint projection |
| H2 | Target-specific whole row | CFRI base + peer bridge + joint projection | Same CFRI base + joint projection, no peer bridge |
| H3 | Target-specific whole row | CFRI base + peer bridge + joint projection | SAITS V2 + same peer bridge, with SAITS-specific validation gain + joint projection |

H1 assesses the base method against SAITS with the same target-only information. H2 measures the value of additional peer information to CFRI. H3 compares methods when both have access to the peer bridge. None of these decisions alone establishes broad superiority across datasets or future market periods. The market-wide result is a prespecified stress test and fallback invariant, not an opportunity to retune the bridge.

## 6. Analysis, completeness, and reporting

- Require a verified checkpoint for every feasible CPU and SAITS condition and every primary method before calculating primary results. Save each condition atomically with a SHA-256 sidecar and a fingerprint covering code, configuration, data frame and condition identity; resume only verified checkpoints. A failed or incomplete run remains incomplete, not a negative or positive hypothesis result.
- Verify CPU/SAITS train, validation and test mask hashes and masked-cell counts match pairwise for each condition. Verify preserved observed cells, finite estimates and zero OHLC violations after projection. Independently recompute primary aggregates from exported condition-level scores.
- Publish all three H1–H3 outcomes, including failures. Show per-target paired differences and wins, mean NRMSE, price and volume breakdowns, each block length and hidden rate, number of eligible/excluded targets and peers with reasons, fallback frequency, validation-selected peer gains, training runtime, package versions, code hashes, and exact data-acquisition time. Report target-only SAITS and market-wide scenarios as secondary context.
- An ETF-level paired bootstrap interval may be reported **descriptively**. These ETFs and their errors share market shocks, so the 16 assets and the repeated masks are not independent replications; do not interpret the interval or the 75% rule as a population-wide guarantee. Any extra models, thresholds, subgroup searches or post-hoc fixes must be labelled exploratory and kept outside the three frozen decisions.
- The private raw Yahoo price files stay out of GitHub and the result export. Release only code, protocol, settings, code/data SHA-256 digests, QC receipt, condition scores and analysis. Do not claim that the overlapping 34 peer raw files reproduce the earlier fold-D snapshot unless all 34 corresponding hashes match; the 16 new targets have no earlier fold-D counterpart. Comparisons within this run are paired by their verified masks and data.

## 7. Freeze and execution order

1. Complete and locally smoke-test the code and Colab notebook on synthetic data. Embed the exact source bundle and SHA-256 manifest in the notebook. On its first **Run all**, the notebook saves the source bundle, digest and UTC time in Google Drive **before** target data acquisition. The data stage rejects a missing or changed local freeze record.
2. Run the data/QC stage, then CPU methods, then SAITS methods, then the locked analysis. Keep private raw data and resumable condition checkpoints under a dedicated Google Drive folder. A restarted Colab run must verify and reuse completed conditions.
3. Export a ZIP with receipts, condition scores, analysis and hashes, excluding raw prices. GitHub may be updated afterward with the code and protocol, but its timestamp then proves publication only, not pre-run registration. Any code repair after the local freeze receives a versioned amendment and its own interpretation.

The next genuinely new-time test is a future period, for example 2027, with code and rules frozen before that period's labels are inspected.
