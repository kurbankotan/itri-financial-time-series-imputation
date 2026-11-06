---
title: "CFRI confirmatory protocol v1: temporal hold-out evaluation on January to September 2026 data"
---

**Study.** Cross-Feature Residual Interpolation (CFRI) for gaps in daily OHLCV series: confirmatory evaluation on fold D.

**Version.** v1, frozen on 27 October 2026, before any 2026 OHLCV observation was retrieved or analyzed.

**Author.** Kurban Kotan, Manisa Celal Bayar University (ORCID 0000-0002-6660-0565).

**Repository.** https://github.com/kurbankotan/itri-financial-time-series-imputation, folder `cfri_confirmatory/`.

**Related protocols.** External validation protocol v1 (`external_validation/protocol/PROTOCOL_v1.md`, locked 23 September 2026) and neural amendment v2 (`NEURAL_PROTOCOL_v2_exploratory.md`, 29 September 2026).

# 1. Purpose and status

The external validation under protocol v1 tested ITRI ($w = 1$, $p = 5$) on 34 country ETFs in three chronological folds whose test periods cover 2020 to 2025. After that study, an extension was developed that uses same-day information from the other OHLC prices of a partially observed trading day (working name ITRI2, here Cross-Feature Residual Interpolation, CFRI). Its development used the same 34 ETFs and folds A to C, and the validation and test years of those folds were inspected. Protocol v1 requires an algorithmic extension to be frozen before it is tested on data that played no role in its development. Every CFRI result on folds A to C is therefore exploratory (Section 10).

This protocol fixes, before data access, how CFRI is evaluated on a new period, 2 January to 30 September 2026 (fold D): data and quality rules, masks, methods, the fixed CFRI setting, the endpoint, four primary hypotheses with their decision rule, the secondary analyses, and the handling of failures and deviations. The code that implements it is frozen and identified by SHA-256 hashes (Section 11).

When this protocol was frozen, no 2026 OHLCV value had been downloaded, viewed or analyzed by the author with any imputation method. The 2026 data are acquired by the run notebook only after this protocol and the code are committed to the public repository; the commit time is the time stamp of the freeze, and the retrieval time of every data file is recorded in the data receipt. Because the 2026 sessions had already traded when the protocol was written, the study is a temporal hold-out evaluation rather than a prospective forecast. General awareness of 2026 market conditions cannot be excluded, but no method, setting or analysis choice used 2026 data.

# 2. Data

- **Assets.** The 34 US-listed iShares country ETFs of `asset_universe_v1.json`, unchanged. No asset is added, removed or replaced on the basis of results.
- **Source.** Yahoo Finance through yfinance 1.7.0; daily bars with `auto_adjust=True` and `actions=False`; interval 1 January 2015 (inclusive) to 1 October 2026 (exclusive). The provider recomputes dividend and split adjustments at download time, so 2015 to 2025 values may differ slightly from the archived v1 snapshot. Fold D uses only the new download.
- **Quality rules** (protocol v1, applied without running any method): all five OHLCV columns present; unique, strictly increasing dates inside the interval; finite positive prices and finite nonnegative volume; valid OHLC order (high and low bracket open and close, relative tolerance $10^{-6}$); at least 95% of the XNYS sessions (exchange_calendars 4.13.2) in every calendar year 2015 to 2025 and in 1 January to 30 September 2026; at most five consecutive absent XNYS sessions; no rows outside the XNYS calendar. Native missing sessions are not filled.
- **Exclusions.** An asset that fails a rule is excluded, logged with its reason and not replaced. A download that fails, or whose provider response cannot be read as unique, increasing daily OHLCV rows inside the interval, is attempted three times; if it still fails, the run stops before any method and the download may be repeated later. Such an asset is excluded only by an explicit, logged decision taken before any method has run. Once the first method output exists, the data receipt is final.
- **Privacy and integrity.** Raw files remain in the author's private storage and are not redistributed. The public data receipt lists for each asset the retrieval time, row count, first and last date, session coverage and the SHA-256 of the file. Every later step verifies these hashes.

# 3. Partition and scaling

- **Fold D:** training 2015 to 2024, validation 2025, test 2 January to 30 September 2026 (187 XNYS sessions). Folds A to C are as in protocol v1.
- **Scaling:** for each feature, the mean and population standard deviation of the observed cells of the masked training partition. They standardize inputs where a method requires it and are the denominators of the endpoint.

# 4. Missingness design

- **Generator:** the strict block mask of protocol v1 (nonadjacent runs of the requested length, one full run per column placed first, exact cell count, edge guard of 20 rows at both ends of each partition).
- **Grid:** missing rates 0.10, 0.20, 0.30; block lengths 1, 2, 3, 5, 10, 20 trading days; mask seeds 11, 23, 37; separate masks for the training, validation and test partitions.
- **Seeds:** the first four bytes of SHA-256(`manifest | ticker | D | partition | block | rate | seed`), as in protocol v1.
- **Feature-wise masks (primary):** each of the five features is masked independently.
- **Whole-row masks (secondary):** the generator is applied to a single column, with the partition label suffixed by `:row`, and the selected days are hidden for all five features.
- **Feasibility rule:** a (block, rate) pair is used only if, in each partition with $R$ rows, $R > 2 \cdot 20 + b$ and $\operatorname{round}(r (R - 40) F) \ge F b$, where $b$ is the block length, $r$ the rate, and $F = 5$ for feature-wise and $F = 1$ for whole-row masks. With $R = 187$ test rows, the pair (20, 0.10) is infeasible under both mask types (74 < 100 cells; 15 < 20 rows). This leaves 17 pairs $\times$ 3 seeds = 51 conditions per asset and mask type. The rule depends only on row counts, is applied to each asset's own counts and applies to every method.

# 5. Methods

## 5.1 Comparators

Definitions are unchanged from the exploratory runs.

- **Linear interpolation** (`linear_raw`).
- **Log-linear interpolation** (`log_linear`): each price is interpolated linearly in log space between the nearest observed days, and volume linearly in $\log(1 + V)$.
- **Original ITRI** with $w = 1$, $p = 5$ (`itri_w1_p5`), as in protocol v1.
- **Chained ridge** (`chained_ridge`): IterativeImputer with Ridge ($\alpha = 1$), mean initialization and at most 10 iterations (tolerance $10^{-3}$), as in protocol v1.

## 5.2 CFRI

Consider a hidden price cell of feature $d$ on day $t$ inside a gap of that feature whose nearest observed days are $a < t < b$. Let $\ell$ denote log prices, $L$ the gap length and $\pi_t = (t - a)/(b - a)$. The base estimate is log-linear interpolation:

$$\hat{\ell}^{(0)}_{t,d} = \ell_{a,d} + \pi_t \left(\ell_{b,d} - \ell_{a,d}\right).$$

For every other price feature $j \ne d$ that is observed on days $a$, $t$ and $b$, the same-day deviation from its own interpolation path is

$$\delta_{t,j} = \ell_{t,j} - \left[\ell_{a,j} + \pi_t \left(\ell_{b,j} - \ell_{a,j}\right)\right],$$

with availability indicator $z_{t,j} = 1$; otherwise $\delta_{t,j} = 0$ and $z_{t,j} = 0$. The predictor vector has nine elements:

$$x_t = \left(\delta_{t,j},\ z_{t,j}\ (j \ne d),\ \log(1 + L),\ \pi_t,\ \frac{\ell_{b,d} - \ell_{a,d}}{b - a}\right).$$

The residual model $f_d$ is a ridge regression ($\alpha = 10$) on standardized predictors. It is trained on the masked training partition only: two additional pseudo-masks (the strict generator with the condition's rate and block length and a separate seed tag) hide further cells, and the label of each pseudo-hidden cell is its true log price minus its base estimate. A model is used only when it has at least 30 labels with nonzero variance. The price estimate is

$$\hat{P}_{t,d} = \exp\left(\hat{\ell}^{(0)}_{t,d} + g\, r_{t,d}\right),\qquad r_{t,d} = \operatorname{clip}\left(f_d(x_t), -0.25, 0.25\right),$$

with $g = 0.75$, and $r_{t,d} = 0$ when no other price feature is observed on all three days $a$, $t$ and $b$.

Volume is interpolated in $u = \log(1 + V)$ and shrunk toward the local median:

$$\hat{V}_t = \max\left\{0,\ \exp\left((1 - \eta)\, \tilde{u}_t + \eta\, m\right) - 1\right\},$$

where $\tilde{u}$ is the linear interpolation of $u$, $m$ is the median of the observed $u$ within 20 days before and after the gap, and $\eta = 0.5$.

**OHLC projection.** On each day with at least one imputed price, only imputed cells are moved so that low $\le$ open, close $\le$ high: an imputed high is raised to the maximum of high, open and close (and an observed low); an imputed low is lowered to the minimum of low, open and close (and an observed high); imputed open and close are clipped to [low, high]. The pass is applied twice. Observed cells never change.

**Fixed setting.** $(\alpha, g, \eta) = (10, 0.75, 0.5)$. The price pair $(\alpha, g)$ is the candidate with the lowest mean validation NRMSE of the four prices, and $\eta$ the volume rule with the lowest mean validation NRMSE of volume, pooled over all feature-wise conditions of folds A to C (validation years 2019, 2021 and 2023). The candidates were $\alpha \in \{0.1, 1, 10\}$ $\times$ $g \in \{0, 0.25, 0.5, 0.75, 1\}$ and six volume rules (linear, ITRI, $\eta \in \{0, 0.25, 0.5, 0.75\}$). Each fold alone selects the same setting. No test-year value entered this choice.

**CFRI in this protocol** is the fixed setting followed by the OHLC projection (`cfri_fixed_proj`).

## 5.3 Secondary variants

CFRI without projection (`cfri_fixed`); CFRI without the residual, that is $g = 0$ (`cfri_fixed_no_cross`); the exploratory version that selects $(\alpha, g)$ and the volume rule per condition on the validation year (`cfri_adaptive`, identical to the archived ITRI2 run, and `cfri_adaptive_no_cross`); and every method followed by the OHLC projection (suffix `_proj`).

## 5.4 SAITS

Normalized SAITS exactly as in amendment v2: 64-day windows, stride 16, batch size 32; 2 layers, $d_{\text{model}} = 64$, 4 heads, $d_k = d_v = 16$, $d_{\text{ffn}} = 128$, dropout 0.1. Each feature of each window is standardized by the mean and standard deviation of its own observed cells in that window (fallback 0 and 1 with fewer than two observed cells or a standard deviation below $10^{-6}$); predictions are de-standardized, observed cells restored and overlapping windows averaged. V1: 30 epochs, patience 5; V2: 200 epochs, patience 20; early stopping on the masked validation partition. PyPOTS 1.5 with the PyTorch build of the Colab GPU runtime (recorded). The seed of each model is taken from SHA-256(`manifest | ticker | D | block | rate | seed | saits | mask type | confirmatory-v1`), with deterministic cuDNN. Each condition is scored raw (`saits_winnorm_V1`, `saits_winnorm_V2`) and after the OHLC projection (suffix `_proj`). BRITS is not run because of its cost.

# 6. Endpoint

For each condition $c$ and method,

$$\text{NRMSE}_c = \frac{1}{5} \sum_{d=1}^{5} \frac{1}{s_d} \sqrt{\frac{1}{|M_{c,d}|} \sum_{t \in M_{c,d}} \left(\hat{x}_{t,d} - x_{t,d}\right)^2},$$

where $M_{c,d}$ is the set of masked test cells of feature $d$ and $s_d$ the training-observed standard deviation. The asset-level score is the mean over the asset's conditions. Methods are summarized by the mean over assets (primary) and the median (secondary).

# 7. Primary hypotheses and decision rule

All primary hypotheses concern fold D with feature-wise masks, with CFRI = `cfri_fixed_proj`.

- **H1.** CFRI has lower NRMSE than log-linear interpolation (`log_linear`).
- **H2.** CFRI has lower NRMSE than the original ITRI (`itri_w1_p5`).
- **H3.** The learned cross-feature residual improves CFRI: CFRI has lower NRMSE than the same pipeline with $g = 0$ (`cfri_fixed_no_cross_proj`, same volume rule and projection).
- **H4.** CFRI has lower NRMSE than normalized SAITS V2 (`saits_winnorm_V2`), paired on the conditions SAITS completed.

The comparators of H1, H2 and H4 are used exactly as defined in the earlier protocols, without OHLC projection. H3 compares two pipelines with the same projection. Comparisons in which every comparator also receives the projection are secondary (Section 8).

**Decision rule.** A hypothesis is met if and only if (i) the mean asset-level NRMSE of CFRI is lower than that of the comparator and (ii) CFRI is lower in at least $k(n)$ of the $n$ analyzed assets, where $k(n)$ is the smallest $k$ with $P(X \ge k) < 0.01$ for $X \sim \text{Binomial}(n, 1/2)$; $k(34) = 25$. The ETFs share market shocks and are not independent, so $k(n)$ is a prespecified consistency threshold, not a significance test. Each hypothesis is decided separately. If the signs of the asset-level differences were independent and equally likely, a given hypothesis would be met by chance with probability below 0.01, and at least one of the four with probability below 0.04 (Bonferroni).

**Interpretation, fixed in advance.**

- All four met: CFRI is confirmed on 2026 data against log-linear interpolation, the original ITRI and SAITS V2, and the learned residual contributes beyond the projection and the volume rule.
- H3 not met: the gain of CFRI on new data is not attributed to the learned residual; the manuscript states which components (projection, volume rule) carry it.
- H1, H2 or H4 not met: CFRI is reported as not confirmed against that comparator on 2026 data.
- In every case all four results are reported together with the full secondary tables. No setting, method, metric, mask or asset is changed after data access.

**Evaluability of H4.** If SAITS V2 completes for fewer than all eligible assets, H4 is evaluated on the completed assets with $k$ recomputed, and the shortfall is reported. Without any SAITS V2 result, H4 is reported as not evaluable.

# 8. Secondary and descriptive analyses

These analyses are prespecified and reported in full, without decision rules.

1. **Projection-equalized comparisons:** CFRI against `log_linear_proj`, `itri_w1_p5_proj` and `saits_winnorm_V2_proj`, with the same statistics as the primary hypotheses (the criterion is shown for information only).
2. **Paired asset-level differences** for every method: mean and median difference, relative difference, 95% percentile bootstrap interval over assets (10,000 resamples, seed 20261027) and the Wilcoxon signed-rank p-value (descriptive; dependence between assets is ignored).
3. **Breakdowns** by block length, missing rate, feature and region, and leave-one-region-out recomputation of H1 to H3.
4. **Alternative summaries:** normalized MAE, price-only NRMSE (four prices), volume NRMSE and medians.
5. **OHLC consistency:** share of rows with an imputed price that violate the OHLC order, for every method and for the true values.
6. **Whole-row masks:** the same tables. With all prices of a day hidden, the CFRI residual is inactive by construction and CFRI reduces to log-linear prices with the volume rule; these results describe that regime and carry no hypothesis.
7. **SAITS** V1 and V2, raw and projected, under both mask types; CFRI-adaptive and every other variant.
8. **Run time and hardware.**

# 9. Failures and deviations

- A condition at which a CPU method raises an error is recorded and excluded for all CPU methods. A SAITS condition that fails is recorded, and SAITS comparisons are paired on completed conditions. Counts are reported.
- Excluded assets are listed with their reasons; $n$ and $k(n)$ follow the analyzed set.
- Any deviation from this protocol, such as a code correction, is recorded in `DEVIATIONS.md` with date, reason and effect. Analyses affected by a deviation made after data access are labeled exploratory. A full rerun is permitted only to repair an execution failure, and both runs are reported.
- Resuming an interrupted session is not a rerun: finished shards are verified by SHA-256 and kept.
- SAITS results can differ slightly across GPU types and PyTorch builds; the single confirmatory run is reported.

# 10. Exploratory evidence from folds A to C (context, not confirmatory)

These results were computed with the frozen code (`analyze_exploratory.py`) on the 2015 to 2025 data of the exploratory ITRI2 run. CFRI-adaptive reproduces the archived ITRI2 results within $5.6 \times 10^{-13}$ (165,240 feature rows). Normalized SAITS values are the archived amendment v2 results on the same masks.

Table 1. Mean asset-level NRMSE under feature-wise masks, 34 ETFs (folds A to C, exploratory).

| Method | Fold A | Fold B | Fold C | Pooled |
|---|---:|---:|---:|---:|
| CFRI (fixed setting, with projection) | 0.2238 | 0.1941 | 0.1699 | 0.1959 |
| CFRI without residual, g = 0 (with projection) | 0.2485 | 0.2149 | 0.1856 | 0.2163 |
| CFRI without projection | 0.2400 | 0.2071 | 0.1806 | 0.2092 |
| CFRI-adaptive (archived ITRI2 rule) | 0.2477 | 0.2158 | 0.1867 | 0.2167 |
| Log-linear interpolation with projection | 0.2592 | 0.2243 | 0.1938 | 0.2258 |
| Log-linear interpolation | 0.3043 | 0.2602 | 0.2220 | 0.2622 |
| Linear interpolation | 0.3126 | 0.2685 | 0.2300 | 0.2704 |
| Original ITRI, w = 1, p = 5 | 0.3085 | 0.2739 | 0.2346 | 0.2723 |
| Chained ridge | 0.3651 | 0.3061 | 0.3278 | 0.3330 |
| Normalized SAITS V2 (amendment v2) | 0.2685 | 0.2155 | 0.1817 | 0.2219 |
| Normalized SAITS V1 (amendment v2) | 0.3053 | 0.2338 | 0.1929 | 0.2440 |

Table 2. The primary comparisons applied to folds A to C (exploratory): mean difference CFRI minus comparator in asset-level NRMSE, 95% bootstrap interval over assets, and the number of ETFs in which CFRI is lower.

| Hypothesis | Fold A | Fold B | Fold C | Pooled |
|---|---|---|---|---|
| H1 vs log-linear | −0.0805 (−0.0858, −0.0754); 34/34 | −0.0661 (−0.0716, −0.0607); 34/34 | −0.0521 (−0.0576, −0.0470); 34/34 | −0.0663 (−0.0699, −0.0627); 34/34 |
| H2 vs original ITRI | −0.0847 (−0.0918, −0.0780); 34/34 | −0.0798 (−0.0878, −0.0724); 34/34 | −0.0647 (−0.0728, −0.0575); 34/34 | −0.0764 (−0.0815, −0.0715); 34/34 |
| H3 vs CFRI without residual | −0.0247 (−0.0267, −0.0228); 34/34 | −0.0208 (−0.0228, −0.0189); 34/34 | −0.0157 (−0.0174, −0.0141); 34/34 | −0.0204 (−0.0216, −0.0192); 34/34 |
| H4 vs SAITS V2 | −0.0447 (−0.0495, −0.0401); 34/34 | −0.0215 (−0.0246, −0.0188); 34/34 | −0.0118 (−0.0141, −0.0096); 34/34 | −0.0260 (−0.0279, −0.0241); 34/34 |

- The advantage of CFRI comes mainly from the prices (pooled price-only NRMSE 0.0668 for CFRI against 0.1378 for log-linear interpolation) and grows with gap length (block 1: 0.1833 against 0.2116; block 20: 0.2238 against 0.3516).
- The margin over SAITS V2 narrows from fold A to fold C (−0.0447, −0.0215, −0.0118) as the training period grows from 4 to 8 years; fold D has 10 training years. SAITS with projection was not available for folds A to C. H4 is therefore the least certain hypothesis.
- Under whole-row masks, CFRI equals log-linear prices plus the volume rule (pooled 0.2515 against 0.2613); chained ridge fails (2.11).
- Among rows with an imputed price under feature-wise masks, the OHLC order is violated by 56% of log-linear, 56% of ITRI, 41% of CFRI without projection and 13% of chained ridge rows; the true data have no violations.

# 11. Software, code and execution

**Code** (folder `cfri_confirmatory/code/`): `cfri_core.py` (methods, masks and scoring), `confirm_data.py` (data acquisition and quality rules), `confirm_cpu.py` (CPU methods on fold D), `confirm_saits.py` (normalized SAITS on fold D), `confirm_analyze.py` (prespecified analysis and export), `run_confirm.py` (command-line driver), `run_exploratory.py` and `analyze_exploratory.py` (exploratory runs and summaries on folds A to C). SHA-256 values, in the format of `sha256sum -c`:

```
ef33fa6f0fdcb8a79b0f364f9ff00b7820a707e8323f6f03bac554de5d657721  cfri_core.py
3cedfb265cf71209ed4234d0e05e22b14a99495311ccebe06a46796413bc92db  confirm_data.py
fecd0fbe7a3f250b6f28ba5cea08f4c326f5799296caf0e547d52c8411179cf6  confirm_cpu.py
86b592b9d380cc0fd694d35f8031f8dac16a130a5bfed022c56b180ca27f07a4  confirm_saits.py
cccc4bb504b518b222cab4ec053f0e94dd4806652f16462c04419e320fd84eb6  confirm_analyze.py
99236ec45445cb6954a4ab657a2c445c6751c063f15fbdc6165d3274b9a96661  run_confirm.py
27d242fcee4ea24ea61bcf63c931af5f638bc60701f6d46fcc1331087268553b  run_exploratory.py
1a0dad07adc200f15f1a493684de4e8f41a661300512a339b41f297db7e9d822  analyze_exploratory.py
```

`run_confirm.py` records the hashes of the code it runs with (`code_sha256.json`) and stops if the code changes between steps. The notebook verifies the six run files against the values above before any step.

**Environment.** Python 3.13 (Colab), NumPy 2.1.3, pandas 2.2.3, SciPy 1.18.1, scikit-learn 1.6.1, yfinance 1.7.0, exchange_calendars 4.13.2, PyPOTS 1.5, pygrinder 0.7, and the PyTorch build of the Colab runtime (recorded in the run metadata).

**Execution.** Colab notebook `CFRI_confirmatory_fold_D_Colab.ipynb` on a GPU runtime, with resumable steps: data and quality control, CPU methods, a one-condition SAITS smoke test, SAITS V2 and V1 under both mask types (up to four processes at a time), analysis and export. A SAITS job stops, to be resumed, on a device error or after three consecutive failed conditions; a shard without any completed condition is never saved. The analysis uses only shards whose SHA-256 and code fingerprint match the frozen code, and stops if a CPU shard of an eligible asset is missing. Results are written to the author's Google Drive and checkpointed per asset.

**Public outputs.** `data_receipt.json`; per-condition metrics (`fold_D_cpu/*.jsonl`, `fold_D_saits/*.csv`) with SHA-256 sidecars; `fold_D_results.json`; `fold_D_report.md`; asset-level tables; the code and its hashes. Raw prices are never included.

# 12. Timeline

1. 27 October 2026: protocol, code and notebook frozen and committed to the public repository.
2. After the commit: data acquisition and the runs of Section 11.
3. After complete outputs: the prespecified analysis; the manuscript reports all primary and secondary results, labeling fold D as confirmatory and folds A to C as exploratory.

# Appendix A. Fold D conditions per (block length, missing rate)

Feature-wise masks: hidden test cells per condition; whole-row masks: hidden test days. Computed for 187 test rows (the XNYS sessions of 2 January to 30 September 2026); each asset's own row counts are used in the run.

| Block length | Rate 0.10 | Rate 0.20 | Rate 0.30 |
|---:|---|---|---|
| 1 | 74 cells / 15 days | 147 cells / 29 days | 220 cells / 44 days |
| 2 | 74 cells / 15 days | 147 cells / 29 days | 220 cells / 44 days |
| 3 | 74 cells / 15 days | 147 cells / 29 days | 220 cells / 44 days |
| 5 | 74 cells / 15 days | 147 cells / 29 days | 220 cells / 44 days |
| 10 | 74 cells / 15 days | 147 cells / 29 days | 220 cells / 44 days |
| 20 | infeasible (excluded for all methods) | 147 cells / 29 days | 220 cells / 44 days |

# Appendix B. Validation ranking behind the fixed setting (folds A to C pooled, feature-wise masks)

| Price candidate | Mean validation NRMSE (four prices) |
|---|---:|
| α = 10, g = 0.75 | 0.059125 |
| α = 1, g = 0.75 | 0.059458 |
| α = 0.1, g = 0.75 | 0.059531 |
| α = 10, g = 1 | 0.064018 |
| α = 1, g = 0.5 | 0.064538 |

| Volume rule | Mean validation NRMSE (volume) |
|---|---:|
| η = 0.5 | 0.659327 |
| η = 0.75 | 0.662404 |
| η = 0.25 | 0.675316 |
| η = 0 (log-linear) | 0.721203 |
| ITRI | 0.733775 |
| linear | 0.765138 |

The five best of 15 price candidates are shown. Selecting on the validation year of each fold alone gives the same setting ($\alpha = 10$, $g = 0.75$, $\eta = 0.5$) in folds A, B and C.
