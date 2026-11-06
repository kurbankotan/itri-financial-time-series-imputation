# CFRI new-asset holdout (study `cfri-heldout-assets-v1`)

This folder contains the protocol, frozen source, Colab notebook, receipts, and results of the locally frozen
evaluation of CFRI on 16 ETFs that were not part of the 34-ETF development universe. In the manuscript: Sections
3.12, 4.6, and 5.7; Supplementary Note S3; Supplementary Tables S23 to S25.

**Status.** The exact OHLC projection and the peer-return bridge tested here were designed after the fold D results
of `../cfri_confirmatory/` had been inspected; an interaction-feature residual model explored at the same time was not
adopted. The notebook saved the source bundle, its SHA-256, and a UTC time stamp in Google Drive at 04:21:09 UTC on
29 October 2026, before the first data request at 04:21:18 UTC, and the analysis refuses a freeze record that is
missing, later than the acquisition, or inconsistent with the source. This folder was published after the run, so the
publication date is not a public pre-run time stamp. The 2026 calendar and the 34 peer ETFs are shared with fold D,
so the study tests new assets, not a new period.

## Design

- **Targets** (fixed in this order, no substitutions): XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, SPY, QQQ, IWM,
  EFA, EEM, IBB, IYR. **Peers:** the 34 country ETFs of the earlier study.
- **Partitions:** training 2015 to 2024, validation 2025, test 2 January to 30 September 2026. The strict block
  generator gives 51 feasible conditions per target and mask type (816 per mask type).
- **Methods.** CFRI with its original fixed setting (alpha = 10, g = 0.75, eta = 0.5) and the exact OHLC projection;
  CFRI with the peer-return bridge for whole-row outages; normalized SAITS V2 with the same projection, with and
  without the same bridge. The exact projection moves the imputed log prices of a day as little as possible, in
  squared distance, to satisfy the order constraints with observed prices fixed. The bridge predicts the target's
  daily log returns from observed peer returns with a ridge regression fitted on training days and adds their
  cumulative sum, detrended to vanish at both observed endpoints, to the four log prices of a whole-row gap; its gain
  is selected on the validation year (Supplementary Note S3 gives every rule).
- **Hypotheses.** Each requires a lower equal-ETF mean NRMSE and lower error in at least 12 of the 16 ETFs.
  H1, feature-wise masks: CFRI against SAITS V2. H2, whole-row masks: CFRI with the bridge against CFRI without it.
  H3, whole-row masks: CFRI with the bridge against SAITS V2 with the same bridge and its own validation-selected gain.
- **Stress test.** In a market-wide outage the peer closes are hidden on the same dates, and the bridge must return
  the no-peer estimate exactly.

`PROTOCOL.md` is the complete protocol.

## Results

| Hypothesis | CFRI | Comparator | CFRI minus comparator (descriptive 95% ETF bootstrap) | ETF wins | Decision |
|---|---:|---:|---:|---:|---|
| H1, feature-wise | 0.13887 | 0.13990 | -0.00103 (-0.00511 to 0.00261) | 9/16 | not met |
| H2, whole-row | 0.15714 | 0.17133 | -0.01419 (-0.02447 to -0.00606) | 16/16 | met |
| H3, whole-row | 0.15714 | 0.25771 | -0.10058 (-0.12040 to -0.08335) | 16/16 | met |

Under feature-wise masks SAITS V2 was more accurate on prices and at five- to twenty-day gaps, and CFRI on volume and
at one- to three-day gaps. Under whole-row masks CFRI had lower error than SAITS V2 in every ETF even without the
bridge (0.17133 against 0.26168). EFA and EEM, whose holdings overlap the peer country funds, supplied 53% of the
bridge's benefit. In the market-wide stress test the bridge found no peer data and CFRI returned exactly to its
no-peer result.

## Contents

| Path | Content |
|---|---|
| `PROTOCOL.md` | Frozen protocol (identical to `code/PROTOCOL.md`) |
| `CFRI_holdout_v1_Colab.ipynb` | Run notebook (instructions in Turkish); it embeds the frozen source bundle and records the local freeze before any data request |
| `requirements-colab.txt` | Pinned packages installed by the notebook; PyTorch comes from the Colab runtime |
| `code/` | Frozen source, byte-identical to `results/frozen_source_bundle.zip`; `code/README.md` holds the original run instructions in Turkish and `code/SOURCE_SHA256.json` the source hashes |
| `results/` | Freeze, data, and code receipts; condition and feature scores; peer-gain tuning log; analysis JSON and report; `CFRI_holdout_v1_results_full.zip` with every condition checkpoint and its SHA-256 sidecar |
| `analysis/verify_reported_numbers.py` | Checks the reported numbers, the checkpoints, and the frozen source; `--write-tables` writes Supplementary Tables S23 to S25 |

## Verify

```bash
python cfri_holdout_v1/analysis/verify_reported_numbers.py     # from the repository root; numpy and pandas only
```

The unit tests of the frozen source run on synthetic data. From `cfri_holdout_v1/code/`, with the packages of
`requirements-colab.txt` installed:

```bash
python -m unittest test_holdout_core test_holdout_analyze
python -m unittest test_holdout_saits                          # also needs PyTorch
```

## Re-running

Open the notebook in Google Colab with a GPU runtime and select *Run all*. It installs the pinned packages, writes the
local freeze record, downloads the data from Yahoo Finance, applies the quality rules, runs the CPU and SAITS
conditions with verified, resumable checkpoints, analyzes them, and downloads a results ZIP without raw prices.
Providers revise historical records, so a fresh download can differ from the study files, whose SHA-256 values are in
`results/data_receipt.json`.

## Data

Raw Yahoo Finance tables are not redistributed and must never be added to this repository; the notebook keeps them in
a private Drive folder. The result files contain errors, hashes, settings, and timings only.
