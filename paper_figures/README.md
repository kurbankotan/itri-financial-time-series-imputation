# Iterative Temporal Regression Imputation (ITRI) and Cross-Feature Residual Interpolation (CFRI)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurbankotan/itri-financial-time-series-imputation/blob/main/notebooks/ITRI_Window_Ablation_Colab.ipynb)
[![Verify published results](https://github.com/kurbankotan/itri-financial-time-series-imputation/actions/workflows/verify-results.yml/badge.svg)](https://github.com/kurbankotan/itri-financial-time-series-imputation/actions/workflows/verify-results.yml)

This repository accompanies the manuscript

> K. Kotan, *Iterative Temporal Regression Imputation for Contiguous Missing Blocks in Multivariate Financial Time Series: A Leakage-Aware Ablation, External Validation, Comparator Failure Analysis, and a Cross-Feature Extension* (manuscript under review).

It contains every piece of code, every frozen protocol, the condition-level results of every method, and the tables and figures reported in the manuscript and its Supplementary Material. Release v2.1.0 adds the cross-feature extension (CFRI) and its two hold-out evaluations to release v2.0.0.

ITRI is a retrospective imputer for contiguous gaps: it initializes a gap by linear interpolation and refines the hidden cells with a feature-wise ordinary least squares model on a centered temporal window of width *w*, applied for *p* passes, restoring the observed cells after every pass. CFRI corrects log-linear interpolation of a hidden price with a ridge model of the same-day deviations of the other observed prices, shrinks volume toward its local median, and projects imputed cells onto the OHLC order; for a whole-row outage of one ETF it can add a peer-return bridge estimated from other ETFs.

## What is where

| Part | Folder | Manuscript |
|---|---|---|
| Development ablation: BIST 30 and S&P 500, 14 features, 28 window and pass configurations | `notebooks/`, `scripts/`, `results/`, `metadata/` | Sections 4.1, 4.2, 5.1; Table 1; Figure 1; Supplementary Tables S1 to S3 and S5 to S7; Supplementary Figures S1 to S3 |
| Matched-mask comparators on the development data: PCHIP, KNN, chained equations | `development_comparators/` | Sections 4.3, 5.2; Supplementary Tables S8 and S9 |
| External validation on 34 country ETFs: frozen protocol, six comparators, exploratory sweep, level-shift diagnostic, normalization amendment | `external_validation/` | Sections 4.4, 5.3 to 5.5, 6.2, 6.3; Tables 2 and 3; Figure 2; Supplementary Note S1; Supplementary Tables S4 and S10 to S18; Supplementary Figures S4 and S5 |
| CFRI: exploratory folds A to C and the prespecified fold D test on January to September 2026 | `cfri_confirmatory/` | Sections 3.12, 4.5, 5.6; Table 4; Supplementary Note S2; Supplementary Tables S19 to S22; Supplementary Figures S6 and S7 |
| CFRI: locally frozen holdout on 16 ETFs outside the development universe, with the exact OHLC projection and the peer-return bridge | `cfri_holdout_v1/` | Sections 3.12, 4.6, 5.7; Supplementary Note S3; Supplementary Tables S23 to S25 |
| Figures of the ITRI studies | `paper_figures/` | Figures 1 and 2; Supplementary Figures S1 to S5 (see the numbering map below) |

### Numbering map

The ITRI scripts, table files, and figure files of release v2.0.0 use the numbering of an earlier, longer version of the manuscript. Their labels correspond to the manuscript as follows.

| Label in scripts and file names | Manuscript |
|---|---|
| Tables 1, 2, 3, 4, 5 | Supplementary Tables S1, S2, S3, S4, S5 |
| Table 6 | Table 1 |
| Tables 7, 8, 9, 10 | Supplementary Tables S6, S7, S8, S9 |
| Table 11 | Table 2 |
| Table 12 | Supplementary Table S10 |
| Table 13 | Table 3 |
| Tables 14, 15 | Supplementary Tables S11, S12 |
| Table 16 | Supplementary Table S17 |
| Tables 17, 18, 19 | Supplementary Tables S13, S14, S15 |
| Table 20 | Supplementary Table S18 |
| Table 21 | Supplementary Table S16 |
| Figure 1 (`figure_1_itri_workflow.png`) | Figure 1 |
| Figures 2, 3, 4 (`figure_2_...`, `figure_3_...`, `figure_4_...`) | Supplementary Figures S1, S2, S3 |
| Figure 5 (`figure_5_external_block_length.png`) | Figure 2 |
| Figures 6, 7 (`figure_6_...`, `figure_7_...`) | Supplementary Figures S4, S5 |
| Sections 4.3 and 4.4; 4.5 | Section 4.3; Section 4.4 |
| Sections 5.1 to 5.5; 5.6 and 5.7 | Section 5.1; Section 5.2 |
| Sections 5.8, 5.9, 5.10 | Sections 5.3, 5.4, 5.5 |
| Sections 6.1 to 6.3; 6.4; 6.5; 6.6 and 6.7 | Section 6.1; Section 6.2; Section 6.3; Section 6.4 |

## Main results

- **Development study.** Validation selected a one-day window with five passes (w = 1, p = 5). On the held-out test partition it reached mean NRMSE 0.198094, 11.03% below the inherited w = 3, p = 3 setting and 14.77% below linear interpolation.
- **External validation (frozen protocol, 34 ETFs, 3 folds, 5,508 matched block masks).** ITRI reached asset-level mean NRMSE 0.2723 against 0.2704 for linear interpolation; the two agree to four decimals on open, high, low, and close, and the whole difference comes from volume. Every frozen comparator (PCHIP, KNN, chained ridge, BRITS, SAITS) had higher error than ITRI.
- **Exploratory analyses.** The fitted one-day price coefficients sum to one, which makes interpolation a fixed point of the refinement; the prespecified setting ranked 28th of 28 on the external panel; every comparator whose output depends on the absolute training-scaled level failed in proportion to the test-period displacement; with per-window normalization SAITS reached 0.2219 (200 epochs), below ITRI in all 34 assets and at every gap length.
- **CFRI, prespecified fold D test (34 ETFs, January to September 2026).** All four primary hypotheses were met: CFRI reached 0.2822 against 0.3616 for log-linear interpolation, 0.3864 for ITRI, 0.3049 for CFRI without its residual, and 0.2950 for normalized SAITS V2 (lower in 34, 34, 34, and 29 of 34 ETFs). With the same OHLC projection, SAITS V2 reached 0.2861 and CFRI was lower in only 13 ETFs, so no overall advantage over projected SAITS was established.
- **CFRI, new-asset holdout (16 ETFs, same 2026 period).** Under feature-wise masks CFRI (0.1389) and projected SAITS V2 (0.1399) were close and CFRI was lower in only 9 of 16 ETFs, so the hypothesis was not met. Under whole-row masks the peer bridge lowered CFRI from 0.1713 to 0.1571 in all 16 ETFs and CFRI with the bridge was lower than SAITS V2 with the same bridge (0.2577) in all 16 ETFs. In the market-wide stress test the bridge has no peer data and CFRI returns exactly to its no-peer result.

## Verify the reported numbers

No market data are needed. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_published_results.py                       # development ablation
python development_comparators/verify_comparator_numbers.py      # development comparators
python external_validation/code/verify_reported_numbers.py       # external validation
python cfri_confirmatory/analysis/verify_reported_numbers.py     # CFRI folds A to C and fold D
python cfri_holdout_v1/analysis/verify_reported_numbers.py       # CFRI new-asset holdout
```

Each script prints one `PASS` or `FAIL` line per check and exits with status 0 only if every check passes. The two CFRI scripts also check the SHA-256 hashes of the frozen code against the hashes recorded by the runs. The GitHub Actions workflow in `.github/workflows/verify-results.yml` runs all five on every push.

The tables and figures can be regenerated from the result files with

```bash
python external_validation/code/analyze_external.py                     # external tables and related text values
python external_validation/code/analyze_brits_subset.py                 # BRITS subset (Supplementary Table S16)
python external_validation/code/cross_hardware_check.py                 # CPU versus A100 check
python external_validation/code/make_figures.py                         # Figure 2 and Supplementary Figures S4 and S5
python cfri_confirmatory/analysis/fold_D_supplement.py                  # Supplementary Tables S19 to S22, Figures S6 and S7
python cfri_holdout_v1/analysis/verify_reported_numbers.py --write-tables   # Supplementary Tables S23 to S25
```

## Rerun the experiments

- **Development ablation.** Open the Colab notebook (badge above) and select *Runtime, Run all*, or run `python scripts/run_full_experiment.py`. Both download the two index series from Yahoo Finance; `metadata/data_manifest.json` records the SHA-256 hashes of the files used for the reported run.
- **Development comparators.** `development_comparators/README.md` explains how to rerun them from the two original study files.
- **External validation.** `external_validation/README.md` describes the acquisition script, the frozen runners, the Colab notebooks, and the exploratory runners. The runs need the vendor snapshot of the 34 ETFs, which is not redistributed (see below).
- **CFRI fold D.** `cfri_confirmatory/README.md` describes the frozen protocol, the run notebook, and the command-line stages; the notebook downloads the data afresh and verifies the SHA-256 hashes of the code before running it.
- **CFRI new-asset holdout.** `cfri_holdout_v1/README.md` describes the protocol and the Colab notebook, which records a local source freeze before downloading any data.

## Data

Daily market records come from Yahoo Finance through `yfinance` and are not redistributed, in line with the provider's terms. The repository records the SHA-256 hash of every input file (`metadata/data_manifest.json`, `external_validation/metadata/acquisition_qc_receipt.json`, `cfri_confirmatory/confirmatory_D/data_receipt.json`, `cfri_holdout_v1/results/data_receipt.json`) and of the private ETF snapshot of the external validation (`48dc63654f63db3ef9d6d5db7ab69ad502aa079e9f6deb4b276cf9b1170f324b`), so that a fresh download can be checked against the study files. Providers revise historical records, so a later download can differ from them. No file in this repository contains price or volume levels: the result files hold reconstruction errors, mask hashes and seeds, timings, fitted coefficients, and diagnostics.

## Software

The development ablation used Python 3.13, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, and yfinance 0.2.66 (`requirements.txt`). The versions used by the external runs, including PyTorch and PyPOTS for BRITS and SAITS, are listed in `external_validation/requirements-runs.txt` and recorded per session in `external_validation/metadata/`. The CFRI runs used Python 3.13.16, NumPy 2.1.3, pandas 2.2.3, yfinance 1.7.0, exchange_calendars 4.13.2, PyPOTS 1.5, and PyTorch 2.11.0, with scikit-learn 1.6.1 and SciPy 1.18.1 for fold D and scikit-learn 1.7.2 for the new-asset holdout; the versions are recorded in the result files of each study.

## Citation and license

Citation metadata are in `CITATION.cff`; please cite the article once it is published. The code is released under the MIT License (`LICENSE`); the result files, tables, and figures are released under CC BY 4.0.

## Author

Kurban KOTAN<br>
Department of Artificial Intelligence and Machine Learning, Faculty of Computer and Information Sciences<br>
Manisa Celal Bayar University, Manisa, Turkey<br>
ORCID: [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565) · kurban.kotan@cbu.edu.tr
