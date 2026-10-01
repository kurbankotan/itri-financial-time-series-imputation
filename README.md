# Iterative Temporal Regression Imputation (ITRI)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurbankotan/itri-financial-time-series-imputation/blob/main/notebooks/ITRI_Window_Ablation_Colab.ipynb)
[![Verify published results](https://github.com/kurbankotan/itri-financial-time-series-imputation/actions/workflows/verify-results.yml/badge.svg)](https://github.com/kurbankotan/itri-financial-time-series-imputation/actions/workflows/verify-results.yml)

This repository accompanies the manuscript

> K. Kotan, *Iterative Temporal Regression Imputation for Contiguous Missing Blocks in Multivariate Financial Time Series: A Leakage-Aware Ablation, External Validation, and Comparator Failure Analysis* (under review).

It contains every piece of code, every frozen protocol, the condition-level results of every method, and the tables and figures reported in the manuscript. ITRI is a retrospective imputer for contiguous gaps: it initializes a gap by linear interpolation and refines the hidden cells with a feature-wise ordinary least squares model on a centered temporal window of width *w*, applied for *p* passes, restoring the observed cells after every pass.

## What is where

| Part | Folder | Manuscript |
|---|---|---|
| Development ablation: BIST 30 and S&P 500, 14 features, 28 window and pass configurations | `notebooks/`, `scripts/`, `results/`, `metadata/` | Sections 4.1-4.3, 5.1-5.5; Tables 1-3, 5-8; Figures 1-4 |
| Matched-mask comparators on the development data: PCHIP, KNN, chained equations | `development_comparators/` | Sections 4.4, 5.6, 5.7; Tables 9-10 |
| External validation on 34 country ETFs: frozen protocol, six comparators, exploratory sweep, level-shift diagnostic, normalization amendment | `external_validation/` | Sections 4.5, 5.8-5.10, 6.4-6.5; Tables 4, 11-21; Figures 5-7 |
| Figures as printed in the manuscript | `paper_figures/` | Figures 1-7 |

## Main results

- **Development study.** Validation selected a one-day window with five passes (w = 1, p = 5). On the held-out test partition it reached mean NRMSE 0.198094, 11.03% below the inherited w = 3, p = 3 setting and 14.77% below linear interpolation.
- **External validation (frozen protocol, 34 ETFs, 3 folds, 5,508 matched block masks).** ITRI reached asset-level mean NRMSE 0.2723 against 0.2704 for linear interpolation; the two agree to four decimals on open, high, low, and close, and the whole difference comes from volume. Every frozen comparator (PCHIP, KNN, chained ridge, BRITS, SAITS) had higher error than ITRI.
- **Exploratory analyses.** The fitted one-day price coefficients sum to one, which makes interpolation a fixed point of the refinement; the prespecified setting ranked 28th of 28 on the external panel; every comparator whose output depends on the absolute training-scaled level failed in proportion to the test-period displacement; with per-window normalization SAITS reached 0.2219 (200 epochs), below ITRI in all 34 assets and at every gap length.

## Verify the reported numbers

No market data are needed. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/verify_published_results.py                     # development ablation
python development_comparators/verify_comparator_numbers.py    # Tables 9 and 10
python external_validation/code/verify_reported_numbers.py     # Sections 5.8-5.10, Tables 11-21
```

Each script prints one `PASS` line per check and exits with status 0 only if every check passes. The GitHub Actions workflow in `.github/workflows/verify-results.yml` runs all three on every push.

The external tables and figures can be regenerated from the result files with

```bash
python external_validation/code/analyze_external.py       # Tables 11-20 and related text values
python external_validation/code/analyze_brits_subset.py   # Table 21
python external_validation/code/cross_hardware_check.py   # CPU versus A100 check (Section 5.10)
python external_validation/code/make_figures.py           # Figures 5-7
```

## Rerun the experiments

- **Development ablation.** Open the Colab notebook (badge above) and select *Runtime, Run all*, or run `python scripts/run_full_experiment.py`. Both download the two index series from Yahoo Finance; `metadata/data_manifest.json` records the SHA-256 hashes of the files used for the reported run.
- **Development comparators.** `development_comparators/README.md` explains how to rerun them from the two original study files.
- **External validation.** `external_validation/README.md` describes the acquisition script, the frozen runners, the Colab notebooks, and the exploratory runners. The runs need the vendor snapshot of the 34 ETFs, which is not redistributed (see below).

## Data

Daily market records come from Yahoo Finance through `yfinance` and are not redistributed, in line with the provider's terms. The repository records the SHA-256 hash of every input file (`metadata/data_manifest.json`, `external_validation/metadata/acquisition_qc_receipt.json`) and of the private ETF snapshot (`48dc63654f63db3ef9d6d5db7ab69ad502aa079e9f6deb4b276cf9b1170f324b`), so that a fresh download can be checked against the study files. Providers revise historical records, so a later download can differ from them. No file in this repository contains price or volume levels: the result files hold reconstruction errors, mask seeds, timings, fitted coefficients, and diagnostics.

## Software

The development ablation used Python 3.13, NumPy 2.1.3, pandas 2.2.3, SciPy 1.16.3, scikit-learn 1.6.1, and yfinance 0.2.66 (`requirements.txt`). The versions used by the external runs, including PyTorch and PyPOTS for BRITS and SAITS, are listed in `external_validation/requirements-runs.txt` and recorded per session in `external_validation/metadata/`.

## Citation and license

Citation metadata are in `CITATION.cff`; please cite the article once it is published. The code is released under the MIT License (`LICENSE`); the result files, tables, and figures are released under CC BY 4.0.

## Author

Kurban KOTAN<br>
Department of Artificial Intelligence and Machine Learning, Faculty of Computer and Information Sciences<br>
Manisa Celal Bayar University, Manisa, Turkey<br>
ORCID: [0000-0002-6660-0565](https://orcid.org/0000-0002-6660-0565) · kurban.kotan@cbu.edu.tr
