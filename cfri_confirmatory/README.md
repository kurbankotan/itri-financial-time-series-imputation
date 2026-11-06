# CFRI confirmatory study (fold D, January to September 2026)

Cross-Feature Residual Interpolation (CFRI) fills gaps in daily OHLCV series by log-linear interpolation plus a
ridge-predicted residual learned from the same-day deviations of the other OHLC prices, a log-local volume rule,
and an OHLC consistency projection of imputed cells. It was developed exploratorily on folds A to C of the external
validation (34 country ETFs, 2015 to 2025; working name ITRI2). This folder freezes a confirmatory test on a period
that played no role in its development.

In the manuscript: Sections 3.12, 4.5, and 5.6; Table 4; Supplementary Note S2; Supplementary Tables S19 to S22;
Supplementary Figures S6 and S7. The later evaluation on 16 new ETFs is in `../cfri_holdout_v1/`.

| Path | Content |
|---|---|
| `CFRI_CONFIRMATORY_PROTOCOL_v1.md` | Frozen protocol: data rules, masks, methods, fixed CFRI setting, endpoint, primary hypotheses H1 to H4 and their decision rule, secondary analyses, deviations policy, code hashes |
| `DEVIATIONS.md` | Log of deviations from the protocol (one: the public commit followed the run) |
| `code/` | Frozen code (SHA-256 values in the protocol) |
| `notebooks/CFRI_confirmatory_fold_D_Colab.ipynb` | Colab notebook used for the run (instructions in Turkish): it embeds the six run files, stops unless their SHA-256 values equal those of the protocol, and runs data and quality control, CPU methods, SAITS, analysis, and the results ZIP |
| `exploratory_AC/` | Exploratory results on folds A to C produced with the same code (no raw prices) |
| `confirmatory_D/` | Confirmatory results on fold D: the results ZIP, extracted summaries, `RESULTS_SUMMARY.md` and `VERIFICATION.md` |
| `analysis/verify_reported_numbers.py` | Checks every CFRI number of the article for folds A to C and fold D, and the code hashes, against the published files |
| `analysis/fold_D_supplement.py` | Recomputes Supplementary Tables S19 to S22 and the data of Supplementary Figures S6 and S7 from the results ZIP and checks them against `fold_D_results.json` |

The two scripts in `analysis/` were added after the run for reporting; they read the published results and do not
change the frozen analysis.

## Status

The confirmatory run on fold D was completed on 27 October 2026 with the frozen code:

- 34 of 34 ETFs were eligible.
- 3,468 CPU conditions and 6,936 SAITS fits ran, with no errors.

All four primary hypotheses were met. When SAITS V2 also receives the OHLC projection, it and CFRI do not differ on
the macro endpoint: SAITS is more accurate on prices and CFRI on volume. See `confirmatory_D/RESULTS_SUMMARY.md` and
`DEVIATIONS.md`.

## Verifying the reported numbers

```bash
python cfri_confirmatory/analysis/verify_reported_numbers.py     # from the repository root; numpy and pandas only
```

## Reproducing the analyses

From this folder:

```bash
python code/run_exploratory.py RAW_DIR OUT_DIR --workers 2        # RAW_DIR: {TICKER}_2015_2025.csv + .meta.json
python code/analyze_exploratory.py OUT_DIR FEATURE_RESULTS_CSV ../external_validation/results/exploratory/windownorm_saits_feature_rows.csv.gz REPORT_DIR
# fold D, from the published results ZIP (no raw data needed):
unzip confirmatory_D/CFRI_fold_D_results.zip -d foldD && python -c "import sys; sys.path.insert(0, 'code'); import confirm_analyze as A; from pathlib import Path; A.analyze(Path('foldD'))"
# Supplementary Tables S19 to S22 and Figures S6 and S7 of the article (fold D), from the same ZIP:
python analysis/fold_D_supplement.py
```

The first two commands rerun the exploratory folds.

- `FEATURE_RESULTS_CSV` is the per-feature output of the archived exploratory ITRI2 run. It is used only for the
  reproduction check.
- The Yahoo Finance files for 2015 to 2026 are not redistributed. Their SHA-256 values are listed in
  `exploratory_AC/run_metadata.json` and `confirmatory_D/data_receipt.json`.

The last two commands rerun the fold D analysis from the published results ZIP and recompute the fold D tables and
figure data of the article's Supplementary Material. A complete rerun of fold D from new downloads uses the notebook
or `python code/run_confirm.py --help` (stages `data`, `cpu`, `saits`, `analyze`).
