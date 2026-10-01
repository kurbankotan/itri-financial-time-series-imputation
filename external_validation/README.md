# External validation on 34 country ETFs

This folder holds everything behind Sections 4.5, 5.8-5.10, 6.4, and 6.5 of the manuscript: the frozen protocols, the code that produced every reported run, the condition-level results of every method, the tables and figure data, and a script that recomputes every reported number. Only the vendor OHLCV snapshot is withheld (see *Data* below).

## Design in brief

| Item | Specification |
|---|---|
| Universe | 34 US-listed country-exposure ETFs that passed prespecified, score-blind eligibility checks (4 Americas, 15 Europe, 13 Asia-Pacific, 2 Middle East and Africa); no result-based substitution |
| Features | Dividend- and split-adjusted open, high, low, close, volume (raw OHLCV, no engineered indicators) |
| Folds | A: train 2015-2018, validation 2019, test 2020-2021; B: 2015-2020, 2021, 2022-2023; C: 2015-2022, 2023, 2024-2025 |
| Masks | Strict non-adjacent runs (`code/masking.py`), one full run per feature first, edge guard 20; block lengths 1, 2, 3, 5, 10, 20; rates 0.10, 0.20, 0.30; seeds 11, 23, 37 |
| Conditions | 54 per ETF and fold, 5,508 in total; identical masks for every method |
| Scaling | Mean and SD of the initially observed training cells of each condition, applied to validation and test |
| Primary endpoint | Macro mean over the five features of RMSE divided by the training-observed SD (`rmse_trainstd`); standardized MAE (`mae_trainstd`) secondary |
| ITRI | w = 1, p = 5, frozen from the development study |
| Comparators | Linear interpolation, PCHIP, KNN (k = 5), chained ridge, BRITS, SAITS (PyPOTS 1.5; 64-day windows, stride 16, at most 30 epochs, patience 5) |
| Unit of analysis | Asset: conditions and folds are averaged within each ETF; descriptive summaries and paired win counts, no inferential tests |

The classical protocol (`protocol/PROTOCOL_v1.md`) and the neural extension (`protocol/NEURAL_PROTOCOL_v1.md`) were frozen on 23 September 2026, after the classical comparators had been run and before any BRITS or SAITS result existed. Three analyses were added after the frozen results had been seen and are labeled exploratory throughout: a 28-configuration ITRI sweep on the identical masks, a level-shift diagnostic, and protocol amendment v2 (`protocol/NEURAL_PROTOCOL_v2_exploratory.md`, 29 September 2026), which reran SAITS (V1: 30 epochs; V2: 200 epochs, patience 20) on the full grid and BRITS (V1) on the fold A, mask seed 11 subset of 612 conditions with per-window instance normalization.

## Verify and regenerate

From the repository root, with the packages in `requirements.txt` (NumPy, pandas, SciPy, Matplotlib):

```bash
python external_validation/code/verify_reported_numbers.py   # every number in Sections 5.8-5.10; exit 0 = all PASS
python external_validation/code/analyze_external.py          # writes tables/ (Tables 11-20)
python external_validation/code/analyze_brits_subset.py      # writes tables/B*.csv (Table 21)
python external_validation/code/cross_hardware_check.py      # writes tables/R1_cross_hardware_check.csv
python external_validation/code/make_figures.py              # writes paper_figures/figure_5-7
cd external_validation/code && python -m pytest -q           # offline unit tests of the mask generator, QC logic and methods
```

## Folder contents

### `protocol/`

| File | Content |
|---|---|
| `PROTOCOL_v1.md` | Classical external-validation protocol, locked 23 September 2026 |
| `NEURAL_PROTOCOL_v1.md` | BRITS and SAITS extension, frozen 23 September 2026 before any neural result |
| `NEURAL_PROTOCOL_v2_exploratory.md` | Amendment v2: per-window normalization, exploratory, with its interpretation rules and final status |

### `notebooks/`

| Notebook | Runs |
|---|---|
| `ITRI_classical_full_Colab.ipynb` | Linear, PCHIP, KNN, chained ridge, and ITRI w = 1, p = 5 on all 5,508 conditions (CPU) |
| `ITRI_BRITS_SAITS_resume_Colab.ipynb` | Frozen BRITS and SAITS runs, resumable with checkpoints (A100) |
| `ITRI_neural_windownorm_Colab.ipynb` | Amendment v2 runs, resumable (A100) |

Each notebook asks for the private snapshot at run time and downloads its results as a ZIP when it finishes.

### `code/`

| File | Role |
|---|---|
| `acquire_and_qc.py` | Downloads the frozen universe through `yfinance` and applies the score-blind eligibility checks; writes the acquisition receipt |
| `audit_received_snapshot.py` | Audits the private snapshot in memory and exports summaries only (`metadata/received_snapshot_audit.json`) |
| `check_external_mask_grid.py` | Checks all 486 prespecified mask shapes |
| `masking.py` | Strict block-mask generator |
| `run_classical_benchmark.py` | Frozen classical runs (linear, PCHIP, KNN, chained ridge, ITRI w = 1, p = 5) |
| `run_neural_benchmark.py` | Frozen BRITS and SAITS runs |
| `analyze_all_benchmarks.py` | Audit and combined tables of the frozen run (`results/v1_frozen/`) |
| `sweep_itri_external.py` | Exploratory 28-configuration sweep; also records the fitted w = 1 coefficients |
| `level_shift_diagnostic.py` | Test-period displacement from the training distribution per asset and fold |
| `pilot_saits_normalization.py` | CPU pilot that motivated the amendment (its results are not reported) |
| `run_neural_windownorm.py` | Amendment v2 runner |
| `test_masking.py`, `test_acquire_and_qc.py`, `test_benchmark_methods.py` | Offline unit tests |
| `repo_paths.py` | Locations of the public result files |
| `analyze_external.py`, `analyze_brits_subset.py`, `cross_hardware_check.py`, `make_figures.py` | Tables and figures from the public result files |
| `verify_reported_numbers.py` | Recomputes and asserts every external number reported in the manuscript |

The runners and the mask generator are byte-identical to the files used for the reported runs, and the notebooks embed the same code. The two frozen protocols are unchanged; the amendment protocol carries a status note added after its runs had finished. In `acquire_and_qc.py` and `check_external_mask_grid.py` only the default location of the metadata files was changed to `../metadata/`.

### `metadata/`

| File | Content |
|---|---|
| `asset_universe_v1.json`, `asset_universe_34_etfs.csv` | Frozen universe with country and region |
| `acquisition_qc_receipt.json` | Acquisition receipt of 23 September 2026: per-asset coverage, QC outcome, and SHA-256 of every raw and cleaned file |
| `received_snapshot_audit.json` | Audit of the snapshot used for all runs (34 of 34 eligible, 68 verified file hashes, 94,044 daily rows) |
| `classical_run_metadata.json`, `classical_COMPLETE.marker` | Settings, versions, fingerprint, and completion marker of the classical run |
| `neural_v1_run_metadata.json`, `neural_v1_environment_history.json`, `neural_v1_COMPLETE.marker` | Frozen neural run: settings, the four Colab sessions with their versions and hardware, completion marker |
| `analysis_audit_v1.json` | Audit of the combined frozen results (PASS: 192,780 feature rows per partition) |
| `sweep_metadata.json` | Exploratory sweep settings |
| `windownorm_run_metadata_*.json`, `cpu_check_run_metadata_*.json` | Amendment v2 runs (SAITS V1 on CPU, SAITS V2 and BRITS V1 on the A100) and the four CPU pieces of the cross-hardware check |
| `shard_manifest.json` | Checkpoint sidecars (fingerprint, SHA-256, row count) of every amendment shard merged into `results/exploratory/` |

### `results/v1_frozen/` (frozen protocol)

| File | Content |
|---|---|
| `test_feature_rows.csv.gz`, `validation_feature_rows.csv.gz` | One row per asset, fold, condition, method, and feature: 34 x 3 x 54 x 7 x 5 = 192,780 rows per partition |
| `condition_macro.csv.gz`, `validation_test_conditions.csv.gz` | Condition-level macro means |
| `overall_method.csv`, `asset_method.csv`, `region_method.csv`, `fold_method.csv`, `block_method.csv`, `block_rate_method.csv`, `feature_method.csv`, `short_gap_method.csv`, `runtime_method.csv`, `paired_asset_differences.csv`, `paired_win_counts.csv`, `validation_test_asset.csv`, `validation_test_method.csv`, `paper_tables.md` | Summaries written by `analyze_all_benchmarks.py`, including the region-wise summaries |

### `results/exploratory/`

| File | Content |
|---|---|
| `sweep_test_feature_rows.part1.csv.gz` to `part3`, `sweep_validation_feature_rows.part1.csv.gz` to `part3` | All 28 ITRI configurations on the identical masks (771,120 rows per partition; each partition is stored in three consecutive parts to stay below GitHub's 25 MB upload limit, and `code/repo_paths.py` reads them back as one table) |
| `grid_*_asset_mean_nrmse.csv`, `asset_level_*_nrmse_full.csv` | Sweep summaries |
| `itri_w1_coefficients.csv.gz` | Fitted lag, lead, and intercept of every one-day fit (Table 16) |
| `level_shift_by_asset_fold.csv`, `level_shift_joined.csv` | Displacement diagnostic per asset and fold |
| `windownorm_saits_feature_rows.csv.gz` | Amendment v2, SAITS V1 and V2, full grid, test and validation |
| `windownorm_brits_subset_feature_rows.csv.gz` | Amendment v2, BRITS V1 on the 612-condition subset |
| `cross_hardware_cpu_feature_rows.csv.gz` | The four amended pieces recomputed on a CPU |
| `pilot_not_reported/` | Pilot rows that motivated the amendment; not reported in the manuscript |

The feature-level files share these columns: `ticker`, `country`, `region`, `fold`, `block_length`, `rate`, `seed`, the derived mask seeds, `method`, `feature`, `masked_cells`, `mae_raw` and `rmse_raw` (errors in the provider's units), `mae_trainstd` and `rmse_trainstd` (errors divided by the training-observed SD), `fit_and_impute_seconds`, and `convergence_warning_count`; neural rows add the model seed and training settings, and amendment rows add `variant`, `partition`, and `window_fallbacks`. Merged files keep every value exactly as written by the runs.

### `tables/` (generated)

| File | Manuscript |
|---|---|
| `E1_external_primary.csv` | Table 11 |
| `E2_external_by_block.csv` | Table 12, Figure 5(a) |
| `E3_external_by_feature.csv` | Table 13 |
| `S1_sweep_test_grid.csv`, `S1v_sweep_validation_grid.csv`, `S1w_sweep_wins.csv`, `sweep_summary.json` | Table 14, Figure 6(b), Section 5.9 |
| `S2_sweep_by_block.csv` | Table 15 |
| `S3_sweep_by_feature.csv` | Section 5.9, feature breakdown |
| `S4_w1_coefficients.csv` | Table 16 |
| `E4_fold_diagnostic.csv` | Table 17 |
| `Z1_level_shift_spearman.csv` | Table 18 |
| `Z2_level_shift_bins.csv` | Table 19 |
| `Z3_level_shift_by_fold.csv`, `level_shift_joined.csv` | Section 5.10, Figure 7 |
| `A1_amended_neural.csv` | Table 20 |
| `A2_amended_by_block.csv`, `A3_amended_by_feature.csv` | Figure 5(b), Section 5.10 |
| `B1_brits_subset.csv` | Table 21 |
| `B2_brits_subset_by_block.csv`, `B3_brits_subset_by_feature.csv`, `B4_brits_subset_spearman.csv` | Section 5.10 |
| `R1_cross_hardware_check.csv` | Section 5.10, cross-hardware check |
| `TABLES.md` | All of the above in one file |

## Rerunning the runs

The runs need the vendor snapshot (`private/snapshot.zip`, SHA-256 `48dc63654f63db3ef9d6d5db7ab69ad502aa079e9f6deb4b276cf9b1170f324b`). Every runner checks this hash and the hashes in the acquisition receipt, and refuses a different file. A new snapshot can be built with `code/acquire_and_qc.py`, but providers revise historical records, so it may not reproduce the recorded hashes.

- Frozen runs: the three notebooks, or `run_classical_benchmark.py --snapshot private/snapshot.zip --out classical`, then `run_neural_benchmark.py --snapshot private/snapshot.zip --classical <classical results ZIP> --out neural`, then `analyze_all_benchmarks.py --classical <classical results ZIP> --neural neural --out analysis`.
- Exploratory runs read `private/snapshot.zip` relative to the working directory. `level_shift_diagnostic.py` also reads `results/analysis/validation_test_conditions.csv` (the decompressed `results/v1_frozen/validation_test_conditions.csv.gz`). From such a working directory: `python <repo>/external_validation/code/sweep_itri_external.py` and `level_shift_diagnostic.py` write `sweep_out/`; `run_neural_windownorm.py --method saits --variant V1` (or `V2`; `--method brits --variant V1 --folds A --seeds 11` for the BRITS subset) writes `neural_windownorm/`.

Bitwise equality across hardware is not claimed for the neural runs; the cross-hardware check quantifies the differences.

## Data

The OHLCV records come from Yahoo Finance through `yfinance` and are not redistributed. No file in this folder contains price or volume levels.
