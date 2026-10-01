# Matched-mask comparators on the development data

This folder holds the comparator analysis of Sections 4.4, 5.6, and 5.7 of the manuscript (Tables 9 and 10). It adds three comparator families to the original window and pass ablation on the BIST 30 and S&P 500 panels, using the same study files, feature engineering, chronological splits, training-only scaling, block masks, and masked-cell metrics as the archived ITRI run.

| Method | Implementation | Setting chosen on validation only |
|---|---|---|
| PCHIP | SciPy `PchipInterpolator`, feature-wise along time | none (no tuned parameter) |
| KNN | scikit-learn `KNNImputer`, nan-euclidean distance, distance weights, training-row donors | k = 10 from {3, 5, 10} |
| Chained equations | scikit-learn `IterativeImputer` with `BayesianRidge`, mean initialization, deterministic | 3 iterations from {3, 5} |

The result files label the chained-equation imputer `MICE`; the manuscript reports it as "Chained ridge, 3 cycles" (Table 9). It produces one point estimate rather than multiple posterior draws.

## Verify

```bash
python development_comparators/verify_comparator_numbers.py
```

The script checks every comparator value in Tables 9 and 10, the selected settings, the win counts, the relative reductions quoted in Section 5.6, the market and gap-length statements, the exact-mask check of Section 4.4, and the partition dates of Section 5.7, using the files in `results/` and the archived ITRI grids in the repository's `results/` folder. No market data are needed.

## Files

| File | Content |
|---|---|
| `run_additional_baselines.py` | Comparator experiment on the exact study files (validation grid, then the locked test settings) |
| `summarize_revision.py` | Matched-mask verification against the archived interpolation baseline and the manuscript summaries |
| `audit_all_experiments.py` | Re-executes all 6,048 archived ITRI metric rows with the functions of `scripts/run_full_experiment.py` |
| `verify_comparator_numbers.py` | Checks the reported numbers from the result files |
| `results/additional_baselines_validation.csv` | Every validation candidate, one row per condition |
| `results/additional_baselines_test.csv` | Locked test settings, one row per condition |
| `results/additional_baselines_summary.csv` | Unpaired means by method, setting, and partition |
| `results/manuscript_comparator_summary.csv` | Means and matched differences against ITRI w = 1, p = 5 (Table 9) |
| `results/validation_test_shift.csv` | Market-wise partition diagnostics (Table 10) |
| `results/selected_vs_inherited_by_market_block.csv` | Selected against inherited ITRI setting by market and gap length |
| `results/revision_integrity_report.json` | Exact-mask check: the reconstructed interpolation baseline matches the archive within 1.2e-16 |
| `results/full_audit_report.json` | Re-execution of all 6,048 archived ITRI rows (largest absolute difference 1.8e-12) |
| `results/additional_baselines_metadata.json` | Settings and software versions (`requirements.txt`) |

## Rerun

Place the two original study files, `BIST30_2015-01-01_2026-01-01.csv` and `SP500_2015-01-01_2026-01-01.csv`, in `reproduction_run/data/` (the folder written by `scripts/run_full_experiment.py`) or point `ITRI_DATA_DIR` to them. The scripts refuse a file whose SHA-256 differs from `metadata/data_manifest.json`, because the provider may have revised the records since the reported download. Then run, from the repository root:

```bash
python development_comparators/run_additional_baselines.py
python development_comparators/summarize_revision.py
python development_comparators/audit_all_experiments.py
```

The 108 constructed conditions reuse the same two market series, so the row-level files support descriptive paired comparisons, not 108 independent market-level replicates.
