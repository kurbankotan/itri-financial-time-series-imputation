# Published result files

These files are the machine-readable outputs used in the manuscript. The full validation and test grids contain one row per market, block length, missing rate, seed, window, and pass-count condition.

`interpolation_baseline_test.csv` was reconstructed on the same 108 test masks used for the selected ITRI comparison. Runtime was not recorded for this baseline reconstruction.

Run `python scripts/verify_published_results.py` from the repository root to check row counts, uniqueness, finite metrics, selected configuration, and reported aggregate values.

