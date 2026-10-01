"""File locations used by the analysis scripts (public repository layout).

Every analysis script in this folder reads only the public result files listed
here; none of them needs the private OHLCV snapshot.
"""
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parent
EXTERNAL = CODE.parent
REPO = EXTERNAL.parent
RESULTS = EXTERNAL / "results"
TABLES = EXTERNAL / "tables"
FIGURES = REPO / "paper_figures"

# frozen protocol v1: classical and neural comparators, feature-level rows
V1_TEST = RESULTS / "v1_frozen" / "test_feature_rows.csv.gz"
V1_VALIDATION = RESULTS / "v1_frozen" / "validation_feature_rows.csv.gz"
# exploratory analyses
# the two sweep files are stored in three consecutive parts each (GitHub upload limit of 25 MB per file)
SWEEP_TEST = [RESULTS / "exploratory" / f"sweep_test_feature_rows.part{i}.csv.gz" for i in (1, 2, 3)]
SWEEP_VALIDATION = [RESULTS / "exploratory" / f"sweep_validation_feature_rows.part{i}.csv.gz" for i in (1, 2, 3)]
W1_COEFFICIENTS = RESULTS / "exploratory" / "itri_w1_coefficients.csv.gz"
LEVEL_SHIFT = RESULTS / "exploratory" / "level_shift_by_asset_fold.csv"
WINDOWNORM_SAITS = RESULTS / "exploratory" / "windownorm_saits_feature_rows.csv.gz"
WINDOWNORM_BRITS_SUBSET = RESULTS / "exploratory" / "windownorm_brits_subset_feature_rows.csv.gz"
CPU_PIECES = RESULTS / "exploratory" / "cross_hardware_cpu_feature_rows.csv.gz"
# development study (repository root)
DEV_TEST_SUMMARY = REPO / "results" / "test_summary.csv"


def read(paths, **kwargs) -> pd.DataFrame:
    """Read one result file, or the parts of a split file in order (identical to the unsplit file)."""
    if isinstance(paths, (list, tuple)):
        return pd.concat([pd.read_csv(x, **kwargs) for x in paths], ignore_index=True)
    return pd.read_csv(paths, **kwargs)
