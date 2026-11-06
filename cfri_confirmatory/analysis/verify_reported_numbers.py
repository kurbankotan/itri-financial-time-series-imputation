#!/usr/bin/env python3
"""Check the CFRI numbers reported in the article for folds A to C and the confirmatory fold D (abstract, Sections
4.5 and 5.6, Table 4) and the integrity of the frozen code, against the published result files only.

No market data are needed. Run from the repository root or from this folder:
    python cfri_confirmatory/analysis/verify_reported_numbers.py
Prints one PASS or FAIL line per check and exits with status 0 only if every check passes.
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "confirmatory_D" / "CFRI_fold_D_results.zip"
PRICES = ["open", "high", "low", "close"]
FAILS = []


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        FAILS.append(label)


def same(value, text, digits):
    """True when value rounds to the reported text (text uses '-' or the Unicode minus)."""
    return f"{value:.{digits}f}" == text.replace("−", "-")


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- code integrity
run_hashes = json.loads((ROOT / "confirmatory_D" / "code_sha256.json").read_text())["files"]
for name, digest in sorted(run_hashes.items()):
    check(f"code/{name} equals the SHA-256 recorded by the fold D run",
          sha((ROOT / "code" / name).read_bytes()) == digest)
protocol = (ROOT / "CFRI_CONFIRMATORY_PROTOCOL_v1.md").read_text(encoding="utf-8")
listed = dict((n, h) for h, n in re.findall(r"([0-9a-f]{64})\s+\*?([A-Za-z_]+\.py)", protocol))
check("the protocol lists the same SHA-256 for the six run files",
      all(listed.get(n) == h for n, h in run_hashes.items()))
check("the protocol lists SHA-256 values that match all eight files in code/",
      len(listed) == 8 and all(sha((ROOT / "code" / n).read_bytes()) == h for n, h in listed.items()))
nb = json.loads((ROOT / "notebooks" / "CFRI_confirmatory_fold_D_Colab.ipynb").read_text(encoding="utf-8"))
cell = next("".join(c["source"]) for c in nb["cells"] if "FILES[" in "".join(c["source"]))
embedded = dict(re.findall(r"FILES\['([^']+)'\] = r'''(.*?)'''\n", cell, re.S))
check("the run notebook embeds exactly the six run files with the recorded SHA-256",
      set(embedded) == set(run_hashes) and all(sha(t.encode("utf-8")) == run_hashes[n] for n, t in embedded.items()))

# ---------------------------------------------------------------- condition-level results
with zipfile.ZipFile(ZIP) as z:
    names = set(z.namelist())
    check("results ZIP passes its CRC check", z.testzip() is None)
    R = json.loads(z.read("fold_D_results.json"))
    check("fold_D_results.json in the ZIP equals the extracted copy",
          z.read("fold_D_results.json") == (ROOT / "confirmatory_D" / "fold_D_results.json").read_bytes())
    rows, missing = [], []
    for n in sorted(names):
        if n.startswith("fold_D_cpu/") and n.endswith(".jsonl"):
            if n + ".done.json" not in names:
                missing.append(n)
            for line in z.read(n).decode().splitlines():
                r = json.loads(line)
                for method, feats in r["scores"].items():
                    for feature, (nrmse, _nmae, _cells) in feats.items():
                        rows.append((r["ticker"], r["block_length"], round(r["rate"], 2), r["seed"], r["mask_type"],
                                     method, feature, nrmse))
        elif n.startswith("fold_D_saits/") and n.endswith(".csv"):
            if n[:-4] + ".done.json" not in names:
                missing.append(n)
            s = pd.read_csv(io.BytesIO(z.read(n)))
            rows += [(t.ticker, int(t.block_length), round(float(t.rate), 2), int(t.seed), t.mask_type, t.method,
                      t.feature, float(t.rmse)) for t in s.itertuples(index=False)]
check("every CPU and SAITS result shard has its completion sidecar", not missing)
d = pd.DataFrame(rows, columns=["ticker", "block", "rate", "seed", "mask", "method", "feature", "nrmse"])


def asset_means(mask, features=None, by=()):
    x = d[d["mask"] == mask]
    if features is not None:
        x = x[x.feature.isin(features)]
    c = x.groupby(["ticker", "block", "rate", "seed", "method"]).nrmse.mean().reset_index()
    return c.groupby(["ticker", *by, "method"]).nrmse.mean().unstack()


A, P, V, W = asset_means("feature"), asset_means("feature", PRICES), asset_means("feature", ["volume"]), asset_means("row")
check("34 assets and 51 conditions per asset under both mask types",
      A.shape[0] == 34 and W.shape[0] == 34 and
      d[d["mask"] == "feature"].groupby("ticker")[["block", "rate", "seed"]].apply(
          lambda g: len(g.drop_duplicates())).eq(51).all())
m = A.mean()
cf = "cfri_fixed_proj"
for meth, text in [(cf, "0.2822"), ("log_linear", "0.3616"), ("itri_w1_p5", "0.3864"),
                   ("cfri_fixed_no_cross_proj", "0.3049"), ("saits_winnorm_V2", "0.2950"),
                   ("log_linear_proj", "0.3230"), ("saits_winnorm_V2_proj", "0.2861")]:
    check(f"Table 4, fold D mean NRMSE of {meth} = {text}", same(m[meth], text, 4))
for comp, text, wins in [("log_linear", "-0.0795", 34), ("itri_w1_p5", "-0.1042", 34),
                         ("cfri_fixed_no_cross_proj", "-0.0228", 34), ("saits_winnorm_V2", "-0.0128", 29),
                         ("log_linear_proj", "-0.0408", 34), ("saits_winnorm_V2_proj", "-0.0040", 13)]:
    diff = A[cf] - A[comp]
    check(f"Table 4, CFRI minus {comp} = {text}, CFRI lower in {wins} of 34",
          same(diff.mean(), text, 4) and int((diff < 0).sum()) == wins)
check("Section 5.6, H1 to H4 all met with threshold 25 of 34",
      all(R["primary"][h]["criterion_met"] and R["primary"][h]["threshold_k"] == 25 for h in ("H1", "H2", "H3", "H4")))
check("Section 5.6, H4 interval -0.0217 to -0.0061",
      same(R["primary"]["H4"]["boot95_low"], "-0.0217", 4) and same(R["primary"]["H4"]["boot95_high"], "-0.0061", 4))
pe = R["projection_equalized"]["saits_winnorm_V2_proj"]
check("Section 5.6, projected SAITS interval -0.0117 to 0.0022",
      same(pe["boot95_low"], "-0.0117", 4) and same(pe["boot95_high"], "0.0022", 4))
check("Section 5.6, H1 to H3 met in every leave-one-region-out subset",
      all(v[h]["criterion_met"] for v in R["by_mask"]["feature"]["leave_one_region_out"].values()
          for h in ("H1", "H2", "H3")))
check("Section 5.6, median NRMSE 0.2194 (CFRI) and 0.2727 (log-linear)",
      same(A[cf].median(), "0.2194", 4) and same(A["log_linear"].median(), "0.2727", 4))
check("Section 5.6, CFRI volume NRMSE about 5 in EWY and EFNL (5.42 and 4.98)",
      same(V.loc["EWY", cf], "5.42", 2) and same(V.loc["EFNL", cf], "4.98", 2))
pm, vm = P.mean(), V.mean()
check("Section 5.6, price NRMSE 0.1074 (g = 0) to 0.0789",
      same(pm["cfri_fixed_no_cross_proj"], "0.1074", 4) and same(pm[cf], "0.0789", 4))
check("Section 5.6, the residual lowered price error in every asset",
      bool((P[cf] < P["cfri_fixed_no_cross_proj"]).all()))
blk = asset_means("feature", PRICES, by=("block",)).groupby("block").mean()
check("Section 5.6, the residual lowered price error at every gap length",
      bool((blk[cf] < blk["cfri_fixed_no_cross_proj"]).all()))
check("Section 5.6, price margin over interpolation 0.039 (1 day) and 0.147 (20 days)",
      same(blk.loc[1, "log_linear"] - blk.loc[1, cf], "0.039", 3) and
      same(blk.loc[20, "log_linear"] - blk.loc[20, cf], "0.147", 3))
ov = R["by_mask"]["feature"]["ohlc_violation_share"]
check("Section 5.6, OHLC violations 38.0% (CFRI unprojected), 54.9% (log-linear), 28.6% (SAITS V2)",
      same(100 * ov["cfri_fixed"], "38.0", 1) and same(100 * ov["log_linear"], "54.9", 1) and
      same(100 * ov["saits_winnorm_V2"], "28.6", 1))
check("Section 5.6, projected SAITS 0.0714 on prices and 1.1448 on volume, CFRI 1.0951 on volume",
      same(pm["saits_winnorm_V2_proj"], "0.0714", 4) and same(vm["saits_winnorm_V2_proj"], "1.1448", 4) and
      same(vm[cf], "1.0951", 4))
mw = W.mean()
check("Section 5.6, whole-row masks 0.3399 (CFRI), 0.3597 (log-linear), 0.4942 (SAITS V2)",
      same(mw[cf], "0.3399", 4) and same(mw["log_linear"], "0.3597", 4) and same(mw["saits_winnorm_V2"], "0.4942", 4))
check("Section 5.6, SAITS V2 took 13.5 s per condition",
      same(R["run_time"]["saits_seconds_per_condition"]["saits_winnorm_V2_feature"], "13.5", 1))

# ---------------------------------------------------------------- exploratory folds A to C (Table 4)
X = json.loads((ROOT / "exploratory_AC" / "exploratory_summary.json").read_text())["by_mask"]["feature"]
table = {"cfri_fixed_proj": ["0.2238", "0.1941", "0.1699"], "log_linear": ["0.3043", "0.2602", "0.2220"],
         "itri_w1_p5": ["0.3085", "0.2739", "0.2346"], "cfri_fixed_no_cross_proj": ["0.2485", "0.2149", "0.1856"],
         "log_linear_proj": ["0.2592", "0.2243", "0.1938"]}
for meth, texts in table.items():
    check(f"Table 4, folds A to C of {meth} = {', '.join(texts)}",
          all(same(X[f]["mean"][meth], t, 4) for f, t in zip("ABC", texts)))
check("Table 4, folds A to C of SAITS V2 = 0.2685, 0.2155, 0.1817",
      all(same(X[f]["saits"]["saits_winnorm_V2"]["mean_comparator"], t, 4)
          for f, t in zip("ABC", ["0.2685", "0.2155", "0.1817"])))
check("Section 5.6, CFRI 0.1959 over folds A to C and lower in all 34 assets in every fold",
      same(X["pooled"]["mean"]["cfri_fixed_proj"], "0.1959", 4) and
      all(X[f]["hypotheses"][h]["method_lower_in_assets"] == 34 for f in "ABC" for h in ("H1", "H2", "H3")) and
      all(X[f]["saits"]["saits_winnorm_V2"]["method_lower_in_assets"] == 34 for f in "ABC"))

print()
print("FAILS:", ", ".join(FAILS) if FAILS else "none")
sys.exit(1 if FAILS else 0)
