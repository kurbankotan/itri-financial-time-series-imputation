#!/usr/bin/env python3
"""Check the new-asset holdout numbers reported in the article (abstract, Sections 4.6 and 5.7) and in the
Supplementary Material (Note S3, Tables S23 to S25), the integrity of the frozen source, and the condition-level
checkpoints, against the published result files only.

No market data are needed. Run from the repository root or from this folder:
    python cfri_holdout_v1/analysis/verify_reported_numbers.py [--write-tables]
With --write-tables, the rows of Supplementary Tables S23 to S25 are written to analysis/supplement_holdout/.
Prints one PASS or FAIL line per check and exits with status 0 only if every check passes.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAILS = []
FEATURES = ["open", "high", "low", "close", "volume"]


def check(label, ok):
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        FAILS.append(label)


def same(value, text, digits):
    return f"{value:.{digits}f}" == text.replace("−", "-")


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- frozen source and receipts
run_hashes = json.loads((RES / "code_sha256.json").read_text())
check("code/ files equal the SHA-256 recorded by the run",
      all(sha((ROOT / "code" / n).read_bytes()) == h for n, h in run_hashes.items()))
freeze = json.loads((RES / "freeze_receipt.json").read_text())
bundle = (RES / "frozen_source_bundle.zip").read_bytes()
check("frozen source bundle matches the SHA-256 in the freeze receipt", sha(bundle) == freeze["embedded_bundle_sha256"])
with zipfile.ZipFile(io.BytesIO(bundle)) as z:
    same_files = all(z.read(n) == (ROOT / "code" / n).read_bytes() for n in z.namelist() if not n.endswith("/"))
check("every file of the frozen bundle is identical to the published code/ copy", same_files)
check("top-level PROTOCOL.md is identical to the frozen code/PROTOCOL.md",
      (ROOT / "PROTOCOL.md").read_bytes() == (ROOT / "code" / "PROTOCOL.md").read_bytes())
receipt = json.loads((RES / "data_receipt.json").read_text())
check("local source freeze precedes the first data request (04:21:09 and 04:21:18 UTC, 29 October 2026)",
      freeze["freeze_utc"] < receipt["acquisition_started_utc"] and freeze["freeze_utc"].startswith("2026-10-29T04:21:09")
      and receipt["acquisition_started_utc"].startswith("2026-10-29T04:21:18"))
check("16 eligible targets and 34 eligible peers, none excluded",
      receipt["eligible_target_count"] == 16 and receipt["eligible_peer_count"] == 34 and not receipt["excluded"])

# ---------------------------------------------------------------- condition-level checkpoints (full archive)
rows, cpu_masks, bad = [], {}, []
with zipfile.ZipFile(RES / "CFRI_holdout_v1_results_full.zip") as z:
    names = set(z.namelist())
    for folder in ("holdout_cpu", "holdout_saits"):
        for n in sorted(x for x in names if x.startswith(folder + "/") and x.endswith(".json")
                        and not x.endswith(".done.json") and not x.endswith("run_manifest.json")):
            data = z.read(n)
            side = z.read(n + ".done.json").decode() if n + ".done.json" in names else ""
            if sha(data) not in side:
                bad.append(n)
            r = json.loads(data)
            key = n.split("/", 1)[1]
            if folder == "holdout_cpu":
                cpu_masks[key] = r["mask_sha256"]
            elif cpu_masks.get(key) != r["mask_sha256"]:
                bad.append(n + " (mask)")
            for method, feats in r["scores"].items():
                rows.append((key.split("_")[0], r["mask_type"], r["block_length"], round(r["rate"], 2), r["seed"], method,
                             float(np.mean([feats[f][0] for f in FEATURES]))))
check("3,264 checkpoints, each matching its SHA-256 sidecar, with identical CPU and SAITS masks",
      not bad and len(cpu_masks) == 1632 and len(rows) > 0)
full = pd.DataFrame(rows, columns=["ticker", "mask", "block", "rate", "seed", "method", "nrmse"])
c = pd.read_csv(RES / "holdout_condition_scores.csv")
c["rate"] = c["rate"].round(2)
merged = c.merge(full, left_on=["ticker", "mask_type", "block_length", "rate", "seed", "method"],
                 right_on=["ticker", "mask", "block", "rate", "seed", "method"], how="outer", indicator=True)
check("holdout_condition_scores.csv equals the checkpoints (6,528 method-condition rows)",
      len(c) == 6528 and (merged["_merge"] == "both").all() and np.allclose(merged["nrmse_x"], merged["nrmse_y"], atol=1e-12))


def asset(mask, metric="nrmse", by=()):
    x = c[c.mask_type == mask]
    return x.groupby(["ticker", *by, "method"])[metric].mean().unstack()


F, W = asset("feature"), asset("row")
FP, FV = asset("feature", "price_nrmse"), asset("feature", "volume_nrmse")
R = json.loads((RES / "holdout_results.json").read_text())
for mt, A in (("feature", F), ("row", W)):
    check(f"{mt} means equal holdout_results.json",
          all(abs(A[m].mean() - v) < 1e-12 for m, v in R["mask_types"][mt]["means"]["nrmse"].items()))

# ---------------------------------------------------------------- Section 5.7 and Note S3
d1 = F["cfri_joint"] - F["saits_joint"]
check("Section 5.7, feature-wise 0.1389 (CFRI) and 0.1399 (SAITS V2), difference -0.0010",
      same(F["cfri_joint"].mean(), "0.1389", 4) and same(F["saits_joint"].mean(), "0.1399", 4) and same(d1.mean(), "-0.0010", 4))
h1 = R["hypotheses"]["H1"]["metrics"]["nrmse"]
check("Section 5.7, H1 interval -0.0051 to 0.0026, CFRI lower in 9 of 16, not met",
      same(h1["boot95_low"], "-0.0051", 4) and same(h1["boot95_high"], "0.0026", 4) and
      int((d1 < 0).sum()) == 9 and not R["hypotheses"]["H1"]["success"])
check("Section 5.7, prices 0.0468 (SAITS) against 0.0509 (CFRI); volume 0.4907 (CFRI) against 0.5121 (SAITS)",
      same(FP["saits_joint"].mean(), "0.0468", 4) and same(FP["cfri_joint"].mean(), "0.0509", 4) and
      same(FV["cfri_joint"].mean(), "0.4907", 4) and same(FV["saits_joint"].mean(), "0.5121", 4))
fb = asset("feature", by=("block_length",)).groupby("block_length").mean()
check("Section 5.7, CFRI led at 1 to 3 days and SAITS at 5 to 20 days",
      all(fb.loc[b, "cfri_joint"] < fb.loc[b, "saits_joint"] for b in (1, 2, 3)) and
      all(fb.loc[b, "cfri_joint"] > fb.loc[b, "saits_joint"] for b in (5, 10, 20)))
check("Section 5.7, original projection 0.1397",
      same(F["cfri_old_proj"].mean(), "0.1397", 4))
d2 = W["cfri_peer_target_joint"] - W["cfri_joint"]
check("Section 5.7, bridge lowered CFRI from 0.1713 to 0.1571 in all 16 ETFs (H2 met)",
      same(W["cfri_joint"].mean(), "0.1713", 4) and same(W["cfri_peer_target_joint"].mean(), "0.1571", 4) and
      int((d2 < 0).sum()) == 16 and R["hypotheses"]["H2"]["success"])
d3 = W["cfri_peer_target_joint"] - W["saits_peer_joint"]
check("Section 5.7, SAITS V2 with the same bridge 0.2577, CFRI lower in all 16 (H3 met)",
      same(W["saits_peer_joint"].mean(), "0.2577", 4) and int((d3 < 0).sum()) == 16 and R["hypotheses"]["H3"]["success"])
d4 = W["cfri_joint"] - W["saits_joint"]
check("Section 5.7, without a bridge CFRI 0.1713 against SAITS V2 0.2617, lower in every ETF",
      same(W["saits_joint"].mean(), "0.2617", 4) and int((d4 < 0).sum()) == 16)
gain = W["cfri_joint"] - W["cfri_peer_target_joint"]
check("Section 5.7, EFA and EEM supplied 53% of the bridge's benefit",
      same(100 * (gain["EFA"] + gain["EEM"]) / gain.sum(), "53", 0))
check("Section 5.7, market-wide stress test equals the no-peer result exactly",
      bool(np.array_equal(W["cfri_peer_marketwide_joint"].values, W["cfri_joint"].values)) and
      R["peer_usage"]["marketwide_used_rows"] == 0)
check("Note S3, H2 change -8.28% and H3 interval -0.12040 to -0.08335",
      same(R["hypotheses"]["H2"]["metrics"]["nrmse"]["relative_difference_pct"], "-8.28", 2) and
      same(R["hypotheses"]["H3"]["metrics"]["nrmse"]["boot95_low"], "-0.12040", 5) and
      same(R["hypotheses"]["H3"]["metrics"]["nrmse"]["boot95_high"], "-0.08335", 5))
check("Note S3, bridge benefit 0.0745 (EEM), 0.0452 (EFA), 0.0004 (XLF and IYR)",
      same(gain["EEM"], "0.0745", 4) and same(gain["EFA"], "0.0452", 4) and same(gain["XLF"], "0.0004", 4)
      and same(gain["IYR"], "0.0004", 4))

# ---------------------------------------------------------------- Supplementary Tables S23 to S25
if "--write-tables" in sys.argv:
    out = ROOT / "analysis" / "supplement_holdout"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"hypothesis": h, **{k: R["hypotheses"][h]["metrics"]["nrmse"][k] for k in
                   ("method", "comparator", "mean_method", "mean_comparator", "mean_difference", "boot95_low",
                    "boot95_high", "method_lower_in_assets", "success")}} for h in ("H1", "H2", "H3")]
                 ).to_csv(out / "table_S23_hypotheses.csv", index=False)
    s24 = pd.DataFrame({"feature_cfri": F["cfri_joint"], "feature_saits_v2": F["saits_joint"],
                        "row_cfri": W["cfri_joint"], "row_cfri_bridge": W["cfri_peer_target_joint"],
                        "row_saits_v2_bridge": W["saits_peer_joint"], "bridge_benefit": gain})
    s24.loc[R["eligible_targets"]].round(6).to_csv(out / "table_S24_by_etf.csv")
    s25 = []
    for key in ("block_length", "rate"):
        Fa, Wa = asset("feature", by=(key,)), asset("row", by=(key,))
        for lev in sorted(Fa.index.get_level_values(key).unique()):
            fs, ws = Fa.xs(lev, level=key), Wa.xs(lev, level=key)
            s25.append({"stratum": f"{key}={lev}", "feature_cfri": fs["cfri_joint"].mean(),
                        "feature_cfri_original_projection": fs["cfri_old_proj"].mean(),
                        "feature_saits_v2": fs["saits_joint"].mean(),
                        "cfri_lower_than_saits_v2": int((fs["cfri_joint"] < fs["saits_joint"]).sum()),
                        "row_cfri": ws["cfri_joint"].mean(), "row_cfri_bridge": ws["cfri_peer_target_joint"].mean(),
                        "row_saits_v2_bridge": ws["saits_peer_joint"].mean()})
    pd.DataFrame(s25).round(6).to_csv(out / "table_S25_by_block_and_rate.csv", index=False)
    print("tables written to", out)

print()
print("FAILS:", ", ".join(FAILS) if FAILS else "none")
sys.exit(1 if FAILS else 0)
