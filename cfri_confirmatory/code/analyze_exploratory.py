#!/usr/bin/env python3
"""Exploratory summaries of CFRI on folds A-C, with the functions used for the confirmatory fold D analysis.

Usage: python analyze_exploratory.py OUT_AC_DIR CODEX_FEATURE_RESULTS_CSV SAITS_ROWS_CSV_GZ REPORT_DIR

Folds A-C were used to develop CFRI (validation and test years were inspected), so every number here is
exploratory. The fixed CFRI setting is re-derived from the pooled validation candidate logs of the feature-wise
conditions and must equal cfri_core.CFRI_FIXED.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfri_core as C  # noqa: E402
import confirm_analyze as A  # noqa: E402

KEY = A.KEY


def load_ac(out_dir: Path):
    rows, cand_rows, sel_rows, viol = [], [], [], []
    for f in sorted(glob.glob(str(out_dir / "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            base = {k: r[k] for k in KEY + ["mask_type"]}
            base["rate"] = round(base["rate"], 2)
            for m, feats in r["scores"].items():
                for feat, (rmse, mae, n) in feats.items():
                    rows.append({**base, "method": m, "feature": feat, "rmse": rmse, "mae": mae})
            for m, v in r["ohlc_violation_share"].items():
                viol.append({**base, "method": m, "ohlc_violation_share": v})
            viol.append({**base, "method": "truth", "ohlc_violation_share": r["truth_ohlc_violation_share"]})
            cand_rows.append({**base, **r["candidates"]})
            sel_rows.append({**base, "price_alpha": r["price_alpha"], "price_gain": r["price_gain"],
                             "volume_choice": r["volume_choice"]})
    return pd.DataFrame(rows), pd.DataFrame(cand_rows), pd.DataFrame(sel_rows), pd.DataFrame(viol)


def fixed_from_validation(cand: pd.DataFrame):
    price_cols = [c for c in cand.columns if c.startswith("val_price_")]
    vol_cols = [c for c in cand.columns if c.startswith("val_volume_")]
    price_rank = cand[price_cols].mean().sort_values()
    vol_rank = cand[vol_cols].mean().sort_values()
    alpha_s, gain_s = price_rank.index[0].replace("val_price_a", "").split("_g")
    return (float(alpha_s), float(gain_s), vol_rank.index[0].replace("val_volume_", "")), price_rank, vol_rank


def main(out_dir, codex_csv, saits_csv, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    long, cand, sel, viol = load_ac(out_dir)
    n_cond = sel.groupby("mask_type").size().to_dict()
    print("conditions per mask type:", n_cond, flush=True)
    assert all(v == 34 * 3 * 54 for v in n_cond.values()), n_cond

    # fixed setting: pooled feature-mask validation logs of folds A-C, and per fold for stability
    fm = cand[cand.mask_type == "feature"]
    fixed, price_rank, vol_rank = fixed_from_validation(fm)
    assert fixed == C.CFRI_FIXED, (fixed, C.CFRI_FIXED)
    per_fold_fixed = {f: fixed_from_validation(fm[fm.fold == f])[0] for f in ("A", "B", "C")}

    # reproduction of the archived exploratory ITRI2 run (Codex): same masks, same numbers
    cx = pd.read_csv(codex_csv, usecols=KEY + ["method", "feature", "rmse_trainstd"], float_precision="round_trip")
    cx["rate"] = cx.rate.round(2)
    cx["method"] = cx.method.replace({"itri2": "cfri_adaptive", "itri2_no_cross": "cfri_adaptive_no_cross"})
    mine = long[long.mask_type == "feature"].merge(cx, on=KEY + ["method", "feature"])
    repro = {"rows_matched": int(len(mine)), "rows_archived": int(len(cx)),
             "max_abs_diff": float((mine.rmse - mine.rmse_trainstd).abs().max())}
    print("reproduction of archived ITRI2 run:", repro, flush=True)

    # archived normalized SAITS (feature-wise masks, test partition)
    s = pd.read_csv(saits_csv, float_precision="round_trip")
    s = s[s.partition == "test"].copy()
    s["rate"] = s.rate.round(2)
    s["mask_type"] = "feature"
    s = s.rename(columns={"rmse_trainstd": "rmse", "mae_trainstd": "mae"})
    long = pd.concat([long, s[KEY + ["mask_type", "method", "feature", "rmse", "mae"]]], ignore_index=True)

    cond = A.condition_level(long)
    summary = {"conditions": n_cond, "reproduction_vs_archived_itri2": repro,
               "fixed_setting": {"pooled": list(fixed), "per_fold": {f: list(v) for f, v in per_fold_fixed.items()},
                                 "price_rank_top5": price_rank.head(5).round(6).to_dict(),
                                 "volume_rank": vol_rank.round(6).to_dict()},
               "adaptive_selection_counts_feature": {
                   "price_alpha": sel[sel.mask_type == "feature"].price_alpha.value_counts().sort_index().to_dict(),
                   "price_gain": sel[sel.mask_type == "feature"].price_gain.value_counts().sort_index().to_dict(),
                   "volume_choice": sel[sel.mask_type == "feature"].volume_choice.value_counts().to_dict()},
               "by_mask": {}}
    tables = []
    for mt in ("feature", "row"):
        res = {}
        for label, folds in (("pooled", ("A", "B", "C")), ("A", ("A",)), ("B", ("B",)), ("C", ("C",))):
            c = cond[(cond.mask_type == mt) & cond.fold.isin(folds)]
            l_ = long[(long.mask_type == mt) & long.fold.isin(folds)]
            v = viol[(viol.mask_type == mt) & viol.fold.isin(folds)]
            sm, asset = A.summarize(c[c.method.isin(A.CPU_ORDER)], l_[l_.method.isin(A.CPU_ORDER)], v,
                                    primary_tests=(mt == "feature"))
            sm["saits"] = A.saits_comparisons(c)
            res[label] = sm
            t = asset.copy()
            t.insert(0, "mask_type", mt)
            t.insert(1, "folds", label)
            tables.append(t.reset_index())
        summary["by_mask"][mt] = res
    pd.concat(tables).to_csv(report_dir / "exploratory_asset_level_nrmse.csv", index=False)
    (report_dir / "exploratory_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # console digest
    for mt in ("feature", "row"):
        p = summary["by_mask"][mt]["pooled"]
        print(f"\n== {mt} masks, folds A-C pooled: mean / median asset-level NRMSE ({p['assets']} assets)")
        print(pd.DataFrame({"mean": p["mean"], "median": p["median"]}).sort_values("mean").round(4).to_string())
        if "hypotheses" in p:
            for label in ("pooled", "A", "B", "C"):
                hy = summary["by_mask"][mt][label]["hypotheses"]
                print(label, {h: (round(r["mean_difference"], 4), f"{r['method_lower_in_assets']}/{r['assets']}",
                                  r["criterion_met"]) for h, r in hy.items()})
        for label in ("pooled", "A", "B", "C"):
            sa = summary["by_mask"][mt][label]["saits"]
            if sa:
                print(label, "SAITS:", {k: (round(r["mean_method"], 4), round(r["mean_comparator"], 4),
                                            f"{r['method_lower_in_assets']}/{r['assets']}") for k, r in sa.items()})
        print("OHLC violation share:", {k: round(v, 3) for k, v in p.get("ohlc_violation_share", {}).items()})
    print("\nfixed setting per fold:", per_fold_fixed)
    return summary


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:5]))
