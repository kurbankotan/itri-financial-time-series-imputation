#!/usr/bin/env python3
"""Recompute Supplementary Tables S19-S22 and the data of Supplementary Figures S6-S7 (CFRI, fold D) from the
condition-level result archive, and check them against the frozen analysis output (fold_D_results.json).

Usage:
    python fold_D_supplement.py [--zip ../confirmatory_D/CFRI_fold_D_results.zip] [--out supplement_fold_D]

Needs numpy and pandas; matplotlib (optional) draws Figures S6 and S7. No raw price data are used.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PRICES = ["open", "high", "low", "close"]
MINUS = "−"
LABEL = {
    "cfri_fixed_proj": "CFRI (primary)", "cfri_fixed": "CFRI without projection",
    "cfri_fixed_no_cross_proj": "CFRI, g = 0 (H3)", "cfri_fixed_no_cross": "CFRI, g = 0, without projection",
    "cfri_adaptive_proj": "Per-condition CFRI", "cfri_adaptive": "Per-condition CFRI without projection",
    "cfri_adaptive_no_cross_proj": "Per-condition CFRI, g = 0",
    "cfri_adaptive_no_cross": "Per-condition CFRI, g = 0, without projection",
    "log_linear": "Log-linear (H1)", "log_linear_proj": "Log-linear + projection (P1)",
    "linear_raw": "Linear", "linear_raw_proj": "Linear + projection",
    "itri_w1_p5": "ITRI, w = 1, p = 5 (H2)", "itri_w1_p5_proj": "ITRI + projection (P2)",
    "chained_ridge": "Chained ridge", "chained_ridge_proj": "Chained ridge + projection",
    "saits_winnorm_V2": "SAITS V2 (H4)", "saits_winnorm_V2_proj": "SAITS V2 + projection (P3)",
    "saits_winnorm_V1": "SAITS V1", "saits_winnorm_V1_proj": "SAITS V1 + projection"}


def f4(v):
    return f"{v:.4f}".replace("-", MINUS)


def load(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        results = json.loads(z.read("fold_D_results.json"))
        receipt = json.loads(z.read("data_receipt.json"))
        for n in sorted(names):
            if n.startswith("fold_D_cpu/") and n.endswith(".jsonl"):
                assert n + ".done.json" in names, f"incomplete shard {n}"
                for line in z.read(n).decode().splitlines():
                    r = json.loads(line)
                    for method, feats in r["scores"].items():
                        for feature, (nrmse, _nmae, _cells) in feats.items():
                            rows.append((r["ticker"], r["block_length"], round(r["rate"], 2), r["seed"],
                                         r["mask_type"], method, feature, nrmse))
            elif n.startswith("fold_D_saits/") and n.endswith(".csv"):
                assert n[:-4] + ".done.json" in names, f"incomplete shard {n}"
                s = pd.read_csv(io.BytesIO(z.read(n)))
                for t in s.itertuples(index=False):
                    rows.append((t.ticker, int(t.block_length), round(float(t.rate), 2), int(t.seed), t.mask_type,
                                 t.method, t.feature, float(t.rmse)))
    d = pd.DataFrame(rows, columns=["ticker", "block_length", "rate", "seed", "mask_type", "method", "feature", "nrmse"])
    region = {a["ticker"]: a["region"] for a in receipt["assets"] if a.get("eligible", True)}
    return d, results, region


def asset_means(d, mask, features=None, by=()):
    x = d[d.mask_type == mask]
    if features is not None:
        x = x[x.feature.isin(features)]
    c = x.groupby(["ticker", "block_length", "rate", "seed", "method"]).nrmse.mean().reset_index()  # macro over features
    return c.groupby(["ticker", *by, "method"]).nrmse.mean().unstack()                              # mean over conditions


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--zip", default=str(here.parent / "confirmatory_D" / "CFRI_fold_D_results.zip"))
    ap.add_argument("--out", default=str(here / "supplement_fold_D"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d, R, region = load(a.zip)
    F = R["by_mask"]["feature"]
    A = {mt: asset_means(d, mt) for mt in ("feature", "row")}

    # checks against the frozen analysis output (rounded to 5 decimals there)
    for mt in ("feature", "row"):
        for m, v in R["by_mask"][mt]["mean"].items():
            assert abs(A[mt][m].mean() - v) < 6e-6, (mt, m)
    for h in ("H1", "H2", "H3", "H4"):
        e = R["primary"][h]
        diff = A["feature"]["cfri_fixed_proj"] - A["feature"][e["comparator"]]
        assert abs(diff.mean() - e["mean_difference"]) < 1e-9 and int((diff < 0).sum()) == e["method_lower_in_assets"], h

    # Table S19: by feature (feature-wise masks) and OHLC violation share
    feat = {f: asset_means(d, "feature", [f]).mean() for f in PRICES + ["volume"]}
    price = asset_means(d, "feature", PRICES).mean()
    ov = F["ohlc_violation_share"]
    s19 = []
    for m in ["cfri_fixed_proj", "cfri_fixed", "cfri_fixed_no_cross_proj", "log_linear", "log_linear_proj", "linear_raw",
              "itri_w1_p5", "itri_w1_p5_proj", "chained_ridge", "saits_winnorm_V1", "saits_winnorm_V2",
              "saits_winnorm_V2_proj"]:
        s19.append({"method": m, **{f: f4(feat[f][m]) for f in PRICES + ["volume"]}, "prices": f4(price[m]),
                    "all_five": f4(A["feature"][m].mean()), "ohlc_violations": f"{100 * ov[m]:.1f}%"})
    pd.DataFrame(s19).to_csv(out / "table_S19_by_feature.csv", index=False)

    # Table S20: all variants, both mask types, CFRI minus method with the frozen bootstrap interval
    s20 = []
    for m in LABEL:
        row = {"method": LABEL[m]}
        for mt in ("feature", "row"):
            row[f"{mt}_mean"] = f4(A[mt][m].mean())
            if m == "cfri_fixed_proj":
                row[f"{mt}_difference"] = row[f"{mt}_cfri_lower"] = "–"
                continue
            blk = R["by_mask"][mt]
            e = blk["saits"][m] if m.startswith("saits") else blk["paired_vs_cfri"][m]
            diff = A[mt]["cfri_fixed_proj"] - A[mt][m]
            assert abs(diff.mean() - e["mean_difference"]) < 1e-9, (mt, m)
            if diff.abs().max() < 1e-12:
                row[f"{mt}_difference"], row[f"{mt}_cfri_lower"] = "identical (rounding only)", "–"
            else:
                row[f"{mt}_difference"] = f"{f4(e['mean_difference'])} ({f4(e['boot95_low'])} to {f4(e['boot95_high'])})"
                row[f"{mt}_cfri_lower"] = int((diff < 0).sum())
        s20.append(row)
    pd.DataFrame(s20).to_csv(out / "table_S20_all_variants.csv", index=False)

    # Table S21: by gap length and missing rate (feature-wise masks)
    cols = ["cfri_fixed_proj", "cfri_fixed_no_cross_proj", "log_linear", "itri_w1_p5", "saits_winnorm_V2",
            "saits_winnorm_V2_proj"]
    s21 = []
    for key in ("block_length", "rate"):
        am = asset_means(d, "feature", by=(key,))
        for level in sorted(am.index.get_level_values(key).unique()):
            sub = am.xs(level, level=key)
            s21.append({"stratum": f"{key}={level}", **{m: f4(sub[m].mean()) for m in cols},
                        "cfri_lower_than_saits_v2_proj": int((sub["cfri_fixed_proj"] < sub["saits_winnorm_V2_proj"]).sum())})
    pd.DataFrame(s21).to_csv(out / "table_S21_by_block_and_rate.csv", index=False)

    # Table S22: regions and leave-one-region-out (frozen output)
    lo = F["leave_one_region_out"]
    s22 = []
    for reg in sorted(set(region.values())):
        tick = [t for t in A["feature"].index if region[t] == reg]
        sub = A["feature"].loc[tick]
        L = lo[f"without_{reg}"]
        s22.append({"region": reg, "assets": len(tick),
                    **{m: f4(sub[m].mean()) for m in ["cfri_fixed_proj", "log_linear", "itri_w1_p5",
                                                       "cfri_fixed_no_cross_proj", "saits_winnorm_V2"]},
                    "assets_left": L["H1"]["assets"],
                    "cfri_lower_H1_H2_H3": ", ".join(str(L[h]["method_lower_in_assets"]) for h in ("H1", "H2", "H3")),
                    "k": L["H1"]["threshold_k"], "met": all(L[h]["criterion_met"] for h in ("H1", "H2", "H3"))})
    pd.DataFrame(s22).to_csv(out / "table_S22_regions.csv", index=False)

    # Figure S6 data: price and volume NRMSE by gap length; Figure S7 data: asset-level paired differences
    pb = asset_means(d, "feature", PRICES, by=("block_length",)).groupby("block_length").mean()
    vb = asset_means(d, "feature", ["volume"], by=("block_length",)).groupby("block_length").mean()
    fig6 = pd.concat({"nrmse_price": pb.stack(), "nrmse_volume": vb.stack()}, axis=1).reset_index()
    fig6.to_csv(out / "figure_S6_by_block.csv", index=False)
    comps = ["log_linear", "itri_w1_p5", "cfri_fixed_no_cross_proj", "saits_winnorm_V2", "log_linear_proj",
             "saits_winnorm_V2_proj"]
    fig7 = pd.DataFrame({c: A["feature"]["cfri_fixed_proj"] - A["feature"][c] for c in comps})
    fig7.to_csv(out / "figure_S7_paired_differences.csv")

    # headline values quoted in the article (Sections 5.6 and 6.2)
    head = {"price_cfri": price["cfri_fixed_proj"], "price_cfri_g0": price["cfri_fixed_no_cross_proj"],
            "price_saits_v2_proj": price["saits_winnorm_V2_proj"], "volume_cfri": feat["volume"]["cfri_fixed_proj"],
            "volume_saits_v2": feat["volume"]["saits_winnorm_V2"],
            "price_margin_vs_log_linear_block1": pb.loc[1, "log_linear"] - pb.loc[1, "cfri_fixed_proj"],
            "price_margin_vs_log_linear_block20": pb.loc[20, "log_linear"] - pb.loc[20, "cfri_fixed_proj"],
            "cfri_price_below_g0_in_assets": int((asset_means(d, "feature", PRICES)["cfri_fixed_proj"]
                                                  < asset_means(d, "feature", PRICES)["cfri_fixed_no_cross_proj"]).sum()),
            "median_cfri": A["feature"]["cfri_fixed_proj"].median(), "median_log_linear": A["feature"]["log_linear"].median()}
    json.dump({k: (round(v, 6) if isinstance(v, float) else v) for k, v in head.items()},
              open(out / "headline_values.json", "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for m in ["log_linear", "cfri_fixed_no_cross_proj", "saits_winnorm_V2", "saits_winnorm_V2_proj", "cfri_fixed_proj"]:
            ax[0].plot([str(b) for b in pb.index], pb[m], marker="o", label=LABEL[m])
        for m in ["itri_w1_p5", "log_linear", "saits_winnorm_V2", "cfri_fixed_proj"]:
            ax[1].plot([str(b) for b in vb.index], vb[m], marker="o", label=LABEL[m])
        for k, t in enumerate(["(a) Open, high, low, close", "(b) Volume"]):
            ax[k].set_title(t, loc="left"); ax[k].set_xlabel("Gap (block) length, trading days")
            ax[k].set_ylabel("Mean asset-level NRMSE"); ax[k].legend(fontsize=7)
        fig.savefig(out / "figure_S6.png", dpi=200)
        fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
        for i, c in enumerate(comps):
            ax.scatter(np.clip(fig7[c], -0.25, 0.1), np.full(len(fig7), len(comps) - 1 - i), s=10)
        ax.set_yticks(range(len(comps)), [LABEL[c] for c in comps][::-1]); ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("CFRI minus comparator, asset-level mean NRMSE (clipped at −0.25)")
        fig.savefig(out / "figure_S7.png", dpi=200)
    except ImportError:
        pass
    print("all checks against fold_D_results.json passed; outputs in", out)


if __name__ == "__main__":
    main()
