"""Confirmatory study, step 4: prespecified analysis of fold D and export of a results ZIP (no raw prices).

Endpoint: for each condition, the mean over the five features of RMSE on masked test cells divided by the
training-observed SD of the feature (NRMSE); per asset, the mean over its conditions; per method, the mean (and
median) over assets.

Primary hypotheses (feature-wise masks), CFRI = cfri_fixed_proj:
  H1 CFRI vs log-linear interpolation (log_linear);
  H2 CFRI vs original ITRI, w = 1, p = 5 (itri_w1_p5);
  H3 CFRI vs CFRI without the cross-feature residual (cfri_fixed_no_cross_proj; same volume rule and projection);
  H4 CFRI vs normalized SAITS V2 (saits_winnorm_V2, protocol amendment v2 settings), paired on the conditions
     SAITS completed.
Comparators of H1, H2 and H4 are used exactly as defined in the earlier protocols (no OHLC projection); the
projection-equalized comparisons (log_linear_proj, itri_w1_p5_proj, saits_winnorm_V2_proj) are secondary.
Each hypothesis is met when CFRI has the lower mean asset-level NRMSE AND is lower in at least k of the n analyzed
assets, k being the smallest count with one-sided binomial tail P(X >= k | n, 1/2) < 0.01 (k = 25 for n = 34).
Assets are not independent; k is a prespecified consistency threshold, not a significance test. Everything else
is descriptive: medians, paired differences with bootstrap intervals over assets, Wilcoxon signed-rank p-values,
block/rate/feature/region breakdowns, leave-one-region-out, MAE, OHLC violations, whole-row masks, SAITS V1.

The same functions summarize the exploratory folds A-C (analyze_exploratory.py).
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C
import confirm_cpu
import confirm_saits

KEY = ["ticker", "fold", "block_length", "rate", "seed"]
PRIMARY = C.PRIMARY_METHOD
HYPOTHESES = (("H1", "log_linear"), ("H2", "itri_w1_p5"), ("H3", "cfri_fixed_no_cross_proj"))
SAITS_HYPOTHESIS = ("H4", "saits_winnorm_V2")
PROJECTION_EQUALIZED = ("log_linear_proj", "itri_w1_p5_proj", "saits_winnorm_V2_proj")
REGION = {t: r for t, _, r in C.ASSETS}
BOOT_SEED, N_BOOT = 20261027, 10000
CPU_ORDER = list(C.METHODS)
SAITS_METHODS = ("saits_winnorm_V1", "saits_winnorm_V2", "saits_winnorm_V1_proj", "saits_winnorm_V2_proj")


def threshold(n: int, alpha: float = 0.01) -> int:
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n < alpha:
            return k
    return n + 1


def condition_level(long: pd.DataFrame) -> pd.DataFrame:
    """Feature rows -> one row per condition and method (macro NRMSE, macro NMAE, price-only, volume)."""
    g = long.groupby(KEY + ["mask_type", "method"])
    out = g.agg(nrmse=("rmse", "mean"), nmae=("mae", "mean"), n_features=("rmse", "size")).reset_index()
    price = long[long.feature != "volume"].groupby(KEY + ["mask_type", "method"]).rmse.mean().rename("nrmse_price")
    vol = long[long.feature == "volume"].groupby(KEY + ["mask_type", "method"]).rmse.mean().rename("nrmse_volume")
    out = out.merge(price.reset_index(), on=KEY + ["mask_type", "method"]).merge(vol.reset_index(), on=KEY + ["mask_type", "method"])
    if (out.n_features != len(C.FEATURES)).any():
        raise ValueError("a condition lacks one of the five features")
    return out.drop(columns="n_features")


def asset_table(cond: pd.DataFrame, value: str = "nrmse") -> pd.DataFrame:
    return cond.groupby(["ticker", "method"])[value].mean().unstack()


def paired(asset: pd.DataFrame, a: str, b: str) -> dict:
    d = (asset[a] - asset[b]).dropna()
    n = len(d)
    if n == 0:
        return {"method": a, "comparator": b, "assets": 0, "criterion_met": False}
    rng = np.random.default_rng(BOOT_SEED)
    boots = d.to_numpy()[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d.to_numpy()).pvalue) if n > 0 and (d != 0).any() else float("nan")
    except Exception:
        p = float("nan")
    k = threshold(n)
    lower_mean = bool(asset.loc[d.index, a].mean() < asset.loc[d.index, b].mean())
    wins = int((d < 0).sum())
    return {"method": a, "comparator": b, "assets": n, "mean_method": float(asset.loc[d.index, a].mean()),
            "mean_comparator": float(asset.loc[d.index, b].mean()), "median_method": float(asset.loc[d.index, a].median()),
            "median_comparator": float(asset.loc[d.index, b].median()), "mean_difference": float(d.mean()),
            "relative_difference_pct": float(100 * d.mean() / asset.loc[d.index, b].mean()),
            "median_difference": float(d.median()), "boot95_low": float(np.percentile(boots, 2.5)),
            "boot95_high": float(np.percentile(boots, 97.5)), "wilcoxon_p_descriptive": p,
            "method_lower_in_assets": wins, "threshold_k": k, "criterion_met": bool(lower_mean and wins >= k)}


def breakdown(cond: pd.DataFrame, long: pd.DataFrame, methods) -> dict:
    c = cond[cond.method.isin(methods)]
    by = {}
    for col in ("block_length", "rate"):
        by[col] = (c.groupby([col, "ticker", "method"]).nrmse.mean().groupby([col, "method"]).mean()
                   .unstack().round(5).to_dict(orient="index"))
    f = long[long.method.isin(methods)]
    by["feature"] = (f.groupby(["feature", "ticker", "method"]).rmse.mean().groupby(["feature", "method"]).mean()
                     .unstack().round(5).to_dict(orient="index"))
    a = asset_table(c)
    a["region"] = a.index.map(REGION)
    by["region_mean"] = a.groupby("region")[list(methods)].mean().round(5).to_dict(orient="index")
    return by


def summarize(cond: pd.DataFrame, long: pd.DataFrame, viol: pd.DataFrame | None, primary_tests: bool):
    """Summary for one mask type (and one fold or pooled folds) of the CPU methods."""
    present = [m for m in CPU_ORDER if m in set(cond.method)]
    counts = cond.groupby("method").size()
    n_cond = int(counts.max())
    methods = [m for m in present if counts.get(m, 0) == n_cond]        # identical conditions for every method
    asset = asset_table(cond[cond.method.isin(methods)])
    out = {"assets": int(len(asset)), "conditions": n_cond, "methods": methods,
           "mean": asset.mean().round(5).to_dict(), "median": asset.median().round(5).to_dict(),
           "mean_nmae": asset_table(cond[cond.method.isin(methods)], "nmae").mean().round(5).to_dict(),
           "mean_price_nrmse": asset_table(cond[cond.method.isin(methods)], "nrmse_price").mean().round(5).to_dict(),
           "mean_volume_nrmse": asset_table(cond[cond.method.isin(methods)], "nrmse_volume").mean().round(5).to_dict(),
           "paired_vs_cfri": {m: paired(asset, PRIMARY, m) for m in methods if m != PRIMARY},
           "breakdowns": breakdown(cond, long, methods)}
    if viol is not None and len(viol):
        out["ohlc_violation_share"] = viol.groupby("method").ohlc_violation_share.mean().round(5).to_dict()
    if primary_tests:
        out["hypotheses"] = {h: paired(asset, PRIMARY, comp) for h, comp in HYPOTHESES}
        loro = {}
        for region in sorted(set(REGION.values())):
            keep = [t for t in asset.index if REGION[t] != region]
            loro[f"without_{region}"] = {h: {k: v for k, v in paired(asset.loc[keep], PRIMARY, comp).items()
                                             if k in ("assets", "mean_difference", "method_lower_in_assets",
                                                      "threshold_k", "criterion_met")}
                                         for h, comp in HYPOTHESES}
        out["leave_one_region_out"] = loro
    return out, asset


def saits_comparisons(cond: pd.DataFrame) -> dict:
    """CFRI vs each SAITS variant, paired on the conditions SAITS completed."""
    res = {}
    cf = cond[cond.method == PRIMARY].set_index(KEY)["nrmse"]
    for s in SAITS_METHODS:
        sc = cond[cond.method == s].set_index(KEY)["nrmse"]
        if sc.empty:
            continue
        both = pd.concat({PRIMARY: cf, s: sc}, axis=1, join="inner").reset_index()
        asset = both.groupby("ticker")[[PRIMARY, s]].mean()
        res[s] = {"conditions_paired": int(len(both)), "conditions_cfri": int(len(cf)), **paired(asset, PRIMARY, s)}
    return res


# ----------------------------------------------------------------------------- fold D loading
def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path, tickers, mask_types=C.MASK_TYPES):
    """Read verified shards. Every eligible asset must have a complete CPU shard per mask type (otherwise the
    analysis stops); SAITS shards are used when their sidecar verifies, and their coverage is reported."""
    rows, viol, errors, problems, seconds = [], [], [], [], []
    for mt in mask_types:
        for t in tickers:
            f = root / "fold_D_cpu" / f"{t}_{mt}.jsonl"
            if not confirm_cpu.shard_complete(f):     # SHA-256, CODE_ID and code digest must match this code
                problems.append(f.name)
                continue
            for line in open(f):
                r = json.loads(line)
                base = {k: r[k] for k in KEY + ["mask_type"]}
                if "error" in r:
                    errors.append({**base, "error": r["error"]})
                    continue
                for m, feats in r["scores"].items():
                    for feat, (rmse, mae, n) in feats.items():
                        rows.append({**base, "method": m, "feature": feat, "rmse": rmse, "mae": mae})
                for m, v in r["ohlc_violation_share"].items():
                    viol.append({**base, "method": m, "ohlc_violation_share": v})
                viol.append({**base, "method": "truth", "ohlc_violation_share": r["truth_ohlc_violation_share"]})
                seconds.append(r["elapsed_seconds"])
    if problems:
        raise RuntimeError(f"CPU results missing, modified or from other code for {problems}; "
                           "run the cpu step to completion first")
    long = pd.DataFrame(rows)
    saits_errors, sv, coverage = [], [], {}
    for f in sorted((root / "fold_D_saits").glob("*.csv")) if (root / "fold_D_saits").is_dir() else []:
        ticker, mt, _, variant = f.stem.rsplit("_", 3)
        info = _read_json(f.with_name(f.stem + ".done.json"))
        if (ticker not in tickers or info is None or info.get("sha256") != _sha(f) or not info.get("rows")
                or info.get("fingerprint") != confirm_saits.job_fingerprint(variant, mt)):
            continue    # unverified, empty or produced by other code: not used
        job = f"{variant}_{mt}"
        coverage[job] = coverage.get(job, 0) + 1
        saits_errors += info.get("errors", [])
        sv.append(pd.read_csv(f, float_precision="round_trip"))
    run_time = {"cpu_all_methods_seconds_per_condition": float(np.mean(seconds)) if seconds else None}
    if sv:
        s = pd.concat(sv, ignore_index=True)
        per = s.drop_duplicates(KEY + ["mask_type", "method"])
        mean_s = per[~per.method.str.endswith("_proj")].groupby(["method", "mask_type"]).seconds.mean().round(3)
        run_time["saits_seconds_per_condition"] = {f"{m}_{mt}": float(v) for (m, mt), v in mean_s.items()}
        long = pd.concat([long, s[KEY + ["mask_type", "method", "feature", "rmse", "mae"]]], ignore_index=True)
        viol += s.drop_duplicates(KEY + ["mask_type", "method"])[KEY + ["mask_type", "method", "ohlc_violation_share"]].to_dict("records")
    long["rate"] = long.rate.round(2)
    viol = pd.DataFrame(viol)
    viol["rate"] = viol.rate.round(2)
    return long, viol, errors, saits_errors, coverage, run_time


def analyze(root: Path) -> dict:
    receipt = json.loads((root / "data_receipt.json").read_text())
    tickers = [a["ticker"] for a in receipt["assets"] if a["eligible"]]
    long, viol, errors, saits_errors, coverage, run_time = load(root, tickers)
    cond = condition_level(long)
    if errors:   # a failed condition is excluded for every method
        bad = pd.DataFrame(errors)[KEY + ["mask_type"]].assign(_bad=1)
        bad["rate"] = bad.rate.round(2)
        cond = cond.merge(bad, on=KEY + ["mask_type"], how="left")
        cond = cond[cond._bad.isna()].drop(columns="_bad")
    cpu_log = _read_json(root / "fold_D_cpu" / "cpu_run_log.json") or {}
    result = {"code_id": C.CODE_ID, "primary_method": PRIMARY, "fixed_setting": list(C.CFRI_FIXED),
              "eligible_assets": receipt["eligible_count"], "excluded_assets": receipt["excluded"],
              "provider_failures_accepted": receipt.get("provider_failures_accepted", False),
              "cpu_environment": cpu_log.get("environment"), "cpu_code_sha256": cpu_log.get("code_sha256"),
              "saits_assets_completed": {f"{v}_{m}": coverage.get(f"{v}_{m}", 0)
                                         for v in ("V2", "V1") for m in C.MASK_TYPES},
              "run_time": run_time,
              "saits_run_metadata": {p.name: _read_json(p) for p in sorted((root / "fold_D_saits").glob("run_metadata_*.json"))}
              if (root / "fold_D_saits").is_dir() else {},
              "cpu_condition_errors": errors, "saits_condition_errors": saits_errors, "by_mask": {}}
    for mt in sorted(cond.mask_type.unique()):
        c, l = cond[cond.mask_type == mt], long[long.mask_type == mt]
        v = viol[(viol.mask_type == mt)]
        summary, asset = summarize(c[c.method.isin(CPU_ORDER)], l[l.method.isin(CPU_ORDER)], v, primary_tests=(mt == "feature"))
        summary["saits"] = saits_comparisons(c)
        if summary["saits"]:
            sa = asset_table(c[c.method.isin(SAITS_METHODS)])
            summary["saits_mean"] = sa.mean().round(5).to_dict()
            summary["saits_assets"] = sa.notna().sum().to_dict()
            asset = asset.join(sa)    # SAITS columns: means over the conditions each SAITS run completed
        result["by_mask"][mt] = summary
        asset.to_csv(root / f"fold_D_asset_level_{mt}.csv")
    if "feature" in result["by_mask"]:
        fs = result["by_mask"]["feature"]
        primary = dict(fs["hypotheses"])
        h, comp = SAITS_HYPOTHESIS
        primary[h] = fs["saits"].get(comp) or {"comparator": comp, "evaluable": False, "criterion_met": False}
        result["primary"] = primary
        result["projection_equalized"] = {c: fs["paired_vs_cfri"].get(c) or fs["saits"].get(c)
                                          for c in PROJECTION_EQUALIZED}
    (root / "fold_D_results.json").write_text(json.dumps(result, indent=1, default=float))
    (root / "fold_D_report.md").write_text(report(result))
    return result


def report(result: dict) -> str:
    excluded = [e["ticker"] for e in result["excluded_assets"]]
    lines = [f"# CFRI confirmatory fold D results ({result['code_id']})", "",
             f"Eligible assets: {result['eligible_assets']}; excluded: {excluded or 'none'} (reasons in "
             f"data_receipt.json); CPU condition errors: {len(result['cpu_condition_errors'])}.",
             "SAITS assets completed (of " + str(result["eligible_assets"]) + "): " +
             ", ".join(f"{k} {v}" for k, v in result["saits_assets_completed"].items()) +
             f"; SAITS condition errors: {len(result['saits_condition_errors'])}.", ""]

    def row(label, r):
        if r is None or r.get("evaluable") is False or not r.get("assets"):
            return f"| {label} | {(r or {}).get('comparator', '')} | not evaluable | | | | | no |"
        return (f"| {label} | {r['comparator']} | {r['mean_method']:.4f} | {r['mean_comparator']:.4f} | "
                f"{r['mean_difference']:+.4f} ({r['boot95_low']:+.4f}, {r['boot95_high']:+.4f}) | "
                f"{r['method_lower_in_assets']}/{r['assets']} | {r['threshold_k']} | {'yes' if r['criterion_met'] else 'no'} |")

    head = ["| | comparator | CFRI mean | comparator mean | difference (95% bootstrap) | CFRI lower in | k | met |",
            "|---|---|---|---|---|---|---|---|"]
    if "primary" in result:
        lines += ["## Primary hypotheses (feature-wise masks)", ""] + head
        lines += [row(h, r) for h, r in result["primary"].items()] + [""]
        lines += ["## Projection-equalized comparisons (secondary, feature-wise masks)", ""] + head
        lines += [row(f"P{i}", r or {"comparator": c, "evaluable": False})
                  for i, (c, r) in enumerate(result["projection_equalized"].items(), 1)] + [""]
    for mt, s in result["by_mask"].items():
        lines += [f"## Mean asset-level NRMSE, {mt} masks ({s['assets']} assets, {s['conditions']} conditions per method)", "",
                  "| method | mean | median | CFRI lower in |", "|---|---|---|---|"]
        for m in s["methods"]:
            r = s["paired_vs_cfri"].get(m)
            w = f"{r['method_lower_in_assets']}/{r['assets']}" if r else ""
            lines.append(f"| {m} | {s['mean'][m]:.4f} | {s['median'][m]:.4f} | {w} |")
        for m, r in s.get("saits", {}).items():
            lines.append(f"| {m} (paired on {r['conditions_paired']} of {r['conditions_cfri']} conditions) | "
                         f"{r['mean_comparator']:.4f} | {r['median_comparator']:.4f} | {r['method_lower_in_assets']}/{r['assets']} |")
        if "ohlc_violation_share" in s:
            v = s["ohlc_violation_share"]
            lines += ["", "OHLC violation share of rows with an imputed price: " +
                      ", ".join(f"{k} {v[k]:.3f}" for k in sorted(v))]
        lines.append("")
    return "\n".join(lines)


def export(root: Path, code_dir: Path, name: str = "CFRI_fold_D_results.zip") -> Path:
    """Metrics, receipts, analysis and code only; files under raw_do_not_publish are never included."""
    zpath = root / name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("fold_D_cpu", "fold_D_saits"):
            if (root / sub).is_dir():
                for f in sorted((root / sub).glob("*")):
                    if f.is_file() and not f.name.endswith(".tmp"):
                        z.write(f, f"{sub}/{f.name}")
        for f in ["data_receipt.json", "fold_D_results.json", "fold_D_report.md", "code_sha256.json"] + \
                 [p.name for p in sorted(root.glob("fold_D_asset_level_*.csv"))]:
            if (root / f).exists():
                z.write(root / f, f)
        for f in sorted(code_dir.glob("*.py")):
            z.write(f, f"code/{f.name}")
        bad = [n for n in z.namelist() if "raw_do_not_publish" in n]
        if bad:
            raise RuntimeError(f"raw data in export: {bad}")
    return zpath


def code_hashes(code_dir: Path) -> dict:
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(code_dir.glob("*.py"))}
