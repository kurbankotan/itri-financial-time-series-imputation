#!/usr/bin/env python3
"""Recompute every external-validation number reported in Sections 5.8-5.10 of the manuscript
(and the related statements in Sections 6.4-6.5) from the public feature-level rows, and assert it.
Run from any directory: python external_validation/code/verify_reported_numbers.py
Exit status 0 means every check passed.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import repo_paths as RP
from cross_hardware_check import compute as cross_hardware

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  <- {detail}"))
    if not cond: fails.append(name)

KEY = ["ticker", "fold", "block_length", "rate", "seed"]
test = pd.read_csv(RP.V1_TEST)
val = pd.read_csv(RP.V1_VALIDATION)
def cond_level(df): return df.groupby(KEY + ["method"])[["rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]].mean().reset_index()
def asset_pivot(c, col="rmse_trainstd"): return c.groupby(["ticker", "method"])[col].mean().unstack("method")
ct, cv = cond_level(test), cond_level(val)
at, av = asset_pivot(ct), asset_pivot(cv)
r4 = lambda v: round(float(v), 4)

# 5.8 primary
check("linear mean 0.2704", r4(at.linear.mean()) == 0.2704); check("linear median 0.2672", r4(at.linear.median()) == 0.2672)
check("itri mean 0.2723", r4(at.itri_w1_p5.mean()) == 0.2723); check("itri median 0.2662", r4(at.itri_w1_p5.median()) == 0.2662)
d = at.itri_w1_p5 - at.linear
check("paired mean diff +0.0020", r4(d.mean()) == 0.0020); check("paired median diff +0.0009", r4(d.median()) == 0.0009)
check("itri better than linear 16/34", int((d < 0).sum()) == 16)
check("validation itri 0.2244 vs linear 0.2307", r4(av.itri_w1_p5.mean()) == 0.2244 and r4(av.linear.mean()) == 0.2307)
check("pchip 0.2785 / 23 wins", r4(at.pchip.mean()) == 0.2785 and int(((at.itri_w1_p5 - at.pchip) < 0).sum()) == 23)
check("chained 0.3330 / 31 wins", r4(at.chained_ridge.mean()) == 0.3330 and int(((at.itri_w1_p5 - at.chained_ridge) < 0).sum()) == 31)
check("knn 1.6014 brits 1.1950 saits 1.1460, 34 wins each", r4(at.knn5.mean()) == 1.6014 and r4(at.brits.mean()) == 1.1950 and r4(at.saits.mean()) == 1.1460
      and all(int(((at.itri_w1_p5 - at[m]) < 0).sum()) == 34 for m in ("knn5", "brits", "saits")))
med_s = ct.groupby("method").fit_and_impute_seconds.median()
check("median seconds itri 0.016, saits 1.4, brits 46", round(med_s.itri_w1_p5, 3) == 0.016 and round(med_s.saits, 1) == 1.4 and round(med_s.brits) == 46)
# by block
def by_block(m):
    return {b: r4(asset_pivot(g)[m].mean()) for b, g in ct.groupby("block_length")}
lb, ib = by_block("linear"), by_block("itri_w1_p5")
wins = {b: int(((asset_pivot(g).itri_w1_p5 - asset_pivot(g).linear) < 0).sum()) for b, g in ct.groupby("block_length")}
check("block1 0.2160 vs 0.2215, 27 wins", ib[1] == 0.2160 and lb[1] == 0.2215 and wins[1] == 27)
check("block2 0.2323 vs 0.2348, 19 wins", ib[2] == 0.2323 and lb[2] == 0.2348 and wins[2] == 19)
check("blocks 5,10,20 wins 14,14,15", (wins[5], wins[10], wins[20]) == (14, 14, 15))
cb = by_block("chained_ridge"); check("chained block1 0.3510, block20 0.2902", cb[1] == 0.3510 and cb[20] == 0.2902)
sb = by_block("saits"); check("saits block1 1.1501 block20 1.1281", sb[1] == 1.1501 and sb[20] == 1.1281)
# by feature
fe = test.groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
check("close itri==linear 0.1423", r4(fe.loc["close", "itri_w1_p5"]) == 0.1423 and r4(fe.loc["close", "linear"]) == 0.1423)
check("volume 0.8111 vs 0.8015", r4(fe.loc["volume", "itri_w1_p5"]) == 0.8111 and r4(fe.loc["volume", "linear"]) == 0.8015)
check("price NRMSE >1.2 for neural, volume 0.89-0.93", all(fe.loc[f, m] > 1.2 for f in ("open", "high", "low", "close") for m in ("brits", "saits"))
      and 0.89 <= fe.loc["volume", "saits"] <= 0.93 and 0.89 <= fe.loc["volume", "brits"] <= 0.93)
check("pchip worse than linear on every feature", all(fe.loc[f, "pchip"] > fe.loc[f, "linear"] for f in fe.index))
# folds
fold = {f: asset_pivot(g) for f, g in ct.groupby("fold")}
check("brits folds 1.73/0.80/1.05", tuple(round(fold[f].brits.mean(), 2) for f in "ABC") == (1.73, 0.80, 1.05))
check("saits folds 1.60/0.75/1.09", tuple(round(fold[f].saits.mean(), 2) for f in "ABC") == (1.60, 0.75, 1.09))
check("linear folds 0.31 -> 0.23", round(fold["A"].linear.mean(), 2) == 0.31 and round(fold["C"].linear.mean(), 2) == 0.23)
gap = {m: r4(at[m].mean() - av[m].mean()) for m in at.columns}
check("test-val gap neural 0.55-0.58, temporal 0.04-0.05", 0.55 <= round(gap["brits"], 2) <= 0.58 and 0.55 <= round(gap["saits"], 2) <= 0.58
      and all(0.04 <= round(gap[m], 2) <= 0.05 for m in ("linear", "pchip", "itri_w1_p5")))
# 5.9 sweep
sw_t = RP.read(RP.SWEEP_TEST); sw_v = RP.read(RP.SWEEP_VALIDATION)
st, sv = asset_pivot(cond_level(sw_t)), asset_pivot(cond_level(sw_v))
tr = {m: st[m].mean() for m in st.columns}; vr = {m: sv[m].mean() for m in sv.columns}
check("grid range 0.2592..0.2723", r4(min(tr.values())) == 0.2592 and r4(max(tr.values())) == 0.2723 and max(tr, key=tr.get) == "itri_w1_p5")
check("w1p5 rank 28/28 on val and test", sorted(tr, key=tr.get)[-1] == "itri_w1_p5" and sorted(vr, key=vr.get)[-1] == "itri_w1_p5")
vsel = min(vr, key=vr.get)
check("val-selected w7p5: val 0.2166 test 0.2599, 32 wins", vsel == "itri_w7_p5" and r4(vr[vsel]) == 0.2166 and r4(tr[vsel]) == 0.2599 and int(((st[vsel] - at.linear) < 0).sum()) == 32)
check("post hoc best w10p2 0.2592", min(tr, key=tr.get) == "itri_w10_p2" and r4(tr["itri_w10_p2"]) == 0.2592)
rho = spearmanr(list(vr.values()), [tr[k] for k in vr]).statistic; check("spearman val-test 0.85", round(rho, 2) == 0.85)
red = {m: 100 * (at.linear.mean() - tr[m]) / at.linear.mean() for m in tr}
check("best reductions 3.9%-4.1%", round(red["itri_w7_p5"], 1) == 3.9 and round(red["itri_w10_p2"], 1) == 4.1 and max(red.values()) < 4.2)
# monotone: every increase in w or decrease in p lowers the mean (checked as: p=1 row decreasing in w? not strictly). Check the claim loosely:
W = [1, 2, 3, 4, 5, 7, 10]; P = [1, 2, 3, 5]
claim_w = all(tr[f"itri_w{W[i+1]}_p{p}"] <= tr[f"itri_w{W[i]}_p{p}"] + 5e-4 for p in P for i in range(len(W) - 1))
claim_p = all(tr[f"itri_w{w}_p{P[j]}"] <= tr[f"itri_w{w}_p{P[j+1]}"] + 5e-4 for w in W for j in range(len(P) - 1))
print("   monotone-in-w (tol 5e-4):", claim_w, " monotone-in-p:", claim_p)
check("w=1,2: each additional pass raises the mean", all(tr[f"itri_w{w}_p{P[j+1]}"] > tr[f"itri_w{w}_p{P[j]}"] for w in (1, 2) for j in range(3)))
check("widening 1->4 lowers the mean at every p", all(tr[f"itri_w{W[i+1]}_p{p}"] < tr[f"itri_w{W[i]}_p{p}"] for p in P for i in range(3)))
check("w>=4 surface flat within 0.002, p=2 or 3 best", all(max(tr[f"itri_w{w}_p{p}"] for w in (4, 5, 7, 10)) - min(tr[f"itri_w{w}_p{p}"] for w in (4, 5, 7, 10)) < 0.002 for p in P)
      and all(min(P, key=lambda p: tr[f"itri_w{w}_p{p}"]) in (2, 3) for w in (4, 5, 7, 10)))
sfe = sw_t.groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
maxdev = max(abs(sfe.loc[f, m] - fe.loc[f, "linear"]) for f in ("open", "high", "low", "close") for m in sfe.columns)
check("prices within 0.004 of linear for all configs", maxdev < 0.004, f"{maxdev:.4f}")
check("volume 0.7641 (w1p1) and 0.7441 (w7p5)", r4(sfe.loc["volume", "itri_w1_p1"]) == 0.7641 and r4(sfe.loc["volume", "itri_w7_p5"]) == 0.7441)
coef = pd.read_csv(RP.W1_COEFFICIENTS); coef["sum"] = coef.coef_lag1 + coef.coef_lead1
g = coef.groupby("feature")
prices = coef[coef.feature != "volume"]
check("price coefs ~0.4994/0.5008, sum 1.0002 sd 0.0005, intercept -0.0001",
      r4(prices.coef_lag1.mean()) in (0.4994, 0.4995, 0.4996) and r4(prices.coef_lead1.mean()) in (0.5006, 0.5007, 0.5008)
      and r4(prices["sum"].mean()) in (1.0002, 1.0003, 1.0004) and round(prices["sum"].std(), 4) <= 0.0006 and r4(prices.intercept.mean()) == -0.0001)
vol = coef[coef.feature == "volume"]
check("volume coefs 0.3172/0.3174 sum 0.6346", r4(vol.coef_lag1.mean()) == 0.3172 and r4(vol.coef_lead1.mean()) == 0.3174 and r4(vol["sum"].mean()) == 0.6346)
# 5.10 level shift
z = pd.read_csv(RP.LEVEL_SHIFT)
af = ct.groupby(["ticker", "fold", "method"]).rmse_trainstd.mean().unstack("method")
m = z.set_index(["ticker", "fold"]).join(af)
zf = m.groupby("fold")[["test_mean_abs_z_price", "test_frac_out2_price"]].mean()
check("mean |z| 2.41/1.81/2.40", tuple(round(v, 2) for v in zf.test_mean_abs_z_price) == (2.41, 1.81, 2.40))
check("share >2sd more than a third every fold", (zf.test_frac_out2_price > 1 / 3).all())
rh = {meth: round(spearmanr(m.test_mean_abs_z_price, m[meth]).statistic, 2) for meth in ("linear", "pchip", "itri_w1_p5", "chained_ridge", "brits", "saits", "knn5")}
check("spearman 0.25/0.24/0.23/0.75/0.89/0.90/0.97", (rh["linear"], rh["pchip"], rh["itri_w1_p5"], rh["chained_ridge"], rh["brits"], rh["saits"], rh["knn5"]) == (0.25, 0.24, 0.23, 0.75, 0.89, 0.90, 0.97))
m["zbin"] = pd.cut(m.test_mean_abs_z_price, [0, 1, 2, 3, 4, 100])
lo = m[m.zbin == pd.Interval(0, 1)]; hi = m[m.zbin == pd.Interval(4, 100)]
check("bin<1: saits 0.363 brits 0.403 linear 0.249 chained ~0.25", round(lo.saits.mean(), 3) == 0.363 and round(lo.brits.mean(), 3) == 0.403 and round(lo.linear.mean(), 3) == 0.249 and abs(lo.chained_ridge.mean() - lo.linear.mean()) < 0.005)
check("bin>4: saits 3.26 knn 4.01 chained 0.52 linear/itri 0.31", round(hi.saits.mean(), 2) == 3.26 and round(hi.knn5.mean(), 2) == 4.01 and round(hi.chained_ridge.mean(), 2) == 0.52 and round(hi.linear.mean(), 2) == 0.31 and round(hi.itri_w1_p5.mean(), 2) == 0.31)
pa = m.groupby("ticker").test_mean_abs_z_price.mean().sort_values()
check("most displaced include EWT, EWC, EDEN, EWL; least THD, EIDO, EWZ", set(pa.index[-4:]) == {"EWT", "EWC", "EDEN", "EWL"} and set(pa.index[:3]) == {"THD", "EIDO", "EWZ"})
# neural budget
wins_ = {f: (n - 64) // 16 + 2 for f, n in z.groupby("fold").n_train.mean().items()}
check("training windows 60/92/123", (wins_["A"], wins_["B"], wins_["C"]) == (60, 92, 123))
# development numbers still as in v5 (unchanged files)
ts = pd.read_csv(RP.DEV_TEST_SUMMARY).set_index(["window", "passes"])
check("dev w10p5 2.633", round(ts.loc[(10, 5), "mean_nrmse"], 3) == 2.633)

# ---------------- amended results (if present)
am = RP.WINDOWNORM_SAITS.exists()
if am:
    A = pd.read_csv(RP.WINDOWNORM_SAITS); A = A[A.partition == "test"]
    n = A.groupby("method")[KEY].apply(lambda g: g.drop_duplicates().shape[0])
    print("   amended methods/conditions:", n.to_dict())


# ---------------- amended SAITS claims (Table 20 / 5.10 / 6.5)
if am and "saits_winnorm_V1" in A.method.unique() and n.get("saits_winnorm_V1") == 5508:
    A = A[A.method == "saits_winnorm_V1"]   # this block checks the 30-epoch variant only
    A_c = A.groupby(KEY + ["method"])[["rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]].mean().reset_index()
    A_at = A_c.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack("method")["saits_winnorm_V1"]
    A_mae = A_c.groupby(["ticker", "method"]).mae_trainstd.mean().unstack("method")["saits_winnorm_V1"]
    check("amended saits mean 0.2440 median 0.2438 nmae 0.1546", r4(A_at.mean()) == 0.2440 and r4(A_at.median()) == 0.2438 and r4(A_mae.mean()) == 0.1546)
    check("amended 10.4% below ITRI, 9.7% below linear", round(100 * (at.itri_w1_p5.mean() - A_at.mean()) / at.itri_w1_p5.mean(), 1) == 10.4
          and round(100 * (at.linear.mean() - A_at.mean()) / at.linear.mean(), 1) == 9.7)
    check("amended better than ITRI and linear in 34/34", int((A_at < at.itri_w1_p5).sum()) == 34 and int((A_at < at.linear).sum()) == 34)
    Aaf = A_c.groupby(["ticker", "fold"]).rmse_trainstd.mean()
    mm = z.set_index(["ticker", "fold"]).join(Aaf.rename("amended"))
    check("amended spearman 0.17", round(spearmanr(mm.test_mean_abs_z_price, mm.amended).statistic, 2) == 0.17)
    mm["zbin"] = pd.cut(mm.test_mean_abs_z_price, [0, 1, 2, 3, 4, 100])
    check("amended bin>4 0.28, bin<1 0.23", round(mm[mm.zbin == pd.Interval(4, 100)].amended.mean(), 2) == 0.28 and round(mm[mm.zbin == pd.Interval(0, 1)].amended.mean(), 2) == 0.23)
    ab = {b: r4(g.groupby("ticker").rmse_trainstd.mean().mean()) for b, g in A_c.groupby("block_length")}
    check("amended by block 0.2180 / 0.2230 / 0.2279 / 0.2334 / 0.2604 / 0.3013", [ab[b] for b in (1, 2, 3, 5, 10, 20)] == [0.2180, 0.2230, 0.2279, 0.2334, 0.2604, 0.3013])
    aw = {}
    for b, g in A_c.groupby("block_length"):
        s_ = g.groupby("ticker").rmse_trainstd.mean(); i_ = ct[(ct.block_length == b) & (ct.method == "itri_w1_p5")].groupby("ticker").rmse_trainstd.mean()
        aw[b] = int((s_ < i_).sum())
    check("amended wins vs ITRI by block 14/28/31/34/34/34", [aw[b] for b in (1, 2, 3, 5, 10, 20)] == [14, 28, 31, 34, 34, 34])
    afe = A.groupby(["ticker", "feature"]).rmse_trainstd.mean().groupby("feature").mean()
    check("amended close 0.1221 volume 0.7479", r4(afe["close"]) == 0.1221 and r4(afe["volume"]) == 0.7479)
    check("amended median seconds 6.2, fallbacks 0", round(A_c.fit_and_impute_seconds.median(), 1) == 6.2 and int(A.window_fallbacks.sum()) == 0)


# ---------------- amendment v2: SAITS V2 full grid and BRITS subset, recomputed from raw shards
def _load(path):
    return pd.read_csv(path, float_precision="round_trip")
_raw = _load(RP.WINDOWNORM_SAITS); _raw = _raw[_raw.partition == "test"]; _raw["rate"] = _raw.rate.round(2)
_v1f = pd.read_csv(RP.V1_TEST, float_precision="round_trip"); _v1f["rate"] = _v1f.rate.round(2)
_cols = KEY + ["feature", "method", "rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]
_all = pd.concat([_v1f[_cols], _raw[_cols]], ignore_index=True)
_c = _all.groupby(KEY + ["method"])[["rmse_trainstd", "mae_trainstd", "fit_and_impute_seconds"]].mean().reset_index()
_a = _c.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack(); _amae = _c.groupby(["ticker", "method"]).mae_trainstd.mean().unstack()
check("V2 conditions 5508", int((_c.method == "saits_winnorm_V2").sum()) == 5508)
check("V2 mean 0.2219 median 0.2218 nmae 0.1398", r4(_a.saits_winnorm_V2.mean()) == 0.2219 and r4(_a.saits_winnorm_V2.median()) == 0.2218 and r4(_amae.saits_winnorm_V2.mean()) == 0.1398)
check("V2 18.5% below ITRI, 17.9% below linear", round(100 * (_a.itri_w1_p5.mean() - _a.saits_winnorm_V2.mean()) / _a.itri_w1_p5.mean(), 1) == 18.5
      and round(100 * (_a.linear.mean() - _a.saits_winnorm_V2.mean()) / _a.linear.mean(), 1) == 17.9)
check("V1 and V2 below ITRI and linear in 34/34", all(int((_a[m] < _a[b]).sum()) == 34 for m in ("saits_winnorm_V1", "saits_winnorm_V2") for b in ("itri_w1_p5", "linear")))
_af = _c.groupby(["ticker", "fold", "method"]).rmse_trainstd.mean().unstack()
_m = z.set_index(["ticker", "fold"]).join(_af)
check("spearman V1 0.17, V2 0.16, SAITS frozen 0.90", round(spearmanr(_m.test_mean_abs_z_price, _m.saits_winnorm_V1).statistic, 2) == 0.17
      and round(spearmanr(_m.test_mean_abs_z_price, _m.saits_winnorm_V2).statistic, 2) == 0.16 and round(spearmanr(_m.test_mean_abs_z_price, _m.saits).statistic, 2) == 0.90)
_m["zbin"] = pd.cut(_m.test_mean_abs_z_price, [0, 1, 2, 3, 4, 100]); _hi = _m[_m.zbin == pd.Interval(4, 100)]
check("bin>4: frozen 3.26, V1 0.28, V2 0.25", round(_hi.saits.mean(), 2) == 3.26 and round(_hi.saits_winnorm_V1.mean(), 2) == 0.28 and round(_hi.saits_winnorm_V2.mean(), 2) == 0.25)
_bb = {b: g.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack() for b, g in _c.groupby("block_length")}
check("block1: V1 0.2180 (14 wins) vs ITRI 0.2160; V2 0.1987 (32 wins)", r4(_bb[1].saits_winnorm_V1.mean()) == 0.2180 and r4(_bb[1].itri_w1_p5.mean()) == 0.2160
      and int((_bb[1].saits_winnorm_V1 < _bb[1].itri_w1_p5).sum()) == 14 and r4(_bb[1].saits_winnorm_V2.mean()) == 0.1987 and int((_bb[1].saits_winnorm_V2 < _bb[1].itri_w1_p5).sum()) == 32)
check("blocks >=2: both variants beat ITRI in 28-34 assets", min(int((_bb[b][m] < _bb[b].itri_w1_p5).sum()) for b in (2, 3, 5, 10, 20) for m in ("saits_winnorm_V1", "saits_winnorm_V2")) == 28)
check("block20: V2 0.2709, ITRI 0.3636, linear 0.3571, chained 0.2902", (r4(_bb[20].saits_winnorm_V2.mean()), r4(_bb[20].itri_w1_p5.mean()), r4(_bb[20].linear.mean()), r4(_bb[20].chained_ridge.mean())) == (0.2709, 0.3636, 0.3571, 0.2902))
_v1m = ["linear", "pchip", "knn5", "chained_ridge", "itri_w1_p5", "brits", "saits", "saits_winnorm_V1"]
check("V2 most accurate at every block length", all(_bb[b].saits_winnorm_V2.mean() < min(_bb[b][x].mean() for x in _v1m) for b in _bb))
_fe = _all.groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
check("V2 close 0.0968 vs 0.1423, volume 0.7415 vs 0.8015, below linear on all features", r4(_fe.loc["close", "saits_winnorm_V2"]) == 0.0968 and r4(_fe.loc["volume", "saits_winnorm_V2"]) == 0.7415
      and bool((_fe.saits_winnorm_V2 < _fe.linear).all()))
check("V2 zero fallbacks", int(_raw[_raw.method == "saits_winnorm_V2"].window_fallbacks.sum()) == 0)
# BRITS subset
_br = _load(RP.WINDOWNORM_BRITS_SUBSET); _br = _br[_br.partition == "test"]; _br["rate"] = _br.rate.round(2)
_sk = _br[KEY].drop_duplicates(); check("BRITS subset 612 conditions, fold A, seed 11", len(_sk) == 612 and set(_sk.fold) == {"A"} and set(_sk.seed) == {11})
_s = pd.concat([_all, _br[_cols]], ignore_index=True).merge(_sk, on=KEY)
_sc = _s.groupby(KEY + ["method"])[["rmse_trainstd", "fit_and_impute_seconds"]].mean().reset_index()
check("every method has 612 subset conditions", bool((_sc.groupby("method").size() == 612).all()))
_sa = _sc.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack()
check("subset: BRITS 1.7305 -> 0.3777; ITRI 0.3083; linear 0.3124; SAITS 0.3014 / 0.2661",
      (r4(_sa.brits.mean()), r4(_sa.brits_winnorm_V1.mean()), r4(_sa.itri_w1_p5.mean()), r4(_sa.linear.mean()), r4(_sa.saits_winnorm_V1.mean()), r4(_sa.saits_winnorm_V2.mean()))
      == (1.7305, 0.3777, 0.3083, 0.3124, 0.3014, 0.2661))
check("subset: ITRI and linear better than normalized BRITS in 34 and 33 assets", int((_sa.itri_w1_p5 < _sa.brits_winnorm_V1).sum()) == 34 and int((_sa.linear < _sa.brits_winnorm_V1).sum()) == 33)
_zA = z[z.fold == "A"].set_index("ticker").loc[_sa.index, "test_mean_abs_z_price"]
check("subset spearman: BRITS frozen 0.99 -> 0.42, linear 0.46", (round(spearmanr(_zA, _sa.brits).statistic, 2), round(spearmanr(_zA, _sa.brits_winnorm_V1).statistic, 2), round(spearmanr(_zA, _sa.linear).statistic, 2)) == (0.99, 0.42, 0.46))
_bw = [int((g.groupby(["ticker", "method"]).rmse_trainstd.mean().unstack().pipe(lambda d: d.brits_winnorm_V1 < d.itri_w1_p5)).sum()) for _, g in _sc.groupby("block_length")]
check("subset: normalized BRITS better than ITRI in 0-8 assets per block", min(_bw) == 0 and max(_bw) == 8)
_sfe = _s.groupby(["ticker", "feature", "method"]).rmse_trainstd.mean().groupby(["feature", "method"]).mean().unstack()
_pr = ["open", "high", "low", "close"]
check("subset: BRITS price 0.25-0.27, interp 0.16-0.17, BRITS better on volume",
      (round(_sfe.loc[_pr, "brits_winnorm_V1"].min(), 2), round(_sfe.loc[_pr, "brits_winnorm_V1"].max(), 2), round(_sfe.loc[_pr, "linear"].min(), 2), round(_sfe.loc[_pr, "linear"].max(), 2)) == (0.25, 0.27, 0.16, 0.17)
      and _sfe.loc["volume", "brits_winnorm_V1"] < min(_sfe.loc["volume", "linear"], _sfe.loc["volume", "itri_w1_p5"]))
check("subset: BRITS median 32 s per condition", round(_sc[_sc.method == "brits_winnorm_V1"].fit_and_impute_seconds.median()) == 32)
_R = cross_hardware()
check("cross-hardware: BRITS <= 2.2e-5, SAITS <= 0.04 per condition, means within 0.0033",
      _R[_R.piece.str.contains("brits")].max_abs_feature_diff.max() <= 2.2e-5 and _R[_R.piece.str.contains("saits")].max_abs_condition_macro_diff.max() <= 0.04
      and round((_R[_R.piece.str.contains("saits")].mean_cpu - _R[_R.piece.str.contains("saits")].mean_gpu).abs().max(), 4) == 0.0033)

print("\nFAILS:", fails if fails else "none")
sys.exit(1 if fails else 0)
