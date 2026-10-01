"""Level-shift diagnostic: test-period z-scores under training scaling vs neural NRMSE."""
import io, json, sys, zipfile
sys.path.insert(0, "results/code_and_protocol")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from run_classical_benchmark import FEATURES, FOLDS, load_verified_asset, split_frame
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.3f}")

with zipfile.ZipFile("private/snapshot.zip") as a:
    receipt = json.loads(a.read("acquisition_qc_receipt.json"))
    rows = []
    for rec in receipt["assets"]:
        frame = load_verified_asset(a, rec)
        for fold in FOLDS:
            parts = split_frame(frame, fold)
            tr, va, te = (parts[k].to_numpy(float) for k in ("train", "val", "test"))
            mu, sd = tr.mean(0), tr.std(0)
            z_te = (te - mu) / sd
            z_va = (va - mu) / sd
            r = dict(ticker=rec["ticker"], fold=fold, n_train=len(tr), n_val=len(va), n_test=len(te))
            for j, f in enumerate(FEATURES):
                r[f"test_mean_abs_z_{f}"] = np.abs(z_te[:, j]).mean()
                r[f"test_frac_out2_{f}"] = (np.abs(z_te[:, j]) > 2).mean()
                r[f"val_mean_abs_z_{f}"] = np.abs(z_va[:, j]).mean()
            r["test_mean_abs_z_price"] = np.mean([r[f"test_mean_abs_z_{f}"] for f in FEATURES[:4]])
            r["test_frac_out2_price"] = np.mean([r[f"test_frac_out2_{f}"] for f in FEATURES[:4]])
            r["val_mean_abs_z_price"] = np.mean([r[f"val_mean_abs_z_{f}"] for f in FEATURES[:4]])
            rows.append(r)
z = pd.DataFrame(rows)
z.to_csv("sweep_out/level_shift_by_asset_fold.csv", index=False)

c = pd.read_csv("results/analysis/validation_test_conditions.csv")
af = c.groupby(["ticker", "fold", "method"])[["nrmse", "validation_nrmse"]].mean().unstack("method")
test_nrmse = af["nrmse"]; val_nrmse = af["validation_nrmse"]
m = z.set_index(["ticker", "fold"]).join(test_nrmse.add_prefix("test_")).join(val_nrmse.add_prefix("val_"))

print("=== windows per partition (train/val/test rows) by fold ===")
print(z.groupby("fold")[["n_train", "n_val", "n_test"]].mean().to_string())
print("\ntrain windows (stride 16, len 64) by fold:", {f: int((n - 64) // 16 + 1 + 1) for f, n in z.groupby("fold").n_train.mean().items()})

print("\n=== test-period price z-score magnitude by fold (mean over assets) ===")
print(m.groupby("fold")[["test_mean_abs_z_price", "test_frac_out2_price", "val_mean_abs_z_price"]].mean().to_string())

print("\n=== Spearman: test_mean_abs_z_price vs test NRMSE (102 asset-folds) ===")
for meth in ["linear", "pchip", "itri_w1_p5", "chained_ridge", "knn5", "brits", "saits"]:
    rho, p = spearmanr(m["test_mean_abs_z_price"], m[f"test_{meth}"])
    print(f"{meth:14s} rho = {rho:+.3f}")

print("\n=== per asset (fold-averaged): z magnitude vs SAITS/BRITS NRMSE ===")
pa = m.groupby("ticker")[["test_mean_abs_z_price", "test_frac_out2_price", "test_linear", "test_brits", "test_saits"]].mean().sort_values("test_mean_abs_z_price")
print(pa.to_string())
rho, p = spearmanr(pa["test_mean_abs_z_price"], pa["test_saits"]); print(f"\nasset-level Spearman z vs SAITS: {rho:+.3f}")
rho, p = spearmanr(pa["test_mean_abs_z_price"], pa["test_brits"]); print(f"asset-level Spearman z vs BRITS: {rho:+.3f}")
rho, p = spearmanr(pa["test_mean_abs_z_price"], pa["test_linear"]); print(f"asset-level Spearman z vs linear: {rho:+.3f}")

# validation: same relation?
print("\n=== Spearman on VALIDATION partition: val_mean_abs_z_price vs val NRMSE ===")
for meth in ["linear", "brits", "saits"]:
    rho, p = spearmanr(m["val_mean_abs_z_price"], m[f"val_{meth}"])
    print(f"{meth:14s} rho = {rho:+.3f}")

# binned view
m["zbin"] = pd.cut(m["test_mean_abs_z_price"], [0, 1, 2, 3, 4, 100], labels=["<1", "1-2", "2-3", "3-4", ">4"])
print("\n=== test NRMSE by |z| bin (asset-fold units) ===")
print(m.groupby("zbin", observed=True)[["test_linear", "test_itri_w1_p5", "test_brits", "test_saits"]].agg(["mean", "count"]).to_string())
m.reset_index().to_csv("sweep_out/level_shift_joined.csv", index=False)
