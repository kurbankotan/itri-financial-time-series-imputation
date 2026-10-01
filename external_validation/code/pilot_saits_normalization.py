#!/usr/bin/env python3
"""EXPLORATORY CPU pilot: does per-window (instance) normalization repair SAITS/BRITS
on trending ETFs?  Not publishable as the prespecified result; labeled exploratory.

Variants
  V0 frozen     : protocol as archived (training-period global scaling, <=30 epochs)
  V1 window     : V0 + per-window normalization from each window's observed cells (RevIN-style)
  V2 window+long: V1 with epochs<=200, patience 20
"""
from __future__ import annotations
import argparse, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "results/code_and_protocol")
from masking import run_lengths, strict_block_mask  # noqa
from run_classical_benchmark import (EDGE_GUARD, FEATURES, FOLDS, corrupt, load_verified_asset,  # noqa
                                     score_features, sha256, split_frame, stable_mask_seed,
                                     train_observed_scaler)
from run_neural_benchmark import (N_STEPS, STRIDE, BATCH_SIZE, BRITS_HIDDEN, SAITS_CONFIG,  # noqa
                                  window_starts, as_windows, fold_windows, set_training_seed,
                                  training_seed)
from pypots.imputation import BRITS, SAITS  # noqa


def window_stats(win: np.ndarray):
    """Per-window, per-feature mean/std from that window's OBSERVED cells only."""
    mu = np.nanmean(win, axis=1, keepdims=True)          # (n_win, 1, n_feat)
    sd = np.nanstd(win, axis=1, keepdims=True)
    n_obs = np.sum(np.isfinite(win), axis=1, keepdims=True)
    bad = (n_obs < 2) | ~np.isfinite(sd) | (sd < 1e-6)
    mu = np.where(bad, 0.0, mu)          # fallback: keep protocol scaling (identity)
    sd = np.where(bad, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32), int(bad.sum())


def fit_predict(method, train_in, val_in, val_truth, test_in, seed, epochs, patience, per_window):
    set_training_seed(seed)
    tr_s, va_s, te_s = window_starts(len(train_in)), window_starts(len(val_in)), window_starts(len(test_in))
    Xtr, Xva, Xva_ori, Xte = (as_windows(a, s) for a, s in ((train_in, tr_s), (val_in, va_s), (val_truth, va_s), (test_in, te_s)))
    fallbacks = 0
    if per_window:
        mtr, str_, f1 = window_stats(Xtr); mva, sva, f2 = window_stats(Xva); mte, ste, f3 = window_stats(Xte)
        fallbacks = f1 + f2 + f3
        Xtr = (Xtr - mtr) / str_; Xva_ori = (Xva_ori - mva) / sva; Xva = (Xva - mva) / sva; Xte = (Xte - mte) / ste
    common = dict(n_steps=N_STEPS, n_features=len(FEATURES), batch_size=BATCH_SIZE, epochs=epochs,
                  patience=min(patience, epochs), device="cpu", model_saving_strategy=None, verbose=False)
    model = BRITS(rnn_hidden_size=BRITS_HIDDEN, **common) if method == "brits" else SAITS(**SAITS_CONFIG, **common)
    t0 = time.perf_counter()
    model.fit({"X": Xtr}, {"X": Xva, "X_ori": Xva_ori})
    Pva = np.asarray(model.predict({"X": Xva})["imputation"])
    Pte = np.asarray(model.predict({"X": Xte})["imputation"])
    if per_window:
        Pva = Pva * sva + mva; Pte = Pte * ste + mte
    elapsed = time.perf_counter() - t0
    return fold_windows(Pva, va_s, val_in), fold_windows(Pte, te_s, test_in), elapsed, fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="THD,EWZ,EWK,EWG,EWT,EDEN")
    ap.add_argument("--folds", default="A")
    ap.add_argument("--blocks", default="1,5,20")
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--methods", default="saits")
    ap.add_argument("--variants", default="V0,V1,V2")
    ap.add_argument("--out", default="pilot_out/pilot_rows.csv")
    a = ap.parse_args()
    variants = {"V0": dict(epochs=30, patience=5, per_window=False),
                "V1": dict(epochs=30, patience=5, per_window=True),
                "V2": dict(epochs=200, patience=20, per_window=True)}
    Path(a.out).parent.mkdir(exist_ok=True)
    rows = []
    with zipfile.ZipFile("private/snapshot.zip") as z:
        manifest_sha = sha256(z.read("asset_universe_v1.json"))
        receipt = json.loads(z.read("acquisition_qc_receipt.json"))
        recs = {r["ticker"]: r for r in receipt["assets"]}
        for ticker in a.tickers.split(","):
            frame = load_verified_asset(z, recs[ticker])
            for fold in a.folds.split(","):
                parts = split_frame(frame, fold)
                train, val, test = (parts[k].to_numpy(float) for k in ("train", "val", "test"))
                for block in map(int, a.blocks.split(",")):
                    rate, rep = a.rate, a.seed
                    seeds = {p: stable_mask_seed(manifest_sha, ticker, fold, p, block, rate, rep) for p in ("train", "val", "test")}
                    trh = strict_block_mask(train.shape, rate, block, seeds["train"], EDGE_GUARD)
                    vah = strict_block_mask(val.shape, rate, block, seeds["val"], EDGE_GUARD)
                    teh = strict_block_mask(test.shape, rate, block, seeds["test"], EDGE_GUARD)
                    mean, scale = train_observed_scaler(train, trh)
                    train_in = corrupt((train - mean) / scale, trh)
                    val_truth = (val - mean) / scale
                    val_in = corrupt(val_truth, vah)
                    test_in = corrupt((test - mean) / scale, teh)
                    common = {"ticker": ticker, "fold": fold, "block_length": block, "rate": rate, "seed": rep}
                    for method in a.methods.split(","):
                        mseed = training_seed(manifest_sha, ticker, fold, block, rate, rep, method)
                        for vname in a.variants.split(","):
                            cfg = variants[vname]
                            v_est, t_est, el, fb = fit_predict(method, train_in, val_in, val_truth, test_in, mseed, **cfg)
                            for part, truth, est, hid in (("validation", val, v_est, vah), ("test", test, t_est, teh)):
                                rows.extend(score_features(truth, est, hid, mean, scale,
                                                           {**common, "method": method, "variant": vname, "partition": part,
                                                            "window_fallbacks": fb}, el, 0))
                            r = [x for x in rows if x["variant"] == vname and x["method"] == method and x["partition"] == "test"
                                 and x["ticker"] == ticker and x["fold"] == fold and x["block_length"] == block]
                            nrmse = np.mean([x["rmse_trainstd"] for x in r])
                            print(f"{ticker}/{fold} b{block} {method} {vname}: test NRMSE {nrmse:.3f} ({el:.0f}s, fallbacks {fb})", flush=True)
                            pd.DataFrame(rows).to_csv(a.out, index=False)
    print("done", a.out)


if __name__ == "__main__":
    main()
