#!/usr/bin/env python3
"""EXPLORATORY neural rerun (protocol v2 amendment): per-window instance normalization.

Identical to the frozen v1 run (same private snapshot, folds, masks, training-period
scaling, windows, architecture, seeds, epoch budget for variant V1) with ONE change:
every 64-day window is standardized by the mean/std of its own observed cells before
the network sees it, and predictions are de-standardized with the same statistics
(RevIN-style instance normalization). Variant V2 additionally raises the epoch budget
(200, patience 20). Output goes to a separate directory and is labeled exploratory.
Resumable: one CSV + sidecar per asset/fold/variant/method.
"""
from __future__ import annotations
import argparse, json, os, sys, time, zipfile, hashlib, platform
from pathlib import Path
import numpy as np, pandas as pd, torch
CODE_DIR = os.environ.get("ITRI_CODE_DIR", "results/code_and_protocol")
SNAPSHOT = os.environ.get("ITRI_SNAPSHOT", "private/snapshot.zip")
sys.path.insert(0, CODE_DIR)
from masking import run_lengths, strict_block_mask  # noqa
from run_classical_benchmark import (BLOCKS, EDGE_GUARD, FEATURES, FOLDS, RATES, SEEDS, corrupt,  # noqa
                                     load_verified_asset, score_features, sha256, split_frame,
                                     stable_mask_seed, train_observed_scaler)
from run_neural_benchmark import (N_STEPS, STRIDE, BATCH_SIZE, BRITS_HIDDEN, SAITS_CONFIG,  # noqa
                                  window_starts, as_windows, fold_windows, set_training_seed,
                                  training_seed)
from pypots.imputation import BRITS, SAITS  # noqa
import logging
from pypots.utils.logging import logger as pypots_logger
pypots_logger.setLevel(logging.ERROR)

VARIANTS = {"V1": dict(epochs=30, patience=5), "V2": dict(epochs=200, patience=20)}
PROTOCOL_ID = "ITRI-neural-protocol-v2-exploratory-2026-09-29|winnorm-observed-cells|fallback<2obs-or-sd<1e-6"


def window_stats(win):
    mu = np.nanmean(win, axis=1, keepdims=True); sd = np.nanstd(win, axis=1, keepdims=True)
    n_obs = np.sum(np.isfinite(win), axis=1, keepdims=True)
    bad = (n_obs < 2) | ~np.isfinite(sd) | (sd < 1e-6)
    return np.where(bad, 0.0, mu).astype(np.float32), np.where(bad, 1.0, sd).astype(np.float32), int(bad.sum())


def fit_predict(method, train_in, val_in, val_truth, test_in, seed, epochs, patience, device):
    set_training_seed(seed)
    tr_s, va_s, te_s = window_starts(len(train_in)), window_starts(len(val_in)), window_starts(len(test_in))
    Xtr, Xva, Xva_ori, Xte = (as_windows(a, s) for a, s in ((train_in, tr_s), (val_in, va_s), (val_truth, va_s), (test_in, te_s)))
    mtr, s_tr, f1 = window_stats(Xtr); mva, s_va, f2 = window_stats(Xva); mte, s_te, f3 = window_stats(Xte)
    Xtr = (Xtr - mtr) / s_tr; Xva_ori = (Xva_ori - mva) / s_va; Xva = (Xva - mva) / s_va; Xte = (Xte - mte) / s_te
    common = dict(n_steps=N_STEPS, n_features=len(FEATURES), batch_size=BATCH_SIZE, epochs=epochs,
                  patience=min(patience, epochs), device=device, model_saving_strategy=None, verbose=False)
    model = BRITS(rnn_hidden_size=BRITS_HIDDEN, **common) if method == "brits" else SAITS(**SAITS_CONFIG, **common)
    t0 = time.perf_counter()
    model.fit({"X": Xtr}, {"X": Xva, "X_ori": Xva_ori})
    Pva = np.asarray(model.predict({"X": Xva})["imputation"]) * s_va + mva
    Pte = np.asarray(model.predict({"X": Xte})["imputation"]) * s_te + mte
    el = time.perf_counter() - t0
    del model
    return fold_windows(Pva, va_s, val_in), fold_windows(Pte, te_s, test_in), el, f1 + f2 + f3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["saits", "brits"])
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--out", default="neural_windownorm")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--folds", default="A,B,C")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = VARIANTS[a.variant]
    out = Path(a.out); shards = out / "shards"; shards.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in a.seeds.split(",")); folds = tuple(a.folds.split(","))
    with zipfile.ZipFile(SNAPSHOT) as z:
        manifest_blob = z.read("asset_universe_v1.json"); manifest_sha = sha256(manifest_blob)
        receipt = json.loads(z.read("acquisition_qc_receipt.json"))
        records = receipt["assets"]
        # Resume key covers the amendment protocol only (not the runner bytes or hardware).
        fingerprint = sha256(PROTOCOL_ID.encode() + manifest_blob + json.dumps(cfg, sort_keys=True).encode()
                             + a.method.encode() + a.variant.encode())
        meta = {"label": "EXPLORATORY protocol v2 amendment: per-window instance normalization",
                "method": a.method, "variant": a.variant, "config": cfg, "fingerprint": fingerprint,
                "manifest_sha256": manifest_sha, "private_zip_sha256": sha256(Path(SNAPSHOT).read_bytes()),
                "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "torch": torch.__version__, "pypots": __import__("pypots").__version__},
                "window_length": N_STEPS, "window_stride": STRIDE, "batch_size": BATCH_SIZE}
        (out / f"run_metadata_{a.method}_{a.variant}.json").write_text(json.dumps(meta, indent=2))
        jobs = [(r, f) for r in records for f in folds]
        jobs = jobs[a.worker_id::a.workers]
        conditions = [(b, r, s) for b in BLOCKS for r in RATES for s in seeds]
        def piece_paths(rec, fold):
            stem = f"{rec['ticker']}_{fold}_{a.method}_{a.variant}"
            return stem, shards / f"{stem}.csv", shards / f"{stem}.meta.json"

        def is_complete(csv_path, meta_path):
            """A piece counts as done only if its sidecar parses and the CSV bytes match the recorded hash."""
            if not (csv_path.exists() and meta_path.exists()):
                return False
            try:
                side = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                return False          # truncated sidecar (e.g. abrupt disconnect): recompute this piece
            return side.get("fingerprint") == fingerprint and side.get("csv_sha256") == sha256(csv_path.read_bytes())

        already = sum(is_complete(*piece_paths(r, f)[1:]) for r, f in jobs)
        print(f"[worker {a.worker_id}] {a.method} {a.variant}: {already}/{len(jobs)} pieces already complete and verified; "
              f"{len(jobs) - already} to compute ({len(conditions)} conditions per piece)", flush=True)
        t_start = time.time(); done = 0
        for rec, fold in jobs:
            ticker = rec["ticker"]
            stem, csv_path, meta_path = piece_paths(rec, fold)
            if is_complete(csv_path, meta_path):
                continue
            frame = load_verified_asset(z, rec)
            parts = split_frame(frame, fold)
            train, val, test = (parts[k].to_numpy(float) for k in ("train", "val", "test"))
            rows = []
            for block, rate, rep in conditions:
                sd = {p: stable_mask_seed(manifest_sha, ticker, fold, p, block, rate, rep) for p in ("train", "val", "test")}
                trh = strict_block_mask(train.shape, rate, block, sd["train"], EDGE_GUARD)
                vah = strict_block_mask(val.shape, rate, block, sd["val"], EDGE_GUARD)
                teh = strict_block_mask(test.shape, rate, block, sd["test"], EDGE_GUARD)
                mean, scale = train_observed_scaler(train, trh)
                train_in = corrupt((train - mean) / scale, trh)
                val_truth = (val - mean) / scale; val_in = corrupt(val_truth, vah)
                test_in = corrupt((test - mean) / scale, teh)
                if int(run_lengths(teh).max()) > block:
                    raise AssertionError("mask invariant")
                mseed = training_seed(manifest_sha, ticker, fold, block, rate, rep, a.method)
                v_est, t_est, el, fb = fit_predict(a.method, train_in, val_in, val_truth, test_in, mseed, device=device, **cfg)
                common = {"ticker": ticker, "country": rec["country"], "region": rec["region"], "fold": fold,
                          "block_length": block, "rate": rate, "seed": rep, "train_mask_seed": sd["train"],
                          "val_mask_seed": sd["val"], "test_mask_seed": sd["test"], "model_seed": mseed,
                          "method": f"{a.method}_winnorm_{a.variant}", "variant": a.variant,
                          "epochs_max": cfg["epochs"], "patience": cfg["patience"], "window_fallbacks": fb}
                rows.extend(score_features(test, t_est, teh, mean, scale, {**common, "partition": "test"}, el, 0))
                rows.extend(score_features(val, v_est, vah, mean, scale, {**common, "partition": "validation"}, el, 0))
            data = pd.DataFrame(rows).to_csv(index=False, float_format="%.17g").encode()
            tmp = csv_path.with_suffix(".csv.tmp"); tmp.write_bytes(data); os.replace(tmp, csv_path)
            mtmp = meta_path.with_name(meta_path.name + ".tmp")
            mtmp.write_text(json.dumps({"fingerprint": fingerprint, "csv_sha256": sha256(data), "rows": len(rows)}, indent=2))
            os.replace(mtmp, meta_path)
            done += 1
            el_total = time.time() - t_start
            total_done = already + done
            left = len(jobs) - total_done
            print(f"[worker {a.worker_id}] {stem} saved | total {total_done}/{len(jobs)} complete "
                  f"({done} this session, {el_total/60:.1f} min) | ~{el_total/done*left/60:.0f} min left for the remaining {left}",
                  flush=True)
    print("worker finished", a.worker_id)


if __name__ == "__main__":
    main()
