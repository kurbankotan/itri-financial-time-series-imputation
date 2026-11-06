"""Confirmatory study, step 3: normalized SAITS on fold D (GPU), protocol amendment v2 settings.

Same windows, architecture, budgets and per-window instance normalization as the archived amendment v2 run
(external_validation/code/run_neural_windownorm.py): 64-day windows, stride 16, batch 32; SAITS with 2 layers,
d_model 64, 4 heads, d_k = d_v = 16, d_ffn 128, dropout 0.1; V1 = 30 epochs (patience 5), V2 = 200 epochs
(patience 20); each window standardized by the mean and SD of its own observed cells, predictions
de-standardized with the same statistics, observed cells restored. Masks and scaling come from cfri_core, so
SAITS sees exactly the conditions of the CPU methods. Each condition is scored raw (saits_winnorm_V*) and after
the same OHLC projection as the CPU methods (saits_winnorm_V*_proj).

One CSV shard per (ticker, mask type, variant) with a SHA-256 sidecar; finished shards are skipped on restart.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C

FOLD = "D"
OUT = "fold_D_saits"
N_STEPS, STRIDE, BATCH_SIZE = 64, 16, 32
SAITS_CONFIG = {"n_layers": 2, "d_model": 64, "n_heads": 4, "d_k": 16, "d_v": 16, "d_ffn": 128, "dropout": 0.1}
VARIANTS = {"V1": dict(epochs=30, patience=5), "V2": dict(epochs=200, patience=20)}


def window_starts(length, n_steps=N_STEPS, stride=STRIDE):
    if length < n_steps:
        raise ValueError("partition shorter than the neural window")
    starts = list(range(0, length - n_steps + 1, stride))
    if starts[-1] != length - n_steps:
        starts.append(length - n_steps)
    return starts


def as_windows(values, starts):
    return np.stack([values[s:s + N_STEPS] for s in starts]).astype(np.float32, copy=False)


def fold_windows(predictions, starts, observed):
    sums = np.zeros(observed.shape, dtype=np.float64)
    counts = np.zeros(observed.shape[0], dtype=np.int32)
    for p, s in zip(predictions, starts):
        sums[s:s + N_STEPS] += p
        counts[s:s + N_STEPS] += 1
    if np.any(counts == 0):
        raise AssertionError("window coverage incomplete")
    out = sums / counts[:, None]
    seen = np.isfinite(observed)
    out[seen] = observed[seen]
    if not np.isfinite(out).all():
        raise ValueError("non-finite neural estimates")
    return out


def window_stats(win):
    mu = np.nanmean(win, axis=1, keepdims=True)
    sd = np.nanstd(win, axis=1, keepdims=True)
    n_obs = np.sum(np.isfinite(win), axis=1, keepdims=True)
    bad = (n_obs < 2) | ~np.isfinite(sd) | (sd < 1e-6)
    return np.where(bad, 0.0, mu).astype(np.float32), np.where(bad, 1.0, sd).astype(np.float32), int(bad.sum())


def set_training_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_seed(ticker, block, rate, rep, mask_type):
    key = f"{C.ORIGINAL_MANIFEST_SHA}|{ticker}|{FOLD}|{block}|{rate:.2f}|{rep}|saits|{mask_type}|confirmatory-v1"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def fit_predict(train_in, val_in, val_truth, test_in, seed, epochs, patience, device):
    import warnings
    from pypots.imputation import SAITS
    set_training_seed(seed)
    tr_s, va_s, te_s = window_starts(len(train_in)), window_starts(len(val_in)), window_starts(len(test_in))
    Xtr, Xva, Xva_ori, Xte = (as_windows(a, s) for a, s in ((train_in, tr_s), (val_in, va_s), (val_truth, va_s), (test_in, te_s)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-missing window rows are handled by the fallback
        mtr, s_tr, f1 = window_stats(Xtr)
        mva, s_va, f2 = window_stats(Xva)
        mte, s_te, f3 = window_stats(Xte)
    Xtr = (Xtr - mtr) / s_tr
    Xva_ori = (Xva_ori - mva) / s_va
    Xva = (Xva - mva) / s_va
    Xte = (Xte - mte) / s_te
    model = SAITS(**SAITS_CONFIG, n_steps=N_STEPS, n_features=len(C.FEATURES), batch_size=BATCH_SIZE, epochs=epochs,
                  patience=min(patience, epochs), device=device, model_saving_strategy=None, verbose=False)
    t0 = time.perf_counter()
    model.fit({"X": Xtr}, {"X": Xva, "X_ori": Xva_ori})
    Pte = np.asarray(model.predict({"X": Xte})["imputation"]) * s_te + mte
    elapsed = time.perf_counter() - t0
    del model
    return fold_windows(Pte, te_s, test_in), elapsed, f1 + f2 + f3


def _environment(device):
    import torch
    import pypots
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "pypots": pypots.__version__, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cudnn": torch.backends.cudnn.version() if device == "cuda" else None}


MAX_CONSECUTIVE_FAILURES = 3


def code_digest() -> str:
    """SHA-256 over the code that determines SAITS results (this file and cfri_core.py)."""
    h = hashlib.sha256()
    for path in (Path(C.__file__), Path(__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()


def job_fingerprint(variant: str, mask_type: str) -> str:
    """Identifies the code, settings and job of a SAITS shard; finished shards are reused only when it matches."""
    return hashlib.sha256(json.dumps({"code": C.CODE_ID, "code_sha256": code_digest(), "variant": variant,
                                      "cfg": VARIANTS[variant], "mask": mask_type, "saits": SAITS_CONFIG,
                                      "win": [N_STEPS, STRIDE, BATCH_SIZE]}, sort_keys=True).encode()).hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_device_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(k in text for k in ("cuda", "cudnn", "cublas", "device-side", "accelerator", "out of memory"))


def run_saits(root: Path, frames: dict, variant: str, mask_type: str, smoke_conditions: int | None = None,
              threads: int = 2):
    """Train and score SAITS for every feasible condition. A condition that fails for a model-level reason is
    recorded and skipped; device errors and three consecutive failures stop the run (resume later). A shard
    without any completed condition is never marked as finished."""
    import logging
    import torch
    torch.set_num_threads(threads)
    try:   # silence the per-model "saving_path not given" notice, as in the archived neural runs
        from pypots.utils.logging import logger as pypots_logger
        pypots_logger.addFilter(lambda record: "saving_path not given" not in record.getMessage())
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = VARIANTS[variant]
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = job_fingerprint(variant, mask_type)
    if not smoke_conditions:
        meta_path = out / f"run_metadata_{variant}_{mask_type}.json"
        history = (_read_json(meta_path) or {}).get("sessions", [])
        history.append({"started_utc": pd.Timestamp.now(tz="UTC").isoformat(), **_environment(device)})
        _atomic(meta_path, json.dumps({"variant": variant, "mask_type": mask_type, "config": cfg, "saits": SAITS_CONFIG,
                                       "fingerprint": fingerprint, "sessions": history}, indent=2).encode())
    t_start, n_done, consecutive = time.time(), 0, 0
    for i, (ticker, frame) in enumerate(frames.items(), 1):
        stem = f"{ticker}_{mask_type}_saits_{variant}"
        csv_path, side = out / f"{stem}.csv", out / f"{stem}.done.json"
        if not smoke_conditions and csv_path.exists() and side.exists():
            info = _read_json(side) or {}
            if info.get("fingerprint") == fingerprint and info.get("sha256") == hashlib.sha256(csv_path.read_bytes()).hexdigest():
                continue
        parts = C.split_frame(frame, FOLD)
        rows_by_part = {k: len(v) for k, v in parts.items()}
        train, val, test = parts["train"], parts["val"], parts["test"]
        rows, errors = [], []
        for block in C.BLOCKS:
            for rate in C.RATES:
                if not C.feasible(rows_by_part, block, rate, mask_type):
                    continue
                for rep in C.SEEDS:
                    m = {n: C.condition_mask(v.shape, ticker, FOLD, n, block, rate, rep, mask_type) for n, v in parts.items()}
                    mean, scale = C.observed_scaler(train, m["train"])
                    train_in = C.corrupt((train - mean) / scale, m["train"])
                    val_truth = (val - mean) / scale
                    val_in = C.corrupt(val_truth, m["val"])
                    test_in = C.corrupt((test - mean) / scale, m["test"])
                    seed = training_seed(ticker, block, rate, rep, mask_type)
                    try:
                        est, elapsed, fallbacks = fit_predict(train_in, val_in, val_truth, test_in, seed, device=device, **cfg)
                    except Exception as exc:
                        if _is_device_error(exc):
                            raise
                        consecutive += 1
                        errors.append({"ticker": ticker, "variant": variant, "mask_type": mask_type, "block_length": block,
                                       "rate": rate, "seed": rep, "error": f"{type(exc).__name__}: {exc}"})
                        print(f"SAITS {variant} {mask_type} {ticker} b{block} r{rate} s{rep} failed: {errors[-1]['error']}",
                              flush=True)
                        if consecutive >= MAX_CONSECUTIVE_FAILURES:
                            raise RuntimeError(f"{consecutive} consecutive SAITS failures; stopping (resume later)") from exc
                        continue
                    consecutive = 0
                    est_raw = est * scale + mean
                    est_raw[~m["test"]] = test[~m["test"]]
                    for name, e in ((f"saits_winnorm_{variant}", est_raw),
                                    (f"saits_winnorm_{variant}_proj", C.ohlc_project(est_raw, m["test"]))):
                        scores = C.score_method(test, e, m["test"], scale)
                        viol, _ = C.ohlc_violations(e, m["test"])
                        for feat, (rmse, mae, n) in scores.items():
                            rows.append({"ticker": ticker, "fold": FOLD, "block_length": block, "rate": rate, "seed": rep,
                                         "mask_type": mask_type, "method": name, "feature": feat, "rmse": rmse,
                                         "mae": mae, "masked_cells": n, "model_seed": seed, "seconds": elapsed,
                                         "window_fallbacks": fallbacks, "ohlc_violation_share": viol})
                    n_done += 1
                    if smoke_conditions and n_done >= smoke_conditions:
                        print(f"SAITS {variant} {mask_type} smoke test passed: {n_done} condition(s), "
                              f"{elapsed:.1f} s for the last one on {device}", flush=True)
                        return pd.DataFrame(rows)
        if not rows:
            raise RuntimeError(f"SAITS {variant} {mask_type} {ticker}: no condition completed; shard not saved")
        data = pd.DataFrame(rows).to_csv(index=False, float_format="%.17g").encode()
        _atomic(csv_path, data)
        _atomic(side, json.dumps({"fingerprint": fingerprint, "sha256": hashlib.sha256(data).hexdigest(),
                                  "rows": len(rows), "errors": errors}).encode())
        el = time.time() - t_start
        print(f"[SAITS {variant} {mask_type} {i}/{len(frames)}] {ticker} saved, {len(errors)} errors | "
              f"{el / 60:.1f} min this session", flush=True)
    return None
