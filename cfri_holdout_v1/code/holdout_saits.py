"""Resumable SAITS V2 comparators for the frozen held-out-asset CFRI study.

Each target is trained separately with the original normalized SAITS V2 recipe.
For whole-row gaps the second comparator receives the same fixed peer pool and
the same frozen bridge as CFRI, with its gain chosen on validation labels only.
No target test label is passed to either model or bridge before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C
import confirm_saits as S
import holdout_data as D
import v2_peer as P
from v2_price import joint_ohlc_project


HERE = Path(__file__).resolve().parent
OUT_DIR = "holdout_saits"
METHODS_BY_MASK = {
    "feature": ("saits_joint",),
    "row": ("saits_joint", "saits_peer_joint"),
}
METHODS_FEATURE = METHODS_BY_MASK["feature"]
METHODS_ROW = METHODS_BY_MASK["row"]
CODE_FILES = ("holdout_saits.py", "holdout_data.py", "cfri_core.py",
              "confirm_saits.py", "v2_peer.py", "v2_price.py",
              "upstream/cfri_core.py")
PEER_ALPHA = 10.0
PEER_MIN_PEERS = 5
PEER_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _code_sha256() -> dict[str, str]:
    return {name: _sha256((HERE / name).read_bytes()) for name in CODE_FILES}


def _frame_sha256(frame: pd.DataFrame) -> str:
    values = frame.loc[:, list(C.FEATURES)].to_numpy(dtype="<f8", copy=True)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite OHLCV frame")
    digest = hashlib.sha256()
    digest.update(np.asarray(frame.index.asi8, dtype="<i8").tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _environment(device: str) -> dict:
    import torch

    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "torch": torch.__version__,
            "pypots": importlib.metadata.version("pypots"), "device": device}


def _manifest(root: Path, targets: dict[str, pd.DataFrame],
              peers: dict[str, pd.DataFrame], environment: dict) -> dict:
    """Bind a restart to the same code, data, settings and package versions."""
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = root / D.RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError("missing data receipt before SAITS run")
    stable = {
        "study": D.STUDY_ID,
        "fold": D.FOLD_TAG,
        "test_end": D.TEST_END,
        "target_tickers": list(targets),
        "peer_tickers": list(peers),
        "methods_by_mask": {mask: list(names) for mask, names in METHODS_BY_MASK.items()},
        "code_sha256": _code_sha256(),
        "data_receipt_sha256": _sha256(receipt_path.read_bytes()),
        "target_frame_sha256": {ticker: _frame_sha256(frame) for ticker, frame in sorted(targets.items())},
        "peer_frame_sha256": {ticker: _frame_sha256(frame) for ticker, frame in sorted(peers.items())},
        "environment": environment,
        "config": {"saits": S.SAITS_CONFIG, "variant": S.VARIANTS["V2"],
                   "windows": [S.N_STEPS, S.STRIDE, S.BATCH_SIZE],
                   "blocks": list(C.BLOCKS), "rates": list(C.RATES), "seeds": list(C.SEEDS),
                   "projection": "joint_ohlc_project(weights=None)",
                   "peer_alpha": PEER_ALPHA, "peer_min_peers": PEER_MIN_PEERS,
                   "peer_gains": list(PEER_GAINS)},
    }
    path = out / "run_manifest.json"
    if path.exists():
        old = _read_json(path)
        if old is None or any(old.get(key) != value for key, value in stable.items()):
            raise RuntimeError("SAITS manifest differs from code, data, settings or runtime; use a new root")
        return old
    manifest = {**stable, "created_utc": datetime.now(timezone.utc).isoformat()}
    _atomic(path, _json_bytes(manifest))
    return manifest


def condition_path(root: Path, ticker: str, mask_type: str, block: int,
                   rate: float, seed: int) -> Path:
    if mask_type not in METHODS_BY_MASK:
        raise ValueError(mask_type)
    return Path(root) / OUT_DIR / f"{ticker}_{mask_type}_b{block}_r{rate:.2f}_s{seed}.json"


def condition_fingerprint(manifest: dict, ticker: str, mask_type: str, block: int,
                          rate: float, seed: int) -> str:
    identity = {"code": manifest["code_sha256"],
                "data_receipt": manifest["data_receipt_sha256"],
                "target_frame": manifest["target_frame_sha256"][ticker],
                "peer_frames": manifest["peer_frame_sha256"],
                "environment": manifest["environment"],
                "config": manifest["config"], "ticker": ticker,
                "mask_type": mask_type, "block": block, "rate": rate, "seed": seed}
    return _sha256(json.dumps(identity, sort_keys=True).encode())


def verified_condition(path: Path, fingerprint: str, mask_type: str) -> bool:
    side = path.with_name(path.name + ".done.json")
    if not path.is_file() or not side.is_file():
        return False
    meta, result = _read_json(side), _read_json(path)
    return bool(meta and result and meta.get("fingerprint") == fingerprint
                and meta.get("sha256") == _sha256(path.read_bytes())
                and result.get("fold") == D.FOLD_TAG
                and result.get("mask_type") == mask_type
                and set(result.get("scores", {})) == set(METHODS_BY_MASK[mask_type])
                and result.get("mask_sha256") == meta.get("mask_sha256"))


def training_seed(ticker: str, block: int, rate: float, seed: int, mask_type: str) -> int:
    key = (f"{C.ORIGINAL_MANIFEST_SHA}|{D.STUDY_ID}|{ticker}|{D.FOLD_TAG}|{block}|"
           f"{rate:.2f}|{seed}|saits|{mask_type}|heldout-v1")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def _fit_predict_both(train_in: np.ndarray, val_in: np.ndarray, val_truth: np.ndarray,
                      test_in: np.ndarray, seed: int, device: str) -> tuple[np.ndarray, np.ndarray, float, int]:
    """The archived V2 training recipe, with validation and test inference."""
    import warnings
    from pypots.imputation import SAITS

    S.set_training_seed(seed)
    tr_starts = S.window_starts(len(train_in))
    va_starts = S.window_starts(len(val_in))
    te_starts = S.window_starts(len(test_in))
    xtr = S.as_windows(train_in, tr_starts)
    xva = S.as_windows(val_in, va_starts)
    xva_truth = S.as_windows(val_truth, va_starts)
    xte = S.as_windows(test_in, te_starts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mtr, str_, ftr = S.window_stats(xtr)
        mva, sva, fva = S.window_stats(xva)
        mte, ste, fte = S.window_stats(xte)
    xtr = (xtr - mtr) / str_
    xva_truth = (xva_truth - mva) / sva
    xva = (xva - mva) / sva
    xte = (xte - mte) / ste
    config = S.VARIANTS["V2"]
    model = SAITS(**S.SAITS_CONFIG, n_steps=S.N_STEPS, n_features=len(C.FEATURES),
                  batch_size=S.BATCH_SIZE, epochs=config["epochs"],
                  patience=min(config["patience"], config["epochs"]), device=device,
                  model_saving_strategy=None, verbose=False)
    t0 = time.perf_counter()
    model.fit({"X": xtr}, {"X": xva, "X_ori": xva_truth})
    pva = np.asarray(model.predict({"X": xva})["imputation"]) * sva + mva
    pte = np.asarray(model.predict({"X": xte})["imputation"]) * ste + mte
    elapsed = time.perf_counter() - t0
    del model
    return (S.fold_windows(pva, va_starts, val_in),
            S.fold_windows(pte, te_starts, test_in),
            float(elapsed), int(ftr + fva + fte))


def _one_condition(frame: pd.DataFrame, peers: dict[str, pd.DataFrame], ticker: str,
                   mask_type: str, block: int, rate: float, seed: int, device: str) -> dict:
    parts = D.split_frame(frame)
    masks = {name: C.condition_mask(values.shape, ticker, D.FOLD_TAG, name,
                                    block, rate, seed, mask_type)
             for name, values in parts.items()}
    train, val, test = (parts[name] for name in ("train", "val", "test"))
    tr_mask, va_mask, te_mask = (masks[name] for name in ("train", "val", "test"))
    mean, scale = C.observed_scaler(train, tr_mask)
    train_in = C.corrupt((train - mean) / scale, tr_mask)
    val_truth_scaled = (val - mean) / scale
    val_in = C.corrupt(val_truth_scaled, va_mask)
    test_in = C.corrupt((test - mean) / scale, te_mask)
    model_seed = training_seed(ticker, block, rate, seed, mask_type)
    val_pred, test_pred, seconds, fallbacks = _fit_predict_both(
        train_in, val_in, val_truth_scaled, test_in, model_seed, device)
    val_raw = np.asarray(val_pred, dtype=float) * scale + mean
    test_raw = np.asarray(test_pred, dtype=float) * scale + mean
    val_raw[~va_mask] = val[~va_mask]
    test_raw[~te_mask] = test[~te_mask]
    # Project first so the peer bridge always receives positive, coherent
    # prices, including when SAITS emitted a nonpositive imputed price.
    val_joint = joint_ohlc_project(val_raw, va_mask)
    test_joint = joint_ohlc_project(test_raw, te_mask)
    estimates = {"saits_joint": test_joint}
    peer_meta: dict = {}
    if mask_type == "row":
        dates = {part: D.partition_dates(frame, part) for part in parts}
        if any(len(dates[part]) != len(parts[part]) for part in parts):
            raise AssertionError("target dates and OHLCV partition rows differ")
        model = P.fit_peer_model(train, tr_mask, dates["train"], peers, ticker,
                                 alpha=PEER_ALPHA, min_peers=PEER_MIN_PEERS)
        qval, nval = P.predict_peer_returns(model, dates["val"], peers)
        gain, validation_scores = P.select_peer_gain(
            val_joint, va_mask, val, scale[:4], qval, nval,
            min_peers=model.min_peers if model is not None else PEER_MIN_PEERS,
            gains=PEER_GAINS, project_fn=joint_ohlc_project)
        peer_est, usage = P.peer_adjust(test_joint, dates["test"], te_mask, peers,
                                         model, gain=gain, market_wide=False,
                                         return_meta=True)
        estimates["saits_peer_joint"] = joint_ohlc_project(peer_est, te_mask)
        peer_meta = {"model_fit_rows": int(model.n_fit) if model is not None else 0,
                     "gain": float(gain),
                     "validation_price_nrmse_by_gain": {str(k): float(v) for k, v in validation_scores.items()},
                     "target_only": usage}
    scores = {name: C.score_method(test, est, te_mask, scale)
              for name, est in estimates.items()}
    violations = {name: C.ohlc_violations(est, te_mask)[0]
                  for name, est in estimates.items()}
    if any(value > 1e-9 for value in violations.values()):
        raise AssertionError("projected SAITS estimate violates OHLC constraints")
    return {"ticker": ticker, "fold": D.FOLD_TAG, "mask_type": mask_type,
            "block_length": block, "rate": rate, "seed": seed,
            "model_seed": model_seed,
            "mask_sha256": {part: _sha256(mask.tobytes()) for part, mask in masks.items()},
            "masked_cells": {part: int(mask.sum()) for part, mask in masks.items()},
            "scores": scores, "ohlc_violation_share": violations,
            "peer": peer_meta, "seconds": seconds, "window_fallbacks": fallbacks,
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def _asset_mask_job(root: Path, ticker: str, frame: pd.DataFrame,
                    peers: dict[str, pd.DataFrame], mask_type: str,
                    manifest: dict, device: str, threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    parts = D.split_frame(frame)
    lengths = {part: len(values) for part, values in parts.items()}
    done = {"ticker": ticker, "mask_type": mask_type, "new": 0, "skipped": 0}
    for block in C.BLOCKS:
        for rate in C.RATES:
            if not C.feasible(lengths, block, rate, mask_type):
                continue
            for seed in C.SEEDS:
                path = condition_path(root, ticker, mask_type, block, rate, seed)
                fingerprint = condition_fingerprint(manifest, ticker, mask_type, block, rate, seed)
                if verified_condition(path, fingerprint, mask_type):
                    done["skipped"] += 1
                    continue
                result = _one_condition(frame, peers, ticker, mask_type, block, rate, seed, device)
                blob = _json_bytes(result)
                _atomic(path, blob)
                _atomic(path.with_name(path.name + ".done.json"),
                        _json_bytes({"fingerprint": fingerprint, "sha256": _sha256(blob),
                                     "mask_sha256": result["mask_sha256"]}))
                done["new"] += 1
    return done


def run_saits(root: Path, targets: dict[str, pd.DataFrame],
              peers: dict[str, pd.DataFrame], mask_types=("feature", "row"),
              max_new_conditions: int | None = None, threads: int = 2,
              device: str | None = None, workers: int | None = None) -> dict:
    """Run all feasible conditions, reusing only verified per-condition files."""
    import torch

    root = Path(root)
    if not targets or not peers:
        raise ValueError("eligible target and peer frames are required")
    if max_new_conditions is not None and max_new_conditions < 1:
        raise ValueError("max_new_conditions must be positive")
    if not mask_types or any(mask not in METHODS_BY_MASK for mask in mask_types):
        raise ValueError("invalid mask types")
    if threads < 1:
        raise ValueError("threads must be positive")
    if set(targets) & set(peers):
        raise ValueError("a target cannot be a member of its peer pool")
    torch.set_num_threads(threads)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if workers is None:
        workers = (4 if torch.cuda.get_device_properties(0).total_memory >= 32 * 1024**3 else 1) if device == "cuda" else 1
    if workers < 1:
        raise ValueError("workers must be positive")
    try:
        from pypots.utils.logging import logger as pypots_logger
        pypots_logger.addFilter(lambda record: "saving_path not given" not in record.getMessage())
    except Exception:
        pass
    manifest = _manifest(root, targets, peers, _environment(device))
    expected = 0
    for mask_type in mask_types:
        for frame in targets.values():
            lengths = {part: len(values) for part, values in D.split_frame(frame).items()}
            expected += sum(len(C.SEEDS) for block in C.BLOCKS for rate in C.RATES
                            if C.feasible(lengths, block, rate, mask_type))
    summary = {"root": str(root), "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "workers": workers, "total": expected, "new": 0,
               "skipped": 0, "stopped_early": False}
    if workers > 1 and max_new_conditions is None:
        jobs = [(root, ticker, frame, peers, mask_type, manifest, device, threads)
                for mask_type in mask_types for ticker, frame in targets.items()]
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(_asset_mask_job, *job) for job in jobs]
            for future in as_completed(pending):
                done = future.result()
                summary["new"] += done["new"]
                summary["skipped"] += done["skipped"]
                print(f"[SAITS H1] {done['ticker']} {done['mask_type']}: "
                      f"{done['new']} new, {done['skipped']} resumed; "
                      f"overall {summary['new'] + summary['skipped']}/{expected}", flush=True)
        return summary
    for mask_type in mask_types:
        for ticker, frame in targets.items():
            lengths = {part: len(values) for part, values in D.split_frame(frame).items()}
            for block in C.BLOCKS:
                for rate in C.RATES:
                    if not C.feasible(lengths, block, rate, mask_type):
                        continue
                    for seed in C.SEEDS:
                        path = condition_path(root, ticker, mask_type, block, rate, seed)
                        fingerprint = condition_fingerprint(manifest, ticker, mask_type, block, rate, seed)
                        if verified_condition(path, fingerprint, mask_type):
                            summary["skipped"] += 1
                            continue
                        if max_new_conditions is not None and summary["new"] >= max_new_conditions:
                            summary["stopped_early"] = True
                            return summary
                        result = _one_condition(frame, peers, ticker, mask_type,
                                                block, rate, seed, device)
                        blob = _json_bytes(result)
                        _atomic(path, blob)
                        _atomic(path.with_name(path.name + ".done.json"),
                                _json_bytes({"fingerprint": fingerprint, "sha256": _sha256(blob),
                                             "mask_sha256": result["mask_sha256"]}))
                        summary["new"] += 1
                        print(f"[SAITS H1 {mask_type}] {ticker} b{block} r{rate:.2f} s{seed}: "
                              f"saved ({summary['new']} new; {summary['skipped']} resumed)", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-new-conditions", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    receipt, targets, peers = D.prepare_data(args.root)
    print(f"Data: {len(targets)} eligible targets; {len(peers)} eligible peers", flush=True)
    print("SAITS:", run_saits(args.root, targets, peers,
                              max_new_conditions=args.max_new_conditions,
                              threads=args.threads, device=args.device,
                              workers=args.workers), flush=True)


if __name__ == "__main__":
    main()
