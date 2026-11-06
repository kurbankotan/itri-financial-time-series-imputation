"""Frozen CPU methods for the CFRI held-out-asset study.

The feature branch is original fixed CFRI with the v2 joint OHLC projection.
The whole-row branch is log interpolation with a target-only, validation-gated
peer bridge; a market-wide outage hides all peer closes on target-gap dates.
No interaction model is fitted.  Each condition is atomically checkpointed on
Drive with a SHA-256 sidecar and resumes only when code, settings and every
target/peer input frame still match its manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C
import holdout_data as D
import v2_peer as P
import v2_price as V

FOLD = D.FOLD_TAG
OUT_DIR = "holdout_cpu"
METHODS_FEATURE = ("cfri_old_proj", "cfri_joint")
METHODS_ROW = ("cfri_joint", "cfri_peer_target_joint", "cfri_peer_marketwide_joint")
CODE_FILES = ("holdout_run.py", "holdout_data.py", "cfri_core.py",
              "upstream/cfri_core.py", "v2_price.py", "v2_peer.py")
SETTINGS = {"price_alpha": 10.0, "base_gain": 0.75, "volume_eta": 0.50,
            "peer_alpha": 10.0, "peer_min_peers": 5,
            "peer_gain_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "blocks": list(C.BLOCKS), "rates": list(C.RATES), "seeds": list(C.SEEDS),
            "fold_tag": FOLD, "partitions": {key: list(span) for key, span in D.PARTITIONS.items()},
            "test_end": D.TEST_END,
            "targets": list(D.TARGETS), "peer_pool": list(D.PEERS)}
HERE = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def frame_hash(frame: pd.DataFrame) -> str:
    x = frame.loc[:, list(C.FEATURES)].to_numpy(dtype="<f8", copy=True)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite frame")
    h = hashlib.sha256()
    h.update(np.asarray(frame.index.asi8, dtype="<i8").tobytes())
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    """Public alias used by the analyzer to recheck the private frame cache."""
    return frame_hash(frame)


def code_hashes() -> dict[str, str]:
    return {name: sha((HERE / name).read_bytes()) for name in CODE_FILES}


def run_manifest(root: Path, targets: dict, peers: dict) -> dict:
    out = Path(root) / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(root) / D.RECEIPT
    if not receipt_path.is_file():
        raise RuntimeError("data receipt required before CPU methods")
    expected = {
        "study": D.STUDY_ID,
        "fold": FOLD,
        "classification": "held-out-asset evaluation; freeze required before data acquisition",
        "code_sha256": code_hashes(),
        "data_receipt_sha256": sha(receipt_path.read_bytes()),
        "target_frame_sha256": {ticker: frame_hash(frame) for ticker, frame in sorted(targets.items())},
        "peer_frame_sha256": {ticker: frame_hash(frame) for ticker, frame in sorted(peers.items())},
        "settings": SETTINGS,
        "methods": {"feature": list(METHODS_FEATURE), "row": list(METHODS_ROW)},
    }
    path = out / "run_manifest.json"
    prior = D.read_json(path) if path.exists() else None
    if prior is not None:
        if any(prior.get(key) != value for key, value in expected.items()):
            raise RuntimeError("CPU code, settings, receipt, target or peer frames changed; use a new Drive root")
        return prior
    if path.exists():
        raise RuntimeError("unreadable CPU manifest; inspect it before resuming")
    expected["created_utc"] = datetime.now(timezone.utc).isoformat()
    D.atomic(path, json_bytes(expected))
    return expected


def condition_path(root: Path, ticker: str, mask_type: str, block: int,
                   rate: float, seed: int) -> Path:
    if mask_type not in C.MASK_TYPES:
        raise ValueError(mask_type)
    return Path(root) / OUT_DIR / f"{ticker}_{mask_type}_b{block}_r{rate:.2f}_s{seed}.json"


def _fingerprint(manifest: dict, ticker: str, mask_type: str,
                 block: int, rate: float, seed: int) -> str:
    identity = {"code": manifest["code_sha256"], "receipt": manifest["data_receipt_sha256"],
                "target_frame": manifest["target_frame_sha256"][ticker],
                "peer_frames": manifest["peer_frame_sha256"], "settings": manifest["settings"],
                "ticker": ticker, "mask_type": mask_type, "block": block,
                "rate": rate, "seed": seed}
    return sha(json.dumps(identity, sort_keys=True).encode())


def verified_condition(path: Path, fingerprint: str,
                       methods: tuple[str, ...]) -> bool:
    side = path.with_name(path.name + ".done.json")
    if not path.is_file() or not side.is_file():
        return False
    meta, value = D.read_json(side), D.read_json(path)
    return bool(meta and value and meta.get("fingerprint") == fingerprint
                and meta.get("sha256") == sha(path.read_bytes())
                and set(value.get("scores", {})) == set(methods)
                and value.get("mask_sha256") == meta.get("mask_sha256"))


def estimate_from_prices(visible: np.ndarray, hidden: np.ndarray,
                         prices: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.empty_like(visible, dtype=float)
    out[:, :4] = prices
    out[:, 4] = volume
    out[~hidden] = visible[~hidden]
    if not np.isfinite(out).all() or (out[:, :4] <= 0).any():
        raise ValueError("nonfinite or nonpositive estimate")
    return out


def one_condition(frame: pd.DataFrame, peers: dict, ticker: str, mask_type: str,
                  block: int, rate: float, seed: int) -> dict:
    t0 = time.perf_counter()
    parts = D.split_frame(frame)
    masks = {part: C.condition_mask(values.shape, ticker, FOLD, part, block, rate, seed, mask_type)
             for part, values in parts.items()}
    train, val, test = (parts[part] for part in ("train", "val", "test"))
    tr_mask, va_mask, te_mask = (masks[part] for part in ("train", "val", "test"))
    _, scales = C.observed_scaler(train, tr_mask)
    val_visible, test_visible = C.corrupt(val, va_mask), C.corrupt(test, te_mask)
    val_volume = C.volume_log_candidate(val, va_mask, SETTINGS["volume_eta"])
    test_volume = C.volume_log_candidate(test, te_mask, SETTINGS["volume_eta"])
    base_val_log, val_designs = C.prepare_price_eval(val, va_mask)
    base_test_log, test_designs = C.prepare_price_eval(test, te_mask)
    if mask_type == "feature":
        training = C.prepare_price_training(train, tr_mask, ticker, FOLD, block, rate, seed)
        models = C.fit_price_models(training, SETTINGS["price_alpha"])
        val_prices = C.price_candidate(base_val_log, val_designs, models, SETTINGS["base_gain"])
        test_prices = C.price_candidate(base_test_log, test_designs, models, SETTINGS["base_gain"])
    elif mask_type == "row":
        # A complete row has zero same-day OHLC support: original CFRI reverts to log interpolation.
        val_prices, test_prices = np.exp(base_val_log), np.exp(base_test_log)
    else:
        raise ValueError(mask_type)
    base_val_est = estimate_from_prices(val_visible, va_mask, val_prices, val_volume)
    base_test_est = estimate_from_prices(test_visible, te_mask, test_prices, test_volume)
    estimates = {"cfri_joint": V.joint_ohlc_project(base_test_est, te_mask)}
    if mask_type == "feature":
        estimates["cfri_old_proj"] = C.ohlc_project(base_test_est, te_mask)
        peer_meta = None
    else:
        train_dates, val_dates, test_dates = (D.partition_dates(frame, part)
                                             for part in ("train", "val", "test"))
        if not (len(train_dates) == len(train) and len(val_dates) == len(val) and
                len(test_dates) == len(test)):
            raise AssertionError("target date and array alignment mismatch")
        model = P.fit_peer_model(train, tr_mask, train_dates, peers, ticker,
                                 alpha=SETTINGS["peer_alpha"], min_peers=SETTINGS["peer_min_peers"])
        val_returns, val_counts = P.predict_peer_returns(model, val_dates, peers)
        gain, score_map = P.select_peer_gain(base_val_est, va_mask, val, scales[:4],
                                             val_returns, val_counts,
                                             min_peers=model.min_peers if model is not None else SETTINGS["peer_min_peers"],
                                             project_fn=V.joint_ohlc_project)
        target_est, target_meta = P.peer_adjust(base_test_est, test_dates, te_mask, peers, model,
                                                 gain=gain, market_wide=False, return_meta=True)
        wide_est, wide_meta = P.peer_adjust(base_test_est, test_dates, te_mask, peers, model,
                                             gain=gain, market_wide=True, return_meta=True)
        estimates["cfri_peer_target_joint"] = V.joint_ohlc_project(target_est, te_mask)
        estimates["cfri_peer_marketwide_joint"] = V.joint_ohlc_project(wide_est, te_mask)
        peer_meta = {"model_fit_rows": int(model.n_fit) if model is not None else 0,
                     "gain": float(gain), "validation_price_nrmse_by_gain": score_map,
                     "target_only": target_meta, "market_wide": wide_meta}
    methods = METHODS_FEATURE if mask_type == "feature" else METHODS_ROW
    scores = {method: C.score_method(test, estimates[method], te_mask, scales) for method in methods}
    violations = {method: C.ohlc_violations(estimates[method], te_mask)[0] for method in methods}
    return {"ticker": ticker, "fold": FOLD, "mask_type": mask_type, "block_length": block,
            "rate": rate, "seed": seed,
            "mask_sha256": {part: sha(mask.tobytes()) for part, mask in masks.items()},
            "masked_cells": {part: int(mask.sum()) for part, mask in masks.items()},
            "scores": scores, "ohlc_violation_share": violations,
            "peer": peer_meta, "seconds": float(time.perf_counter() - t0),
            "completed_utc": datetime.now(timezone.utc).isoformat()}


def expected_conditions(targets: dict, mask_types=C.MASK_TYPES) -> int:
    total = 0
    for mask_type in mask_types:
        for frame in targets.values():
            lengths = {part: len(value) for part, value in D.split_frame(frame).items()}
            total += sum(len(C.SEEDS) for block in C.BLOCKS for rate in C.RATES
                         if C.feasible(lengths, block, rate, mask_type))
    return total


def run_cpu(root: Path, targets: dict, peers: dict, mask_types=C.MASK_TYPES,
            max_new_conditions: int | None = None) -> dict:
    if not targets or len(peers) < 5:
        raise ValueError("need eligible held-out targets and at least five peers")
    if not mask_types or any(mt not in C.MASK_TYPES for mt in mask_types):
        raise ValueError("invalid mask types")
    if max_new_conditions is not None and max_new_conditions < 1:
        raise ValueError("max_new_conditions must be positive")
    root = Path(root)
    manifest = run_manifest(root, targets, peers)
    total = expected_conditions(targets, mask_types)
    summary = {"total": total, "new": 0, "skipped": 0, "stopped_early": False}
    for mask_type in mask_types:
        methods = METHODS_FEATURE if mask_type == "feature" else METHODS_ROW
        for ticker, frame in targets.items():
            lengths = {part: len(value) for part, value in D.split_frame(frame).items()}
            for block in C.BLOCKS:
                for rate in C.RATES:
                    if not C.feasible(lengths, block, rate, mask_type):
                        continue
                    for seed in C.SEEDS:
                        path = condition_path(root, ticker, mask_type, block, rate, seed)
                        fingerprint = _fingerprint(manifest, ticker, mask_type, block, rate, seed)
                        if verified_condition(path, fingerprint, methods):
                            summary["skipped"] += 1
                            continue
                        if max_new_conditions is not None and summary["new"] >= max_new_conditions:
                            summary["stopped_early"] = True
                            return summary
                        result = one_condition(frame, peers, ticker, mask_type, block, rate, seed)
                        blob = json_bytes(result)
                        D.atomic(path, blob)
                        D.atomic(path.with_name(path.name + ".done.json"),
                                 json_bytes({"fingerprint": fingerprint, "sha256": sha(blob),
                                             "mask_sha256": result["mask_sha256"]}))
                        summary["new"] += 1
                        if summary["new"] == 1 or (summary["new"] + summary["skipped"]) % 25 == 0:
                            print(f"[CPU {mask_type}] {ticker} b{block} r{rate:.2f} s{seed}: "
                                  f"{summary['new'] + summary['skipped']}/{total} saved/verified", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("data", "cpu"), default="cpu")
    parser.add_argument("--max-new-conditions", type=int)
    args = parser.parse_args()
    receipt, targets, peers = D.prepare_data(args.root)
    print(f"Data: {len(targets)} eligible targets; {len(peers)} eligible peers", flush=True)
    if args.stage == "cpu":
        print("CPU:", run_cpu(args.root, targets, peers,
                              max_new_conditions=args.max_new_conditions), flush=True)


if __name__ == "__main__":
    main()
