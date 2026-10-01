#!/usr/bin/env python3
"""BRITS/SAITS external validation on the frozen classical benchmark masks.

One neural model is fitted for each asset, fold, block length, missing rate,
mask seed and method. Only the corresponding masked training partition is
used for fitting; the separate validation partition selects the epoch. Test
truth is used only by score_features after inference. Checkpoints contain
metrics and metadata, never raw OHLCV or model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import platform
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pypots
import pygrinder
import scipy
import sklearn
import torch
from pypots.imputation import BRITS, SAITS
from pypots.utils.logging import logger as pypots_logger

from masking import run_lengths, strict_block_mask
from run_classical_benchmark import (
    BLOCKS, EDGE_GUARD, FEATURES, FOLDS, RATES, SEEDS, METHODS,
    corrupt, load_verified_asset, run_methods, score_features, sha256, split_frame,
    stable_mask_seed, train_observed_scaler,
)


NEURAL_METHODS = ("brits", "saits")
N_STEPS = 64
STRIDE = 16
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5
BRITS_HIDDEN = 64
SAITS_CONFIG = {"n_layers": 2, "d_model": 64, "n_heads": 4,
                "d_k": 16, "d_v": 16, "d_ffn": 128, "dropout": 0.1}


class _NoSavingPathWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "saving_path not given" not in record.getMessage()


pypots_logger.addFilter(_NoSavingPathWarning())


def window_starts(length: int, n_steps: int = N_STEPS,
                  stride: int = STRIDE) -> list[int]:
    if length < n_steps or n_steps < 2 or stride < 1:
        raise ValueError("partition is too short for the frozen neural window")
    starts = list(range(0, length - n_steps + 1, stride))
    if starts[-1] != length - n_steps:
        starts.append(length - n_steps)
    return starts


def as_windows(values: np.ndarray, starts: list[int],
               n_steps: int = N_STEPS) -> np.ndarray:
    return np.stack([values[start:start + n_steps] for start in starts]).astype(
        np.float32, copy=False)


def fold_windows(predictions: np.ndarray, starts: list[int],
                 observed: np.ndarray) -> np.ndarray:
    """Average overlapping predictions; restore every originally observed cell."""
    if predictions.shape != (len(starts), N_STEPS, observed.shape[1]):
        raise ValueError("neural prediction shape mismatch")
    sums = np.zeros(observed.shape, dtype=np.float64)
    counts = np.zeros(observed.shape[0], dtype=np.int32)
    for prediction, start in zip(predictions, starts):
        sums[start:start + N_STEPS] += prediction
        counts[start:start + N_STEPS] += 1
    if np.any(counts == 0):
        raise AssertionError("test window coverage is incomplete")
    completed = sums / counts[:, None]
    originally_observed = np.isfinite(observed)
    completed[originally_observed] = observed[originally_observed]
    if not np.isfinite(completed).all():
        raise ValueError("neural model produced non-finite estimates")
    return completed


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training_seed(manifest_sha: str, ticker: str, fold: str,
                  block: int, rate: float, repetition_seed: int,
                  method: str) -> int:
    key = (f"{manifest_sha}|{ticker}|{fold}|{block}|{rate:.2f}|"
           f"{repetition_seed}|{method}|neural-v1")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def fit_and_predict(method: str, train_input: np.ndarray,
                    val_input: np.ndarray, val_truth: np.ndarray,
                    test_input: np.ndarray, seed: int,
                    device: str, epochs: int = EPOCHS) -> tuple[np.ndarray, np.ndarray, float]:
    set_training_seed(seed)
    train_starts = window_starts(len(train_input))
    val_starts = window_starts(len(val_input))
    test_starts = window_starts(len(test_input))
    train_set = {"X": as_windows(train_input, train_starts)}
    val_set = {"X": as_windows(val_input, val_starts),
               "X_ori": as_windows(val_truth, val_starts)}
    test_set = {"X": as_windows(test_input, test_starts)}
    common = {"n_steps": N_STEPS, "n_features": len(FEATURES),
              "batch_size": BATCH_SIZE, "epochs": epochs,
              "patience": min(PATIENCE, epochs), "device": device,
              "model_saving_strategy": None, "verbose": False}
    if method == "brits":
        model = BRITS(rnn_hidden_size=BRITS_HIDDEN, **common)
    elif method == "saits":
        model = SAITS(**SAITS_CONFIG, **common)
    else:
        raise ValueError(f"unknown neural method: {method}")
    started = time.perf_counter()
    model.fit(train_set, val_set)
    val_predictions = np.asarray(model.predict({"X": val_set["X"]})["imputation"])
    test_predictions = np.asarray(model.predict(test_set)["imputation"])
    val_completed = fold_windows(val_predictions, val_starts, val_input)
    test_completed = fold_windows(test_predictions, test_starts, test_input)
    elapsed = time.perf_counter() - started
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return val_completed, test_completed, elapsed


def verified_classical_shard(archive: zipfile.ZipFile, ticker: str,
                             fold: str, expected_fingerprint: str) -> dict:
    name = f"results/shards/{ticker}_{fold}.csv"
    sidecar = json.loads(archive.read(name.replace(".csv", ".meta.json")))
    blob = archive.read(name)
    if (sidecar["csv_sha256"] != sha256(blob) or
            sidecar["run_fingerprint"] != expected_fingerprint):
        raise ValueError(f"classical checkpoint checksum mismatch: {ticker}/{fold}")
    # Pandas' default parser can turn the archived 0.29999999999999999 into
    # 0.2999999999999999, which does not equal the prespecified Python 0.3.
    frame = pd.read_csv(io.BytesIO(blob), float_precision="round_trip")
    reference = frame.loc[(frame.method == "linear") & (frame.feature == "open")]
    if len(reference) != len(BLOCKS) * len(RATES) * len(SEEDS):
        raise ValueError(f"incomplete classical condition grid: {ticker}/{fold}")
    return {(int(row.block_length), float(row.rate), int(row.seed)):
            (int(row.train_mask_seed), int(row.test_mask_seed))
            for row in reference.itertuples()}


def migrate_checkpoints(output_dir: Path, fingerprint: str,
                        allowed_previous: set[str]) -> dict:
    """Relabel intact shards written under an earlier environment fingerprint.

    This is a one-time migration for checkpoints produced before the resume key
    was made hardware independent. Only a fingerprint listed in
    ``allowed_previous`` is accepted, so a genuine protocol change can never be
    laundered into the current run. Only the provenance key is rewritten. Every
    CSV is re-hashed first, so a shard is kept only when its bytes still match
    the digest recorded beside it. Anything that fails verification is deleted
    and recomputed.
    """
    migrated = 0
    dropped = 0
    superseded: set[str] = set()
    for directory, expected in ((output_dir / "shards", 10),
                                (output_dir / "validation_shards", 35)):
        if not directory.is_dir():
            continue
        for meta_path in sorted(directory.glob("*.meta.json")):
            csv_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
            if not csv_path.exists():
                meta_path.unlink(missing_ok=True)
                dropped += 1
                continue
            try:
                sidecar = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta_path.unlink(missing_ok=True)
                csv_path.unlink(missing_ok=True)
                dropped += 1
                continue
            if sidecar.get("run_fingerprint") == fingerprint:
                continue
            if sidecar.get("run_fingerprint") not in allowed_previous:
                continue
            if (sidecar.get("expected_rows") != expected or
                    sidecar.get("csv_sha256") != sha256(csv_path.read_bytes())):
                meta_path.unlink(missing_ok=True)
                csv_path.unlink(missing_ok=True)
                dropped += 1
                continue
            previous = str(sidecar.get("run_fingerprint"))
            superseded.add(previous)
            sidecar.setdefault("migrated_from", []).append(previous)
            sidecar["run_fingerprint"] = fingerprint
            atomic_write_bytes(meta_path, json.dumps(sidecar, indent=2).encode())
            migrated += 1
    return {"migrated": migrated, "dropped": dropped,
            "superseded_fingerprints": sorted(superseded)}


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def run(snapshot: Path, classical_zip: Path, output_dir: Path,
        pilot: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with zipfile.ZipFile(snapshot) as private, zipfile.ZipFile(classical_zip) as classical:
        manifest_blob = private.read("asset_universe_v1.json")
        receipt_blob = private.read("acquisition_qc_receipt.json")
        manifest_sha = sha256(manifest_blob)
        receipt = json.loads(receipt_blob)
        classical_meta = json.loads(classical.read("results/run_metadata.json"))
        classical_marker = json.loads(classical.read("results/COMPLETE.marker"))
        if (receipt["manifest_sha256"] != manifest_sha or
                classical_meta["manifest_sha256"] != manifest_sha or
                classical_meta["private_zip_sha256"] != sha256(snapshot.read_bytes()) or
                classical_marker["expected_shards"] != 102 or
                classical_marker["pilot_not_publishable"] or
                classical_marker["run_fingerprint"] != classical_meta["run_fingerprint"]):
            raise ValueError("private/classical inputs are incomplete or do not match")
        if (receipt["eligible_count"] != 34 or receipt["excluded_count"] or
                tuple(classical_meta["methods"]) != METHODS or
                classical_meta["edge_guard"] != EDGE_GUARD):
            raise ValueError("classical protocol or panel differs from the frozen run")

        versions = {"python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                    "torch": torch.__version__, "pypots": pypots.__version__,
                    "pygrinder": pygrinder.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version()}
        hardware = {"device": device,
                    "gpu_name": torch.cuda.get_device_name(0)
                    if device == "cuda" else None}
        config = {"window_length": N_STEPS, "window_stride": STRIDE,
                  "batch_size": BATCH_SIZE, "epochs_max": EPOCHS,
                  "patience": PATIENCE, "brits_hidden": BRITS_HIDDEN,
                  "saits": SAITS_CONFIG, "methods": NEURAL_METHODS,
                  "device": device, "hardware": hardware}
        # The resume key covers the protocol only. Accelerator model, driver
        # stack and framework build are recorded in the metadata but excluded,
        # so a run may legitimately span several sessions and GPUs.
        protocol_config = {"window_length": N_STEPS, "window_stride": STRIDE,
                           "batch_size": BATCH_SIZE, "epochs_max": EPOCHS,
                           "patience": PATIENCE, "brits_hidden": BRITS_HIDDEN,
                           "saits": SAITS_CONFIG, "methods": NEURAL_METHODS}
        protocol_versions = {key: versions[key] for key in
                             ("numpy", "pandas", "scipy", "sklearn",
                              "pypots", "pygrinder")}
        fingerprint = sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("masking.py").read_bytes()
            + Path(__file__).with_name("run_classical_benchmark.py").read_bytes()
            + manifest_blob + receipt_blob + classical_marker["run_fingerprint"].encode()
            + json.dumps(protocol_config, sort_keys=True).encode()
            + json.dumps(protocol_versions, sort_keys=True).encode()
            + str(pilot).encode())
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        validation_dir = output_dir / "validation_shards"
        validation_dir.mkdir(exist_ok=True)
        meta_path = output_dir / "run_metadata.json"
        previous_meta = (json.loads(meta_path.read_text())
                         if meta_path.exists() else {})
        previous_key = previous_meta.get("run_fingerprint")
        legacy = bool(previous_meta) and "resume_key_scope" not in previous_meta
        migration = {"migrated": 0, "dropped": 0, "superseded_fingerprints": []}
        if legacy and previous_key and previous_key != fingerprint:
            # One-time move from the old hardware-dependent key to the new one.
            migration = migrate_checkpoints(output_dir, fingerprint,
                                            {previous_key})
            print(f"Checkpoint migration: {migration['migrated']} relabelled, "
                  f"{migration['dropped']} discarded as unverifiable",
                  flush=True)
        elif previous_key and previous_key != fingerprint:
            changed = {k: (previous_meta.get("protocol_versions", {}).get(k), v)
                       for k, v in protocol_versions.items()
                       if previous_meta.get("protocol_versions", {}).get(k) != v}
            raise SystemExit(
                "Protokol anahtari degisti, tamamlanmis shard'lar bu kosuya "
                "dahil edilemez. Farkli kutuphaneler: " + json.dumps(changed)
                + ". Ya ortami eski surumlere sabitleyin ya da sonuc klasorunu "
                "yeni bir isimle bastan baslatin.")

        history_path = output_dir / "environment_history.json"
        history = (json.loads(history_path.read_text())
                   if history_path.exists() else [])
        history.append({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()),
                        "versions": versions, "hardware": hardware,
                        "run_fingerprint": fingerprint,
                        "checkpoint_migration": migration})
        atomic_write_bytes(history_path, json.dumps(history, indent=2).encode())

        metadata = {"run_fingerprint": fingerprint,
                    "resume_key_scope": "protocol only; accelerator, driver "
                                        "and framework build excluded",
                    "sessions": len(history),
                    "protocol_versions": protocol_versions,
                    "environment_history": "environment_history.json",
                    "classical_fingerprint": classical_marker["run_fingerprint"],
                    "manifest_sha256": manifest_sha,
                    "private_zip_sha256": classical_meta["private_zip_sha256"],
                    "classical_zip_sha256": sha256(classical_zip.read_bytes()),
                    "pilot_not_publishable": pilot,
                    "model_selection": "separate masked validation partition",
                    "train_and_val_data": "condition-specific masks, no test labels",
                    "window_overlap_aggregation": "arithmetic mean",
                    "config": config, "versions": versions,
                    "hardware": hardware}
        atomic_write_bytes(output_dir / "run_metadata.json",
                           json.dumps(metadata, indent=2).encode())

        records = receipt["assets"][:1] if pilot else receipt["assets"]
        folds = ("A",) if pilot else tuple(FOLDS)
        conditions = [(b, r, s) for b in BLOCKS for r in RATES for s in SEEDS]
        if pilot:
            conditions = conditions[:1]
        expected_shards = len(records) * len(folds) * len(conditions)
        session_started = time.time()
        session_done = 0
        already = sum(1 for _ in shard_dir.glob("*.csv"))
        print(f"Resuming: {already}/{expected_shards} shards already on disk "
              f"({100 * already / expected_shards:.1f}%)", flush=True)
        for asset_index, record in enumerate(records, 1):
            ticker = record["ticker"]
            frame = load_verified_asset(private, record)
            for fold in folds:
                reference = verified_classical_shard(
                    classical, ticker, fold, classical_marker["run_fingerprint"])
                partitions = split_frame(frame, fold)
                train = partitions["train"].to_numpy(float)
                val = partitions["val"].to_numpy(float)
                test = partitions["test"].to_numpy(float)
                for condition_index, (block, rate, repetition_seed) in enumerate(conditions, 1):
                    stem = f"{ticker}_{fold}_b{block}_r{round(rate*100)}_s{repetition_seed}"
                    csv_path = shard_dir / (stem + ".csv")
                    meta_path = shard_dir / (stem + ".meta.json")
                    val_csv_path = validation_dir / (stem + ".csv")
                    val_meta_path = validation_dir / (stem + ".meta.json")
                    if (csv_path.exists() and meta_path.exists() and
                            val_csv_path.exists() and val_meta_path.exists()):
                        sidecar = json.loads(meta_path.read_text())
                        val_sidecar = json.loads(val_meta_path.read_text())
                        if (sidecar.get("run_fingerprint") == fingerprint and
                                sidecar.get("expected_rows") == 10 and
                                sidecar.get("csv_sha256") == sha256(csv_path.read_bytes()) and
                                val_sidecar.get("run_fingerprint") == fingerprint and
                                val_sidecar.get("expected_rows") == 35 and
                                val_sidecar.get("csv_sha256") == sha256(val_csv_path.read_bytes())):
                            continue
                    train_seed = stable_mask_seed(manifest_sha, ticker, fold, "train",
                                                  block, rate, repetition_seed)
                    val_seed = stable_mask_seed(manifest_sha, ticker, fold, "val",
                                                block, rate, repetition_seed)
                    test_seed = stable_mask_seed(manifest_sha, ticker, fold, "test",
                                                 block, rate, repetition_seed)
                    if reference[(block, rate, repetition_seed)] != (train_seed, test_seed):
                        raise AssertionError("neural and classical mask seeds differ")
                    train_hidden = strict_block_mask(train.shape, rate, block,
                                                     train_seed, EDGE_GUARD)
                    val_hidden = strict_block_mask(val.shape, rate, block,
                                                   val_seed, EDGE_GUARD)
                    test_hidden = strict_block_mask(test.shape, rate, block,
                                                    test_seed, EDGE_GUARD)
                    mean, scale = train_observed_scaler(train, train_hidden)
                    train_input = corrupt((train - mean) / scale, train_hidden)
                    val_truth = (val - mean) / scale
                    val_input = corrupt(val_truth, val_hidden)
                    test_input = corrupt((test - mean) / scale, test_hidden)
                    lengths = run_lengths(test_hidden)
                    if int(lengths.max()) > block:
                        raise AssertionError("test mask exceeds requested run length")
                    common = {"ticker": ticker, "country": record["country"],
                              "region": record["region"], "fold": fold,
                              "test_start_year": FOLDS[fold]["test"][0],
                              "test_end_year": FOLDS[fold]["test"][1],
                              "block_length": block, "rate": rate,
                              "seed": repetition_seed,
                              "train_mask_seed": train_seed,
                              "val_mask_seed": val_seed,
                              "test_mask_seed": test_seed,
                              "actual_max_test_run": int(lengths.max())}
                    rows = []
                    val_common = {"ticker": ticker, "country": record["country"],
                                  "region": record["region"], "fold": fold,
                                  "block_length": block, "rate": rate,
                                  "seed": repetition_seed,
                                  "train_mask_seed": train_seed,
                                  "val_mask_seed": val_seed,
                                  "partition": "validation"}
                    val_rows = []
                    for method, (val_estimate, val_elapsed, val_warnings) in run_methods(
                            train_input, val_input, repetition_seed).items():
                        val_rows.extend(score_features(
                            val, val_estimate, val_hidden, mean, scale,
                            {**val_common, "method": method},
                            val_elapsed, val_warnings))
                    for method in NEURAL_METHODS:
                        model_seed = training_seed(manifest_sha, ticker, fold,
                                                   block, rate, repetition_seed, method)
                        val_estimate, estimate, elapsed = fit_and_predict(
                            method, train_input, val_input, val_truth,
                            test_input, model_seed, device,
                            epochs=1 if pilot else EPOCHS)
                        val_rows.extend(score_features(
                            val, val_estimate, val_hidden, mean, scale,
                            {**val_common, "method": method,
                             "model_seed": model_seed}, elapsed, 0))
                        scored = score_features(test, estimate, test_hidden,
                                                mean, scale,
                                                {**common, "method": method,
                                                 "model_seed": model_seed,
                                                 "window_length": N_STEPS,
                                                 "window_stride": STRIDE,
                                                 "epochs_max": 1 if pilot else EPOCHS,
                                                 "patience": 1 if pilot else PATIENCE},
                                                elapsed, 0)
                        rows.extend(scored)
                    if len(rows) != len(NEURAL_METHODS) * len(FEATURES):
                        raise AssertionError("neural condition row count mismatch")
                    if len(val_rows) != (len(METHODS) + len(NEURAL_METHODS)) * len(FEATURES):
                        raise AssertionError("validation condition row count mismatch")
                    val_data = pd.DataFrame(val_rows).to_csv(
                        index=False, float_format="%.17g").encode()
                    atomic_write_bytes(val_csv_path, val_data)
                    val_sidecar = {"run_fingerprint": fingerprint,
                                   "expected_rows": len(val_rows),
                                   "csv_sha256": sha256(val_data),
                                   "pilot_not_publishable": pilot}
                    atomic_write_bytes(val_meta_path,
                                       json.dumps(val_sidecar, indent=2).encode())
                    data = pd.DataFrame(rows).to_csv(
                        index=False, float_format="%.17g").encode()
                    atomic_write_bytes(csv_path, data)
                    sidecar = {"run_fingerprint": fingerprint,
                               "expected_rows": len(rows),
                               "csv_sha256": sha256(data),
                               "pilot_not_publishable": pilot}
                    atomic_write_bytes(meta_path, json.dumps(sidecar, indent=2).encode())
                    session_done += 1
                    if session_done % 10 == 0:
                        rate_s = (time.time() - session_started) / session_done
                        left = expected_shards - already - session_done
                        print(f"[{asset_index}/{len(records)}] {ticker}/{fold} "
                              f"{condition_index}/{len(conditions)} | "
                              f"toplam {already + session_done}/{expected_shards} "
                              f"(%{100 * (already + session_done) / expected_shards:.1f}) | "
                              f"{rate_s:.0f} s/shard | kalan ~{rate_s * left / 3600:.1f} saat",
                              flush=True)

        verified = 0
        for record in records:
            for fold in folds:
                for block, rate, repetition_seed in conditions:
                    stem = (f"{record['ticker']}_{fold}_b{block}_"
                            f"r{round(rate*100)}_s{repetition_seed}")
                    csv_path = shard_dir / (stem + ".csv")
                    sidecar = json.loads((shard_dir / (stem + ".meta.json")).read_text())
                    val_csv_path = validation_dir / (stem + ".csv")
                    val_sidecar = json.loads(
                        (validation_dir / (stem + ".meta.json")).read_text())
                    if (sidecar["run_fingerprint"] != fingerprint or
                            sidecar["csv_sha256"] != sha256(csv_path.read_bytes()) or
                            val_sidecar["run_fingerprint"] != fingerprint or
                            val_sidecar["csv_sha256"] != sha256(val_csv_path.read_bytes())):
                        raise AssertionError(f"invalid neural checkpoint: {stem}")
                    verified += 1
        if verified != expected_shards:
            raise AssertionError("missing neural result shards")
        marker = {"expected_shards": expected_shards,
                  "expected_rows": expected_shards * len(NEURAL_METHODS) * len(FEATURES),
                  "expected_validation_rows": expected_shards *
                  (len(METHODS) + len(NEURAL_METHODS)) * len(FEATURES),
                  "pilot_not_publishable": pilot,
                  "run_fingerprint": fingerprint}
        atomic_write_bytes(output_dir / "COMPLETE.marker",
                           json.dumps(marker, indent=2).encode())
        print("COMPLETE", output_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true",
                        help="one asset/fold/condition and one epoch; not publishable")
    args = parser.parse_args()
    run(args.snapshot, args.classical, args.out, pilot=args.pilot)
