#!/usr/bin/env python3
"""Exploratory CFRI runs on folds A-C (feature-wise and whole-row masks) from cached OHLCV CSVs.

Usage: python run_exploratory.py RAW_DIR OUT_DIR [--workers 2] [--limit N]
Each (ticker, fold, mask_type) shard is written as JSON lines with a SHA-256 sidecar; finished shards are
skipped on restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfri_core as C  # noqa: E402


def load_frame(raw_dir: Path, ticker: str) -> pd.DataFrame:
    csv_path = raw_dir / f"{ticker}_2015_2025.csv"
    meta = json.loads((raw_dir / f"{ticker}_2015_2025.meta.json").read_text())
    blob = csv_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
        raise ValueError(f"hash mismatch {ticker}")
    frame = pd.read_csv(csv_path, index_col="date", parse_dates=True, float_precision="round_trip")
    frame = frame.loc[:, list(C.FEATURES)].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"nonfinite {ticker}")
    return frame


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def run_shard(job):
    raw_dir, out_dir, ticker, fold, mask_type = job
    path = Path(out_dir) / f"{ticker}_{fold}_{mask_type}.jsonl"
    side = path.with_suffix(".done.json")
    if path.exists() and side.exists():
        info = json.loads(side.read_text())
        if info.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest() and info.get("code") == C.CODE_ID:
            return ticker, fold, mask_type, "skip"
    frame = load_frame(Path(raw_dir), ticker)
    parts = C.split_frame(frame, fold)
    lines = []
    for block in C.BLOCKS:
        for rate in C.RATES:
            for seed in C.SEEDS:
                res = C.run_condition(parts, ticker, fold, block, rate, seed, mask_type)
                lines.append(json.dumps(jsonable(res), sort_keys=True, allow_nan=False))
    data = ("\n".join(lines) + "\n").encode()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    side.write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "code": C.CODE_ID, "conditions": len(lines)}))
    return ticker, fold, mask_type, "done"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mask-types", default="feature,row")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(a.raw_dir, a.out_dir, t, f, m) for m in a.mask_types.split(",") for t, _, _ in C.ASSETS for f in ("A", "B", "C")]
    if a.limit:
        jobs = jobs[:a.limit]
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_shard, jobs), 1):
            print(i, len(jobs), *r, flush=True)
