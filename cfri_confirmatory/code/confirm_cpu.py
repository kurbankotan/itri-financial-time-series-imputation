"""Confirmatory study, step 2: CPU methods on fold D, feature-wise and whole-row masks.

Every condition runs cfri_core.run_condition, which scores all methods on identical masks: linear_raw,
log_linear, itri_w1_p5, chained_ridge, cfri_adaptive(_no_cross), cfri_fixed(_no_cross), and each of them
followed by the OHLC projection of imputed cells (suffix _proj). CFRI as evaluated in the protocol is
cfri_fixed_proj with the setting C.CFRI_FIXED frozen from folds A-C validation data.

One JSON-lines shard per (ticker, mask type) with a SHA-256 sidecar; finished shards are skipped on restart.
Infeasible conditions (C.feasible) are listed in the sidecar. A condition that raises an error is written as an
error line; the analysis excludes it for every method and reports it.
"""
from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cfri_core as C

FOLD = "D"
OUT = "fold_D_cpu"


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def code_digest() -> str:
    """SHA-256 over the code that determines the CPU results (this file and cfri_core.py)."""
    h = hashlib.sha256()
    for path in (Path(C.__file__), Path(__file__)):
        h.update(path.read_bytes())
    return h.hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def shard_complete(path: Path, n_conditions: int | None = None) -> bool:
    """True when the shard and its sidecar agree (SHA-256, code digest and, if given, the condition count)."""
    side = path.with_name(path.name + ".done.json")
    if not (path.exists() and side.exists()):
        return False
    try:
        info = json.loads(side.read_text())
    except (OSError, ValueError):
        return False
    return (info.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest() and info.get("code") == C.CODE_ID
            and info.get("code_sha256") == code_digest()
            and (n_conditions is None or info.get("conditions") == n_conditions))


def conditions_for(parts, mask_type):
    rows = {k: len(v) for k, v in parts.items()}
    keep, dropped = [], []
    for block in C.BLOCKS:
        for rate in C.RATES:
            ok = C.feasible(rows, block, rate, mask_type)
            for seed in C.SEEDS:
                (keep if ok else dropped).append((block, rate, seed))
    return keep, dropped


def _shard(job):
    root, ticker, mask_type, frame = job
    out = Path(root) / OUT
    path = out / f"{ticker}_{mask_type}.jsonl"
    side = path.with_name(path.name + ".done.json")
    parts = C.split_frame(frame, FOLD)
    keep, dropped = conditions_for(parts, mask_type)
    if shard_complete(path, len(keep)):
        return ticker, mask_type, "skip", 0
    lines, errors = [], 0
    for b, r, s in keep:
        try:
            res = C.run_condition(parts, ticker, FOLD, b, r, s, mask_type)
        except Exception as exc:  # reported, excluded for all methods in the analysis
            res = {"ticker": ticker, "fold": FOLD, "block_length": b, "rate": r, "seed": s, "mask_type": mask_type,
                   "error": f"{type(exc).__name__}: {exc}"}
            errors += 1
        lines.append(json.dumps(_jsonable(res), sort_keys=True, allow_nan=False))
    data = ("\n".join(lines) + "\n").encode()
    _atomic(path, data)
    _atomic(side, json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "code": C.CODE_ID,
                              "code_sha256": code_digest(), "conditions": len(keep), "errors": errors,
                              "rows_by_partition": {k: len(v) for k, v in parts.items()},
                              "dropped_infeasible": dropped}).encode())
    return ticker, mask_type, "done", errors


def run_cpu(root: Path, frames: dict, mask_types=C.MASK_TYPES, workers: int | None = None) -> dict:
    (root / OUT).mkdir(parents=True, exist_ok=True)
    jobs = [(str(root), t, m, f) for m in mask_types for t, f in frames.items()]
    workers = workers or os.cpu_count() or 1
    status = {}
    with Pool(workers) as pool:
        for i, (ticker, mask_type, state, errors) in enumerate(pool.imap_unordered(_shard, jobs), 1):
            status[f"{ticker}_{mask_type}"] = {"state": state, "errors": errors}
            print(f"[CPU {i}/{len(jobs)}] {ticker} {mask_type}: {state}" + (f", {errors} errors" if errors else ""),
                  flush=True)
    complete = all(shard_complete(root / OUT / f"{t}_{m}.jsonl") for m in mask_types for t in frames)
    import platform, scipy, sklearn, pandas
    log = {"code": C.CODE_ID, "code_sha256": code_digest(), "workers": workers, "shards": len(jobs),
           "complete": complete, "status": status,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
                           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "cpu_count": os.cpu_count()}}
    _atomic(root / OUT / "cpu_run_log.json", json.dumps(log, indent=1).encode())
    return log
