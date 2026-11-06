"""Score-blind acquisition for the frozen CFRI held-out-asset study.

The target ETFs are disjoint from the 34 fold-D development targets. This module
never reads prior study roots, scores, or snapshots.  A private cache in this
run's root makes interrupted downloads resumable; the receipt binds each file
to its SHA-256 hash before any method may run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C

STUDY_ID = "cfri-heldout-assets-v1"
FOLD_TAG = "H1"
START = "2015-01-01"
END_EXCLUSIVE = "2026-10-01"
TEST_END = "2026-09-30"
YEARS = tuple(range(2015, 2027))
PARTITIONS = {"train": (2015, 2024), "val": (2025, 2025), "test": (2026, 2026)}
RAW_DIR = "raw_do_not_publish"
RECEIPT = "data_receipt.json"

# Frozen before any held-out target data are downloaded or scored.
TARGETS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV",
           "XLY", "SPY", "QQQ", "IWM", "EFA", "EEM", "IBB", "IYR")
PEERS = tuple(ticker for ticker, _, _ in C.ASSETS)
CANDIDATES = tuple((ticker, "target") for ticker in TARGETS) + tuple(
    (ticker, "peer") for ticker in PEERS)
HERE = Path(__file__).resolve().parent
REQUIRED_FROZEN_FILES = (
    "PROTOCOL.md", "cfri_core.py", "upstream/cfri_core.py", "confirm_saits.py",
    "v2_price.py", "v2_peer.py", "holdout_data.py", "holdout_run.py",
    "holdout_saits.py", "holdout_analyze.py",
)
assert len(TARGETS) == 16 and len(PEERS) == 34
assert len(set(TARGETS) | set(PEERS)) == 50


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def validate_freeze(root: Path) -> dict:
    """Require the recorded, byte-identical source bundle before reading target data."""
    root = Path(root)
    receipt = read_json(root / "freeze_receipt.json")
    if not isinstance(receipt, dict):
        raise RuntimeError("freeze_receipt.json is required before data acquisition")
    commit = receipt.get("git_commit")
    digest = receipt.get("embedded_bundle_sha256")
    if commit is not None and (not isinstance(commit, str) or
                               re.fullmatch(r"[0-9a-fA-F]{40}", commit) is None):
        raise RuntimeError("optional Git commit SHA must have 40 hex characters")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise RuntimeError("freeze receipt must contain a 64-hex source bundle SHA")
    try:
        frozen_at = datetime.fromisoformat(receipt["freeze_utc"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as exc:
        raise RuntimeError("freeze receipt must contain an ISO UTC timestamp") from exc
    if frozen_at.utcoffset() != timezone.utc.utcoffset(frozen_at):
        raise RuntimeError("freeze timestamp must be UTC")
    bundle = root / "frozen_source_bundle.zip"
    if not bundle.is_file() or sha256(bundle.read_bytes()).lower() != digest.lower():
        raise RuntimeError("frozen source bundle is absent or its SHA does not match the receipt")
    try:
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or archive.testzip() is not None:
                raise RuntimeError("frozen source bundle is corrupt or contains duplicate paths")
            if any("raw_do_not_publish" in name or "raw" in Path(name).parts for name in names):
                raise RuntimeError("frozen source bundle contains a raw-data path")
            if "SOURCE_SHA256.json" not in names:
                raise RuntimeError("frozen source bundle lacks SOURCE_SHA256.json")
            manifest = json.loads(archive.read("SOURCE_SHA256.json"))
            if not isinstance(manifest, dict) or not set(REQUIRED_FROZEN_FILES).issubset(manifest):
                raise RuntimeError("frozen source manifest lacks required study files")
            for name, expected in manifest.items():
                if (not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts or
                        not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None or
                        name not in names):
                    raise RuntimeError("invalid frozen source manifest entry")
                local = HERE / name
                if not local.is_file():
                    raise RuntimeError(f"frozen source file is missing: {name}")
                content = local.read_bytes()
                if sha256(content) != expected or sha256(archive.read(name)) != expected:
                    raise RuntimeError(f"source changed after freeze: {name}")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen source bundle is unreadable") from exc
    return receipt


def normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("empty provider response")
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            names = [str(n).lower().replace(" ", "_") for n in out.columns.get_level_values(level)]
            if all(name in names for name in C.FEATURES):
                out.columns = names
                break
        else:
            raise ValueError("cannot locate OHLCV level in provider columns")
    else:
        out.columns = [str(n).lower().replace(" ", "_") for n in out.columns]
    missing = set(C.FEATURES).difference(out.columns)
    if missing:
        raise ValueError(f"missing provider columns: {sorted(missing)}")
    out = out.loc[:, list(C.FEATURES)]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("duplicate or unordered dates")
    if (out.index < pd.Timestamp(START)).any() or (out.index >= pd.Timestamp(END_EXCLUSIVE)).any():
        raise ValueError("provider returned dates outside requested interval")
    return out.astype(float)


def xnys_sessions() -> pd.DatetimeIndex:
    import exchange_calendars as xcals

    sessions = pd.DatetimeIndex(xcals.get_calendar("XNYS").sessions_in_range(START, TEST_END))
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def max_true_run(flags) -> int:
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    lengths = edges[1::2] - edges[::2]
    return int(lengths.max()) if lengths.size else 0


def assess_quality(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    reasons: list[str] = []
    values = frame.loc[:, list(C.FEATURES)].to_numpy(float)
    if not np.isfinite(values).all():
        reasons.append("nonfinite_ohlcv")
    else:
        if (frame.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
            reasons.append("nonpositive_price")
        if (frame["volume"] < 0).any():
            reasons.append("negative_volume")
        tol = 1e-6 * frame.loc[:, ("open", "high", "low", "close")].max(axis=1)
        if ((frame["high"] + tol < frame[["open", "low", "close"]].max(axis=1)) |
                (frame["low"] - tol > frame[["open", "high", "close"]].min(axis=1))).any():
            reasons.append("invalid_ohlc_order")
    observed = pd.DatetimeIndex(frame.index)
    missing = ~sessions.isin(observed)
    non_us = observed.difference(sessions)
    yearly = {}
    for year in YEARS:
        expected = sessions[sessions.year == year]
        present = int(expected.isin(observed).sum())
        coverage = present / len(expected)
        yearly[str(year)] = {"expected_sessions": len(expected), "observed_sessions": present, "coverage": coverage}
        if coverage < 0.95:
            reasons.append(f"coverage_below_95pct_{year}")
    longest = max_true_run(missing)
    if longest > 5:
        reasons.append("more_than_five_consecutive_absent_us_sessions")
    if len(non_us):
        reasons.append("rows_outside_us_exchange_calendar")
    return {"eligible": not reasons, "reasons": sorted(set(reasons)), "row_count": int(len(frame)),
            "first_date": frame.index.min().strftime("%Y-%m-%d"),
            "last_date": frame.index.max().strftime("%Y-%m-%d"),
            "sessions_2026": int((frame.index.year == 2026).sum()),
            "longest_absent_us_session_run": longest, "non_us_session_count": int(len(non_us)),
            "yearly_coverage": yearly}


def _download(yf, ticker: str, attempts: int = 3) -> pd.DataFrame:
    scale = float(os.environ.get("CFRI_RETRY_PAUSE_SCALE", "1"))
    for attempt in range(attempts):
        try:
            got = yf.download(ticker, start=START, end=END_EXCLUSIVE,
                              auto_adjust=True, actions=False, progress=False, threads=False)
            return normalized_ohlcv(got)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(scale * (10, 30)[attempt])
    raise AssertionError("unreachable")


def methods_started(root: Path) -> bool:
    return any(p.is_file() for dirname in ("holdout_cpu", "holdout_saits", "holdout_saits_peer")
               if (root / dirname).is_dir() for p in (root / dirname).iterdir())


def provider_failures(receipt: dict) -> list[str]:
    return [asset["ticker"] for asset in receipt["assets"] if
            any(reason.startswith("acquisition_or_qc_error:") for reason in asset.get("reasons", []))]


def _validate_receipt_identity(receipt: dict) -> None:
    if (receipt.get("study") != STUDY_ID or receipt.get("start_inclusive") != START or
            receipt.get("end_exclusive") != END_EXCLUSIVE or
            receipt.get("target_tickers") != list(TARGETS) or
            receipt.get("peer_tickers") != list(PEERS)):
        raise RuntimeError("data receipt belongs to different study, candidate list, or date range")
    found = [(asset.get("ticker"), asset.get("role")) for asset in receipt.get("assets", [])]
    if found != list(CANDIDATES):
        raise RuntimeError("data receipt candidate rows changed or are incomplete")


def acquire(root: Path) -> dict:
    """Download only into this run root; retry provider failures before methods start."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate_freeze(root)
    import exchange_calendars as xcals
    import yfinance as yf
    old_path = root / RECEIPT
    if old_path.exists():
        old = read_json(old_path)
        if old is None:
            raise RuntimeError("unreadable data receipt; inspect it before continuing")
        _validate_receipt_identity(old)
        if methods_started(root) or not provider_failures(old):
            print(f"data receipt final: {old['eligible_target_count']} targets, "
                  f"{old['eligible_peer_count']} peers", flush=True)
            return old
        print("retrying provider failures before any method run:", provider_failures(old), flush=True)

    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    sessions = xnys_sessions()
    receipt = {"study": STUDY_ID, "start_inclusive": START, "end_exclusive": END_EXCLUSIVE,
               "test_end": TEST_END, "target_tickers": list(TARGETS), "peer_tickers": list(PEERS),
               "provider": "Yahoo Finance via yfinance; auto_adjust=True; actions=False",
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "yfinance": yf.__version__,
                               "exchange_calendars": xcals.__version__},
               "acquisition_started_utc": datetime.now(timezone.utc).isoformat(), "assets": []}
    for i, (ticker, role) in enumerate(CANDIDATES, 1):
        path = raw / f"{ticker}_{START}_{END_EXCLUSIVE}.csv"
        meta_path = path.with_suffix(".meta.json")
        rec = {"ticker": ticker, "role": role, "sequence": i}
        try:
            if path.exists():
                frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
                meta = read_json(meta_path) or {}
                if meta.get("sha256") and meta["sha256"] != sha256(path.read_bytes()):
                    raise RuntimeError("same-run cached CSV does not match its retrieval hash")
                rec["source"] = "same_run_drive_cache"
                rec["retrieved_utc"] = meta.get("retrieved_utc")
            else:
                frame = _download(yf, ticker)
                retrieved = datetime.now(timezone.utc).isoformat()
                payload = frame.to_csv(float_format="%.17g", index_label="date").encode()
                atomic(path, payload)
                atomic(meta_path, json.dumps({"ticker": ticker, "retrieved_utc": retrieved,
                                              "yfinance": yf.__version__,
                                              "sha256": sha256(payload)}, sort_keys=True).encode())
                rec["source"] = "fresh_yfinance_download"
                rec["retrieved_utc"] = retrieved
            rec["sha256"] = sha256(path.read_bytes())
            frame = normalized_ohlcv(frame)
            rec.update(assess_quality(frame, sessions))
        except Exception as exc:
            rec.update({"eligible": False,
                        "reasons": [f"acquisition_or_qc_error:{type(exc).__name__}:{exc}"]})
        receipt["assets"].append(rec)
        print(f"[{i:02d}/{len(CANDIDATES)}] {ticker} ({role}): "
              f"{'eligible' if rec['eligible'] else 'EXCLUDED ' + str(rec['reasons'])}", flush=True)
    receipt["acquisition_finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["eligible_target_count"] = sum(a["eligible"] and a["role"] == "target" for a in receipt["assets"])
    receipt["eligible_peer_count"] = sum(a["eligible"] and a["role"] == "peer" for a in receipt["assets"])
    receipt["excluded"] = [{"ticker": a["ticker"], "role": a["role"], "reasons": a["reasons"]}
                           for a in receipt["assets"] if not a["eligible"]]
    atomic(old_path, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return receipt


def load_frames(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    receipt = read_json(Path(root) / RECEIPT)
    if receipt is None:
        raise RuntimeError("missing or unreadable data receipt")
    validate_freeze(root)
    _validate_receipt_identity(receipt)
    targets, peers = {}, {}
    for rec in receipt["assets"]:
        if not rec["eligible"]:
            continue
        path = Path(root) / RAW_DIR / f"{rec['ticker']}_{START}_{END_EXCLUSIVE}.csv"
        blob = path.read_bytes()
        if sha256(blob) != rec["sha256"]:
            raise RuntimeError(f"{rec['ticker']}: raw file changed after data receipt")
        frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        (targets if rec["role"] == "target" else peers)[rec["ticker"]] = normalized_ohlcv(frame)
    return targets, peers


def eligible_targets(receipt: dict) -> list[str]:
    return [a["ticker"] for a in receipt["assets"] if a["role"] == "target" and a["eligible"]]


def eligible_peers(receipt: dict) -> list[str]:
    return [a["ticker"] for a in receipt["assets"] if a["role"] == "peer" and a["eligible"]]


def partition_dates(frame: pd.DataFrame, part: str) -> pd.DatetimeIndex:
    if part not in PARTITIONS:
        raise ValueError(part)
    start, end = PARTITIONS[part]
    dates = frame.index[(frame.index.year >= start) & (frame.index.year <= end)]
    if part == "test":
        dates = dates[dates <= pd.Timestamp(TEST_END)]
    return pd.DatetimeIndex(dates)


def split_frame(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    parts = {}
    for part in PARTITIONS:
        dates = partition_dates(frame, part)
        if len(dates) < 100:
            raise ValueError(f"short {FOLD_TAG}/{part}: {len(dates)} rows")
        parts[part] = frame.loc[dates, list(C.FEATURES)].to_numpy(dtype=float, copy=True)
    return parts


def prepare_data(root: Path) -> tuple[dict, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    receipt = acquire(root)
    if provider_failures(receipt):
        raise RuntimeError("provider errors remain; rerun the data stage before methods")
    if receipt["eligible_target_count"] < 12 or receipt["eligible_peer_count"] < 5:
        raise RuntimeError("fewer than 12 held-out targets or 5 eligible peers; stop and report QC")
    targets, peers = load_frames(root)
    return receipt, targets, peers
