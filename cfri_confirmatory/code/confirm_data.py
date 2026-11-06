"""Confirmatory study, step 1: acquire daily OHLCV for the 34 ETFs (2015-01-01 to 2026-09-30) and apply the
score-blind quality rules of the external validation protocol v1 (external_validation/code/acquire_and_qc.py),
extended to the 2026 partial year.

Rules (unchanged from protocol v1 except for the end date): Yahoo Finance through yfinance 1.7.0 with
auto_adjust=True and actions=False; all five OHLCV columns; unique, strictly increasing dates inside the
requested interval; finite positive prices and finite nonnegative volume; valid OHLC order (relative tolerance
1e-6); at least 95% of the XNYS sessions of every calendar year 2015-2025 and of 2026-01-01..2026-09-30; at most
five consecutive absent XNYS sessions; no rows outside the XNYS calendar. An asset that fails is excluded and
logged, never replaced. No imputation method is run and no score is read at this step.

Raw files stay in the user's private Google Drive (never in the public repository); only hashes and QC results
are exported.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C

START, END_EXCLUSIVE, LAST_SESSION = "2015-01-01", "2026-10-01", "2026-09-30"
YEARS = tuple(range(2015, 2027))          # 2026 is the partial year up to LAST_SESSION
COLUMNS = C.FEATURES
RAW_DIR = "raw_do_not_publish"
RECEIPT = "data_receipt.json"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def max_true_run(flags) -> int:  # as in acquire_and_qc.py
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    lengths = edges[1::2] - edges[::2]
    return int(lengths.max()) if lengths.size else 0


def normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:  # as in acquire_and_qc.py, new end date
    if frame.empty:
        raise ValueError("empty provider response")
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            names = [str(n).lower().replace(" ", "_") for n in out.columns.get_level_values(level)]
            if all(n in names for n in COLUMNS):
                out.columns = names
                break
        else:
            raise ValueError("cannot locate OHLCV level in provider columns")
    else:
        out.columns = [str(n).lower().replace(" ", "_") for n in out.columns]
    missing = set(COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"missing provider columns: {sorted(missing)}")
    out = out.loc[:, list(COLUMNS)]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    if out.index.has_duplicates:
        raise ValueError("duplicate trading dates")
    if not out.index.is_monotonic_increasing:
        raise ValueError("trading dates not strictly increasing")
    if (out.index < pd.Timestamp(START)).any() or (out.index >= pd.Timestamp(END_EXCLUSIVE)).any():
        raise ValueError("provider returned dates outside requested interval")
    return out.astype(float)


def xnys_sessions() -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    s = pd.DatetimeIndex(xcals.get_calendar("XNYS").sessions_in_range(START, LAST_SESSION))
    if s.tz is not None:
        s = s.tz_localize(None)
    return s.normalize()


def assess_quality(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:  # as in acquire_and_qc.py
    reasons = []
    values = frame.loc[:, list(COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all():
        reasons.append("nonfinite_ohlcv")
    if np.isfinite(values).all():
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
    return {"eligible": not reasons, "reasons": sorted(set(reasons)), "row_count": len(frame),
            "first_date": frame.index.min().strftime("%Y-%m-%d"), "last_date": frame.index.max().strftime("%Y-%m-%d"),
            "sessions_2026": int((frame.index.year == 2026).sum()), "longest_absent_us_session_run": longest,
            "non_us_session_count": len(non_us), "yearly_coverage": yearly}


def methods_started(root: Path) -> bool:
    return any(p.is_file() for d in ("fold_D_cpu", "fold_D_saits") if (root / d).is_dir() for p in (root / d).iterdir())


def provider_failures(receipt: dict) -> list:
    return [a["ticker"] for a in receipt["assets"]
            if not a["eligible"] and any(r.startswith("acquisition_or_qc_error") for r in a["reasons"])]


def _download(yf, ticker: str, attempts: int = 3) -> pd.DataFrame:
    """Up to three attempts (pauses of 10 and 30 seconds; CFRI_RETRY_PAUSE_SCALE rescales them for offline tests)."""
    scale = float(os.environ.get("CFRI_RETRY_PAUSE_SCALE", "1"))
    for attempt in range(attempts):
        try:
            got = yf.download(ticker, start=START, end=END_EXCLUSIVE, auto_adjust=True, actions=False,
                              progress=False, threads=False)
            return normalized_ohlcv(got)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(scale * (10, 30)[attempt])


def acquire(root: Path) -> dict:
    """Download (or reuse the Drive cache), check, and write data_receipt.json.

    Once any method output exists the receipt is final and is returned unchanged. Before that, a repeated call
    re-attempts only assets that failed with a download or provider error (they have no cached file); cached files
    are reused with their original retrieval record, so every other QC decision is reproduced exactly."""
    import exchange_calendars as xcals
    import yfinance as yf
    if (root / RECEIPT).exists():
        old = json.loads((root / RECEIPT).read_text())
        if methods_started(root) or not provider_failures(old) or old.get("provider_failures_accepted"):
            print("data_receipt.json exists and is final:", old["eligible_count"], "eligible assets", flush=True)
            return old
        print("retrying provider failures before any method has run:", provider_failures(old), flush=True)
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    sessions = xnys_sessions()
    receipt = {"study": "CFRI confirmatory fold D", "start_inclusive": START, "end_exclusive": END_EXCLUSIVE,
               "provider": "Yahoo Finance via yfinance; auto_adjust=True; actions=False",
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                               "yfinance": yf.__version__, "exchange_calendars": xcals.__version__},
               "code_id": C.CODE_ID, "xnys_sessions_2026_to_cutoff": int((sessions.year == 2026).sum()),
               "acquisition_started_utc": datetime.now(timezone.utc).isoformat(), "assets": []}
    for i, (ticker, country, region) in enumerate(C.ASSETS, 1):
        path = raw / f"{ticker}_{START}_{END_EXCLUSIVE}.csv"
        meta_path = path.with_suffix(".meta.json")
        rec = {"ticker": ticker, "country": country, "region": region, "sequence": i}
        try:
            if path.exists():
                frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                rec["source"] = "drive_cache"
                rec["retrieved_utc"] = meta.get("retrieved_utc")
            else:
                frame = _download(yf, ticker)
                retrieved = datetime.now(timezone.utc).isoformat()
                payload = frame.to_csv(float_format="%.17g", index_label="date").encode()
                _atomic(path, payload)
                _atomic(meta_path, json.dumps({"ticker": ticker, "retrieved_utc": retrieved,
                                               "yfinance": yf.__version__, "sha256": sha256_bytes(payload)}).encode())
                rec["source"] = "fresh_yfinance_download"
                rec["retrieved_utc"] = retrieved
            rec["sha256"] = sha256_bytes(path.read_bytes())
            frame = normalized_ohlcv(frame)
            rec.update(assess_quality(frame, sessions))
        except Exception as exc:  # logged exclusion, never a replacement
            rec.update({"eligible": False, "reasons": [f"acquisition_or_qc_error:{type(exc).__name__}:{exc}"]})
        receipt["assets"].append(rec)
        print(f"[{i:02d}/34] {ticker}: {'eligible' if rec['eligible'] else 'EXCLUDED ' + str(rec['reasons'])}",
              flush=True)
    receipt["acquisition_finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["eligible_count"] = sum(a["eligible"] for a in receipt["assets"])
    receipt["excluded"] = [{"ticker": a["ticker"], "reasons": a["reasons"]} for a in receipt["assets"] if not a["eligible"]]
    receipt["provider_failures_accepted"] = False
    _atomic(root / RECEIPT, json.dumps(receipt, ensure_ascii=False, indent=2).encode())
    return receipt


def accept_provider_failures(root: Path) -> dict:
    """Record that repeated provider failures are accepted as exclusions (score-blind; before any method runs)."""
    receipt = json.loads((root / RECEIPT).read_text())
    if methods_started(root):
        raise RuntimeError("methods have already run; the receipt is final")
    receipt["provider_failures_accepted"] = True
    receipt["provider_failures_accepted_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic(root / RECEIPT, json.dumps(receipt, ensure_ascii=False, indent=2).encode())
    return receipt


def load_frames(root: Path) -> dict:
    """Eligible assets from data_receipt.json, each CSV verified against its recorded SHA-256."""
    receipt = json.loads((root / RECEIPT).read_text())
    frames = {}
    for rec in receipt["assets"]:
        if not rec["eligible"]:
            continue
        path = root / RAW_DIR / f"{rec['ticker']}_{START}_{END_EXCLUSIVE}.csv"
        blob = path.read_bytes()
        if sha256_bytes(blob) != rec["sha256"]:
            raise ValueError(f"{rec['ticker']}: data file changed after the receipt was written")
        frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
        frames[rec["ticker"]] = normalized_ohlcv(frame)
    return frames
