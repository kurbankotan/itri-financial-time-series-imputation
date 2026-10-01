#!/usr/bin/env python3
"""Acquire the frozen ETF universe and log objective, score-blind QC.

This stage never runs an imputation model or reads any imputation scores.
Historical vendor data may change; the output receipt records SHA-256 hashes.
Do not upload raw Yahoo data publicly without checking redistribution rights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf


COLUMNS = ("open", "high", "low", "close", "volume")
START = "2015-01-01"
END_EXCLUSIVE = "2026-01-01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_true_run(flags: np.ndarray) -> int:
    flags = np.asarray(flags, dtype=bool)
    if flags.size == 0:
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    lengths = edges[1::2] - edges[::2]
    return int(lengths.max()) if lengths.size else 0


def normalized_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one Yahoo ticker; reject ambiguous or duplicated dates."""
    if frame.empty:
        raise ValueError("empty provider response")
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # Single-ticker yfinance typically puts Price on level 0.
        for level in range(out.columns.nlevels):
            names = [str(name).lower().replace(" ", "_")
                     for name in out.columns.get_level_values(level)]
            if all(name in names for name in COLUMNS):
                out.columns = names
                break
        else:
            raise ValueError("cannot locate OHLCV level in provider columns")
    else:
        out.columns = [str(name).lower().replace(" ", "_")
                       for name in out.columns]
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
    if (out.index < pd.Timestamp(START)).any() or \
            (out.index >= pd.Timestamp(END_EXCLUSIVE)).any():
        raise ValueError("provider returned dates outside requested interval")
    return out.astype(float)


def assess_quality(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    """Apply prespecified session coverage and OHLCV integrity thresholds."""
    reasons: list[str] = []
    values = frame.loc[:, list(COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all():
        reasons.append("nonfinite_ohlcv")
    if np.isfinite(values).all():
        if (frame.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
            reasons.append("nonpositive_price")
        if (frame["volume"] < 0).any():
            reasons.append("negative_volume")
        tolerance = 1e-6 * frame.loc[:, ("open", "high", "low", "close")].max(axis=1)
        if ((frame["high"] + tolerance < frame[["open", "low", "close"]].max(axis=1)) |
                (frame["low"] - tolerance > frame[["open", "high", "close"]].min(axis=1))).any():
            reasons.append("invalid_ohlc_order")

    observed = pd.DatetimeIndex(frame.index)
    missing = ~sessions.isin(observed)
    non_us_sessions = observed.difference(sessions)
    yearly: dict[str, dict] = {}
    for year in range(2015, 2026):
        expected_year = sessions[sessions.year == year]
        present = int(expected_year.isin(observed).sum())
        coverage = present / len(expected_year)
        yearly[str(year)] = {
            "expected_sessions": len(expected_year),
            "observed_sessions": present,
            "coverage": coverage,
        }
        if coverage < 0.95:
            reasons.append(f"coverage_below_95pct_{year}")
    longest_gap = max_true_run(missing)
    if longest_gap > 5:
        reasons.append("more_than_five_consecutive_absent_us_sessions")
    if len(non_us_sessions):
        reasons.append("rows_outside_us_exchange_calendar")
    return {
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "row_count": len(frame),
        "first_date": frame.index.min().strftime("%Y-%m-%d"),
        "last_date": frame.index.max().strftime("%Y-%m-%d"),
        "longest_absent_us_session_run": longest_gap,
        "non_us_session_count": len(non_us_sessions),
        "yearly_coverage": yearly,
    }


def run(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["data_start_inclusive"] != START or \
            manifest["data_end_exclusive"] != END_EXCLUSIVE:
        raise ValueError("manifest dates disagree with acquisition code")
    assets = manifest["assets"]
    tickers = [entry["ticker"] for entry in assets]
    if len(tickers) != len(set(tickers)):
        raise ValueError("duplicate symbols in locked manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_vendor_cache_do_not_publish"
    clean_dir = output_dir / "clean_vendor_cache_do_not_publish"
    raw_dir.mkdir(exist_ok=True)
    clean_dir.mkdir(exist_ok=True)
    cal = xcals.get_calendar("XNYS")
    sessions = pd.DatetimeIndex(cal.sessions_in_range(START, "2025-12-31"))
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    sessions = sessions.normalize()
    result = {
        "protocol_version": manifest["protocol_version"],
        "manifest_sha256": sha256_file(manifest_path),
        "acquisition_started_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Yahoo Finance via yfinance; auto_adjust=True; actions=False",
        "environment": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "yfinance": yf.__version__,
            "exchange_calendars": xcals.__version__,
        },
        "assets": [],
    }
    for index, entry in enumerate(assets, start=1):
        ticker = entry["ticker"]
        raw_path = raw_dir / f"{ticker}_{START}_{END_EXCLUSIVE}.csv"
        clean_path = clean_dir / f"{ticker}_{START}_{END_EXCLUSIVE}.csv"
        record = {**entry, "sequence": index}
        try:
            if raw_path.exists():
                frame = pd.read_csv(raw_path, index_col=0, parse_dates=True,
                                    float_precision="round_trip")
                record["source"] = "immutable_local_cache"
            else:
                frame = yf.download(ticker, start=START, end=END_EXCLUSIVE,
                                    auto_adjust=True, actions=False,
                                    progress=False, threads=False)
                record["source"] = "fresh_yfinance_download"
                record["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
                frame = normalized_ohlcv(frame)
                frame.to_csv(raw_path, float_format="%.17g", index_label="date")
            record["raw_sha256"] = sha256_file(raw_path)
            frame = normalized_ohlcv(frame)
            record.update(assess_quality(frame, sessions))
            if record["eligible"]:
                if clean_path.exists():
                    existing = pd.read_csv(clean_path, index_col=0,
                                           parse_dates=True,
                                           float_precision="round_trip")
                    if not frame.equals(existing):
                        raise ValueError("clean cache conflict; refusing overwrite")
                else:
                    frame.to_csv(clean_path, float_format="%.17g", index_label="date")
                record["clean_sha256"] = sha256_file(clean_path)
        except Exception as exc:
            record.update({"eligible": False,
                           "reasons": [f"acquisition_or_qc_error:{type(exc).__name__}:{exc}"]})
        result["assets"].append(record)
        print(f"[{index:02d}/{len(assets)}] {ticker}: "
              f"{'eligible' if record['eligible'] else 'excluded'}")

    result["acquisition_finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["eligible_count"] = sum(entry["eligible"] for entry in result["assets"])
    result["excluded_count"] = len(assets) - result["eligible_count"]
    receipt_path = output_dir / "acquisition_qc_receipt.json"
    receipt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"QC receipt: {receipt_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent.parent / "metadata" / "asset_universe_v1.json")
    parser.add_argument("--out", type=Path, default=Path("expanded_validation_run"))
    args = parser.parse_args()
    run(args.manifest, args.out)
