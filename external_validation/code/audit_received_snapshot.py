"""Audit received private OHLCV ZIP in memory; export summaries only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main(receipt_zip: Path, private_zip: Path, output: Path) -> None:
    with zipfile.ZipFile(receipt_zip) as public, zipfile.ZipFile(private_zip) as private:
        if public.testzip() is not None or private.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        public_manifest = public.read("asset_universe_v1.json")
        private_manifest = private.read("asset_universe_v1.json")
        public_receipt = public.read("acquisition_qc_receipt.json")
        private_receipt = private.read("acquisition_qc_receipt.json")
        if public_manifest != private_manifest or public_receipt != private_receipt:
            raise ValueError("public and private manifests/receipts differ")

        manifest = json.loads(public_manifest)
        receipt = json.loads(public_receipt)
        if digest(public_manifest) != receipt["manifest_sha256"]:
            raise ValueError("manifest SHA-256 differs from receipt")
        if receipt["protocol_version"] != manifest["protocol_version"]:
            raise ValueError("protocol version mismatch")
        if [entry["ticker"] for entry in manifest["assets"]] != [
                entry["ticker"] for entry in receipt["assets"]]:
            raise ValueError("asset order differs")

        summaries = []
        clean_hashes = []
        for record in receipt["assets"]:
            ticker = record["ticker"]
            filename = f"{ticker}_2015-01-01_2026-01-01.csv"
            raw_blob = private.read(f"raw_vendor_cache_do_not_publish/{filename}")
            clean_blob = private.read(f"clean_vendor_cache_do_not_publish/{filename}")
            if digest(raw_blob) != record["raw_sha256"]:
                raise ValueError(f"{ticker} raw SHA-256 mismatch")
            if digest(clean_blob) != record["clean_sha256"]:
                raise ValueError(f"{ticker} clean SHA-256 mismatch")
            if raw_blob != clean_blob:
                raise ValueError(f"{ticker} raw/clean cache differ unexpectedly")
            frame = pd.read_csv(io.BytesIO(clean_blob), index_col=0,
                                parse_dates=True, float_precision="round_trip")
            if len(frame) != record["row_count"]:
                raise ValueError(f"{ticker} row-count mismatch")
            if frame.columns.tolist() != ["open", "high", "low", "close", "volume"]:
                raise ValueError(f"{ticker} OHLCV columns mismatch")
            if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                raise ValueError(f"{ticker} dates invalid")
            if not np.isfinite(frame.to_numpy(float)).all():
                raise ValueError(f"{ticker} nonfinite data")
            if (frame[["open", "high", "low", "close"]] <= 0).any().any():
                raise ValueError(f"{ticker} nonpositive price")
            if (frame["volume"] < 0).any():
                raise ValueError(f"{ticker} negative volume")
            years = frame.index.year
            splits = {
                "A": (int(((years >= 2015) & (years <= 2018)).sum()),
                      int((years == 2019).sum()),
                      int(((years >= 2020) & (years <= 2021)).sum())),
                "B": (int(((years >= 2015) & (years <= 2020)).sum()),
                      int((years == 2021).sum()),
                      int(((years >= 2022) & (years <= 2023)).sum())),
                "C": (int(((years >= 2015) & (years <= 2022)).sum()),
                      int((years == 2023).sum()),
                      int(((years >= 2024) & (years <= 2025)).sum())),
            }
            for label, (train_n, val_n, test_n) in splits.items():
                if min(train_n, val_n, test_n) <= 60:
                    raise ValueError(f"{ticker} fold {label} too short")
            summaries.append({
                "ticker": ticker,
                "country": record["country"],
                "region": record["region"],
                "rows": len(frame),
                "start": frame.index.min().strftime("%Y-%m-%d"),
                "end": frame.index.max().strftime("%Y-%m-%d"),
                "zero_volume_rows": int((frame["volume"] == 0).sum()),
                "zero_volume_dates": [date.strftime("%Y-%m-%d") for date in
                                      frame.index[frame["volume"] == 0]],
                "fold_counts_train_val_test": splits,
                "sha256": record["clean_sha256"],
            })
            clean_hashes.append(record["clean_sha256"])

        if len(set(clean_hashes)) != len(clean_hashes):
            raise ValueError("duplicate clean data files across tickers")
        if len(summaries) != len(manifest["assets"]):
            raise ValueError("missing asset in private snapshot")
        if receipt["eligible_count"] != len(summaries) or receipt["excluded_count"]:
            raise ValueError("receipt eligibility disagrees with snapshot")

        report = {
            "snapshot_audit": "PASS",
            "asset_count": len(summaries),
            "public_receipt_sha256": digest(public_receipt),
            "private_zip_sha256": digest(private_zip.read_bytes()),
            "data_files_verified": 2 * len(summaries),
            "rows_total_clean": sum(x["rows"] for x in summaries),
            "zero_volume_total": sum(x["zero_volume_rows"] for x in summaries),
            "assets": summaries,
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
                          if key != "assets"}, indent=2))
        print("Summary written to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_zip", type=Path)
    parser.add_argument("private_zip", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.receipt_zip, args.private_zip, args.output)
