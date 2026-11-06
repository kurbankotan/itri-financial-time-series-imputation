"""Offline synthetic tests; no Yahoo request or held-out target score is read."""
from __future__ import annotations

import json
import hashlib
import io
import math
import sys
import tempfile
import types
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import cfri_core as C
import build_notebook as B
import holdout_data as D
import holdout_run as R


def make_freeze(root: Path) -> None:
    manifest = {name: hashlib.sha256((B.HERE / name).read_bytes()).hexdigest()
                for name in B.SOURCES}
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in B.SOURCES:
            archive.write(B.HERE / name, name)
        archive.writestr("SOURCE_SHA256.json", json.dumps(manifest))
    payload = blob.getvalue()
    D.atomic(root / "frozen_source_bundle.zip", payload)
    receipt = {"embedded_bundle_sha256": hashlib.sha256(payload).hexdigest(),
               "freeze_utc": datetime.now(timezone.utc).isoformat()}
    D.atomic(root / "freeze_receipt.json", json.dumps(receipt).encode())


def frame_for_dates(dates: pd.DatetimeIndex, phase: float = 0.0) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n, dtype=float)
    close = np.exp(np.log(80.0 + phase) + 0.00015 * t + 0.015 * np.sin(t / 19.0 + phase))
    opened = close * (1.0 + 0.002 * np.sin(t / 7.0 + phase))
    high = np.maximum(opened, close) * 1.005
    low = np.minimum(opened, close) * 0.995
    volume = 1_000_000.0 * (1.0 + 0.2 * np.sin(t / 13.0 + phase))
    return pd.DataFrame({"open": opened, "high": high, "low": low,
                         "close": close, "volume": volume}, index=dates)


class HoldoutDataTests(unittest.TestCase):
    def test_freeze_required_before_any_data_request(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "freeze_receipt.json is required"):
                D.prepare_data(Path(temp))

    def test_tampered_source_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_freeze(root)
            path = root / "frozen_source_bundle.zip"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA does not match"):
                D.prepare_data(root)

    def test_score_blind_fresh_acquisition_cache_and_hashes(self):
        dates = pd.bdate_range(D.START, D.TEST_END)
        calls = []

        def download(ticker, **kwargs):
            calls.append(ticker)
            return frame_for_dates(dates, phase=float(len(calls)) / 10.0)

        yf = types.ModuleType("yfinance")
        yf.__version__ = "1.7.0"
        yf.download = download
        xcals = types.ModuleType("exchange_calendars")
        xcals.__version__ = "4.13.2"
        xcals.get_calendar = lambda name: types.SimpleNamespace(
            sessions_in_range=lambda start, end: pd.bdate_range(start, end))
        with tempfile.TemporaryDirectory() as temp, patch.dict(
                sys.modules, {"yfinance": yf, "exchange_calendars": xcals}):
            root = Path(temp)
            make_freeze(root)
            receipt, targets, peers = D.prepare_data(root)
            self.assertEqual(len(calls), 50)
            self.assertEqual(len(targets), 16)
            self.assertEqual(len(peers), 34)
            self.assertEqual(D.eligible_targets(receipt), list(D.TARGETS))
            self.assertEqual(D.eligible_peers(receipt), list(D.PEERS))
            self.assertEqual({a["source"] for a in receipt["assets"]}, {"fresh_yfinance_download"})
            again, _, _ = D.prepare_data(root)
            self.assertEqual(len(calls), 50)
            self.assertEqual(again, receipt)
            raw = root / D.RAW_DIR / f"{D.TARGETS[0]}_{D.START}_{D.END_EXCLUSIVE}.csv"
            raw.write_bytes(raw.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "raw file changed"):
                D.load_frames(root)

    def test_partition_dates_use_only_frozen_periods(self):
        dates = pd.bdate_range(D.START, "2026-10-15")
        frame = frame_for_dates(dates)
        parts = D.split_frame(frame)
        self.assertEqual(parts["train"].shape[0], len(dates[dates.year <= 2024]))
        self.assertEqual(parts["val"].shape[0], len(dates[dates.year == 2025]))
        self.assertEqual(parts["test"].shape[0], len(dates[(dates.year == 2026) &
                                                              (dates <= pd.Timestamp(D.TEST_END))]))
        self.assertEqual(D.FOLD_TAG, "H1")


class HoldoutRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.bdate_range(D.START, D.TEST_END)
        cls.target = frame_for_dates(dates, phase=0.2)
        cls.peers = {f"P{i}": frame_for_dates(dates, phase=0.1 * i) for i in range(1, 7)}

    def test_feature_and_row_scores_serialize_and_marketwide_falls_back(self):
        feature = R.one_condition(self.target, self.peers, "TEST", "feature", 1, 0.10, 11)
        row = R.one_condition(self.target, self.peers, "TEST", "row", 1, 0.10, 11)
        self.assertEqual(set(feature["scores"]), set(R.METHODS_FEATURE))
        self.assertEqual(set(row["scores"]), set(R.METHODS_ROW))
        self.assertEqual(feature["fold"], D.FOLD_TAG)
        self.assertEqual(row["fold"], D.FOLD_TAG)
        self.assertEqual(len(feature["mask_sha256"]), 3)
        self.assertEqual(len(row["mask_sha256"]), 3)
        self.assertEqual(row["scores"]["cfri_joint"],
                         row["scores"]["cfri_peer_marketwide_joint"])
        self.assertEqual(row["peer"]["market_wide"]["used_rows"], 0)
        for result in (feature, row):
            self.assertIsInstance(json.loads(R.json_bytes(result)), dict)
            for scores in result["scores"].values():
                for item in scores.values():
                    self.assertTrue(math.isfinite(item[0]))
                    self.assertGreater(item[2], 0)

    def test_manifest_checkpoint_resume_and_peer_hash_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            D.atomic(root / D.RECEIPT, b'{"synthetic":true}\n')
            targets = {"TEST": self.target}
            manifest = R.run_manifest(root, targets, self.peers)
            self.assertEqual(manifest["target_frame_sha256"]["TEST"], R.frame_sha256(self.target))
            self.assertEqual(len(manifest["peer_frame_sha256"]), len(self.peers))
            # The JSON manifest must round-trip without tuple/list mismatches.
            self.assertEqual(R.run_manifest(root, targets, self.peers), manifest)
            summary = R.run_cpu(root, targets, self.peers,
                                mask_types=("row",), max_new_conditions=1)
            self.assertEqual(summary["new"], 1)
            path = R.condition_path(root, "TEST", "row", 1, 0.10, 11)
            fingerprint = R._fingerprint(manifest, "TEST", "row", 1, 0.10, 11)
            self.assertTrue(R.verified_condition(path, fingerprint, R.METHODS_ROW))
            summary2 = R.run_cpu(root, targets, self.peers,
                                 mask_types=("row",), max_new_conditions=1)
            self.assertGreaterEqual(summary2["skipped"], 1)
            self.assertEqual(summary2["new"], 1)
            changed = {k: v.copy() for k, v in self.peers.items()}
            changed["P1"].iloc[0, 3] *= 1.00001
            with self.assertRaisesRegex(RuntimeError, "peer frames changed"):
                R.run_manifest(root, targets, changed)


if __name__ == "__main__":
    unittest.main()
