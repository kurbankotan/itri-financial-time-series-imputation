"""Synthetic end-to-end checks for the frozen holdout analysis gate."""
from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import holdout_analyze as A
import build_notebook as B
import holdout_data as D
import holdout_run as R
import holdout_saits as S


class HoldoutAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.keys = [("XLB", "feature", 1, 0.1, 11),
                     ("XLB", "row", 1, 0.1, 11)]
        index = pd.date_range("2025-01-01", periods=30, freq="B")
        values = np.tile([10.0, 11.0, 9.0, 10.5, 100.0], (len(index), 1))
        self.frame = pd.DataFrame(values, index=index, columns=list(A.FEATURES))
        self.targets = {"XLB": self.frame}
        self.peers = {ticker: self.frame.copy() for ticker in D.PEERS[:5]}
        self.receipt = {
            "study": D.STUDY_ID, "start_inclusive": D.START,
            "end_exclusive": D.END_EXCLUSIVE,
            "target_tickers": list(D.TARGETS), "peer_tickers": list(D.PEERS),
            "acquisition_started_utc": "2026-10-28T20:00:00+00:00",
            "assets": [{"ticker": ticker, "role": role,
                        "eligible": ticker == "XLB" or ticker in self.peers,
                        "reasons": [] if ticker == "XLB" or ticker in self.peers
                        else ["synthetic_qc_exclusion"],
                        "sha256": ("a" if role == "target" else "b") * 64}
                       for ticker, role in D.CANDIDATES],
            "eligible_target_count": 1, "eligible_peer_count": 5,
            "excluded": [{"ticker": ticker, "role": role,
                          "reasons": ["synthetic_qc_exclusion"]}
                         for ticker, role in D.CANDIDATES
                         if ticker != "XLB" and ticker not in self.peers],
        }
        D.atomic(self.root / D.RECEIPT, (json.dumps(self.receipt) + "\n").encode())
        self._freeze()
        self._manifests()
        self._conditions()
        self.patches = [patch.object(A, "MIN_TARGETS", 1),
                        patch.object(A, "expected_keys", return_value=self.keys),
                        patch.object(D, "load_frames", return_value=(self.targets, self.peers))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def _freeze(self) -> None:
        bundle = self.root / "frozen_source_bundle.zip"
        manifest = {name: hashlib.sha256((B.HERE / name).read_bytes()).hexdigest()
                    for name in B.SOURCES}
        with zipfile.ZipFile(bundle, "w") as archive:
            for name in B.SOURCES:
                archive.write(B.HERE / name, name)
            archive.writestr("SOURCE_SHA256.json", json.dumps(manifest))
        freeze = {"embedded_bundle_sha256": A.sha256(bundle.read_bytes()),
                  "freeze_utc": "2026-10-28T19:00:00+00:00"}
        D.atomic(self.root / "freeze_receipt.json", (json.dumps(freeze) + "\n").encode())

    def _manifests(self) -> None:
        receipt_hash = A.sha256((self.root / D.RECEIPT).read_bytes())
        target_hash = {"XLB": R.frame_sha256(self.frame)}
        peer_hash = {ticker: R.frame_sha256(frame) for ticker, frame in self.peers.items()}
        self.cpu_manifest = {
            "study": D.STUDY_ID, "fold": D.FOLD_TAG,
            "code_sha256": R.code_hashes(), "data_receipt_sha256": receipt_hash,
            "target_frame_sha256": target_hash, "peer_frame_sha256": peer_hash,
            "settings": R.SETTINGS,
            "methods": {"feature": list(R.METHODS_FEATURE), "row": list(R.METHODS_ROW)},
        }
        self.saits_manifest = {
            "study": D.STUDY_ID, "fold": D.FOLD_TAG,
            "code_sha256": S._code_sha256(), "data_receipt_sha256": receipt_hash,
            "target_frame_sha256": target_hash, "peer_frame_sha256": peer_hash,
            "target_tickers": list(self.targets), "peer_tickers": list(self.peers),
            "methods_by_mask": {k: list(v) for k, v in S.METHODS_BY_MASK.items()},
            "environment": {"device": "cuda", "test": True}, "config": {"test": True},
        }
        for directory, data in ((R.OUT_DIR, self.cpu_manifest), (S.OUT_DIR, self.saits_manifest)):
            D.atomic(self.root / directory / "run_manifest.json", (json.dumps(data) + "\n").encode())

    @staticmethod
    def _scores(rmse: float) -> dict:
        return {feature: [rmse, rmse / 2, 1] for feature in A.FEATURES}

    def _write_condition(self, path: Path, fingerprint: str, methods: dict,
                         key: tuple, peer: dict | None = None) -> None:
        ticker, mask_type, block, rate, seed = key
        masks = {part: str(i) * 64 for i, part in enumerate(A.PARTS, 1)}
        data = {"ticker": ticker, "fold": D.FOLD_TAG,
                "mask_type": mask_type, "block_length": block, "rate": rate, "seed": seed,
                "mask_sha256": masks, "masked_cells": {part: 5 for part in A.PARTS},
                "scores": {method: self._scores(value) for method, value in methods.items()},
                "ohlc_violation_share": {method: 0.0 for method in methods},
                "peer": peer}
        payload = (json.dumps(data) + "\n").encode()
        side = {"fingerprint": fingerprint, "sha256": A.sha256(payload),
                "mask_sha256": masks}
        D.atomic(path, payload)
        D.atomic(path.with_name(path.name + ".done.json"), (json.dumps(side) + "\n").encode())

    def _conditions(self) -> None:
        for key in self.keys:
            mask_type = key[1]
            if mask_type == "feature":
                cpu_methods = {"cfri_old_proj": 0.3, "cfri_joint": 0.2}
                saits_methods = {"saits_joint": 0.25}
                cpu_peer = saits_peer = None
            else:
                cpu_methods = {"cfri_joint": 0.4,
                               "cfri_peer_target_joint": 0.3,
                               "cfri_peer_marketwide_joint": 0.4}
                saits_methods = {"saits_joint": 0.5, "saits_peer_joint": 0.35}
                cpu_peer = {"gain": 0.5, "model_fit_rows": 200,
                            "target_only": {"used_rows": 5}, "market_wide": {"used_rows": 0}}
                saits_peer = {"gain": 0.5, "model_fit_rows": 200,
                              "target_only": {"used_rows": 5}}
            self._write_condition(R.condition_path(self.root, *key),
                                  R._fingerprint(self.cpu_manifest, *key),
                                  cpu_methods, key, cpu_peer)
            self._write_condition(S.condition_path(self.root, *key),
                                  S.condition_fingerprint(self.saits_manifest, *key),
                                  saits_methods, key, saits_peer)

    def _rewrite(self, path: Path, mutate) -> None:
        data = json.loads(path.read_text())
        mutate(data)
        payload = (json.dumps(data) + "\n").encode()
        sidepath = path.with_name(path.name + ".done.json")
        side = json.loads(sidepath.read_text())
        side["sha256"] = A.sha256(payload)
        side["mask_sha256"] = data["mask_sha256"]
        D.atomic(path, payload)
        D.atomic(sidepath, (json.dumps(side) + "\n").encode())

    def test_complete_run_and_raw_free_export(self) -> None:
        output = A.analyze_and_export(self.root)
        result = json.loads((self.root / "holdout_results.json").read_text())
        self.assertEqual(result["assets"], 1)
        self.assertEqual(result["classification"], "asset_holdout_local_freeze")
        self.assertEqual(result["conditions_by_mask"], {"feature": 1, "row": 1})
        self.assertEqual({key: val["success"] for key, val in result["hypotheses"].items()},
                         {"H1": True, "H2": True, "H3": True})
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("freeze_receipt.json", archive.namelist())
            self.assertIn("frozen_source_bundle.zip", archive.namelist())
            self.assertNotIn("raw_do_not_publish/secret.csv", archive.namelist())

    def test_cpu_gpu_mask_mismatch_rejected(self) -> None:
        path = R.condition_path(self.root, *self.keys[0])
        self._rewrite(path, lambda data: data["mask_sha256"].update(test="c" * 64))
        with self.assertRaisesRegex(ValueError, "masks or counts differ"):
            A.load(self.root)

    def test_marketwide_peer_use_rejected(self) -> None:
        path = R.condition_path(self.root, *self.keys[1])
        self._rewrite(path, lambda data: data["peer"]["market_wide"].update(used_rows=1))
        with self.assertRaisesRegex(ValueError, "marketwide peer used"):
            A.load(self.root)

    def test_freeze_after_acquisition_rejected(self) -> None:
        path = self.root / "freeze_receipt.json"
        freeze = json.loads(path.read_text())
        freeze["freeze_utc"] = "2026-10-28T21:00:00+00:00"
        D.atomic(path, (json.dumps(freeze) + "\n").encode())
        with self.assertRaisesRegex(ValueError, "later than data acquisition"):
            A.load(self.root)

    def test_source_change_after_freeze_rejected(self) -> None:
        bundle = self.root / "frozen_source_bundle.zip"
        bundle.write_bytes(bundle.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA does not match the receipt"):
            A.load(self.root)


if __name__ == "__main__":
    unittest.main()
