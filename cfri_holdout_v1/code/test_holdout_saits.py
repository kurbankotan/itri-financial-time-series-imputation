"""Focused tests for SAITS holdout identity, masks and peer comparator wiring.

The lightweight local test stubs scikit-learn only when it is absent; Colab
installs the real library.  The expensive neural fit is replaced by a fixed
masked-input interpolation, so this does not claim to reproduce SAITS scores.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


if importlib.util.find_spec("sklearn") is None:
    names = ("sklearn", "sklearn.exceptions", "sklearn.experimental",
             "sklearn.experimental.enable_iterative_imputer", "sklearn.impute",
             "sklearn.linear_model", "sklearn.preprocessing")
    for name in names:
        module = types.ModuleType(name)
        if name in ("sklearn", "sklearn.experimental"):
            module.__path__ = []
        sys.modules[name] = module
    for module, symbol in (("sklearn.exceptions", "ConvergenceWarning"),
                           ("sklearn.impute", "IterativeImputer"),
                           ("sklearn.linear_model", "Ridge"),
                           ("sklearn.preprocessing", "StandardScaler")):
        setattr(sys.modules[module], symbol, type(symbol, (), {}))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cfri_core as C  # noqa: E402
import holdout_data as D  # noqa: E402
import holdout_saits as H  # noqa: E402


def synthetic_frame(rows: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=rows)
    t = np.arange(rows, dtype=float)
    close = 100 + 0.1 * t + 0.1 * np.sin(t / 5)
    return pd.DataFrame({"open": close * 1.001, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1_000_000 + 1000 * t}, index=dates)


class HoldoutSaitsTests(unittest.TestCase):
    def test_seed_is_deterministic_and_new_fold(self):
        first = H.training_seed("XLB", 5, 0.2, 23, "feature")
        self.assertEqual(first, H.training_seed("XLB", 5, 0.2, 23, "feature"))
        self.assertNotEqual(first, H.training_seed("XLB", 5, 0.2, 23, "row"))
        self.assertNotEqual(first, H.S.training_seed("XLB", 5, 0.2, 23, "feature"))

    def test_manifest_binds_peer_frames_and_verified_condition_detects_tamper(self):
        frame = synthetic_frame()
        peer = synthetic_frame()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            D.atomic(root / D.RECEIPT, b'{"study":"synthetic"}\n')
            manifest = H._manifest(root, {"XLB": frame}, {"EWC": peer},
                                   {"python": "synthetic", "torch": "synthetic"})
            self.assertEqual(manifest["fold"], "H1")
            fingerprint = H.condition_fingerprint(manifest, "XLB", "row", 1, 0.1, 11)
            path = H.condition_path(root, "XLB", "row", 1, 0.1, 11)
            payload = {"fold": "H1", "mask_type": "row",
                       "scores": {name: {} for name in H.METHODS_ROW},
                       "mask_sha256": {"train": "a", "val": "b", "test": "c"}}
            blob = H._json_bytes(payload)
            H._atomic(path, blob)
            H._atomic(path.with_name(path.name + ".done.json"), H._json_bytes({
                "fingerprint": fingerprint, "sha256": H._sha256(blob),
                "mask_sha256": payload["mask_sha256"]}))
            self.assertTrue(H.verified_condition(path, fingerprint, "row"))
            path.write_text(path.read_text() + " ")
            self.assertFalse(H.verified_condition(path, fingerprint, "row"))
            changed = peer.copy()
            changed.iloc[0, 3] += 1
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                H._manifest(root, {"XLB": frame}, {"EWC": changed},
                            {"python": "synthetic", "torch": "synthetic"})

    def test_same_masks_and_validation_only_peer_gain(self):
        frame = synthetic_frame()
        arrays = {"train": frame.iloc[:200].to_numpy(float),
                  "val": frame.iloc[200:300].to_numpy(float),
                  "test": frame.iloc[300:].to_numpy(float)}
        dates = {"train": frame.index[:200], "val": frame.index[200:300],
                 "test": frame.index[300:]}

        def fake_fit(train_in, val_in, val_truth_scaled, test_in, model_seed, device):
            self.assertTrue(np.isnan(train_in).any())
            self.assertTrue(np.isnan(val_in).any())
            self.assertTrue(np.isnan(test_in).any())
            self.assertTrue(np.isfinite(val_truth_scaled).all())
            return C.linear_fill(val_in), C.linear_fill(test_in), 0.01, 0

        original_select = H.P.select_peer_gain

        def check_validation(baseline_val, hidden_val, truth_val, *args, **kwargs):
            self.assertTrue(np.array_equal(truth_val, arrays["val"]))
            self.assertFalse(np.array_equal(truth_val, arrays["test"]))
            self.assertTrue(np.array_equal(hidden_val,
                C.condition_mask(arrays["val"].shape, "XLB", "H1", "val", 1, 0.1, 11, "row")))
            return original_select(baseline_val, hidden_val, truth_val, *args, **kwargs)

        with (patch.object(D, "split_frame", return_value=arrays),
              patch.object(D, "partition_dates", side_effect=lambda _frame, part: dates[part]),
              patch.object(H, "_fit_predict_both", side_effect=fake_fit),
              patch.object(H.P, "select_peer_gain", side_effect=check_validation)):
            result = H._one_condition(frame, {}, "XLB", "row", 1, 0.1, 11, "cpu")
        self.assertEqual(set(result["scores"]), set(H.METHODS_ROW))
        self.assertEqual(result["fold"], "H1")
        self.assertEqual(result["peer"]["gain"], 0.0)
        self.assertEqual(result["peer"]["target_only"]["used_rows"], 0)
        self.assertEqual(result["scores"]["saits_joint"], result["scores"]["saits_peer_joint"])
        self.assertTrue(all(value == 0.0 for value in result["ohlc_violation_share"].values()))
        self.assertEqual(result["mask_sha256"]["test"],
                         H._sha256(C.condition_mask(arrays["test"].shape, "XLB", "H1",
                                                     "test", 1, 0.1, 11, "row").tobytes()))
        json.dumps(result, allow_nan=False)

    def test_neural_recipe_predicts_validation_and_test_once_after_fit(self):
        class FakeSAITS:
            instances = []

            def __init__(self, **config):
                self.config = config
                self.predictions = 0
                self.instances.append(self)

            def fit(self, train, validation):
                self.assert_train = train
                self.assert_validation = validation

            def predict(self, inputs):
                self.predictions += 1
                return {"imputation": np.nan_to_num(inputs["X"], nan=0.0)}

        fake_package = types.ModuleType("pypots")
        fake_imputation = types.ModuleType("pypots.imputation")
        fake_imputation.SAITS = FakeSAITS
        fake_package.imputation = fake_imputation
        base = synthetic_frame(400).to_numpy(float)
        train = base[:140]
        val = base[140:270]
        test = base[270:]
        train_mask = np.zeros_like(train, dtype=bool)
        val_mask = np.zeros_like(val, dtype=bool)
        test_mask = np.zeros_like(test, dtype=bool)
        train_mask[30:33] = True
        val_mask[30:33] = True
        test_mask[30:33] = True
        train_in = C.corrupt(train, train_mask)
        val_in = C.corrupt(val, val_mask)
        test_in = C.corrupt(test, test_mask)
        with (patch.dict(sys.modules, {"pypots": fake_package, "pypots.imputation": fake_imputation}),
              patch.object(H.S, "set_training_seed")):
            got_val, got_test, elapsed, fallbacks = H._fit_predict_both(
                train_in, val_in, val, test_in, 123, "cpu")
        self.assertEqual(len(FakeSAITS.instances), 1)
        self.assertEqual(FakeSAITS.instances[0].predictions, 2)
        self.assertEqual(got_val.shape, val.shape)
        self.assertEqual(got_test.shape, test.shape)
        self.assertTrue(np.isfinite(got_val).all())
        self.assertTrue(np.isfinite(got_test).all())
        self.assertTrue(np.array_equal(got_val[~val_mask], val[~val_mask]))
        self.assertTrue(np.array_equal(got_test[~test_mask], test[~test_mask]))
        self.assertGreaterEqual(elapsed, 0)
        self.assertGreaterEqual(fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
