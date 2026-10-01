"""Offline unit checks for the acquisition/QC logic (no network needed)."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def load_module():
    for name in ("exchange_calendars", "yfinance"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            sys.modules[name].__version__ = "test-stub"
    path = Path(__file__).with_name("acquire_and_qc.py")
    spec = importlib.util.spec_from_file_location("acquire_and_qc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_module()


class AcquisitionQCTests(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2015-01-01", "2025-12-31")
        self.frame = pd.DataFrame({
            "Open": 100.0, "High": 101.0, "Low": 99.0,
            "Close": 100.0, "Volume": 1000.0,
        }, index=self.sessions)

    def test_full_panel_is_eligible(self):
        clean = mod.normalized_ohlcv(self.frame)
        result = mod.assess_quality(clean, self.sessions)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["longest_absent_us_session_run"], 0)

    def test_five_missing_sessions_allowed_but_six_not(self):
        five = self.frame.drop(index=self.sessions[500:505])
        six = self.frame.drop(index=self.sessions[500:506])
        self.assertTrue(mod.assess_quality(mod.normalized_ohlcv(five),
                                           self.sessions)["eligible"])
        verdict = mod.assess_quality(mod.normalized_ohlcv(six), self.sessions)
        self.assertFalse(verdict["eligible"])
        self.assertIn("more_than_five_consecutive_absent_us_sessions",
                      verdict["reasons"])

    def test_bad_price_and_duplicate_date_are_rejected(self):
        bad = self.frame.copy()
        bad.loc[self.sessions[100], "Low"] = np.nan
        verdict = mod.assess_quality(mod.normalized_ohlcv(bad), self.sessions)
        self.assertIn("nonfinite_ohlcv", verdict["reasons"])
        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]]).sort_index()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            mod.normalized_ohlcv(duplicate)


if __name__ == "__main__":
    unittest.main()
