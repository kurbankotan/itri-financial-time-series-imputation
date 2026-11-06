"""Locked analysis and raw-free export for the CFRI held-out-asset study.

The analyzer refuses incomplete or modified checkpoints. It runs only after the
frozen source bundle, freeze receipt, private data receipt, and both method
manifests are present. All three prespecified comparisons are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cfri_core as C
import holdout_data as D
import holdout_run as R
import holdout_saits as S


HERE = Path(__file__).resolve().parent
FEATURES = tuple(C.FEATURES)
KEYS = ("ticker", "mask_type", "block_length", "rate", "seed")
PARTS = ("train", "val", "test")
MIN_TARGETS = 12
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261028
HYPOTHESES = {
    "H1": ("feature", "cfri_joint", "saits_joint"),
    "H2": ("row", "cfri_peer_target_joint", "cfri_joint"),
    "H3": ("row", "cfri_peer_target_joint", "saits_peer_joint"),
}
FROZEN_SOURCE_FILES = (
    "PROTOCOL.md", "cfri_core.py", "upstream/cfri_core.py", "confirm_saits.py", "holdout_data.py",
    "holdout_run.py", "holdout_saits.py", "holdout_analyze.py",
    "v2_peer.py", "v2_price.py",
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"missing or unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must include UTC timezone")
    return parsed


def verify_freeze(root: Path, data_receipt: dict) -> dict:
    """Check the local pre-data freeze and the actual embedded, raw-free source."""
    try:
        freeze = D.validate_freeze(root)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    frozen_at = _utc(freeze.get("freeze_utc"), "freeze_utc")
    acquired_at = _utc(data_receipt.get("acquisition_started_utc"), "acquisition_started_utc")
    if frozen_at > acquired_at:
        raise ValueError("freeze receipt is later than data acquisition")
    return freeze


def _manifest_code(manifest: dict, expected: dict[str, str], label: str) -> None:
    if manifest.get("code_sha256") != expected:
        raise ValueError(f"{label} code changed after checkpoint creation")
    for name, digest in expected.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{label} invalid code digest for {name}")


def verify_manifests(root: Path, receipt: dict, targets: dict, peers: dict) -> tuple[dict, dict]:
    cpu = read_json(root / R.OUT_DIR / "run_manifest.json")
    saits = read_json(root / S.OUT_DIR / "run_manifest.json")
    for label, manifest in (("CPU", cpu), ("SAITS", saits)):
        if manifest.get("study") != D.STUDY_ID or manifest.get("fold") != D.FOLD_TAG:
            raise ValueError(f"{label} manifest belongs to another study")
        if manifest.get("data_receipt_sha256") != sha256((root / D.RECEIPT).read_bytes()):
            raise ValueError(f"{label} receipt changed after checkpoints")
    _manifest_code(cpu, R.code_hashes(), "CPU")
    _manifest_code(saits, S._code_sha256(), "SAITS")
    actual_target = {ticker: R.frame_sha256(frame) for ticker, frame in sorted(targets.items())}
    actual_peer = {ticker: R.frame_sha256(frame) for ticker, frame in sorted(peers.items())}
    if cpu.get("target_frame_sha256") != actual_target or saits.get("target_frame_sha256") != actual_target:
        raise ValueError("CPU/SAITS target frames differ from verified raw cache")
    if cpu.get("peer_frame_sha256") != actual_peer or saits.get("peer_frame_sha256") != actual_peer:
        raise ValueError("CPU/SAITS peer frames differ from verified raw cache")
    if saits.get("target_tickers") != list(targets) or saits.get("peer_tickers") != list(peers):
        raise ValueError("SAITS target or peer ordering changed")
    if cpu.get("methods") != {"feature": list(R.METHODS_FEATURE), "row": list(R.METHODS_ROW)}:
        raise ValueError("CPU method set changed")
    if saits.get("methods_by_mask") != {key: list(value) for key, value in S.METHODS_BY_MASK.items()}:
        raise ValueError("SAITS method set changed")
    return cpu, saits


def expected_keys(targets: dict) -> list[tuple[str, str, int, float, int]]:
    """Recompute feasibility from actual target partition lengths, not a constant."""
    keys = []
    for ticker in D.TARGETS:
        if ticker not in targets:
            continue
        parts = D.split_frame(targets[ticker])
        lengths = {name: len(value) for name, value in parts.items()}
        for mask_type in C.MASK_TYPES:
            for block in C.BLOCKS:
                for rate in C.RATES:
                    if C.feasible(lengths, block, rate, mask_type):
                        for seed in C.SEEDS:
                            keys.append((ticker, mask_type, block, rate, seed))
    if not keys:
        raise ValueError("no feasible target conditions")
    return keys


def _assert_exact_files(root: Path, expected: list[tuple]) -> None:
    for module, directory in ((R, R.OUT_DIR), (S, S.OUT_DIR)):
        wanted = {module.condition_path(root, *key).name for key in expected}
        found = {path.name for path in (root / directory).glob("*.json")
                 if path.name != "run_manifest.json" and not path.name.endswith(".done.json")}
        if found != wanted:
            missing, extra = sorted(wanted - found), sorted(found - wanted)
            raise ValueError(f"{directory} incomplete/extraneous conditions: "
                             f"missing={missing[:3]} ({len(missing)}), extra={extra[:3]} ({len(extra)})")


def _checkpoint(path: Path, methods: tuple[str, ...]) -> dict:
    data, side = read_json(path), read_json(path.with_name(path.name + ".done.json"))
    if side.get("sha256") != sha256(path.read_bytes()):
        raise ValueError(f"checkpoint SHA differs: {path}")
    if data.get("mask_sha256") != side.get("mask_sha256"):
        raise ValueError(f"checkpoint mask digest differs: {path}")
    if set(data.get("scores", {})) != set(methods):
        raise ValueError(f"checkpoint method set differs: {path}")
    for part in PARTS:
        digest = data.get("mask_sha256", {}).get(part)
        count = data.get("masked_cells", {}).get(part)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid {part} mask digest: {path}")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"invalid {part} masked-cell count: {path}")
    if set(data.get("ohlc_violation_share", {})) != set(methods):
        raise ValueError(f"checkpoint OHLC diagnostic set differs: {path}")
    for method in methods:
        by_feature = data["scores"][method]
        if set(by_feature) != set(FEATURES):
            raise ValueError(f"feature score set differs: {path}, {method}")
        counts = 0
        for feature in FEATURES:
            values = by_feature[feature]
            if (not isinstance(values, (list, tuple)) or len(values) != 3 or
                    not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
                            for x in values[:2]) or
                    not isinstance(values[2], int) or values[2] <= 0):
                raise ValueError(f"invalid feature score: {path}, {method}, {feature}")
            counts += values[2]
        if counts != data["masked_cells"]["test"]:
            raise ValueError(f"feature score counts disagree with test mask: {path}, {method}")
        violation = data["ohlc_violation_share"][method]
        if not isinstance(violation, (int, float)) or not math.isfinite(violation) or abs(violation) > 1e-9:
            raise ValueError(f"OHLC violation after projection: {path}, {method}")
    return data


def _identity(data: dict, key: tuple, path: Path) -> None:
    ticker, mask_type, block, rate, seed = key
    if (data.get("ticker") != ticker or data.get("fold") != D.FOLD_TAG or
            data.get("mask_type") != mask_type or data.get("block_length") != block or
            not isinstance(data.get("rate"), (int, float)) or
            not math.isclose(data["rate"], rate, rel_tol=0, abs_tol=1e-12) or
            data.get("seed") != seed):
        raise ValueError(f"condition identity differs: {path}")


def _validate_marketwide(cpu: dict, path: Path) -> None:
    if cpu["scores"]["cfri_peer_marketwide_joint"] != cpu["scores"]["cfri_joint"]:
        raise ValueError(f"marketwide peer score must equal no-peer score: {path}")
    meta = cpu.get("peer") or {}
    wide = meta.get("market_wide") or {}
    if wide.get("used_rows") != 0:
        raise ValueError(f"marketwide peer used a hidden row: {path}")


def _score_rows(key: tuple, result: dict) -> tuple[list[dict], list[dict]]:
    condition_rows, feature_rows = [], []
    identity = dict(zip(KEYS, key))
    for method, scores in result["scores"].items():
        rmse = [float(scores[name][0]) for name in FEATURES]
        mae = [float(scores[name][1]) for name in FEATURES]
        condition_rows.append({**identity, "method": method,
                               "nrmse": float(np.mean(rmse)), "nmae": float(np.mean(mae)),
                               "price_nrmse": float(np.mean(rmse[:4])), "volume_nrmse": rmse[4],
                               "ohlc_violation_share": float(result["ohlc_violation_share"][method])})
        for feature in FEATURES:
            x = scores[feature]
            feature_rows.append({**identity, "method": method, "feature": feature,
                                 "rmse": float(x[0]), "mae": float(x[1]),
                                 "masked_cells": int(x[2])})
    return condition_rows, feature_rows


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple], dict, dict]:
    """Verify the entire run before returning any test score to analysis."""
    root = Path(root)
    receipt = read_json(root / D.RECEIPT)
    D._validate_receipt_identity(receipt)
    freeze = verify_freeze(root, receipt)
    if D.provider_failures(receipt):
        raise ValueError("provider errors remain in data receipt")
    targets, peers = D.load_frames(root)  # rechecks every eligible private raw SHA
    if len(targets) < MIN_TARGETS or len(peers) < 5:
        raise ValueError("fewer than 12 eligible targets or five eligible peers; inconclusive QC run")
    if list(targets) != D.eligible_targets(receipt) or list(peers) != D.eligible_peers(receipt):
        raise ValueError("loaded frames differ from receipt eligibility/order")
    cpu_manifest, saits_manifest = verify_manifests(root, receipt, targets, peers)
    expected = expected_keys(targets)
    _assert_exact_files(root, expected)
    condition_rows: list[dict] = []
    feature_rows: list[dict] = []
    tuning_rows: list[dict] = []
    for key in expected:
        ticker, mask_type, block, rate, seed = key
        cp, sp = R.condition_path(root, *key), S.condition_path(root, *key)
        cm = R.METHODS_FEATURE if mask_type == "feature" else R.METHODS_ROW
        sm = S.METHODS_BY_MASK[mask_type]
        if not R.verified_condition(cp, R._fingerprint(cpu_manifest, *key), cm):
            raise ValueError(f"CPU checkpoint/fingerprint failed: {cp}")
        if not S.verified_condition(sp, S.condition_fingerprint(saits_manifest, *key), mask_type):
            raise ValueError(f"SAITS checkpoint/fingerprint failed: {sp}")
        cpu, saits = _checkpoint(cp, cm), _checkpoint(sp, sm)
        for result, path in ((cpu, cp), (saits, sp)):
            _identity(result, key, path)
        if (cpu["mask_sha256"] != saits["mask_sha256"] or
                cpu["masked_cells"] != saits["masked_cells"]):
            raise ValueError(f"CPU/SAITS masks or counts differ: {key}")
        for feature in FEATURES:
            cpu_n = cpu["scores"][cm[0]][feature][2]
            saits_n = saits["scores"][sm[0]][feature][2]
            if cpu_n != saits_n:
                raise ValueError(f"CPU/SAITS hidden feature count differs: {key}, {feature}")
        if mask_type == "row":
            _validate_marketwide(cpu, cp)
        for result in (cpu, saits):
            crows, frows = _score_rows(key, result)
            condition_rows.extend(crows)
            feature_rows.extend(frows)
        cpu_peer = cpu.get("peer") or {}
        saits_peer = saits.get("peer") or {}
        tuning_rows.append({**dict(zip(KEYS, key)),
                            "cfri_peer_gain": cpu_peer.get("gain"),
                            "cfri_peer_model_fit_rows": cpu_peer.get("model_fit_rows", 0),
                            "cfri_peer_used_rows": (cpu_peer.get("target_only") or {}).get("used_rows", 0),
                            "cfri_marketwide_peer_used_rows": (cpu_peer.get("market_wide") or {}).get("used_rows", 0),
                            "saits_peer_gain": saits_peer.get("gain"),
                            "saits_peer_model_fit_rows": saits_peer.get("model_fit_rows", 0),
                            "saits_peer_used_rows": (saits_peer.get("target_only") or {}).get("used_rows", 0)})
    cond, feature, tuning = (pd.DataFrame(rows) for rows in (condition_rows, feature_rows, tuning_rows))
    if cond.empty or cond.duplicated(list(KEYS) + ["method"]).any():
        raise ValueError("missing or duplicate condition-level scores")
    return cond, feature, tuning, expected, receipt, freeze


def paired(asset: pd.DataFrame, method: str, comparator: str, metric: str) -> dict:
    method_score = asset[(metric, method)]
    comparator_score = asset[(metric, comparator)]
    if method_score.isna().any() or comparator_score.isna().any():
        raise ValueError(f"incomplete asset pairing: {method}/{comparator}/{metric}")
    difference = method_score.to_numpy(dtype=float) - comparator_score.to_numpy(dtype=float)
    n = len(difference)
    if n < 1:
        raise ValueError("empty paired asset comparison")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = difference[rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))].mean(axis=1)
    other_mean = float(comparator_score.mean())
    return {"method": method, "comparator": comparator, "metric": metric,
            "assets": n, "mean_method": float(method_score.mean()),
            "mean_comparator": other_mean,
            "mean_difference": float(difference.mean()),
            "relative_difference_pct": float(100 * difference.mean() / other_mean) if other_mean else None,
            "method_lower_in_assets": int(np.sum(difference < 0)),
            "boot95_low": float(np.percentile(draws, 2.5)),
            "boot95_high": float(np.percentile(draws, 97.5))}


def _by_asset(cond: pd.DataFrame, mask_type: str) -> pd.DataFrame:
    one = cond.loc[cond.mask_type == mask_type]
    counts = one.groupby(["ticker", "method"]).size().unstack("method")
    if counts.isna().any().any() or counts.nunique(axis=1).max() != 1:
        raise ValueError(f"method coverage differs in {mask_type}")
    return one.groupby(["ticker", "method"])[["nrmse", "nmae", "price_nrmse", "volume_nrmse"]].mean().unstack("method")


def summarize(cond: pd.DataFrame, tuning: pd.DataFrame, expected: list[tuple],
              receipt: dict, freeze: dict) -> dict:
    n = len(D.eligible_targets(receipt))
    if n < MIN_TARGETS:
        raise ValueError("not enough eligible targets for prespecified decision")
    threshold = math.ceil(0.75 * n)
    out = {"study": D.STUDY_ID, "classification": "asset_holdout_local_freeze",
           "scope": "new target assets; January–September 2026 calendar shared with development fold D",
           "freeze": freeze, "eligible_targets": D.eligible_targets(receipt),
           "eligible_peers": D.eligible_peers(receipt), "excluded_assets": receipt.get("excluded", []),
           "assets": n, "target_win_threshold": threshold,
           "conditions": len(expected), "conditions_by_mask": {}, "mask_types": {}, "hypotheses": {}}
    assets = {}
    for mask_type in C.MASK_TYPES:
        one = cond.loc[cond.mask_type == mask_type]
        per_method = one.groupby("method").size()
        expected_count = sum(key[1] == mask_type for key in expected)
        if per_method.empty or not (per_method == expected_count).all():
            raise ValueError(f"incomplete method coverage in {mask_type}")
        assets[mask_type] = _by_asset(cond, mask_type)
        per_asset = assets[mask_type]
        means = {metric: {method: float(per_asset[(metric, method)].mean())
                          for method in per_method.index}
                 for metric in ("nrmse", "nmae", "price_nrmse", "volume_nrmse")}
        out["conditions_by_mask"][mask_type] = expected_count
        out["mask_types"][mask_type] = {
            "means": means,
            "ohlc_violation_share": one.groupby("method").ohlc_violation_share.mean().to_dict(),
        }
    for name, (mask_type, method, comparator) in HYPOTHESES.items():
        per_asset = assets[mask_type]
        metrics = {metric: paired(per_asset, method, comparator, metric)
                   for metric in ("nrmse", "price_nrmse", "volume_nrmse")}
        primary = metrics["nrmse"]
        primary["required_asset_wins"] = threshold
        primary["success"] = bool(primary["mean_difference"] < 0 and
                                  primary["method_lower_in_assets"] >= threshold)
        out["hypotheses"][name] = {"mask_type": mask_type, "metrics": metrics,
                                    "success": primary["success"]}
    row_tuning = tuning.loc[tuning.mask_type == "row"]
    out["peer_usage"] = {
        "cfri_used_rows": int(row_tuning.cfri_peer_used_rows.sum()),
        "saits_used_rows": int(row_tuning.saits_peer_used_rows.sum()),
        "marketwide_used_rows": int(row_tuning.cfri_marketwide_peer_used_rows.sum()),
        "cfri_gain_counts": row_tuning.cfri_peer_gain.value_counts().sort_index().to_dict(),
        "saits_gain_counts": row_tuning.saits_peer_gain.value_counts().sort_index().to_dict(),
    }
    if out["peer_usage"]["marketwide_used_rows"] != 0:
        raise ValueError("marketwide peer usage must be zero")
    return out


def report(result: dict) -> str:
    lines = ["# CFRI simple held-out-asset evaluation", "",
             f"Study: `{result['study']}`. Classification: **locally frozen asset holdout**. "
             "The target ETFs were not in the earlier 34-ETF development study, but the January–September "
             "2026 market period is shared. This is not a future-period replication.", "",
             f"Local source freeze UTC: `{result['freeze']['freeze_utc']}`; embedded bundle SHA-256: "
             f"`{result['freeze']['embedded_bundle_sha256']}`. GitHub publication after this run "
             "does not establish a public pre-run timestamp.", "",
             f"Eligible target ETFs: {result['assets']}/16; eligible fixed peers: "
             f"{len(result['eligible_peers'])}/34. Required ETF wins: "
             f"{result['target_win_threshold']}/{result['assets']}.", "",
             "The endpoint averages five feature NRMSEs per condition, conditions within ETF, "
             "then ETFs equally. Differences are proposed method minus comparator; lower is better.", ""]
    if result["excluded_assets"]:
        lines += ["Excluded by score-blind QC:", ""]
        for excluded in result["excluded_assets"]:
            lines.append(f"- {excluded['ticker']} ({excluded['role']}): {', '.join(excluded['reasons'])}")
        lines.append("")
    for mask_type in C.MASK_TYPES:
        item = result["mask_types"][mask_type]
        lines += [f"## {mask_type} masks", "",
                  f"Verified conditions per method: {result['conditions_by_mask'][mask_type]}.", "",
                  "| Method | NRMSE | Price NRMSE | Volume NRMSE | OHLC violations |",
                  "|---|---:|---:|---:|---:|"]
        for method, mean in item["means"]["nrmse"].items():
            lines.append(f"| {method} | {mean:.5f} | {item['means']['price_nrmse'][method]:.5f} | "
                         f"{item['means']['volume_nrmse'][method]:.5f} | "
                         f"{item['ohlc_violation_share'][method]:.4f} |")
        lines.append("")
    lines += ["## Prespecified hypotheses", "",
              "Each succeeds only when the equal-ETF mean error is lower and at least 75% of eligible "
              "ETFs have a lower mean error.", "",
              "| Hypothesis | Comparison | Difference | Descriptive 95% ETF bootstrap interval | ETF wins | Decision |",
              "|---|---|---:|---:|---:|---|"]
    for name, item in result["hypotheses"].items():
        p = item["metrics"]["nrmse"]
        lines.append(f"| {name} | {p['method']} vs {p['comparator']} ({item['mask_type']}) | "
                     f"{p['mean_difference']:+.5f} | [{p['boot95_low']:+.5f}, {p['boot95_high']:+.5f}] | "
                     f"{p['method_lower_in_assets']}/{p['assets']} | "
                     f"{'met' if item['success'] else 'not met'} |")
    lines += ["", "H1 uses target-only information for both methods. H2 measures the added peer bridge "
              "on whole-row outages. H3 gives both methods the same peer pool and bridge, with each gain "
              "selected on 2025 validation labels.", "",
              "## Interpretation limits", "",
              "- The 34 peer ETFs and the 2026 calendar were used in the earlier study; only target assets "
              "are held out here. A future-period replication is still needed.",
              "- Target-specific row results assume other ETFs' same-day closes are available. "
              "The market-wide stress case hides those closes and must fall back to no-peer CFRI.",
              "- The 16 ETFs share market shocks; the asset bootstrap is descriptive, not an "
              "independent-sample confidence statement. Repeated masks are not independent samples.",
              "- This is offline gap imputation using observed values on both sides of a gap, not forecasting.",
              "- Raw Yahoo OHLCV tables are retained privately and excluded from this ZIP.", ""]
    return "\n".join(lines)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, float_format="%.17g").encode()


def analyze_and_export(root: Path) -> Path:
    root = Path(root)
    condition, feature, tuning, expected, receipt, freeze = load(root)
    result = summarize(condition, tuning, expected, receipt, freeze)
    outputs = {
        "holdout_results.json": (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        "holdout_report.md": report(result).encode(),
        "holdout_condition_scores.csv": _csv_bytes(condition),
        "holdout_feature_scores.csv": _csv_bytes(feature),
        "holdout_tuning.csv": _csv_bytes(tuning),
    }
    code_files = sorted(
        [path for path in HERE.iterdir() if path.is_file() and
         (path.suffix == ".py" or path.name in ("PROTOCOL.md", "requirements-colab.txt")) and
         not path.name.startswith("test_")]
        + list((HERE / "upstream").glob("*.py")))
    code_manifest = {path.relative_to(HERE).as_posix(): sha256(path.read_bytes())
                     for path in code_files}
    outputs["code_sha256.json"] = (json.dumps(code_manifest, sort_keys=True, indent=2) + "\n").encode()
    for name, payload in outputs.items():
        D.atomic(root / name, payload)
    included = [root / name for name in ("freeze_receipt.json", "frozen_source_bundle.zip",
                D.RECEIPT, *outputs.keys())]
    for directory in (R.OUT_DIR, S.OUT_DIR):
        included.extend(sorted(path for path in (root / directory).glob("*.json") if path.is_file()))
    output = root / "CFRI_holdout_v1_results.zip"
    temp = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in included:
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, path.relative_to(root).as_posix())
        for path in code_files:
            archive.write(path, "code/" + path.relative_to(HERE).as_posix())
    os.replace(temp, output)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise IOError(f"result ZIP CRC failed: {bad}")
        if any("raw_do_not_publish" in name for name in archive.namelist()):
            raise AssertionError("raw prices appeared in result ZIP")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    output = analyze_and_export(args.root)
    print(f"Complete verified holdout results: {output}", flush=True)


if __name__ == "__main__":
    main()
