"""Cross-Feature Residual Interpolation (CFRI) and its comparators: frozen core for the confirmatory study.

CFRI is the method explored under the working name "ITRI2" (notebook ITRI_2_Colab_Drive_Resume.ipynb,
code version "itri2-residual-colab-v1"). Its price and volume rules, candidate grids, pseudo-mask seeds and
per-condition validation selection are reproduced here unchanged; the functions marked [verbatim] are copied
from that notebook. This module adds: a whole-row mask type, candidate-level logging (needed for the
fixed-setting ablation), an OHLC consistency diagnostic, an OHLC projection variant, and fold D.

Masks, folds A-C, scaling and the primary endpoint are identical to the external validation study
(external_validation/code/masking.py and run_classical_benchmark.py of the public repository).
"""

from __future__ import annotations

import hashlib
import math
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CODE_ID = "cfri-core-v2"
PSEUDO_SEED_TAG = "itri2-residual-colab-v1"   # keeps CFRI pseudo masks identical to the exploratory ITRI2 run
ORIGINAL_MANIFEST_SHA = "37ed5c7becd54665c108ecf296bca1a0dd94ffc0a20788a3b9dd42679eb70d64"
FEATURES = ("open", "high", "low", "close", "volume")
ASSETS = (
    ("EWC", "Canada", "Americas"), ("EWZ", "Brazil", "Americas"),
    ("ECH", "Chile", "Americas"), ("EWW", "Mexico", "Americas"),
    ("EWO", "Austria", "Europe"), ("EWK", "Belgium", "Europe"),
    ("EDEN", "Denmark", "Europe"), ("EFNL", "Finland", "Europe"),
    ("EWQ", "France", "Europe"), ("EWG", "Germany", "Europe"),
    ("EIRL", "Ireland", "Europe"), ("EWI", "Italy", "Europe"),
    ("EWN", "Netherlands", "Europe"), ("ENOR", "Norway", "Europe"),
    ("EPOL", "Poland", "Europe"), ("EWP", "Spain", "Europe"),
    ("EWD", "Sweden", "Europe"), ("EWL", "Switzerland", "Europe"),
    ("EWU", "United Kingdom", "Europe"), ("EWA", "Australia", "Asia-Pacific"),
    ("MCHI", "China", "Asia-Pacific"), ("EWH", "Hong Kong", "Asia-Pacific"),
    ("INDA", "India", "Asia-Pacific"), ("EIDO", "Indonesia", "Asia-Pacific"),
    ("EWJ", "Japan", "Asia-Pacific"), ("EWM", "Malaysia", "Asia-Pacific"),
    ("ENZL", "New Zealand", "Asia-Pacific"), ("EPHE", "Philippines", "Asia-Pacific"),
    ("EWS", "Singapore", "Asia-Pacific"), ("EWY", "South Korea", "Asia-Pacific"),
    ("EWT", "Taiwan", "Asia-Pacific"), ("THD", "Thailand", "Asia-Pacific"),
    ("EIS", "Israel", "Middle East and Africa"),
    ("EZA", "South Africa", "Middle East and Africa"),
)
FOLDS = {
    "A": {"train": (2015, 2018), "val": (2019, 2019), "test": (2020, 2021)},
    "B": {"train": (2015, 2020), "val": (2021, 2021), "test": (2022, 2023)},
    "C": {"train": (2015, 2022), "val": (2023, 2023), "test": (2024, 2025)},
    # Confirmatory fold: the test year is 2026, used only up to the protocol cut-off date.
    "D": {"train": (2015, 2024), "val": (2025, 2025), "test": (2026, 2026)},
}
FOLD_D_TEST_END = "2026-09-30"
BLOCKS = (1, 2, 3, 5, 10, 20)
RATES = (0.10, 0.20, 0.30)
SEEDS = (11, 23, 37)
EDGE_GUARD = 20
MASK_TYPES = ("feature", "row")
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
PRICE_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)
VOLUME_SHRINK = (0.0, 0.25, 0.5, 0.75)
PRICE_CANDIDATES = tuple((a, g) for a in RIDGE_ALPHAS for g in PRICE_GAINS)
VOLUME_CANDIDATES = ("linear_raw", "itri_w1_p5") + tuple(f"log_local_{e:.2f}" for e in VOLUME_SHRINK)
# Fixed CFRI setting, selected before the confirmatory study as the candidate pair and volume rule with the
# lowest mean validation NRMSE pooled over all feature-wise conditions of folds A-C (validation years 2019,
# 2021 and 2023; no test data involved).
CFRI_FIXED = (10.0, 0.75, "log_local_0.50")
BASE_METHODS = ("linear_raw", "log_linear", "itri_w1_p5", "chained_ridge", "cfri_adaptive_no_cross",
                "cfri_adaptive", "cfri_fixed_no_cross", "cfri_fixed")
METHODS = BASE_METHODS + tuple(m + "_proj" for m in BASE_METHODS)
PRIMARY_METHOD = "cfri_fixed_proj"    # CFRI as evaluated in the confirmatory study


# ----------------------------------------------------------------------------- masks and seeds
def run_lengths(mask: np.ndarray) -> np.ndarray:  # [verbatim]
    lengths = []
    for feature in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[:, feature], False]))
        lengths.extend((edges[1::2] - edges[::2]).tolist())
    return np.asarray(lengths, dtype=int)


def strict_block_mask(shape, rate, block_length, seed, edge_guard=EDGE_GUARD):  # [verbatim]
    rows, features = shape
    if rows <= 2 * edge_guard + block_length or features < 1:
        raise ValueError("partition too short for this mask")
    if not 0 < rate < 1 or block_length < 1 or edge_guard < 0:
        raise ValueError("invalid mask setting")
    eligible_rows = rows - 2 * edge_guard
    target = max(1, round(rate * eligible_rows * features))
    if target < features * block_length:
        raise ValueError("target too small for one run per feature")
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    for feature in rng.permutation(features):
        start = int(rng.integers(edge_guard, rows - edge_guard - block_length + 1))
        mask[start:start + block_length, feature] = True
    remaining = target - features * block_length
    attempts = 0
    while remaining:
        length = min(block_length, remaining)
        feature = int(rng.integers(features))
        start = int(rng.integers(edge_guard, rows - edge_guard - length + 1))
        neighborhood = mask[max(edge_guard, start - 1):min(rows - edge_guard, start + length + 1), feature]
        if not neighborhood.any():
            mask[start:start + length, feature] = True
            remaining -= length
        attempts += 1
        if attempts >= 1000 * target:
            raise RuntimeError("could not place nonadjacent blocks")
    lengths = run_lengths(mask)
    if (int(mask.sum()) != target or lengths.max() > block_length or
            not np.all(mask.sum(axis=0) >= block_length) or
            mask[:edge_guard].any() or mask[rows - edge_guard:].any()):
        raise AssertionError("mask invariant failed")
    return mask


def stable_seed(ticker, fold, partition, block, rate, seed):  # [verbatim] (original study derivation)
    key = f"{ORIGINAL_MANIFEST_SHA}|{ticker}|{fold}|{partition}|{block}|{rate:.2f}|{seed}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def auxiliary_seed(ticker, fold, block, rate, seed, repetition):  # [verbatim] with the original tag
    key = f"{PSEUDO_SEED_TAG}|{ticker}|{fold}|pseudo|{block}|{rate:.2f}|{seed}|{repetition}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def condition_mask(shape, ticker, fold, partition, block, rate, seed, mask_type):
    """feature: each feature hidden independently (original generator and seeds).
    row: whole trading days hidden for all five features; the rows follow the same strict generator applied
    to one column, with seeds derived from the partition label suffixed by ':row'."""
    if mask_type == "feature":
        return strict_block_mask(shape, rate, block, stable_seed(ticker, fold, partition, block, rate, seed))
    if mask_type == "row":
        rows = strict_block_mask((shape[0], 1), rate, block,
                                 stable_seed(ticker, fold, partition + ":row", block, rate, seed))
        return np.repeat(rows, shape[1], axis=1)
    raise ValueError(mask_type)


def feasible(rows_by_part, block, rate, mask_type, edge_guard=EDGE_GUARD):
    """True when the strict generator can place the requested runs in every partition."""
    nfeat = len(FEATURES) if mask_type == "feature" else 1
    for rows in rows_by_part.values():
        if rows <= 2 * edge_guard + block:
            return False
        if max(1, round(rate * (rows - 2 * edge_guard) * nfeat)) < nfeat * block:
            return False
    return True


# ----------------------------------------------------------------------------- data helpers
def split_frame(frame: pd.DataFrame, fold: str) -> dict:
    parts = {}
    for name, (start, end) in FOLDS[fold].items():
        part = frame.loc[(frame.index.year >= start) & (frame.index.year <= end), list(FEATURES)]
        if fold == "D" and name == "test":
            part = part.loc[part.index <= pd.Timestamp(FOLD_D_TEST_END)]
        if len(part) < 100:
            raise ValueError(f"empty/short {fold}/{name}")
        parts[name] = part.to_numpy(float, copy=True)
    return parts


def observed_scaler(train, hidden):  # [verbatim]
    means = np.array([train[~hidden[:, d], d].mean() for d in range(train.shape[1])])
    scales = np.array([train[~hidden[:, d], d].std(ddof=0) for d in range(train.shape[1])])
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 1e-12).any():
        raise ValueError("invalid training-only scaler")
    return means, scales


def corrupt(array, hidden):  # [verbatim]
    out = array.copy()
    out[hidden] = np.nan
    return out


def linear_fill(array):  # [verbatim]
    filled = pd.DataFrame(array).interpolate(method="linear", limit_direction="both")
    out = filled.ffill().bfill().to_numpy(float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError("interpolation left nonfinite values")
    return out


class OriginalITRI:  # [verbatim]
    def fit(self, train_corrupt):
        mask = np.isnan(train_corrupt)
        filled = linear_fill(train_corrupt)
        self.models = []
        for d in range(filled.shape[1]):
            col = filled[:, d]
            z = np.column_stack((np.roll(col, 1), np.roll(col, -1)))
            z[[0, -1], :] = np.nan
            good = (~mask[:, d]) & np.isfinite(z).all(axis=1)
            if good.sum() < 4:
                self.models.append(None)
            else:
                design = np.column_stack((np.ones(good.sum()), z[good]))
                self.models.append(np.linalg.lstsq(design, col[good], rcond=None)[0])
        return self

    def transform(self, target_corrupt):
        mask = np.isnan(target_corrupt)
        out = linear_fill(target_corrupt)
        for _ in range(5):
            for d, beta in enumerate(self.models):
                if beta is None:
                    continue
                col = out[:, d]
                z = np.column_stack((np.roll(col, 1), np.roll(col, -1)))
                z[[0, -1], :] = np.nan
                rows = mask[:, d] & np.isfinite(z).all(axis=1)
                out[rows, d] = beta[0] + z[rows] @ beta[1:]
            out[~mask] = target_corrupt[~mask]
        return out


# ----------------------------------------------------------------------------- CFRI [verbatim functions]
def mask_runs(col_mask):
    edges = np.flatnonzero(np.diff(np.r_[False, col_mask, False]))
    return zip(edges[::2], edges[1::2])  # [start, stop)


def price_design(corrupt_log, target_feature, selected=None, truth_log=None):
    """Predictors use only visible cells; hidden target truth is used only as a label."""
    missing = np.isnan(corrupt_log)
    if selected is None:
        selected = missing[:, target_feature]
    xs, ys, rows, bases, supports = [], [], [], [], []
    n = len(corrupt_log)
    for start, stop in mask_runs(missing[:, target_feature]):
        a, b = start - 1, stop
        if a < 0 or b >= n:
            continue
        target_a, target_b = corrupt_log[a, target_feature], corrupt_log[b, target_feature]
        if not np.isfinite(target_a) or not np.isfinite(target_b):
            continue
        length = stop - start
        for t in range(start, stop):
            if not selected[t]:
                continue
            pos = (t - a) / (b - a)
            base = target_a + pos * (target_b - target_a)
            delta, availability = [], []
            for j in range(4):
                if j == target_feature:
                    continue
                visible = np.isfinite(corrupt_log[[a, t, b], j]).all()
                if visible:
                    aux_base = corrupt_log[a, j] + pos * (corrupt_log[b, j] - corrupt_log[a, j])
                    delta.append(float(corrupt_log[t, j] - aux_base))
                    availability.append(1.0)
                else:
                    delta.append(0.0)
                    availability.append(0.0)
            features = delta + availability + [math.log1p(length), pos, (target_b - target_a) / (b - a)]
            xs.append(features)
            rows.append(t)
            bases.append(base)
            supports.append(sum(availability))
            if truth_log is not None:
                ys.append(float(truth_log[t, target_feature] - base))
    x = np.asarray(xs, dtype=float).reshape((-1, 9))
    y = np.asarray(ys, dtype=float) if truth_log is not None else None
    return x, y, np.asarray(rows, dtype=int), np.asarray(bases), np.asarray(supports)


def prepare_price_training(train_raw, train_mask, ticker, fold, block, rate, seed):
    truth_log = np.log(train_raw[:, :4])
    datasets = [[] for _ in range(4)]
    labels = [[] for _ in range(4)]
    for repetition in range(2):
        pseudo = strict_block_mask(train_raw.shape, rate, block,
                                   auxiliary_seed(ticker, fold, block, rate, seed, repetition))
        effective = train_mask | pseudo
        visible_log = corrupt(truth_log, effective[:, :4])
        for d in range(4):
            selected = pseudo[:, d] & ~train_mask[:, d]
            x, y, _, _, _ = price_design(visible_log, d, selected, truth_log)
            if len(x):
                datasets[d].append(x)
                labels[d].append(y)
    return [(np.vstack(datasets[d]), np.concatenate(labels[d]))
            if datasets[d] else (np.empty((0, 9)), np.empty(0)) for d in range(4)]


def prepare_price_eval(raw, hidden):
    visible_log = np.log(corrupt(raw[:, :4], hidden[:, :4]))
    base_log = linear_fill(visible_log)
    designs = [price_design(visible_log, d) for d in range(4)]
    return base_log, designs


def fit_price_models(training, alpha):
    models = []
    for x, y in training:
        if len(y) < 30 or np.std(y) < 1e-10:
            models.append(None)
        else:
            scaler = StandardScaler().fit(x)
            model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
            models.append((scaler, model))
    return models


def price_candidate(base_log, designs, models, gain):
    out = np.exp(base_log.copy())
    if gain <= 0:
        return out
    for d, pack in enumerate(models):
        if pack is None:
            continue
        x, _, rows, bases, supports = designs[d]
        if not len(rows):
            continue
        scaler, model = pack
        residual = np.clip(model.predict(scaler.transform(x)), -0.25, 0.25)
        residual[supports == 0] = 0.0
        out[rows, d] = np.exp(bases + gain * residual)
    return out


def volume_log_candidate(raw, hidden, eta, local_radius=20):
    visible = np.log1p(corrupt(raw[:, 4], hidden[:, 4]))
    base = linear_fill(visible[:, None])[:, 0]
    out = base.copy()
    for start, stop in mask_runs(hidden[:, 4]):
        neighborhood = visible[max(0, start - local_radius):min(len(visible), stop + local_radius)]
        nearby = neighborhood[np.isfinite(neighborhood)]
        if len(nearby):
            out[start:stop] = (1 - eta) * base[start:stop] + eta * np.median(nearby)
    result = np.expm1(out)
    result[~hidden[:, 4]] = raw[~hidden[:, 4], 4]
    return np.maximum(result, 0.0)


# ----------------------------------------------------------------------------- scoring, OHLC checks
def feature_rmse(truth, estimate, mask, scales, d):
    selected = mask[:, d]
    if selected.sum() == 0:
        raise ValueError("no masked evaluation cells")
    return float(np.sqrt(np.mean((estimate[selected, d] - truth[selected, d]) ** 2)) / scales[d])


def score_method(truth, estimate, hidden, scales):
    if estimate.shape != truth.shape or not np.isfinite(estimate).all():
        raise ValueError("invalid estimate")
    if not np.allclose(estimate[~hidden], truth[~hidden], rtol=1e-9, atol=1e-9):
        raise ValueError("method changed an observed cell")
    out = {}
    for d, feature in enumerate(FEATURES):
        selected = hidden[:, d]
        err = estimate[selected, d] - truth[selected, d]
        out[feature] = (float(np.sqrt(np.mean(err ** 2)) / scales[d]), float(np.mean(np.abs(err)) / scales[d]),
                        int(selected.sum()))
    return out


def ohlc_violations(estimate, hidden, tol=1e-9):
    """Share of rows with at least one imputed price cell that break high >= max(open, close), low <=
    min(open, close) or high >= low (relative tolerance)."""
    rows = hidden[:, :4].any(axis=1)
    if not rows.any():
        return 0.0, 0
    o, h, l, c = (estimate[rows, k] for k in range(4))
    bad = ((h < np.maximum(o, c) * (1 - tol)) | (l > np.minimum(o, c) * (1 + tol)) | (h < l * (1 - tol)))
    return float(bad.mean()), int(rows.sum())


def ohlc_project(estimate, hidden):
    """Move only imputed price cells so that each day satisfies low <= open, close <= high."""
    out = estimate.copy()
    o, h, l, c = 0, 1, 2, 3
    for t in np.flatnonzero(hidden[:, :4].any(axis=1)):
        hid = hidden[t, :4]
        v = out[t, :4].copy()
        for _ in range(2):
            if hid[h]:
                v[h] = max(v[h], v[o], v[c], v[l] if not hid[l] else -np.inf)
            if hid[l]:
                v[l] = min(v[l], v[o], v[c], v[h] if not hid[h] else np.inf)
            if hid[o]:
                v[o] = min(max(v[o], v[l]), v[h])
            if hid[c]:
                v[c] = min(max(v[c], v[l]), v[h])
        out[t, :4] = v
    return out


# ----------------------------------------------------------------------------- one condition
def run_condition(parts, ticker, fold, block, rate, seed, mask_type, fixed=None):
    """All methods on one condition. `fixed` overrides CFRI_FIXED = (alpha, gain, volume_rule)."""
    t0 = time.perf_counter()
    masks = {name: condition_mask(values.shape, ticker, fold, name, block, rate, seed, mask_type)
             for name, values in parts.items()}
    train, val, test = (parts[n] for n in ("train", "val", "test"))
    train_mask, val_mask, test_mask = (masks[n] for n in ("train", "val", "test"))
    means, scales = observed_scaler(train, train_mask)
    tr_scaled = corrupt((train - means) / scales, train_mask)
    va_scaled = corrupt((val - means) / scales, val_mask)
    te_scaled = corrupt((test - means) / scales, test_mask)
    linear_val = linear_fill(corrupt(val, val_mask))
    linear_test = linear_fill(corrupt(test, test_mask))
    itri = OriginalITRI().fit(tr_scaled)
    itri_val = itri.transform(va_scaled) * scales + means
    itri_test = itri.transform(te_scaled) * scales + means
    chained = IterativeImputer(estimator=Ridge(alpha=1.0), max_iter=10, tol=1e-3, initial_strategy="mean",
                               imputation_order="ascending", sample_posterior=False, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        chained.fit(tr_scaled)
        chained_test = chained.transform(te_scaled) * scales + means
    chained_test[~test_mask] = test[~test_mask]

    # CFRI price candidates: validation selection (as in the exploratory run) with full logging
    training = prepare_price_training(train, train_mask, ticker, fold, block, rate, seed)
    val_base, val_designs = prepare_price_eval(val, val_mask)
    test_base, test_designs = prepare_price_eval(test, test_mask)
    log = {}
    best_price = (float("inf"), None, None)
    models_by_alpha = {}
    for alpha in RIDGE_ALPHAS:
        models = fit_price_models(training, alpha)
        models_by_alpha[alpha] = models
        for gain in PRICE_GAINS:
            cand_val = val.copy()
            cand_val[:, :4] = price_candidate(val_base, val_designs, models, gain)
            err = float(np.mean([feature_rmse(val, cand_val, val_mask, scales, d) for d in range(4)]))
            log[f"val_price_a{alpha:g}_g{gain:g}"] = err
            cand_test = test.copy()
            cand_test[:, :4] = price_candidate(test_base, test_designs, models, gain)
            for d in range(4):
                log[f"test_{FEATURES[d]}_a{alpha:g}_g{gain:g}"] = feature_rmse(test, cand_test, test_mask, scales, d)
            if err < best_price[0] - 1e-12:
                best_price = (err, alpha, gain)
    price_alpha, price_gain = best_price[1:]
    # refit (deterministic) exactly as the exploratory run did
    chosen_models = fit_price_models(training, price_alpha)
    selected_prices = price_candidate(test_base, test_designs, chosen_models, price_gain)
    uncrossed_prices = price_candidate(test_base, test_designs, chosen_models, 0.0)

    volume_val = {"linear_raw": linear_val[:, 4], "itri_w1_p5": itri_val[:, 4]}
    volume_test = {"linear_raw": linear_test[:, 4], "itri_w1_p5": itri_test[:, 4]}
    for eta in VOLUME_SHRINK:
        volume_val[f"log_local_{eta:.2f}"] = volume_log_candidate(val, val_mask, eta)
        volume_test[f"log_local_{eta:.2f}"] = volume_log_candidate(test, test_mask, eta)
    volume_errors = {name: float(np.sqrt(np.mean((series[val_mask[:, 4]] - val[val_mask[:, 4], 4]) ** 2)) / scales[4])
                     for name, series in volume_val.items()}
    for name in VOLUME_CANDIDATES:
        log[f"val_volume_{name}"] = volume_errors[name]
        sel = test_mask[:, 4]
        log[f"test_volume_{name}"] = float(np.sqrt(np.mean((volume_test[name][sel] - test[sel, 4]) ** 2)) / scales[4])
    volume_choice = min(volume_errors, key=lambda name: (volume_errors[name], name))

    log_linear = test.copy()
    log_linear[:, :4] = uncrossed_prices
    log_linear[:, 4] = volume_log_candidate(test, test_mask, 0.0)
    adaptive_no_cross = test.copy()
    adaptive_no_cross[:, :4] = uncrossed_prices
    adaptive_no_cross[:, 4] = volume_test[volume_choice]
    adaptive = test.copy()
    adaptive[:, :4] = selected_prices
    adaptive[:, 4] = volume_test[volume_choice]
    f_alpha, f_gain, f_volume = CFRI_FIXED if fixed is None else fixed
    fixed_est = test.copy()
    fixed_est[:, :4] = price_candidate(test_base, test_designs, models_by_alpha[f_alpha], f_gain)
    fixed_est[:, 4] = volume_test[f_volume]
    fixed_no_cross = test.copy()
    fixed_no_cross[:, :4] = uncrossed_prices
    fixed_no_cross[:, 4] = volume_test[f_volume]
    estimates = {"linear_raw": linear_test, "log_linear": log_linear, "itri_w1_p5": itri_test,
                 "chained_ridge": chained_test, "cfri_adaptive_no_cross": adaptive_no_cross,
                 "cfri_adaptive": adaptive, "cfri_fixed_no_cross": fixed_no_cross, "cfri_fixed": fixed_est}
    for value in estimates.values():
        value[~test_mask] = test[~test_mask]
    for name in list(estimates):                      # OHLC projection of imputed cells, for every method
        estimates[name + "_proj"] = ohlc_project(estimates[name], test_mask)
    scores = {m: score_method(test, est, test_mask, scales) for m, est in estimates.items()}
    violations = {m: ohlc_violations(est, test_mask)[0] for m, est in estimates.items()}
    truth_violation, price_rows = ohlc_violations(test, test_mask)
    return {
        "ticker": ticker, "fold": fold, "block_length": block, "rate": rate, "seed": seed, "mask_type": mask_type,
        "mask_sha256": {n: hashlib.sha256(m.tobytes()).hexdigest() for n, m in masks.items()},
        "masked_cells": {n: int(m.sum()) for n, m in masks.items()},
        "price_alpha": price_alpha, "price_gain": price_gain, "volume_choice": volume_choice,
        "scores": scores, "ohlc_violation_share": violations, "truth_ohlc_violation_share": truth_violation,
        "imputed_price_rows": price_rows, "candidates": log, "elapsed_seconds": time.perf_counter() - t0,
    }
