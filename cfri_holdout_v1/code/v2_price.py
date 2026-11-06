"""Exploratory price improvements for CFRI.

The 2026 fold D results informed this module, so fold D scores produced with it
are exploratory.  Training predictors see only cells visible after the training
mask and additional pseudo mask; validation labels select the small gain gate.
Neither the feature construction nor the OHLC projection reads hidden test truth.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Callable, Sequence

import numpy as np

from upstream import cfri_core as core


# Constraints A @ [open, high, low, close] >= 0 in log-price space.
_OHLC_A = np.array(
    [[-1.0, 1.0, 0.0, 0.0],   # high >= open
     [0.0, 1.0, 0.0, -1.0],   # high >= close
     [1.0, 0.0, -1.0, 0.0],   # open >= low
     [0.0, 0.0, -1.0, 1.0]],  # close >= low
    dtype=float,
)


def _nearby_visible_volatility(visible_log_col: np.ndarray, rows: np.ndarray, radius: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Median absolute adjacent log return, using visible pairs only.

    Inside a masked run all hidden returns are NaN, so the statistic uses only
    observations outside that run.  The second output marks a usable window.
    """
    if radius < 1:
        raise ValueError("radius must be positive")
    col = np.asarray(visible_log_col, dtype=float)
    returns = np.abs(np.diff(col))
    returns[~(np.isfinite(col[:-1]) & np.isfinite(col[1:]))] = np.nan
    vol = np.zeros(len(rows), dtype=float)
    available = np.zeros(len(rows), dtype=float)
    for i, t in enumerate(rows):
        near = returns[max(0, int(t) - radius):min(len(returns), int(t) + radius)]
        usable = near[np.isfinite(near)]
        if len(usable):
            vol[i] = float(np.median(usable))
            available[i] = 1.0
    return vol, available


def _augment_design(visible_log: np.ndarray, target_feature: int, design: tuple) -> tuple:
    """Add cross-feature interactions without accessing a target label."""
    base_x, y, rows, bases, supports = design
    if base_x.shape[1] != 9:
        raise ValueError("expected the nine-column original CFRI design")
    if len(rows) == 0:
        return np.empty((0, 23), dtype=float), y, rows, bases, supports
    vol, vol_available = _nearby_visible_volatility(visible_log[:, target_feature], rows)
    log_vol = np.log1p(100.0 * vol)
    delta = base_x[:, :3]
    log_length = base_x[:, 6]
    position = base_x[:, 7]
    centered_position = 2.0 * position - 1.0
    hump = 4.0 * position * (1.0 - position)
    x = np.column_stack(
        (base_x,
         log_vol, vol_available,
         delta * centered_position[:, None],
         delta * hump[:, None],
         delta * log_length[:, None],
         delta * log_vol[:, None])
    )
    if x.shape[1] != 23 or not np.isfinite(x).all():
        raise AssertionError("invalid interaction design")
    return x, y, rows, bases, supports


def prepare_interaction_training(
    train_raw: np.ndarray, train_mask: np.ndarray, ticker: str,
    fold: str, block: int, rate: float, seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build pseudo-mask labels solely from originally visible training cells.

    This keeps the original CFRI auxiliary seeds and two pseudo-mask repetitions.
    Hidden training cells are excluded as labels; original values of all masked
    cells are absent from every design matrix.
    """
    train_raw = np.asarray(train_raw, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_raw.shape != train_mask.shape or train_raw.shape[1] < 5:
        raise ValueError("training values and mask must be matching OHLCV arrays")
    if np.any(train_raw[:, :4] <= 0) or not np.isfinite(train_raw).all():
        raise ValueError("training prices must be positive and finite")
    truth_log = np.log(train_raw[:, :4])
    datasets: list[list[np.ndarray]] = [[] for _ in range(4)]
    labels: list[list[np.ndarray]] = [[] for _ in range(4)]
    for repetition in range(2):
        pseudo = core.strict_block_mask(
            train_raw.shape, rate, block,
            core.auxiliary_seed(ticker, fold, block, rate, seed, repetition),
        )
        effective = train_mask | pseudo
        visible_log = core.corrupt(truth_log, effective[:, :4])
        for d in range(4):
            selected = pseudo[:, d] & ~train_mask[:, d]
            augmented = _augment_design(
                visible_log, d,
                core.price_design(visible_log, d, selected, truth_log),
            )
            x, y = augmented[:2]
            if len(x):
                datasets[d].append(x)
                labels[d].append(y)
    return [
        (np.vstack(datasets[d]), np.concatenate(labels[d]))
        if datasets[d] else (np.empty((0, 23), dtype=float), np.empty(0, dtype=float))
        for d in range(4)
    ]


def prepare_interaction_eval(raw: np.ndarray, hidden: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
    """Return log-linear paths and feature designs for a masked partition.

    Hidden entries of ``raw`` are replaced by NaN before any predictor is made.
    ``raw`` is used only to copy the visible entries, never for a hidden label.
    """
    raw = np.asarray(raw, dtype=float)
    hidden = np.asarray(hidden, dtype=bool)
    if raw.shape != hidden.shape or raw.shape[1] < 5:
        raise ValueError("values and mask must be matching OHLCV arrays")
    if np.any(raw[:, :4] <= 0) or not np.isfinite(raw).all():
        raise ValueError("prices must be positive and finite")
    visible_log = np.log(core.corrupt(raw[:, :4], hidden[:, :4]))
    base_log = core.linear_fill(visible_log)
    designs = [
        _augment_design(visible_log, d, core.price_design(visible_log, d))
        for d in range(4)
    ]
    return base_log, designs


def fit_interaction_models(training: Sequence[tuple[np.ndarray, np.ndarray]], alpha: float = 10.0) -> list:
    """Fit four regularized residual models using training pseudo-mask labels."""
    if len(training) != 4 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("expected four training sets and a positive ridge alpha")
    return core.fit_price_models(training, alpha)


def interaction_price_candidate(
    base_log: np.ndarray, designs: Sequence[tuple], models: Sequence,
    gain: float = 0.75, support_power: float = 0.0,
) -> np.ndarray:
    """Apply the price residual with an availability gate.

    The gate is ``gain * (visible_helpers / 3) ** support_power`` and is zero
    when no same-day helper is visible.  The small gain/power grid should be
    chosen on validation only via :func:`select_price_gate`.
    """
    if len(designs) != 4 or len(models) != 4:
        raise ValueError("expected four designs and four models")
    if not np.isfinite(gain) or gain < 0 or not np.isfinite(support_power) or support_power < 0:
        raise ValueError("gain and support_power must be finite and nonnegative")
    out = np.exp(np.array(base_log, dtype=float, copy=True))
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError("base_log must have four price columns")
    if gain == 0:
        return out
    for d, pack in enumerate(models):
        if pack is None:
            continue
        x, _, rows, bases, supports = designs[d]
        if len(rows) == 0:
            continue
        scaler, model = pack
        residual = np.clip(model.predict(scaler.transform(x)), -0.25, 0.25)
        gate = np.zeros(len(rows), dtype=float)
        has_help = supports > 0
        gate[has_help] = gain * (np.minimum(supports[has_help], 3.0) / 3.0) ** support_power
        out[rows, d] = np.exp(bases + gate * residual)
    return out


def select_price_gate(
    val_truth: np.ndarray, val_hidden: np.ndarray, scales: np.ndarray,
    val_base: np.ndarray, val_designs: Sequence[tuple], models: Sequence,
    gains: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    support_powers: Sequence[float] = (0.0, 0.5, 1.0),
    projection: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Choose gain and support gate using masked validation labels only.

    If supplied, ``projection`` is applied before validation scoring.  Use the
    same projection when selecting the gate and evaluating that variant.
    """
    val_truth = np.asarray(val_truth, dtype=float)
    val_hidden = np.asarray(val_hidden, dtype=bool)
    scales = np.asarray(scales, dtype=float)
    if val_truth.shape != val_hidden.shape or val_truth.shape[1] < 4 or scales.shape[0] < 4:
        raise ValueError("invalid validation arrays")
    if len(gains) == 0 or len(support_powers) == 0:
        raise ValueError("gate grid must be nonempty")
    for d in range(4):
        if not np.any(val_hidden[:, d]) or not np.isfinite(scales[d]) or scales[d] <= 0:
            raise ValueError("every price needs masked validation cells and a positive scale")
    best: tuple[float, float, float] | None = None
    for gain in gains:
        for power in support_powers:
            candidate = interaction_price_candidate(val_base, val_designs, models, gain, power)
            # The candidate receives only visible observations before any
            # projection; hidden labels enter solely in the score below.
            candidate[~val_hidden[:, :4]] = val_truth[:, :4][~val_hidden[:, :4]]
            if projection is not None:
                candidate = projection(candidate, val_hidden[:, :4])
            errors = []
            for d in range(4):
                m = val_hidden[:, d]
                errors.append(float(np.sqrt(np.mean((candidate[m, d] - val_truth[m, d]) ** 2)) / scales[d]))
            mean_error = float(np.mean(errors))
            record = (mean_error, float(gain), float(power))
            if best is None or record < best:
                best = record
    assert best is not None
    return {"val_price_nrmse": best[0], "gain": best[1], "support_power": best[2]}


@lru_cache(maxsize=64)
def _projector_config(hidden_bits: tuple[bool, bool, bool, bool], weight_key: tuple[float, float, float, float]):
    hidden = np.asarray(hidden_bits, dtype=bool)
    fixed = ~hidden
    a_hidden = _OHLC_A[:, hidden]
    a_fixed = _OHLC_A[:, fixed]
    weights = np.asarray(weight_key, dtype=float)[hidden]
    inv_weights = 1.0 / weights
    movable = np.flatnonzero(np.any(np.abs(a_hidden) > 0, axis=1))
    active_sets = []
    for count in range(len(movable) + 1):
        for subset in combinations(movable.tolist(), count):
            if not subset:
                active_sets.append((subset, None, None))
                continue
            active = a_hidden[list(subset)]
            gram = (active * inv_weights[None, :]) @ active.T
            inverse = np.linalg.pinv(gram, rcond=1e-12)
            adjustment = inv_weights[:, None] * active.T @ inverse
            active_sets.append((subset, inverse, adjustment))
    return a_hidden, a_fixed, active_sets


def _weighted_project_log_row(y: np.ndarray, hidden: np.ndarray, weights: np.ndarray) -> np.ndarray:
    hidden_bits = tuple(bool(v) for v in hidden)
    weight_key = tuple(float(v) for v in weights)
    a_hidden, a_fixed, active_sets = _projector_config(hidden_bits, weight_key)
    p = y[hidden]
    b = -a_fixed @ y[~hidden]
    tol = 1e-10
    # An inequality involving only observed prices cannot be repaired.
    immovable = ~np.any(np.abs(a_hidden) > 0, axis=1)
    if np.any(b[immovable] > tol):
        raise ValueError("observed OHLC cells are mutually inconsistent")
    best_x = None
    best_obj = np.inf
    w = weights[hidden]
    for subset, inverse, adjustment in active_sets:
        if subset:
            idx = list(subset)
            rhs = b[idx] - a_hidden[idx] @ p
            multipliers = inverse @ rhs
            if np.any(multipliers < -tol):
                continue
            x = p + adjustment @ rhs
            if np.max(np.abs(a_hidden[idx] @ x - b[idx])) > tol:
                continue
        else:
            x = p
        if np.any(a_hidden @ x < b - tol):
            continue
        objective = float(np.dot(w, (x - p) ** 2))
        if objective < best_obj - 1e-14:
            best_obj = objective
            best_x = x
    if best_x is None:
        raise RuntimeError("OHLC active-set projection failed")
    result = y.copy()
    result[hidden] = best_x
    return result


def joint_ohlc_project(
    estimate: np.ndarray, hidden: np.ndarray, weights: np.ndarray | None = None,
) -> np.ndarray:
    """Exact weighted OHLC projection with all observed cells fixed.

    For each row, minimize squared movement of *hidden* log-prices under the
    four order constraints.  The active-set problem has at most four price
    coordinates and four inequalities; all active sets are checked exactly.
    Uniform weights are the default and should be used for fair head-to-head
    postprocessing of CFRI and SAITS.  Optional weights must be fixed before
    test scoring, e.g. from validation residual variances.

    Nonpositive/nonfinite *imputed predictions* are initialized by temporal
    interpolation of visible positive prices in that column; observations are
    never replaced.  This fallback is part of the projection definition and
    is applied identically to every compared method.
    """
    estimate = np.asarray(estimate)
    hidden = np.asarray(hidden, dtype=bool)
    if estimate.ndim != 2 or estimate.shape[1] < 4 or estimate.shape != hidden.shape:
        raise ValueError("estimate and hidden must be matching arrays with at least four columns")
    if not np.issubdtype(estimate.dtype, np.floating):
        raise TypeError("estimate must be a floating array")
    if weights is None:
        w = np.ones(4, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (4,) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("weights must be four finite positive values")
    out = estimate.copy()
    observed_prices = out[:, :4][~hidden[:, :4]]
    if np.any(~np.isfinite(observed_prices)) or np.any(observed_prices <= 0):
        raise ValueError("observed prices must be finite and positive")
    fallback_pool = observed_prices if len(observed_prices) else out[:, :4][
        np.isfinite(out[:, :4]) & (out[:, :4] > 0)]
    if len(fallback_pool) == 0:
        raise ValueError("no positive price is available for an invalid prediction fallback")
    global_fallback = float(np.median(fallback_pool))
    for d in range(4):
        bad = hidden[:, d] & (~np.isfinite(out[:, d]) | (out[:, d] <= 0))
        if not np.any(bad):
            continue
        visible = (~hidden[:, d]) & np.isfinite(out[:, d]) & (out[:, d] > 0)
        if np.any(visible):
            out[bad, d] = np.interp(
                np.flatnonzero(bad), np.flatnonzero(visible), out[visible, d],
            )
        else:
            out[bad, d] = global_fallback
    for t in np.flatnonzero(hidden[:, :4].any(axis=1)):
        h = hidden[t, :4]
        y = np.log(np.asarray(out[t, :4], dtype=float))
        projected = _weighted_project_log_row(y, h, w)
        indices = np.flatnonzero(h)
        out[t, indices] = np.exp(projected[indices])
        # Exact inequalities can differ by a few ulps after exp.  Repair only
        # roundoff on hidden cells, never a substantive solver discrepancy.
        row = out[t, :4]
        if row[1] < max(row[0], row[3]):
            if h[1] and max(row[0], row[3]) - row[1] <= 1e-9 * max(row[0], row[3]):
                out[t, 1] = max(row[0], row[3])
            elif h[0] and row[0] > row[1] and row[0] - row[1] <= 1e-9 * row[1]:
                out[t, 0] = row[1]
            elif h[3] and row[3] > row[1] and row[3] - row[1] <= 1e-9 * row[1]:
                out[t, 3] = row[1]
        row = out[t, :4]
        if row[2] > min(row[0], row[3]):
            if h[2] and row[2] - min(row[0], row[3]) <= 1e-9 * row[2]:
                out[t, 2] = min(row[0], row[3])
            elif h[0] and row[0] < row[2] and row[2] - row[0] <= 1e-9 * row[2]:
                out[t, 0] = row[2]
            elif h[3] and row[3] < row[2] and row[2] - row[3] <= 1e-9 * row[2]:
                out[t, 3] = row[2]
        row = out[t, :4]
        if row[1] < max(row[0], row[3]) - 1e-9 or row[2] > min(row[0], row[3]) + 1e-9:
            raise RuntimeError("OHLC projection left an order violation")
    return out
