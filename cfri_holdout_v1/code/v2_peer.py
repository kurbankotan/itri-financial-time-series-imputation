"""Train-only peer ETF return bridge for *asset-specific* whole-row outages.

This module is an exploratory extension of CFRI.  The target's missing prices are
never used as regressors or labels.  Test-period peer closes are permitted only
in the explicitly labelled asset-specific outage setting, where those closes
would be observable while one target ETF's row is missing.  Pass
``synchronized_hidden=target_row_mask`` to model a market-wide outage.

The bridge adds a *shape* to CFRI's log-linear price path.  Its correction is
exactly zero at the observed left and right endpoints.  If fewer than
``min_peers`` peer returns are available on enough days of a gap, it leaves the
baseline unchanged.  Volume is left to CFRI's existing rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeerReturnModel:
    """Ridge mapping from observed peer close returns to target close returns."""

    target_ticker: str
    peer_names: tuple[str, ...]
    peer_mean: np.ndarray
    peer_scale: np.ndarray
    standardized_mean: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    min_peers: int
    n_fit: int


def _dates(index, n: int) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index)
    if len(dates) != n or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("target dates must match the target rows, be unique and increasing")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def aligned_peer_returns(
    target_dates,
    peer_frames: Mapping[str, pd.DataFrame],
    peer_names: tuple[str, ...],
    synchronized_hidden: np.ndarray | None = None,
) -> np.ndarray:
    """Peer log returns between *adjacent target dates*, NaN when unavailable.

    ``synchronized_hidden`` removes peer closes on those target dates *before*
    calculating returns.  This prevents a return on the first observed day
    after a market-wide outage from using a peer close inside the outage.
    """
    dates = _dates(target_dates, len(target_dates))
    n = len(dates)
    hidden = np.zeros(n, dtype=bool) if synchronized_hidden is None else np.asarray(synchronized_hidden, bool)
    if hidden.shape != (n,):
        raise ValueError("synchronized_hidden must have one boolean per target date")
    result = np.full((n, len(peer_names)), np.nan, dtype=float)
    for j, name in enumerate(peer_names):
        frame = peer_frames.get(name)
        if frame is None:
            continue
        if "close" not in frame.columns or frame.index.has_duplicates:
            raise ValueError(f"{name}: peer frame needs a unique date index and close column")
        peer = frame["close"].copy()
        peer.index = pd.DatetimeIndex(peer.index)
        if peer.index.tz is not None:
            peer.index = peer.index.tz_localize(None)
        peer.index = peer.index.normalize()
        if peer.index.has_duplicates:
            raise ValueError(f"{name}: normalized peer dates are duplicated")
        close = peer.reindex(dates).to_numpy(dtype=float, copy=True)
        close[hidden] = np.nan
        close[~np.isfinite(close) | (close <= 0)] = np.nan
        log_close = np.log(close)
        result[1:, j] = log_close[1:] - log_close[:-1]
    return result


def fit_peer_model(
    target_train: np.ndarray,
    train_hidden: np.ndarray,
    train_dates,
    peer_frames: Mapping[str, pd.DataFrame],
    target_ticker: str,
    *,
    alpha: float = 10.0,
    min_peers: int = 5,
    min_fit_rows: int = 100,
) -> PeerReturnModel | None:
    """Fit on the training partition only, ignoring all hidden target labels.

    A target daily-return label is eligible only when both target closes used
    to calculate it are observed under ``train_hidden``.  Peer availability
    and standardization are estimated from training dates only.
    """
    target = np.asarray(target_train, float)
    mask = np.asarray(train_hidden, bool)
    if target.ndim != 2 or target.shape[1] < 4 or mask.shape != target.shape:
        raise ValueError("target_train/train_hidden must have matching OHLCV rows")
    if alpha < 0 or min_peers < 1:
        raise ValueError("alpha must be nonnegative and min_peers positive")
    dates = _dates(train_dates, len(target))
    names = tuple(sorted(name for name in peer_frames if name != target_ticker))
    if len(names) < min_peers:
        return None
    x_raw = aligned_peer_returns(dates, peer_frames, names)
    valid_peer = np.isfinite(x_raw)
    eligible_peer = valid_peer.sum(axis=0) >= max(30, min_fit_rows // 2)
    names = tuple(name for name, keep in zip(names, eligible_peer) if keep)
    x_raw = x_raw[:, eligible_peer]
    valid_peer = valid_peer[:, eligible_peer]
    if len(names) < min_peers:
        return None

    # Mean-fill missing returns in standardized space.  The number and
    # identity of peers are selected solely from training availability.
    means = np.array([x_raw[valid_peer[:, j], j].mean() for j in range(len(names))])
    scales = np.array([x_raw[valid_peer[:, j], j].std(ddof=0) for j in range(len(names))])
    scales[scales < 1e-8] = 1.0
    x = np.where(valid_peer, (x_raw - means) / scales, 0.0)

    close = target[:, 3]
    safe_close = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    observed = (~mask[:, 3]) & np.isfinite(safe_close)
    y = np.full(len(target), np.nan)
    y[1:] = np.log(safe_close[1:]) - np.log(safe_close[:-1])
    eligible = np.zeros(len(target), dtype=bool)
    eligible[1:] = observed[1:] & observed[:-1]
    eligible &= valid_peer.sum(axis=1) >= min_peers
    eligible &= np.isfinite(y)
    if int(eligible.sum()) < max(min_fit_rows, len(names) + 2):
        return None

    fit_x = x[eligible]
    fit_y = y[eligible]
    center_x = fit_x.mean(axis=0)
    center_y = float(fit_y.mean())
    xc = fit_x - center_x
    gram = xc.T @ xc + alpha * np.eye(len(names))
    beta = np.linalg.solve(gram, xc.T @ (fit_y - center_y))
    intercept = float(center_y - center_x @ beta)
    return PeerReturnModel(target_ticker, names, means, scales, center_x,
                           beta, intercept, float(alpha), min_peers, int(eligible.sum()))


def predict_peer_returns(
    model: PeerReturnModel | None,
    target_dates,
    peer_frames: Mapping[str, pd.DataFrame],
    *,
    synchronized_hidden: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted target close returns and available-peer counts.

    A day with too few observed peer returns receives a zero prediction.
    ``model=None`` also yields zero, giving an exact no-peer fallback.
    """
    n = len(target_dates)
    if model is None:
        return np.zeros(n), np.zeros(n, dtype=int)
    raw = aligned_peer_returns(target_dates, peer_frames, model.peer_names, synchronized_hidden)
    valid = np.isfinite(raw)
    counts = valid.sum(axis=1)
    x = np.where(valid, (raw - model.peer_mean) / model.peer_scale, 0.0)
    predicted = model.intercept + x @ model.coefficients
    predicted[counts < model.min_peers] = 0.0
    predicted[0] = 0.0
    return predicted, counts


def apply_peer_bridge(
    baseline: np.ndarray,
    hidden: np.ndarray,
    peer_return: np.ndarray,
    peer_count: np.ndarray,
    *,
    gain: float = 1.0,
    min_peers: int = 5,
    min_coverage: float = 0.5,
    max_abs_log_shift: float = 0.25,
    return_meta: bool = False,
):
    """Add an endpoint-anchored peer shape to hidden OHLC rows only.

    The correction at hidden time ``t`` in a run bounded by observed rows
    ``a`` and ``b`` is ``gain * [sum(q[a+1:t]) - (t-a)/(b-a)*sum(q[a+1:b])]``.
    Thus the two observed endpoints remain unchanged.  The same log shift is
    applied to all four prices, preserving each day's relative OHLC shape.
    A later constraint projection may still be needed for the baseline.
    """
    base = np.asarray(baseline, float)
    mask = np.asarray(hidden, bool)
    q = np.asarray(peer_return, float)
    counts = np.asarray(peer_count, int)
    if base.ndim != 2 or base.shape[1] < 4 or mask.shape != base.shape:
        raise ValueError("baseline and hidden must have matching OHLCV shape")
    n = len(base)
    if q.shape != (n,) or counts.shape != (n,):
        raise ValueError("peer arrays must have one entry per target row")
    if not (0 <= min_coverage <= 1) or min_peers < 1 or max_abs_log_shift <= 0:
        raise ValueError("invalid peer bridge coverage or shift limit")
    if not np.isfinite(q).all() or not np.isfinite(base).all() or (base[:, :4] <= 0).any():
        raise ValueError("baseline prices and peer predictions must be finite and positive")
    out = base.copy()
    row_hidden = mask[:, :4].all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, row_hidden, False]))
    meta = {"candidate_runs": len(edges) // 2, "used_runs": 0, "used_rows": 0,
            "fallback_runs": 0}
    for start, stop in zip(edges[::2], edges[1::2]):
        a, b = start - 1, stop
        if a < 0 or b >= n:
            meta["fallback_runs"] += 1
            continue
        span_count = counts[a + 1:b + 1]
        if float(np.mean(span_count >= min_peers)) < min_coverage:
            meta["fallback_runs"] += 1
            continue
        span_q = q[a + 1:b + 1]
        cum = np.cumsum(span_q)[:-1]
        frac = np.arange(1, stop - start + 1) / (b - a)
        shape = cum - frac * span_q.sum()
        log_shift = np.clip(gain * shape, -max_abs_log_shift, max_abs_log_shift)
        out[start:stop, :4] *= np.exp(log_shift[:, None])
        meta["used_runs"] += 1
        meta["used_rows"] += int(stop - start)
    if not np.array_equal(out[~mask], base[~mask]):
        raise AssertionError("peer bridge changed an observed cell")
    return (out, meta) if return_meta else out


def peer_adjust(
    baseline: np.ndarray,
    target_dates,
    hidden: np.ndarray,
    peer_frames: Mapping[str, pd.DataFrame],
    model: PeerReturnModel | None,
    *,
    gain: float,
    market_wide: bool = False,
    return_meta: bool = False,
):
    """Convenience wrapper for target-only or synchronized-market row gaps.

    With ``market_wide=True``, peer closes at every target-hidden row are
    removed before daily peer returns are calculated.  For a full-row gap,
    this normally makes the branch fall back exactly to ``baseline``.
    """
    mask = np.asarray(hidden, bool)
    synchronized = mask[:, :4].all(axis=1) if market_wide else None
    q, count = predict_peer_returns(model, target_dates, peer_frames,
                                    synchronized_hidden=synchronized)
    minimum = model.min_peers if model is not None else 1
    return apply_peer_bridge(baseline, mask, q, count, gain=gain,
                             min_peers=minimum, return_meta=return_meta)


def select_peer_gain(
    baseline_val: np.ndarray,
    hidden_val: np.ndarray,
    truth_val: np.ndarray,
    price_scales: np.ndarray,
    peer_return_val: np.ndarray,
    peer_count_val: np.ndarray,
    *,
    min_peers: int,
    gains: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    project_fn=None,
) -> tuple[float, dict[float, float]]:
    """Select the branch gain using validation labels only.

    ``project_fn(estimate, hidden)`` may be CFRI's frozen OHLC projection or
    the new joint constraint solver.  The same projection must be used for
    every candidate, including gain zero.  Selection minimizes the mean of
    four price NRMSEs; ties prefer the smaller gain.
    """
    baseline = np.asarray(baseline_val, float)
    hidden = np.asarray(hidden_val, bool)
    truth = np.asarray(truth_val, float)
    scales = np.asarray(price_scales, float)
    if baseline.shape != hidden.shape or truth.shape != baseline.shape or scales.shape != (4,):
        raise ValueError("validation arrays or four price scales have incompatible shape")
    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise ValueError("validation scales must be finite and positive")
    scores: dict[float, float] = {}
    for gain in gains:
        est = apply_peer_bridge(baseline, hidden, peer_return_val, peer_count_val,
                                gain=gain, min_peers=min_peers)
        if project_fn is not None:
            est = project_fn(est, hidden)
        terms = []
        for d in range(4):
            selected = hidden[:, d]
            if selected.any():
                terms.append(float(np.sqrt(np.mean((est[selected, d] - truth[selected, d]) ** 2)) / scales[d]))
        if not terms:
            raise ValueError("validation mask contains no price labels")
        scores[float(gain)] = float(np.mean(terms))
    best = min(scores, key=lambda g: (scores[g], g))
    return best, scores
