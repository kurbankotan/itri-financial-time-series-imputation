"""Controlled missing-run masks for a new, separately reported validation study.

The archived ITRI experiment permits adjacent sampled segments. This module
does not alter those historical masks or results. It defines strict runs for a
future experiment in which the requested block length has a literal meaning.
"""

from __future__ import annotations

import numpy as np


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return every maximal missing-run length, separately for each feature."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    lengths: list[int] = []
    for feature in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[:, feature], False]))
        lengths.extend((edges[1::2] - edges[::2]).tolist())
    return np.asarray(lengths, dtype=int)


def strict_block_mask(
    shape: tuple[int, int], rate: float, block_length: int, seed: int,
    edge_guard: int = 10,
) -> np.ndarray:
    """Place nonadjacent runs of length ``block_length`` at an exact cell rate.

    Every feature receives one full run before the remaining runs are sampled,
    so feature-wise errors are defined for every condition. A final shorter
    run is allowed only when the rounded target cell count is not divisible
    by the requested block length. One observed row separates every pair of
    runs within a feature. Other features may be missing on the same dates.
    """
    rows, features = shape
    if rows <= 2 * edge_guard + block_length or features < 1:
        raise ValueError("not enough rows/features for the block and guard")
    if not 0 < rate < 1 or block_length < 1 or edge_guard < 0:
        raise ValueError("invalid rate, block length, or edge guard")
    eligible_rows = rows - 2 * edge_guard
    target = max(1, round(rate * eligible_rows * features))
    if target < features * block_length:
        raise ValueError("target too small for one full masked run per feature")
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    # The endpoint is a five-feature macro mean, so a zero-mask feature is
    # invalid. Reserve exactly one nominal-length run per feature first.
    for feature in rng.permutation(features):
        start = int(rng.integers(edge_guard,
                                 rows - edge_guard - block_length + 1))
        mask[start:start + block_length, feature] = True
    remaining = target - features * block_length
    attempts = 0
    max_attempts = 1000 * target

    while remaining:
        length = min(block_length, remaining)
        feature = int(rng.integers(features))
        start = int(rng.integers(edge_guard, rows - edge_guard - length + 1))
        neighborhood = mask[max(edge_guard, start - 1):
                            min(rows - edge_guard, start + length + 1), feature]
        if not neighborhood.any():
            mask[start:start + length, feature] = True
            remaining -= length
        attempts += 1
        if attempts >= max_attempts:
            raise RuntimeError("could not place the target runs without adjacency")

    lengths = run_lengths(mask)
    if (int(mask.sum()) != target or lengths.max() > block_length or
            not np.all(mask.sum(axis=0) >= block_length)):
        raise AssertionError("strict-block mask invariant failed")
    if np.any(mask[:edge_guard]) or np.any(mask[rows - edge_guard:]):
        raise AssertionError("strict-block mask crossed the edge guard")
    return mask
