from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    embargo: int = 10,
) -> Iterator[Fold]:
    """
    Walk-forward CV:
      - Data is assumed time-ordered.
      - For each fold, test is a contiguous block.
      - Train is all data strictly before test start, with an embargo gap.

    Parameters
    ----------
    df : DataFrame
        Must be time-ordered already (e.g., by ts).
    n_folds : int
        Number of folds.
    embargo : int
        Number of rows to drop right before test block (reduces leakage for overlapping targets).

    Yields
    ------
    Fold
    """
    n = len(df)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    # Split indices into n_folds contiguous blocks
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: (n % n_folds)] += 1

    starts = np.cumsum(np.r_[0, fold_sizes[:-1]])
    ends = starts + fold_sizes

    for k, (s, e) in enumerate(zip(starts, ends)):
        test_idx = np.arange(s, e)

        train_end = max(0, s - embargo)
        train_idx = np.arange(0, train_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield Fold(fold=k, train_idx=train_idx, test_idx=test_idx)


def assert_time_ordered(df: pd.DataFrame, ts_col: str = "ts") -> None:
    ts = pd.to_datetime(df[ts_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("DataFrame is not time-ordered by ts.")
