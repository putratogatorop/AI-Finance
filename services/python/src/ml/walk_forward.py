"""Walk-forward time-series cross-validation splitter.

Generates chronological train/val/test windows with embargo gaps
to prevent target leakage. Windows slide forward by step_size.
"""

from __future__ import annotations


def walk_forward_splits(
    n_samples: int,
    train_size: int = 26208,
    val_size: int = 2880,
    test_size: int = 2880,
    step_size: int = 1344,
    embargo: int = 16,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Generate walk-forward train/val/test index ranges.

    Parameters
    ----------
    n_samples : total rows in dataset
    train_size : training window (6 months of 15min = 26,208)
    val_size : validation window (1 month = 2,880)
    test_size : test window (1 month = 2,880)
    step_size : slide step (2 weeks = 1,344)
    embargo : gap between windows to prevent target leakage (= target horizon)

    Returns
    -------
    List of (train, val, test) tuples, each a (start, end) index pair.
    Indices are [start, end) — use df.iloc[start:end].
    """
    splits = []
    offset = 0

    while True:
        train_start = offset
        train_end = train_start + train_size

        val_start = train_end + embargo
        val_end = val_start + val_size

        test_start = val_end + embargo
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        splits.append((
            (train_start, train_end),
            (val_start, val_end),
            (test_start, test_end),
        ))

        offset += step_size

    return splits
