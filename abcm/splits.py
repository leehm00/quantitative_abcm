from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DateFold:
    fold_id: int
    train_dates: list[str]
    valid_dates: list[str]


def make_time_block_folds(dates: list[str], n_folds: int) -> list[DateFold]:
    unique_dates = sorted(dict.fromkeys(str(date) for date in dates))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(unique_dates) < n_folds:
        raise ValueError("Need at least one date per fold")
    fold_sizes = [len(unique_dates) // n_folds] * n_folds
    for idx in range(len(unique_dates) % n_folds):
        fold_sizes[idx] += 1
    folds = []
    start = 0
    for fold_id, size in enumerate(fold_sizes):
        end = start + size
        valid_dates = unique_dates[start:end]
        train_dates = unique_dates[:start] + unique_dates[end:]
        folds.append(DateFold(fold_id=fold_id, train_dates=train_dates, valid_dates=valid_dates))
        start = end
    return folds


def select_validation_fold(
    dates: list[str],
    n_folds: int = 5,
    fold_id: int | None = None,
) -> tuple[list[str], list[str]]:
    folds = make_time_block_folds(dates, n_folds=n_folds)
    selected = folds[-1] if fold_id is None else folds[fold_id]
    return selected.train_dates, selected.valid_dates
