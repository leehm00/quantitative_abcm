from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ABCMBatch:
    x: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    dates: list[str]
    codes: list[list[str]]


@dataclass(frozen=True)
class ABCMPairedBatch:
    current: ABCMBatch
    previous: ABCMBatch


class ABCMWindowSampler:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        lookback: int = 60,
        min_history: int | None = None,
        date_col: str = "TRADE_DT",
        code_col: str = "S_INFO_WINDCODE",
    ):
        self.df = df.sort_values([date_col, code_col]).copy()
        self.df[date_col] = self.df[date_col].astype(str)
        self.df[code_col] = self.df[code_col].astype(str)
        self.feature_columns = feature_columns
        self.lookback = lookback
        self.min_history = min_history or max(1, int(lookback * 0.9))
        self.date_col = date_col
        self.code_col = code_col
        self.y1_col = "y1_train" if "y1_train" in self.df.columns else "y1_raw"
        self.y2_col = "y2_train" if "y2_train" in self.df.columns else "y2_raw"
        self._by_key = {}
        self._pos_by_key: dict[tuple[object, str], dict[str, int]] = {}
        self._features_by_key: dict[tuple[object, str], np.ndarray] = {}
        self._y1_by_key: dict[tuple[object, str], np.ndarray] = {}
        self._y2_by_key: dict[tuple[object, str], np.ndarray] = {}
        for key, group in self.df.groupby(["segment_id", code_col], sort=False):
            history = group.sort_values(date_col).reset_index(drop=True)
            self._by_key[key] = history
            self._pos_by_key[key] = {
                str(value): idx for idx, value in enumerate(history[date_col].tolist())
            }
            self._features_by_key[key] = history[self.feature_columns].to_numpy(dtype=np.float32)
            self._y1_by_key[key] = history[self.y1_col].to_numpy(dtype=np.float32)
            self._y2_by_key[key] = history[self.y2_col].to_numpy(dtype=np.float32)
        self._entries_by_date: dict[str, list[tuple[tuple[object, str], str]]] = {}
        for date, group in self.df.groupby(date_col, sort=False):
            entries = []
            for row in group[["segment_id", code_col]].itertuples(index=False):
                key = (row.segment_id, getattr(row, code_col))
                entries.append((key, key[1]))
            self._entries_by_date[str(date)] = entries
        self.available_dates = sorted(self.df[date_col].astype(str).unique().tolist())
        self._date_to_index = {date: idx for idx, date in enumerate(self.available_dates)}

    @staticmethod
    def _trim_batch(batch: ABCMBatch, n_stocks: int) -> ABCMBatch:
        return ABCMBatch(
            x=batch.x[:, :n_stocks, :, :],
            y1=batch.y1[:, :n_stocks],
            y2=batch.y2[:, :n_stocks],
            dates=batch.dates,
            codes=[batch.codes[0][:n_stocks]],
        )

    @classmethod
    def _stack_batches(cls, batches: list[ABCMBatch]) -> ABCMBatch:
        if not batches:
            raise ValueError("No batches to stack")
        n_stocks = min(batch.x.shape[1] for batch in batches)
        trimmed = [cls._trim_batch(batch, n_stocks) for batch in batches]
        dates: list[str] = []
        codes: list[list[str]] = []
        for batch in trimmed:
            dates.extend(batch.dates)
            codes.extend(batch.codes)
        return ABCMBatch(
            x=np.concatenate([batch.x for batch in trimmed], axis=0),
            y1=np.concatenate([batch.y1 for batch in trimmed], axis=0),
            y2=np.concatenate([batch.y2 for batch in trimmed], axis=0),
            dates=dates,
            codes=codes,
        )

    def sample_for_date(
        self,
        date: str,
        stock_limit: int = 512,
        seed: int | None = None,
        codes: list[str] | None = None,
        require_labels: bool = True,
    ) -> ABCMBatch:
        rng = np.random.default_rng(seed)
        entries = self._entries_by_date.get(str(date))
        if entries is None:
            raise ValueError(f"No samples available for date {date}")
        code_set = set(codes) if codes is not None else None
        if codes is not None:
            entries = [entry for entry in entries if entry[1] in code_set]
        rows = []
        codes = []
        for key, code in entries:
            end = self._pos_by_key[key].get(str(date))
            if end is None:
                continue
            start = end - self.lookback + 1
            if start < 0:
                continue
            features = self._features_by_key[key][start : end + 1]
            valid_count = np.isfinite(features).all(axis=1).sum()
            if valid_count < self.min_history:
                continue
            y1 = float(self._y1_by_key[key][end])
            y2 = float(self._y2_by_key[key][end])
            if require_labels and (not np.isfinite(y1) or not np.isfinite(y2)):
                continue
            rows.append((features, y1, y2))
            codes.append(code)
        if not rows:
            raise ValueError(f"No samples available for date {date}")
        if len(rows) > stock_limit:
            indices = rng.choice(len(rows), size=stock_limit, replace=False)
            rows = [rows[int(idx)] for idx in indices]
            codes = [codes[int(idx)] for idx in indices]
        x = np.stack([item[0] for item in rows])[None, :, :, :]
        y1 = np.asarray([item[1] for item in rows], dtype=np.float32)[None, :]
        y2 = np.asarray([item[2] for item in rows], dtype=np.float32)[None, :]
        return ABCMBatch(x=x, y1=y1, y2=y2, dates=[str(date)], codes=[codes])

    def sample_for_dates(
        self,
        dates: list[str],
        stock_limit: int = 512,
        seed: int | None = None,
        require_labels: bool = True,
    ) -> ABCMBatch:
        batches = []
        for idx, date in enumerate(dates):
            date_seed = None if seed is None else seed + idx
            batches.append(
                self.sample_for_date(
                    date,
                    stock_limit=stock_limit,
                    seed=date_seed,
                    require_labels=require_labels,
                )
            )
        return self._stack_batches(batches)

    def date_at_lag(self, date: str, lag_periods: int) -> str:
        dates = self.available_dates
        idx = self._date_to_index.get(str(date))
        if idx is None:
            raise ValueError(f"Date {date} is not available")
        previous_idx = idx - lag_periods
        if previous_idx < 0:
            raise ValueError(f"Date {date} does not have lag {lag_periods}")
        return dates[previous_idx]

    def sample_pair_for_date(
        self,
        date: str,
        lag_periods: int = 5,
        stock_limit: int = 512,
        seed: int | None = None,
    ) -> ABCMPairedBatch:
        current = self.sample_for_date(date, stock_limit=stock_limit, seed=seed)
        previous_date = self.date_at_lag(date, lag_periods)
        previous = self.sample_for_date(
            previous_date,
            stock_limit=stock_limit,
            seed=seed,
            codes=current.codes[0],
            require_labels=False,
        )
        previous_by_code = {code: idx for idx, code in enumerate(previous.codes[0])}
        shared_codes = [code for code in current.codes[0] if code in previous_by_code]
        if not shared_codes:
            raise ValueError(f"No overlapping stocks for {date} and lag {lag_periods}")
        current_by_code = {code: idx for idx, code in enumerate(current.codes[0])}
        current_indices = [current_by_code[code] for code in shared_codes]
        previous_indices = [previous_by_code[code] for code in shared_codes]
        aligned_current = ABCMBatch(
            x=current.x[:, current_indices, :, :],
            y1=current.y1[:, current_indices],
            y2=current.y2[:, current_indices],
            dates=current.dates,
            codes=[shared_codes],
        )
        aligned_previous = ABCMBatch(
            x=previous.x[:, previous_indices, :, :],
            y1=previous.y1[:, previous_indices],
            y2=previous.y2[:, previous_indices],
            dates=previous.dates,
            codes=[shared_codes],
        )
        return ABCMPairedBatch(current=aligned_current, previous=aligned_previous)

    def sample_pair_for_dates(
        self,
        dates: list[str],
        lag_periods: int = 5,
        stock_limit: int = 512,
        seed: int | None = None,
    ) -> ABCMPairedBatch:
        pairs = []
        for idx, date in enumerate(dates):
            date_seed = None if seed is None else seed + idx
            pairs.append(
                self.sample_pair_for_date(
                    date,
                    lag_periods=lag_periods,
                    stock_limit=stock_limit,
                    seed=date_seed,
                )
            )
        n_stocks = min(pair.current.x.shape[1] for pair in pairs)
        current = self._stack_batches([self._trim_batch(pair.current, n_stocks) for pair in pairs])
        previous = self._stack_batches([self._trim_batch(pair.previous, n_stocks) for pair in pairs])
        return ABCMPairedBatch(current=current, previous=previous)
