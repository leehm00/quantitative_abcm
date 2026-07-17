from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_FILE_PATTERN = "testdata_*.pkl"

PRICE_COLUMNS = {
    "S_DQ_PRECLOSE": "adj_preclose",
    "S_DQ_OPEN": "adj_open",
    "S_DQ_HIGH": "adj_high",
    "S_DQ_LOW": "adj_low",
    "S_DQ_CLOSE": "adj_close",
    "S_DQ_AVGPRICE": "adj_vwap",
}


@dataclass(frozen=True)
class DataInventory:
    data_root: Path
    file_count: int
    file_numbers: list[int]
    missing_file_numbers: list[int]
    date_min: str | None
    date_max: str | None
    n_dates: int
    columns: list[str]


def _file_number(path: Path) -> int:
    match = re.search(r"testdata_(\d+)\.pkl$", path.name)
    if match is None:
        raise ValueError(f"Unexpected data file name: {path.name}")
    return int(match.group(1))


def list_data_files(data_root: str | Path) -> list[Path]:
    root = Path(data_root)
    return sorted(root.glob(DATA_FILE_PATTERN), key=_file_number)


def summarize_data_files(data_root: str | Path) -> DataInventory:
    root = Path(data_root)
    files = list_data_files(root)
    numbers = [_file_number(path) for path in files]
    missing = sorted(set(range(max(numbers) + 1)) - set(numbers)) if numbers else []
    dates: list[str] = []
    columns: list[str] = []
    for path in files:
        df = pd.read_pickle(path)
        if not columns:
            columns = list(df.columns)
        if "TRADE_DT" in df.columns:
            dates.extend(df["TRADE_DT"].astype(str).unique().tolist())
    unique_dates = sorted(set(dates))
    return DataInventory(
        data_root=root,
        file_count=len(files),
        file_numbers=numbers,
        missing_file_numbers=missing,
        date_min=unique_dates[0] if unique_dates else None,
        date_max=unique_dates[-1] if unique_dates else None,
        n_dates=len(unique_dates),
        columns=columns,
    )


def load_data_files(data_root: str | Path, files: Iterable[Path] | None = None) -> pd.DataFrame:
    selected = list(files) if files is not None else list_data_files(data_root)
    if not selected:
        raise FileNotFoundError(f"No {DATA_FILE_PATTERN} files found under {data_root}")
    df = pd.concat([pd.read_pickle(path) for path in selected], ignore_index=True)
    return df.sort_values(["TRADE_DT", "S_INFO_WINDCODE"]).reset_index(drop=True)


def add_adjusted_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in [*PRICE_COLUMNS, "S_DQ_ADJFACTOR"] if col not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for adjusted prices: {missing}")
    for raw_col, adjusted_col in PRICE_COLUMNS.items():
        out[adjusted_col] = out[raw_col].astype(float) * out["S_DQ_ADJFACTOR"].astype(float)
    return out


def add_calendar_gap_segments(
    df: pd.DataFrame,
    date_col: str = "TRADE_DT",
    max_calendar_gap_days: int = 30,
) -> pd.DataFrame:
    out = df.copy()
    unique_dates = pd.Series(sorted(out[date_col].astype(str).unique()), name=date_col)
    parsed = pd.to_datetime(unique_dates, format="%Y%m%d")
    segment_ids = parsed.diff().dt.days.gt(max_calendar_gap_days).cumsum().astype(int)
    date_to_segment = dict(zip(unique_dates.tolist(), segment_ids.tolist()))
    out["segment_id"] = out[date_col].astype(str).map(date_to_segment).astype(int)
    return out
