from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from abcm.data import add_adjusted_prices, add_calendar_gap_segments, list_data_files, load_data_files
from abcm.dataset import ABCMWindowSampler
from abcm.features import (
    FEATURE_COLUMNS,
    add_forward_return_labels,
    add_price_volume_features,
    add_training_label_columns,
    prepare_feature_frame,
)
from abcm.losses import abc_loss


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text()
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _parse_simple_yaml(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value.strip())
    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"').strip("'")


def prepare_abcm_frame(
    raw_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    y1_horizon: int = 11,
    y2_horizon: int = 21,
    entry_lag: int = 1,
    label_clip_abs: float | None = None,
    label_transform: str | None = None,
) -> pd.DataFrame:
    features = feature_columns or FEATURE_COLUMNS
    frame = add_calendar_gap_segments(raw_df)
    frame = add_adjusted_prices(frame)
    frame = add_price_volume_features(frame)
    frame = add_forward_return_labels(
        frame,
        y1_horizon=y1_horizon,
        y2_horizon=y2_horizon,
        entry_lag=entry_lag,
        label_clip_abs=label_clip_abs,
        label_transform=label_transform,
    )
    frame = prepare_feature_frame(frame, features)
    return frame.sort_values(["TRADE_DT", "S_INFO_WINDCODE"]).reset_index(drop=True)


def load_or_prepare_abcm_frame(
    raw_df: pd.DataFrame,
    cache_path: str | Path | None = None,
    feature_columns: list[str] | None = None,
    y1_horizon: int = 11,
    y2_horizon: int = 21,
    entry_lag: int = 1,
    label_clip_abs: float | None = None,
    label_transform: str | None = None,
    wait_seconds: int = 7200,
) -> pd.DataFrame:
    if cache_path is None or str(cache_path) == "":
        return prepare_abcm_frame(
            raw_df,
            feature_columns=feature_columns,
            y1_horizon=y1_horizon,
            y2_horizon=y2_horizon,
            entry_lag=entry_lag,
            label_clip_abs=label_clip_abs,
            label_transform=label_transform,
        )

    path = Path(cache_path)
    if path.exists():
        return add_training_label_columns(pd.read_pickle(path), label_transform=label_transform)

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    lock_fd: int | None = None
    start = time.monotonic()
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if path.exists():
                return add_training_label_columns(pd.read_pickle(path), label_transform=label_transform)
            if time.monotonic() - start > wait_seconds:
                raise TimeoutError(f"Timed out waiting for prepared frame cache: {path}")
            time.sleep(5)

    try:
        if path.exists():
            return add_training_label_columns(pd.read_pickle(path), label_transform=label_transform)
        frame = prepare_abcm_frame(
            raw_df,
            feature_columns=feature_columns,
            y1_horizon=y1_horizon,
            y2_horizon=y2_horizon,
            entry_lag=entry_lag,
            label_clip_abs=label_clip_abs,
            label_transform=label_transform,
        )
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        frame.to_pickle(tmp_path)
        os.replace(tmp_path, path)
        return frame
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_or_prepare_abcm_frame_from_files(
    data_root: str | Path,
    max_files: int | None = None,
    cache_path: str | Path | None = None,
    feature_columns: list[str] | None = None,
    y1_horizon: int = 11,
    y2_horizon: int = 21,
    entry_lag: int = 1,
    label_clip_abs: float | None = None,
    label_transform: str | None = None,
) -> pd.DataFrame:
    if cache_path is not None and str(cache_path) != "":
        path = Path(cache_path)
        if path.exists():
            return add_training_label_columns(pd.read_pickle(path), label_transform=label_transform)

    selected_files = list_data_files(data_root)
    if max_files is not None:
        selected_files = selected_files[:max_files]
    raw = load_data_files(data_root, selected_files)
    return load_or_prepare_abcm_frame(
        raw,
        cache_path=cache_path,
        feature_columns=feature_columns,
        y1_horizon=y1_horizon,
        y2_horizon=y2_horizon,
        entry_lag=entry_lag,
        label_clip_abs=label_clip_abs,
        label_transform=label_transform,
    )


def _training_label_columns(frame: pd.DataFrame) -> tuple[str, str]:
    y1_col = "y1_train" if "y1_train" in frame.columns else "y1_raw"
    y2_col = "y2_train" if "y2_train" in frame.columns else "y2_raw"
    return y1_col, y2_col


def available_training_dates(
    frame: pd.DataFrame,
    lookback: int,
    min_stocks: int = 32,
    date_col: str = "TRADE_DT",
) -> list[str]:
    y1_col, y2_col = _training_label_columns(frame)
    counts = frame.loc[frame[y1_col].notna() & frame[y2_col].notna()].groupby(date_col).size()
    eligible = set(counts[counts >= min_stocks].index.astype(str).tolist())
    if "segment_id" not in frame.columns:
        all_dates = sorted(frame[date_col].astype(str).unique().tolist())
        min_date = all_dates[min(lookback - 1, len(all_dates) - 1)] if all_dates else None
        return [date for date in all_dates if min_date is not None and date >= min_date and date in eligible]

    selected = []
    segment_dates = (
        frame[["segment_id", date_col]]
        .drop_duplicates()
        .assign(**{date_col: lambda data: data[date_col].astype(str)})
        .sort_values(["segment_id", date_col])
    )
    for _, group in segment_dates.groupby("segment_id", sort=False):
        dates = group[date_col].tolist()
        for idx, date in enumerate(dates):
            if idx >= lookback - 1 and date in eligible:
                selected.append(date)
    return sorted(selected)


def available_paired_training_dates(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    lookback: int = 60,
    min_stocks: int = 32,
    turnover_lag: int = 5,
    stock_limit: int = 512,
) -> list[str]:
    candidate_dates = available_training_dates(frame, lookback=lookback, min_stocks=min_stocks)
    candidate_set = set(candidate_dates)
    if "segment_id" not in frame.columns:
        all_dates = sorted(frame["TRADE_DT"].astype(str).unique().tolist())
        date_to_index = {date: idx for idx, date in enumerate(all_dates)}
        min_index = lookback - 1 + turnover_lag
        return [
            date
            for date in candidate_dates
            if date_to_index.get(date, -1) >= min_index
        ]

    selected = []
    segment_dates = (
        frame[["segment_id", "TRADE_DT"]]
        .drop_duplicates()
        .assign(TRADE_DT=lambda data: data["TRADE_DT"].astype(str))
        .sort_values(["segment_id", "TRADE_DT"])
    )
    min_index = lookback - 1 + turnover_lag
    for _, group in segment_dates.groupby("segment_id", sort=False):
        dates = group["TRADE_DT"].tolist()
        for idx, date in enumerate(dates):
            if idx >= min_index and date in candidate_set:
                selected.append(date)
    return sorted(selected)


def train_one_batch(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    date: str | None = None,
    dates: list[str] | None = None,
    lookback: int = 60,
    stock_limit: int = 512,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    turnover_lag: int | None = None,
    sampler: ABCMWindowSampler | None = None,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    features = feature_columns or FEATURE_COLUMNS
    sampler = sampler or ABCMWindowSampler(frame, features, lookback=lookback)
    if dates is not None:
        date_list = [str(item) for item in dates]
    elif date is None:
        eligible_dates = available_training_dates(frame, lookback=lookback, min_stocks=1)
        if not eligible_dates:
            raise ValueError("No eligible training dates found")
        date_list = [eligible_dates[0]]
    else:
        date_list = [str(date)]
    paired = None
    if turnover_lag is not None and turnover_lag > 0:
        try:
            paired = sampler.sample_pair_for_dates(date_list, lag_periods=turnover_lag, stock_limit=stock_limit, seed=42)
            batch = paired.current
        except ValueError:
            batch = sampler.sample_for_dates(date_list, stock_limit=stock_limit, seed=42)
    else:
        batch = sampler.sample_for_dates(date_list, stock_limit=stock_limit, seed=42)
    x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
    y1 = torch.as_tensor(batch.y1, dtype=torch.float32, device=device)
    y2 = torch.as_tensor(batch.y2, dtype=torch.float32, device=device)
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    factors, alpha, beta = model(x)
    beta_prev = None
    if paired is not None:
        x_prev = torch.as_tensor(paired.previous.x, dtype=torch.float32, device=device)
        _, _, beta_prev = model(x_prev)
        beta_prev = beta_prev.detach()
    loss = abc_loss(factors, alpha, beta, y1, y2, beta_prev=beta_prev, **(loss_weights or {}))
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": float(loss.total.detach().cpu()),
        "mse": float(loss.mse.detach().cpu()),
        "r2_residual": float(loss.r2_residual.detach().cpu()),
        "corr": float(loss.corr.detach().cpu()),
        "turnover": float(loss.turnover.detach().cpu()),
        "n_stocks": float(batch.x.shape[1]),
        "n_dates": float(batch.x.shape[0]),
        "date": float(batch.dates[0]),
    }


def evaluate_loss_for_dates(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    dates: list[str],
    feature_columns: list[str] | None = None,
    lookback: int = 60,
    stock_limit: int = 512,
    device: str | torch.device = "cpu",
    turnover_lag: int | None = None,
    sampler: ABCMWindowSampler | None = None,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    features = feature_columns or FEATURE_COLUMNS
    sampler = sampler or ABCMWindowSampler(frame, features, lookback=lookback)
    model.to(device)
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for date in dates:
            paired = None
            if turnover_lag is not None and turnover_lag > 0:
                try:
                    paired = sampler.sample_pair_for_date(date, lag_periods=turnover_lag, stock_limit=stock_limit, seed=42)
                    batch = paired.current
                except ValueError:
                    batch = sampler.sample_for_date(date, stock_limit=stock_limit, seed=42)
            else:
                batch = sampler.sample_for_date(date, stock_limit=stock_limit, seed=42)
            x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
            y1 = torch.as_tensor(batch.y1, dtype=torch.float32, device=device)
            y2 = torch.as_tensor(batch.y2, dtype=torch.float32, device=device)
            factors, alpha, beta = model(x)
            beta_prev = None
            if paired is not None:
                x_prev = torch.as_tensor(paired.previous.x, dtype=torch.float32, device=device)
                _, _, beta_prev = model(x_prev)
            loss = abc_loss(factors, alpha, beta, y1, y2, beta_prev=beta_prev, **(loss_weights or {}))
            rows.append(
                {
                    "loss": float(loss.total.cpu()),
                    "mse": float(loss.mse.cpu()),
                    "r2_residual": float(loss.r2_residual.cpu()),
                    "corr": float(loss.corr.cpu()),
                    "turnover": float(loss.turnover.cpu()),
                    "n_stocks": float(batch.x.shape[1]),
                }
            )
    if not rows:
        raise ValueError("No validation dates evaluated")
    keys = rows[0].keys()
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    result["n_dates"] = float(len(rows))
    return result


def export_factors_for_dates(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    dates: list[str],
    feature_columns: list[str] | None = None,
    lookback: int = 60,
    stock_limit: int = 512,
    device: str | torch.device = "cpu",
    sampler: ABCMWindowSampler | None = None,
) -> pd.DataFrame:
    features = feature_columns or FEATURE_COLUMNS
    sampler = sampler or ABCMWindowSampler(frame, features, lookback=lookback)
    label_lookup = (
        frame[["TRADE_DT", "S_INFO_WINDCODE", "y1_raw", "y2_raw"]]
        .assign(
            TRADE_DT=lambda data: data["TRADE_DT"].astype(str),
            S_INFO_WINDCODE=lambda data: data["S_INFO_WINDCODE"].astype(str),
        )
        .drop_duplicates(["TRADE_DT", "S_INFO_WINDCODE"], keep="last")
        .set_index(["TRADE_DT", "S_INFO_WINDCODE"])
    )
    date_to_segment: dict[str, Any] = {}
    if "segment_id" in frame.columns:
        date_segments = (
            frame[["TRADE_DT", "segment_id"]]
            .drop_duplicates()
            .assign(TRADE_DT=lambda data: data["TRADE_DT"].astype(str))
        )
        segment_counts = date_segments.groupby("TRADE_DT")["segment_id"].nunique()
        ambiguous_dates = segment_counts[segment_counts > 1].index.tolist()
        if ambiguous_dates:
            raise ValueError(f"Dates appear in multiple segments: {ambiguous_dates[:5]}")
        date_to_segment = dict(zip(date_segments["TRADE_DT"], date_segments["segment_id"]))
    model.to(device)
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for date in dates:
            batch = sampler.sample_for_date(date, stock_limit=stock_limit, seed=42)
            x = torch.as_tensor(batch.x, dtype=torch.float32, device=device)
            factors, alpha, beta = model(x)
            alpha_np = alpha.detach().cpu().numpy()[0]
            beta_np = beta.detach().cpu().numpy()[0]
            for idx, code in enumerate(batch.codes[0]):
                raw_labels = label_lookup.loc[(batch.dates[0], code)]
                row: dict[str, object] = {
                    "TRADE_DT": batch.dates[0],
                    "S_INFO_WINDCODE": code,
                    "alpha_0": float(alpha_np[idx, 0]),
                    "y1_raw": float(raw_labels["y1_raw"]),
                    "y2_raw": float(raw_labels["y2_raw"]),
                }
                if date_to_segment:
                    row["segment_id"] = date_to_segment[batch.dates[0]]
                for beta_idx in range(beta_np.shape[1]):
                    row[f"beta_{beta_idx}"] = float(beta_np[idx, beta_idx])
                rows.append(row)
    return pd.DataFrame(rows)
