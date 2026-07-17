from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d",
    "open_gap",
    "close_open_ret",
    "high_low_range",
    "vwap_close_gap",
    "log_volume",
    "log_amount",
    "volume_ratio_20",
    "amount_ratio_20",
    "ret_5d",
    "ret_20d",
    "vol_20d",
]


def _group_keys(df: pd.DataFrame) -> list[str]:
    keys = []
    if "segment_id" in df.columns:
        keys.append("segment_id")
    keys.append("S_INFO_WINDCODE")
    return keys


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator.astype(float) / denominator.astype(float) - 1.0
    return result.replace([np.inf, -np.inf], np.nan)


def add_price_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values([*_group_keys(df), "TRADE_DT"]).copy()
    grouped = out.groupby(_group_keys(out), sort=False, group_keys=False)
    out["ret_1d"] = grouped["adj_close"].pct_change()
    out["open_gap"] = _safe_divide(out["adj_open"], out["adj_preclose"])
    out["close_open_ret"] = _safe_divide(out["adj_close"], out["adj_open"])
    out["high_low_range"] = _safe_divide(out["adj_high"], out["adj_low"])
    out["vwap_close_gap"] = _safe_divide(out["adj_vwap"], out["adj_close"])
    out["log_volume"] = np.log1p(out["S_DQ_VOLUME"].astype(float).clip(lower=0))
    out["log_amount"] = np.log1p(out["S_DQ_AMOUNT"].astype(float).clip(lower=0))
    out["volume_ratio_20"] = grouped["S_DQ_VOLUME"].transform(
        lambda s: s.astype(float) / s.astype(float).rolling(20, min_periods=1).mean() - 1.0
    )
    out["amount_ratio_20"] = grouped["S_DQ_AMOUNT"].transform(
        lambda s: s.astype(float) / s.astype(float).rolling(20, min_periods=1).mean() - 1.0
    )
    out["ret_5d"] = grouped["adj_close"].pct_change(5)
    out["ret_20d"] = grouped["adj_close"].pct_change(20)
    out["vol_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=2).std())
    return out.replace([np.inf, -np.inf], np.nan)


def add_forward_return_labels(
    df: pd.DataFrame,
    y1_horizon: int = 11,
    y2_horizon: int = 21,
    entry_lag: int = 1,
    label_clip_abs: float | None = None,
    label_transform: str | None = None,
) -> pd.DataFrame:
    out = df.sort_values([*_group_keys(df), "TRADE_DT"]).copy()
    grouped = out.groupby(_group_keys(out), sort=False, group_keys=False)
    entry = grouped["adj_close"].shift(-entry_lag)
    y1_exit = grouped["adj_close"].shift(-y1_horizon)
    y2_exit = grouped["adj_close"].shift(-y2_horizon)
    out["y1_raw"] = _safe_divide(y1_exit, entry)
    out["y2_raw"] = _safe_divide(y2_exit, entry)
    if label_clip_abs is not None:
        clip_abs = abs(float(label_clip_abs))
        out["y1_raw"] = out["y1_raw"].clip(lower=-clip_abs, upper=clip_abs)
        out["y2_raw"] = out["y2_raw"].clip(lower=-clip_abs, upper=clip_abs)
    out = add_training_label_columns(out, label_transform=label_transform)
    return out


def _rank_centered_one_date(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.dropna()
    n_valid = len(valid)
    if n_valid == 0:
        return out
    if n_valid == 1:
        out.loc[valid.index] = 0.0
        return out
    ranks = valid.rank(method="average")
    out.loc[valid.index] = ((ranks - 1.0) / (n_valid - 1.0)) * 2.0 - 1.0
    return out


def _zscore_one_date(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return out
    std = valid.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        out.loc[valid.index] = 0.0
        return out
    out.loc[valid.index] = (valid - valid.mean()) / std
    return out


def add_training_label_columns(
    df: pd.DataFrame,
    label_transform: str | None = None,
    date_col: str = "TRADE_DT",
) -> pd.DataFrame:
    transform = "raw" if label_transform in {None, ""} else str(label_transform)
    out = df.copy()
    if transform == "raw":
        out["y1_train"] = out["y1_raw"]
        out["y2_train"] = out["y2_raw"]
        return out
    if transform == "rank_centered":
        transform_fn = _rank_centered_one_date
    elif transform == "zscore":
        transform_fn = _zscore_one_date
    else:
        raise ValueError(f"Unsupported label_transform: {label_transform}")
    out["y1_train"] = out.groupby(date_col, group_keys=False)["y1_raw"].transform(transform_fn)
    out["y2_train"] = out.groupby(date_col, group_keys=False)["y2_raw"].transform(transform_fn)
    return out


def _mad_zscore_one_date(series: pd.Series, n_sigma: float) -> pd.Series:
    values = series.astype(float)
    median = values.median()
    if np.isnan(median):
        return pd.Series(0.0, index=series.index)
    filled = values.fillna(median)
    mad = np.median(np.abs(filled.to_numpy() - median))
    if mad > 0:
        width = n_sigma * mad * 1.4826
        filled = filled.clip(median - width, median + width)
    mean = filled.mean()
    std = filled.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return ((filled - mean) / std).fillna(0.0)


def cross_sectional_mad_zscore(
    df: pd.DataFrame,
    columns: list[str],
    date_col: str = "TRADE_DT",
    n_sigma: float = 5.0,
) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = out.groupby(date_col, group_keys=False)[column].transform(
            lambda s: _mad_zscore_one_date(s, n_sigma)
        )
    return out


def prepare_feature_frame(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    cols = feature_columns or FEATURE_COLUMNS
    return cross_sectional_mad_zscore(df, cols)
