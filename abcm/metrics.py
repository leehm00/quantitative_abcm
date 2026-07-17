from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "TRADE_DT",
) -> pd.Series:
    values = {}
    for date, group in df.groupby(date_col):
        clean = group[[factor_col, return_col]].dropna()
        if len(clean) < 2:
            values[date] = np.nan
        else:
            factor = clean[factor_col].astype(float)
            returns = clean[return_col].astype(float)
            if factor.nunique(dropna=True) < 2 or returns.nunique(dropna=True) < 2:
                values[date] = np.nan
            else:
                values[date] = factor.corr(returns, method="spearman")
    return pd.Series(values, name=factor_col)


def r_square_cross_section(x: np.ndarray, y: np.ndarray, ridge_eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y[:, 0])
    x = x[valid]
    y = y[valid]
    if x.shape[0] <= x.shape[1]:
        return np.nan
    xtx = x.T @ x
    coef = np.linalg.solve(xtx + ridge_eps * np.eye(x.shape[1]), x.T @ y)
    y_hat = x @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum(y**2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def factor_autocorrelation(
    df: pd.DataFrame,
    factor_col: str,
    lag_periods: int = 5,
    date_col: str = "TRADE_DT",
    code_col: str = "S_INFO_WINDCODE",
    segment_col: str = "segment_id",
) -> pd.Series:
    if segment_col in df.columns:
        series = [
            _factor_autocorrelation_one_segment(group, factor_col, lag_periods, date_col, code_col)
            for _, group in df.groupby(segment_col, sort=False)
        ]
        series = [item for item in series if not item.empty]
        if not series:
            return pd.Series(dtype=float, name=factor_col)
        out = pd.concat(series)
        out.index = out.index.astype(str)
        return out.sort_index().rename(factor_col)
    return _factor_autocorrelation_one_segment(df, factor_col, lag_periods, date_col, code_col)


def _factor_autocorrelation_one_segment(
    df: pd.DataFrame,
    factor_col: str,
    lag_periods: int,
    date_col: str,
    code_col: str,
) -> pd.Series:
    pivot = df.pivot_table(
        index=date_col,
        columns=code_col,
        values=factor_col,
        aggfunc="first",
    )
    pivot.index = pivot.index.astype(str)
    pivot = pivot.sort_index()
    dates = pivot.index.tolist()
    values = {}
    for idx in range(lag_periods, len(dates)):
        current_date = dates[idx]
        current = pivot.iloc[idx].to_numpy(dtype=float)
        previous = pivot.iloc[idx - lag_periods].to_numpy(dtype=float)
        valid = np.isfinite(current) & np.isfinite(previous)
        if valid.sum() < 2:
            values[current_date] = np.nan
        else:
            current_values = current[valid]
            previous_values = previous[valid]
            if current_values.std() == 0 or previous_values.std() == 0:
                values[current_date] = np.nan
            else:
                values[current_date] = float(np.corrcoef(current_values, previous_values)[0, 1])
    return pd.Series(values, name=factor_col)


def ic_summary(ic: pd.Series) -> dict[str, float]:
    clean = ic.dropna()
    if clean.empty:
        return {"rankic": np.nan, "abs_rankic": np.nan, "icir": np.nan, "win_rate": np.nan}
    std = clean.std()
    return {
        "rankic": float(clean.mean()),
        "abs_rankic": float(clean.abs().mean()),
        "icir": float(clean.mean() / std) if std and np.isfinite(std) else np.nan,
        "win_rate": float((clean > 0).mean()),
    }
