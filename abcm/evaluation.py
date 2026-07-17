from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from abcm.metrics import factor_autocorrelation, ic_summary, r_square_cross_section, rank_ic


@dataclass(frozen=True)
class EvaluationResult:
    beta_metrics: pd.DataFrame
    alpha_metrics: pd.DataFrame
    daily_rsquare: pd.DataFrame
    rolling_rsquare: pd.DataFrame
    prediction_accuracy: pd.DataFrame
    alpha_group_returns: pd.DataFrame
    alpha_long_short: pd.DataFrame


def _factor_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted(
        [col for col in df.columns if col.startswith(prefix)],
        key=lambda name: int(name.rsplit("_", 1)[1]) if name.rsplit("_", 1)[1].isdigit() else name,
    )


def _metric_table(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str,
    autocorr_lag: int,
) -> pd.DataFrame:
    rows = []
    for factor_col in factor_cols:
        ic = rank_ic(df, factor_col, return_col)
        summary = ic_summary(ic)
        autocorr = factor_autocorrelation(df, factor_col, lag_periods=autocorr_lag).mean()
        rows.append({"factor": factor_col, **summary, "autocorrelation": float(autocorr)})
    return pd.DataFrame(rows)


def _rebalance_dates(df: pd.DataFrame, rebalance_every: int, date_col: str, segment_col: str) -> list[str]:
    rebalance_every = max(int(rebalance_every), 1)
    if segment_col in df.columns:
        date_frame = (
            df[[segment_col, date_col]]
            .drop_duplicates()
            .assign(**{date_col: lambda data: data[date_col].astype(str)})
            .sort_values([segment_col, date_col])
        )
        selected: list[str] = []
        for _, group in date_frame.groupby(segment_col, sort=False):
            selected.extend(group[date_col].iloc[::rebalance_every].tolist())
        return selected
    dates = sorted(df[date_col].astype(str).unique().tolist())
    return dates[::rebalance_every]


def prediction_accuracy_table(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "y1_raw",
    date_col: str = "TRADE_DT",
) -> pd.DataFrame:
    rows = []
    for factor_col in factor_cols:
        ic = rank_ic(df, factor_col, return_col, date_col=date_col).dropna()
        rankic_mean = float(ic.mean()) if not ic.empty else np.nan
        orientation = -1.0 if np.isfinite(rankic_mean) and rankic_mean < 0 else 1.0
        hit_values = []
        daily_hit_values = []
        for _, group in df.groupby(date_col):
            clean = group[[factor_col, return_col]].dropna()
            if len(clean) < 2:
                continue
            oriented = orientation * clean[factor_col].astype(float)
            factor_median = oriented.median()
            return_values = clean[return_col].astype(float)
            return_median = return_values.median()
            hits = (oriented >= factor_median).to_numpy() == (return_values >= return_median).to_numpy()
            hit_values.extend(hits.tolist())
            daily_hit_values.append(float(np.mean(hits)))
        rows.append(
            {
                "factor": factor_col,
                "orientation": orientation,
                "rankic": rankic_mean,
                "cross_sectional_hit_rate": float(np.mean(hit_values)) if hit_values else np.nan,
                "mean_daily_hit_rate": float(np.mean(daily_hit_values)) if daily_hit_values else np.nan,
                "n_dates": float(len(daily_hit_values)),
                "n_observations": float(len(hit_values)),
            }
        )
    return pd.DataFrame(rows)


def alpha_group_return_tables(
    df: pd.DataFrame,
    factor_col: str = "alpha_0",
    return_col: str = "y1_raw",
    n_groups: int = 20,
    rebalance_every: int = 5,
    date_col: str = "TRADE_DT",
    segment_col: str = "segment_id",
    orientation: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if orientation is None:
        ic = rank_ic(df, factor_col, return_col, date_col=date_col).dropna()
        rankic_mean = float(ic.mean()) if not ic.empty else np.nan
        orientation = -1.0 if np.isfinite(rankic_mean) and rankic_mean < 0 else 1.0
    selected_dates = set(_rebalance_dates(df, rebalance_every, date_col, segment_col))
    frame = df.loc[df[date_col].astype(str).isin(selected_dates)].copy()
    rows = []
    long_short_rows = []
    for date, group in frame.groupby(date_col):
        clean = group[[factor_col, return_col]].dropna().copy()
        if len(clean) < n_groups:
            continue
        clean["_score"] = float(orientation) * clean[factor_col].astype(float)
        clean["_group"] = pd.qcut(
            clean["_score"].rank(method="first"),
            q=n_groups,
            labels=False,
            duplicates="drop",
        )
        universe_return = float(clean[return_col].mean())
        grouped = clean.groupby("_group")[return_col].agg(["mean", "count"]).reset_index()
        if grouped.empty:
            continue
        top_group = int(grouped["_group"].max())
        bottom_group = int(grouped["_group"].min())
        top_return = float(grouped.loc[grouped["_group"] == top_group, "mean"].iloc[0])
        bottom_return = float(grouped.loc[grouped["_group"] == bottom_group, "mean"].iloc[0])
        long_short_rows.append(
            {
                "TRADE_DT": str(date),
                "orientation": float(orientation),
                "top_group": top_group,
                "bottom_group": bottom_group,
                "top_return": top_return,
                "bottom_return": bottom_return,
                "universe_return": universe_return,
                "top_excess_return": top_return - universe_return,
                "bottom_excess_return": bottom_return - universe_return,
                "long_short_return": top_return - bottom_return,
            }
        )
        for item in grouped.itertuples(index=False):
            group_id = int(getattr(item, "_0"))
            mean_return = float(item.mean)
            rows.append(
                {
                    "TRADE_DT": str(date),
                    "orientation": float(orientation),
                    "group": group_id,
                    "mean_return": mean_return,
                    "excess_return": mean_return - universe_return,
                    "n_stocks": int(item.count),
                }
            )
    group_frame = pd.DataFrame(rows)
    if group_frame.empty:
        group_summary = pd.DataFrame(
            columns=[
                "group",
                "orientation",
                "mean_return",
                "mean_excess_return",
                "positive_rate",
                "mean_n_stocks",
                "n_periods",
            ]
        )
    else:
        group_summary = (
            group_frame.groupby("group")
            .agg(
                orientation=("orientation", "first"),
                mean_return=("mean_return", "mean"),
                mean_excess_return=("excess_return", "mean"),
                positive_rate=("mean_return", lambda values: float((values > 0).mean())),
                mean_n_stocks=("n_stocks", "mean"),
                n_periods=("TRADE_DT", "nunique"),
            )
            .reset_index()
        )
    return group_summary, pd.DataFrame(long_short_rows)


def daily_rsquare_table(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "y2_raw",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    rows = []
    group_cols = [segment_col, "TRADE_DT"] if segment_col in df.columns else ["TRADE_DT"]
    for key, group in df.groupby(group_cols):
        if segment_col in df.columns:
            segment_id, date = key
        else:
            date = key
            segment_id = None
        x = group[factor_cols].to_numpy(dtype=float)
        y = group[return_col].to_numpy(dtype=float)
        row = {"TRADE_DT": str(date), "r_square": r_square_cross_section(x, y)}
        if segment_id is not None:
            row[segment_col] = segment_id
        rows.append(row)
    sort_cols = [segment_col, "TRADE_DT"] if rows and segment_col in rows[0] else ["TRADE_DT"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def rolling_rsquare_table(
    daily: pd.DataFrame,
    rolling_window: int = 243,
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    out = daily.copy()
    window = max(int(rolling_window), 1)
    if segment_col in out.columns:
        out = out.sort_values([segment_col, "TRADE_DT"]).reset_index(drop=True)
        out["rolling_rsquare"] = out.groupby(segment_col, sort=False)["r_square"].transform(
            lambda values: values.rolling(window=window, min_periods=1).mean()
        )
        return out[[segment_col, "TRADE_DT", "rolling_rsquare"]]
    if len(out):
        window = min(window, len(out))
    out["rolling_rsquare"] = out["r_square"].rolling(window=window, min_periods=1).mean()
    return out[["TRADE_DT", "rolling_rsquare"]]


def evaluate_factor_frame(
    df: pd.DataFrame,
    rolling_window: int = 243,
    autocorr_lag: int = 5,
) -> EvaluationResult:
    required = {"TRADE_DT", "S_INFO_WINDCODE", "y1_raw", "y2_raw"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required evaluation columns: {sorted(missing)}")
    alpha_cols = _factor_columns(df, "alpha_")
    beta_cols = _factor_columns(df, "beta_")
    factor_cols = alpha_cols + beta_cols
    if not alpha_cols:
        raise ValueError("No alpha factor columns found")
    if not beta_cols:
        raise ValueError("No beta factor columns found")
    beta_metrics = _metric_table(df, beta_cols, "y1_raw", autocorr_lag=autocorr_lag)
    alpha_metrics = _metric_table(df, alpha_cols, "y1_raw", autocorr_lag=autocorr_lag)
    daily = daily_rsquare_table(df, factor_cols, return_col="y2_raw")
    rolling = rolling_rsquare_table(daily, rolling_window=rolling_window)
    prediction_accuracy = prediction_accuracy_table(df, factor_cols, return_col="y1_raw")
    alpha_group_returns, alpha_long_short = alpha_group_return_tables(
        df,
        factor_col=alpha_cols[0],
        return_col="y1_raw",
    )
    return EvaluationResult(
        beta_metrics=beta_metrics,
        alpha_metrics=alpha_metrics,
        daily_rsquare=daily,
        rolling_rsquare=rolling,
        prediction_accuracy=prediction_accuracy,
        alpha_group_returns=alpha_group_returns,
        alpha_long_short=alpha_long_short,
    )


def write_evaluation_outputs(result: EvaluationResult, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result.beta_metrics.to_csv(root / "metrics_beta.csv", index=False)
    result.alpha_metrics.to_csv(root / "metrics_alpha.csv", index=False)
    result.daily_rsquare.to_csv(root / "rsquare_daily.csv", index=False)
    result.rolling_rsquare.to_csv(root / "rolling_rsquare.csv", index=False)
    result.prediction_accuracy.to_csv(root / "prediction_accuracy.csv", index=False)
    result.alpha_group_returns.to_csv(root / "alpha_group_returns.csv", index=False)
    result.alpha_long_short.to_csv(root / "alpha_long_short.csv", index=False)
