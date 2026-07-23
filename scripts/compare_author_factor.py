from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KEY_COLUMNS = ["TRADE_DT", "S_INFO_WINDCODE"]
PDF_ALPHA_METRICS = {
    "rankic": 0.1269,
    "icir": 0.96,
    "win_rate": 0.8663,
    "top_annualized": 0.3451,
}
MODEL_LABELS = {
    "author_factor": "作者 ABCM1 因子",
    "h32_tuned": "本地 ABCM h32",
    "h48_tuned": "本地 ABCM h48",
    "h64_tuned": "本地 ABCM h64",
    "h32_pdf_loss": "本地 ABCM h32 PDF 损失",
    "lightgbm": "LightGBM",
}
RETURN_HORIZONS = {"y1_raw": 11, "y2_raw": 21}


def _as_string_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["TRADE_DT"] = out["TRADE_DT"].astype(str)
    out["S_INFO_WINDCODE"] = out["S_INFO_WINDCODE"].astype(str)
    return out


def load_author_factors(author_dir: str | Path) -> pd.DataFrame:
    root = Path(author_dir)
    files = sorted(root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No author factor CSV files found under {root}")

    parts: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, dtype={"stock": "string"})
        if list(frame.columns) != ["stock", "factor_value"]:
            raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
        frame.insert(0, "TRADE_DT", path.stem)
        frame = frame.rename(
            columns={"stock": "S_INFO_WINDCODE", "factor_value": "author_factor"}
        )
        parts.append(frame)

    result = _as_string_keys(pd.concat(parts, ignore_index=True))
    if result.duplicated(KEY_COLUMNS).any():
        duplicate = result.loc[result.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"Duplicate author factor keys found:\n{duplicate}")
    values = pd.to_numeric(result["author_factor"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Author factor contains missing or non-finite values")
    result["author_factor"] = values.astype(np.float32)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_labels(cache_path: str | Path, min_date: str) -> pd.DataFrame:
    frame = pd.read_pickle(cache_path)
    required = {*KEY_COLUMNS, "S_DQ_PCTCHANGE", "y1_raw", "y2_raw"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Prepared cache is missing columns: {sorted(missing)}")
    labels = frame.loc[
        frame["TRADE_DT"].astype(str).ge(str(min_date)),
        [*KEY_COLUMNS, "S_DQ_PCTCHANGE", "y1_raw", "y2_raw"],
    ].copy()
    labels["daily_return"] = pd.to_numeric(
        labels.pop("S_DQ_PCTCHANGE"), errors="coerce"
    ) / 100.0
    labels = _as_string_keys(labels)
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("Prepared cache contains duplicate date-stock keys")
    return labels


def _daily_pearson_from_ranked(
    dates: pd.Series,
    x: pd.Series,
    y: pd.Series,
) -> pd.Series:
    work = pd.DataFrame(
        {
            "TRADE_DT": dates.astype(str),
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    if work.empty:
        return pd.Series(dtype=float)
    work["xy"] = work["x"] * work["y"]
    work["x2"] = work["x"] * work["x"]
    work["y2"] = work["y"] * work["y"]
    grouped = work.groupby("TRADE_DT", sort=True).agg(
        n=("x", "size"),
        sx=("x", "sum"),
        sy=("y", "sum"),
        sxy=("xy", "sum"),
        sx2=("x2", "sum"),
        sy2=("y2", "sum"),
    )
    covariance = grouped["sxy"] - grouped["sx"] * grouped["sy"] / grouped["n"]
    variance_x = grouped["sx2"] - grouped["sx"] ** 2 / grouped["n"]
    variance_y = grouped["sy2"] - grouped["sy"] ** 2 / grouped["n"]
    denominator = np.sqrt(variance_x.clip(lower=0.0) * variance_y.clip(lower=0.0))
    result = covariance / denominator.replace(0.0, np.nan)
    return result.rename("correlation")


def daily_spearman(
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str = "y1_raw",
) -> pd.Series:
    work = frame[["TRADE_DT", factor_col, return_col]].dropna().copy()
    if work.empty:
        return pd.Series(dtype=float, name=factor_col)
    factor_rank = work.groupby("TRADE_DT", sort=False)[factor_col].rank(method="average")
    return_rank = work.groupby("TRADE_DT", sort=False)[return_col].rank(method="average")
    result = _daily_pearson_from_ranked(work["TRADE_DT"], factor_rank, return_rank)
    return result.rename(factor_col)


def summarize_ic(ic: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(ic, errors="coerce").dropna()
    std = float(clean.std(ddof=1)) if len(clean) > 1 else math.nan
    mean = float(clean.mean()) if len(clean) else math.nan
    return {
        "n_dates": int(len(clean)),
        "rankic": mean,
        "rankic_std": std,
        "icir": mean / std if np.isfinite(std) and std > 0 else math.nan,
        "win_rate": float((clean > 0).mean()) if len(clean) else math.nan,
        "rankic_median": float(clean.median()) if len(clean) else math.nan,
        "rankic_p05": float(clean.quantile(0.05)) if len(clean) else math.nan,
        "rankic_p95": float(clean.quantile(0.95)) if len(clean) else math.nan,
    }


def cross_sectional_hit_rate(
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str = "y1_raw",
    orientation: float = 1.0,
) -> float:
    work = frame[["TRADE_DT", factor_col, return_col]].dropna().copy()
    if work.empty:
        return math.nan
    score = float(orientation) * work[factor_col].astype(float)
    factor_median = score.groupby(work["TRADE_DT"]).transform("median")
    return_median = work[return_col].groupby(work["TRADE_DT"]).transform("median")
    return float(((score >= factor_median) == (work[return_col] >= return_median)).mean())


def group_return_series(
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str = "y1_raw",
    orientation: float = 1.0,
    n_groups: int = 20,
    rebalance_every: int = 5,
) -> pd.DataFrame:
    work = frame[["TRADE_DT", factor_col, return_col]].dropna().copy()
    dates = sorted(work["TRADE_DT"].astype(str).unique().tolist())
    selected_dates = set(dates[:: max(int(rebalance_every), 1)])
    work = work.loc[work["TRADE_DT"].astype(str).isin(selected_dates)]
    rows: list[dict[str, float | int | str]] = []
    for date, group in work.groupby("TRADE_DT", sort=True):
        if len(group) < n_groups:
            continue
        score = float(orientation) * group[factor_col].to_numpy(dtype=float)
        returns = group[return_col].to_numpy(dtype=float)
        order = np.argsort(score, kind="mergesort")
        group_size = max(int(math.ceil(len(group) / n_groups)), 1)
        top = returns[order[-group_size:]]
        bottom = returns[order[:group_size]]
        universe_return = float(np.mean(returns))
        top_return = float(np.mean(top))
        bottom_return = float(np.mean(bottom))
        rows.append(
            {
                "TRADE_DT": str(date),
                "n_stocks": int(len(group)),
                "top_return": top_return,
                "bottom_return": bottom_return,
                "universe_return": universe_return,
                "top_excess": top_return - universe_return,
                "long_short": top_return - bottom_return,
            }
        )
    return pd.DataFrame(rows)


def trimmed_mean(values: Iterable[float], fraction: float = 0.05) -> float:
    clean = np.sort(pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy())
    if clean.size == 0:
        return math.nan
    cut = int(math.floor(clean.size * float(fraction)))
    if cut == 0:
        return float(clean.mean())
    if clean.size <= 2 * cut:
        return math.nan
    return float(clean[cut:-cut].mean())


def summarize_factor(
    frame: pd.DataFrame,
    factor_col: str,
    return_col: str = "y1_raw",
    orientation: float | None = None,
    n_groups: int = 20,
    rebalance_every: int = 5,
) -> tuple[dict[str, float | int], pd.Series, pd.DataFrame]:
    daily_ic = daily_spearman(frame, factor_col, return_col=return_col)
    summary = summarize_ic(daily_ic)
    if orientation is None:
        orientation = 1.0 if float(summary["rankic"]) >= 0 else -1.0
    groups = group_return_series(
        frame,
        factor_col,
        return_col=return_col,
        orientation=orientation,
        n_groups=n_groups,
        rebalance_every=rebalance_every,
    )
    summary.update(
        {
            "orientation": float(orientation),
            "n_observations": int(frame[[factor_col, return_col]].dropna().shape[0]),
            "mean_stocks": float(
                frame[["TRADE_DT", factor_col, return_col]]
                .dropna()
                .groupby("TRADE_DT")
                .size()
                .mean()
            ),
            "hit_rate": cross_sectional_hit_rate(
                frame,
                factor_col,
                return_col=return_col,
                orientation=orientation,
            ),
            "n_rebalance_periods": int(len(groups)),
            "top_excess_mean": float(groups["top_excess"].mean()) if len(groups) else math.nan,
            "top_excess_trimmed_mean": (
                trimmed_mean(groups["top_excess"]) if len(groups) else math.nan
            ),
            "top_annualized_approx": (
                float(groups["top_excess"].mean()) * 252.0 / 11.0
                if len(groups)
                else math.nan
            ),
            "top_trimmed_annualized_approx": (
                trimmed_mean(groups["top_excess"]) * 252.0 / 11.0
                if len(groups)
                else math.nan
            ),
            "long_short_mean": float(groups["long_short"].mean()) if len(groups) else math.nan,
            "long_short_annualized_approx": (
                float(groups["long_short"].mean()) * 252.0 / 11.0
                if len(groups)
                else math.nan
            ),
        }
    )
    return summary, daily_ic, groups


def load_local_factor_runs(
    run_dirs: Iterable[str | Path],
    factor_name: str,
    min_date: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        factor_path = Path(run_dir) / "factors.csv"
        if not factor_path.exists():
            raise FileNotFoundError(f"Missing factors file: {factor_path}")
        frame = pd.read_csv(
            factor_path,
            usecols=["TRADE_DT", "S_INFO_WINDCODE", "alpha_0", "y1_raw", "y2_raw"],
            dtype={"TRADE_DT": str, "S_INFO_WINDCODE": str},
        )
        frame = frame.loc[frame["TRADE_DT"].astype(str).ge(str(min_date))]
        frame = frame.rename(columns={"alpha_0": factor_name})
        parts.append(frame)
    result = _as_string_keys(pd.concat(parts, ignore_index=True))
    if result.duplicated(KEY_COLUMNS).any():
        duplicate = result.loc[result.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head()
        raise ValueError(f"Duplicate local factor keys for {factor_name}:\n{duplicate}")
    result[factor_name] = pd.to_numeric(result[factor_name], errors="coerce").astype(np.float32)
    return result.sort_values(KEY_COLUMNS).reset_index(drop=True)


def local_run_map(
    manifest_path: str | Path,
    lightgbm_summary_path: str | Path,
    candidates: Iterable[str],
) -> dict[str, list[str]]:
    manifest = pd.read_csv(manifest_path)
    result: dict[str, list[str]] = {}
    for candidate in candidates:
        rows = manifest.loc[manifest["candidate"] == candidate]
        if rows.empty:
            raise ValueError(f"Candidate {candidate!r} was not found in {manifest_path}")
        result[candidate] = rows.sort_values("validation_fold")["run_dir"].astype(str).tolist()
    lightgbm = pd.read_csv(lightgbm_summary_path)
    result["lightgbm"] = lightgbm.sort_values("validation_fold")["run_dir"].astype(str).tolist()
    return result


def build_common_frame(
    author_labeled: pd.DataFrame,
    runs: dict[str, list[str]],
    min_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    author_base = author_labeled.loc[
        author_labeled["TRADE_DT"].astype(str).ge(str(min_date))
        & author_labeled["y1_raw"].notna(),
        [*KEY_COLUMNS, "author_factor", "y1_raw", "y2_raw"],
    ].copy()
    common = author_base.copy()
    pairwise_frames: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, float | int | str]] = []
    for name, run_dirs in runs.items():
        local = load_local_factor_runs(run_dirs, name, min_date=min_date)
        pairwise = author_base.merge(
            local[[*KEY_COLUMNS, name, "y1_raw", "y2_raw"]],
            on=KEY_COLUMNS,
            how="inner",
            suffixes=("", f"_{name}"),
            validate="one_to_one",
        )
        y1_other = f"y1_raw_{name}"
        y2_other = f"y2_raw_{name}"
        y1_diff = (
            np.nanmax(np.abs(pairwise["y1_raw"] - pairwise[y1_other]))
            if len(pairwise)
            else math.nan
        )
        y2_diff = (
            np.nanmax(np.abs(pairwise["y2_raw"] - pairwise[y2_other]))
            if len(pairwise)
            else math.nan
        )
        pairwise = pairwise.drop(columns=[y1_other, y2_other])
        pairwise_frames[name] = pairwise.sort_values(KEY_COLUMNS).reset_index(drop=True)
        coverage_rows.append(
            {
                "model": name,
                "model_rows": int(len(local)),
                "model_dates": int(local["TRADE_DT"].nunique()),
                "pairwise_rows": int(len(pairwise)),
                "pairwise_dates": int(pairwise["TRADE_DT"].nunique()),
                "pairwise_mean_stocks": float(pairwise.groupby("TRADE_DT").size().mean()),
                "max_abs_y1_difference": float(y1_diff),
                "max_abs_y2_difference": float(y2_diff),
            }
        )
        common = common.merge(
            local[[*KEY_COLUMNS, name]],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    if common.empty:
        raise ValueError("No all-model common date-stock observations were found")
    coverage = pd.DataFrame(coverage_rows)
    coverage["all_common_rows"] = int(len(common))
    coverage["all_common_dates"] = int(common["TRADE_DT"].nunique())
    coverage["all_common_start"] = common["TRADE_DT"].min()
    coverage["all_common_end"] = common["TRADE_DT"].max()
    coverage["all_common_mean_stocks"] = float(common.groupby("TRADE_DT").size().mean())
    return common.sort_values(KEY_COLUMNS).reset_index(drop=True), coverage, pairwise_frames


def factor_autocorrelation(
    frame: pd.DataFrame,
    factor_col: str,
    lag: int,
) -> tuple[pd.Series, float]:
    dates = sorted(frame["TRADE_DT"].astype(str).unique().tolist())
    if lag <= 0 or len(dates) <= lag:
        return pd.Series(dtype=float), math.nan
    pairs = pd.DataFrame({"TRADE_DT": dates[lag:], "LAG_DT": dates[:-lag]})
    current = frame[[*KEY_COLUMNS, factor_col]].merge(pairs, on="TRADE_DT", how="inner")
    lagged = frame[[*KEY_COLUMNS, factor_col]].rename(
        columns={"TRADE_DT": "LAG_DT", factor_col: "lag_factor"}
    )
    merged = current.merge(lagged, on=["LAG_DT", "S_INFO_WINDCODE"], how="inner")
    current_rank = merged.groupby("TRADE_DT", sort=False)[factor_col].rank(method="average")
    lag_rank = merged.groupby("TRADE_DT", sort=False)["lag_factor"].rank(method="average")
    correlation = _daily_pearson_from_ranked(merged["TRADE_DT"], current_rank, lag_rank)
    mean_overlap = float(merged.groupby("TRADE_DT").size().mean())
    return correlation, mean_overlap


def _top_mask(
    frame: pd.DataFrame,
    factor_col: str,
    top_fraction: float,
) -> pd.Series:
    fraction = float(top_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    values = pd.to_numeric(frame[factor_col], errors="coerce")
    valid = values.notna()
    counts = valid.groupby(frame["TRADE_DT"], sort=False).transform("sum")
    top_counts = np.ceil(counts.astype(float) * fraction).clip(lower=1.0)
    ranks = values.groupby(frame["TRADE_DT"], sort=False).rank(
        method="first",
        ascending=False,
    )
    return valid & ranks.le(top_counts)


def top_overlap_daily(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
    top_fraction: float,
) -> pd.DataFrame:
    work = (
        frame[[*KEY_COLUMNS, left_col, right_col]]
        .dropna()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    if work.empty:
        return pd.DataFrame()
    left_top = _top_mask(work, left_col, top_fraction)
    right_top = _top_mask(work, right_col, top_fraction)
    work["left_top"] = left_top
    work["right_top"] = right_top
    work["intersection"] = left_top & right_top
    work["union"] = left_top | right_top
    daily = work.groupby("TRADE_DT", sort=True).agg(
        n_stocks=("S_INFO_WINDCODE", "size"),
        left_top_count=("left_top", "sum"),
        right_top_count=("right_top", "sum"),
        intersection_count=("intersection", "sum"),
        union_count=("union", "sum"),
    )
    minimum_top = daily[["left_top_count", "right_top_count"]].min(axis=1)
    daily["overlap_rate"] = daily["intersection_count"] / minimum_top.replace(0, np.nan)
    daily["jaccard"] = daily["intersection_count"] / daily["union_count"].replace(0, np.nan)
    daily.insert(0, "top_fraction", float(top_fraction))
    return daily.reset_index()


def disagreement_group_returns(
    frame: pd.DataFrame,
    local_col: str,
    top_fraction: float,
    return_columns: Iterable[str] = ("y1_raw", "y2_raw"),
) -> pd.DataFrame:
    selected_columns = [*KEY_COLUMNS, "author_factor", local_col, *return_columns]
    work = (
        frame[selected_columns]
        .dropna(subset=["author_factor", local_col])
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    if work.empty:
        return pd.DataFrame()
    author_top = _top_mask(work, "author_factor", top_fraction)
    local_top = _top_mask(work, local_col, top_fraction)
    work["selection_group"] = np.select(
        [
            author_top & local_top,
            author_top & ~local_top,
            ~author_top & local_top,
        ],
        ["both_top", "author_only", "local_only"],
        default="neither_top",
    )
    parts: list[pd.DataFrame] = []
    for return_col in return_columns:
        valid = work[["TRADE_DT", "selection_group", return_col]].dropna().copy()
        if valid.empty:
            continue
        universe = valid.groupby("TRADE_DT", sort=True)[return_col].mean().rename(
            "universe_return"
        )
        grouped = (
            valid.groupby(["TRADE_DT", "selection_group"], sort=True)[return_col]
            .agg(group_return="mean", n_stocks="size")
            .reset_index()
            .merge(universe.reset_index(), on="TRADE_DT", how="left")
        )
        grouped["excess_return"] = grouped["group_return"] - grouped["universe_return"]
        grouped.insert(1, "return_horizon", return_col)
        grouped.insert(2, "top_fraction", float(top_fraction))
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cross_sectional_rank_residual(
    frame: pd.DataFrame,
    target_col: str,
    control_col: str,
) -> pd.Series:
    work = frame[["TRADE_DT", target_col, control_col]].copy()
    target_rank = work.groupby("TRADE_DT", sort=False)[target_col].rank(
        method="average",
        pct=True,
    )
    control_rank = work.groupby("TRADE_DT", sort=False)[control_col].rank(
        method="average",
        pct=True,
    )
    target_centered = target_rank - target_rank.groupby(work["TRADE_DT"], sort=False).transform(
        "mean"
    )
    control_centered = control_rank - control_rank.groupby(
        work["TRADE_DT"], sort=False
    ).transform("mean")
    numerator = (target_centered * control_centered).groupby(
        work["TRADE_DT"], sort=False
    ).transform("sum")
    denominator = control_centered.pow(2).groupby(
        work["TRADE_DT"], sort=False
    ).transform("sum")
    beta = numerator / denominator.replace(0.0, np.nan)
    return (target_centered - beta * control_centered).rename(
        f"{target_col}_residual_after_{control_col}"
    )


def residual_rankic_outputs(
    frame: pd.DataFrame,
    local_col: str,
    periods: dict[str, tuple[str, str]],
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame[[*KEY_COLUMNS, "author_factor", local_col, "y1_raw", "y2_raw"]].copy()
    residual_specs = [
        (local_col, "author_factor", "local_residual"),
        ("author_factor", local_col, "author_residual"),
    ]
    for target_col, control_col, residual_name in residual_specs:
        work[residual_name] = cross_sectional_rank_residual(work, target_col, control_col)

    metric_rows: list[dict[str, float | int | str]] = []
    daily_parts: list[pd.DataFrame] = []
    for period_name, (start, end) in periods.items():
        period = _period_frame(work, start, end)
        for target_col, control_col, residual_name in residual_specs:
            for return_col, horizon in RETURN_HORIZONS.items():
                daily = daily_spearman(period, residual_name, return_col=return_col)
                summary = summarize_ic(daily)
                ci_low, ci_high = moving_block_bootstrap_mean_ci(
                    daily,
                    block_length=horizon,
                    n_bootstrap=bootstrap_samples,
                    seed=42,
                )
                metric_rows.append(
                    {
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "residual": residual_name,
                        "target_factor": target_col,
                        "control_factor": control_col,
                        "return_horizon": return_col,
                        **summary,
                        "block_bootstrap_ci_low": ci_low,
                        "block_bootstrap_ci_high": ci_high,
                        "block_length": horizon,
                        "bootstrap_samples": int(bootstrap_samples),
                    }
                )
                if len(daily):
                    item = daily.rename_axis("TRADE_DT").rename("rankic").reset_index()
                    item.insert(0, "return_horizon", return_col)
                    item.insert(0, "residual", residual_name)
                    item.insert(0, "period", period_name)
                    daily_parts.append(item)
    daily_frame = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    return pd.DataFrame(metric_rows), daily_frame


def _target_weight_turnover(
    previous: dict[str, float] | None,
    current: dict[str, float],
) -> float:
    if previous is None:
        return 1.0
    stocks = set(previous) | set(current)
    return 0.5 * float(
        sum(abs(current.get(stock, 0.0) - previous.get(stock, 0.0)) for stock in stocks)
    )


def _maximum_drawdown(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if clean.size == 0:
        return math.nan
    nav = np.cumprod(1.0 + clean)
    nav_with_start = np.concatenate(([1.0], nav))
    running_peak = np.maximum.accumulate(nav_with_start)
    drawdown = nav_with_start / running_peak - 1.0
    return float(-np.min(drawdown))


def _annualized_compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if clean.size == 0:
        return math.nan
    total = float(np.prod(1.0 + clean))
    if total <= 0.0:
        return math.nan
    return total ** (252.0 / clean.size) - 1.0


def top_portfolio_backtest(
    frame: pd.DataFrame,
    factor_col: str,
    daily_return_series: pd.Series,
    market_dates: list[str],
    top_fraction: float,
    rebalance_every: int,
    cost_bps_values: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_frame = (
        frame[[*KEY_COLUMNS, factor_col]]
        .dropna()
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )
    if factor_frame.empty or not market_dates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    cross_sections = {
        str(date): group.reset_index(drop=True)
        for date, group in factor_frame.groupby("TRADE_DT", sort=True)
    }
    signal_dates = list(cross_sections)[:: max(rebalance_every, 1)]
    market_array = np.asarray(market_dates, dtype=str)
    targets: list[dict[str, object]] = []
    for signal_date in signal_dates:
        execution_index = int(np.searchsorted(market_array, str(signal_date), side="right"))
        if execution_index >= len(market_dates):
            continue
        cross_section = cross_sections[str(signal_date)]
        mask = _top_mask(cross_section, factor_col, top_fraction)
        stocks = cross_section.loc[mask, "S_INFO_WINDCODE"].astype(str).tolist()
        if not stocks:
            continue
        weight = 1.0 / len(stocks)
        targets.append(
            {
                "signal_date": str(signal_date),
                "execution_date": str(market_dates[execution_index]),
                "execution_index": execution_index,
                "weights": {stock: weight for stock in stocks},
                "n_holdings": len(stocks),
                "n_universe": int(len(cross_section)),
            }
        )
    if not targets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    gross_rows: list[dict[str, float | int | str]] = []
    turnover_rows: list[dict[str, float | int | str]] = []
    previous_weights: dict[str, float] | None = None
    for idx, target in enumerate(targets):
        current_weights = target["weights"]
        if not isinstance(current_weights, dict):
            raise TypeError("portfolio target weights must be a dictionary")
        turnover = _target_weight_turnover(previous_weights, current_weights)
        turnover_rows.append(
            {
                "signal_date": str(target["signal_date"]),
                "execution_date": str(target["execution_date"]),
                "target_turnover": turnover,
                "n_holdings": int(target["n_holdings"]),
                "n_universe": int(target["n_universe"]),
            }
        )
        start_index = int(target["execution_index"])
        end_index = (
            int(targets[idx + 1]["execution_index"])
            if idx + 1 < len(targets)
            else min(start_index + max(int(rebalance_every), 1), len(market_dates) - 1)
        )
        period_dates = market_dates[start_index + 1 : end_index + 1]
        stocks = list(current_weights)
        if period_dates:
            lookup = pd.MultiIndex.from_product(
                [period_dates, stocks],
                names=KEY_COLUMNS,
            )
            selected_returns = daily_return_series.reindex(lookup)
            observed = selected_returns.notna().groupby(level="TRADE_DT").mean()
            gross = selected_returns.fillna(0.0).groupby(level="TRADE_DT").mean()
            for return_date in period_dates:
                gross_rows.append(
                    {
                        "TRADE_DT": str(return_date),
                        "signal_date": str(target["signal_date"]),
                        "gross_return": float(gross.get(return_date, 0.0)),
                        "return_coverage": float(observed.get(return_date, 0.0)),
                        "n_holdings": int(target["n_holdings"]),
                    }
                )
        previous_weights = current_weights

    gross_frame = pd.DataFrame(gross_rows)
    if gross_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(turnover_rows)
    gross_frame = gross_frame.sort_values("TRADE_DT").drop_duplicates("TRADE_DT", keep="last")
    initial_date = str(targets[0]["execution_date"])
    if initial_date not in set(gross_frame["TRADE_DT"]):
        gross_frame = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "TRADE_DT": initial_date,
                            "signal_date": str(targets[0]["signal_date"]),
                            "gross_return": 0.0,
                            "return_coverage": 1.0,
                            "n_holdings": int(targets[0]["n_holdings"]),
                        }
                    ]
                ),
                gross_frame,
            ],
            ignore_index=True,
        ).sort_values("TRADE_DT")
    turnover_frame = pd.DataFrame(turnover_rows)
    turnover_by_date = turnover_frame.set_index("execution_date")["target_turnover"]
    gross_frame["target_turnover"] = (
        gross_frame["TRADE_DT"].map(turnover_by_date).fillna(0.0).astype(float)
    )

    metric_rows: list[dict[str, float | int]] = []
    daily_parts: list[pd.DataFrame] = []
    turnover_after_initial = turnover_frame["target_turnover"].iloc[1:]
    for cost_bps in cost_bps_values:
        daily = gross_frame.copy()
        daily["cost_bps"] = float(cost_bps)
        daily["transaction_cost"] = daily["target_turnover"] * float(cost_bps) / 10000.0
        daily["net_return"] = daily["gross_return"] - daily["transaction_cost"]
        daily["gross_nav"] = (1.0 + daily["gross_return"]).cumprod()
        daily["net_nav"] = (1.0 + daily["net_return"]).cumprod()
        net_std = float(daily["net_return"].std(ddof=1)) if len(daily) > 1 else math.nan
        metric_rows.append(
            {
                "top_fraction": float(top_fraction),
                "cost_bps": float(cost_bps),
                "start": str(daily["TRADE_DT"].min()),
                "end": str(daily["TRADE_DT"].max()),
                "n_return_days": int(len(daily)),
                "n_rebalances": int(len(turnover_frame)),
                "mean_holdings": float(turnover_frame["n_holdings"].mean()),
                "mean_universe": float(turnover_frame["n_universe"].mean()),
                "mean_target_turnover": (
                    float(turnover_after_initial.mean()) if len(turnover_after_initial) else math.nan
                ),
                "median_target_turnover": (
                    float(turnover_after_initial.median())
                    if len(turnover_after_initial)
                    else math.nan
                ),
                "mean_return_coverage": float(daily["return_coverage"].mean()),
                "gross_total_return": float(daily["gross_nav"].iloc[-1] - 1.0),
                "net_total_return": float(daily["net_nav"].iloc[-1] - 1.0),
                "gross_annualized_return": _annualized_compound_return(daily["gross_return"]),
                "net_annualized_return": _annualized_compound_return(daily["net_return"]),
                "net_annualized_volatility": (
                    net_std * math.sqrt(252.0) if np.isfinite(net_std) else math.nan
                ),
                "net_sharpe": (
                    float(daily["net_return"].mean()) / net_std * math.sqrt(252.0)
                    if np.isfinite(net_std) and net_std > 0.0
                    else math.nan
                ),
                "gross_max_drawdown": _maximum_drawdown(daily["gross_return"]),
                "net_max_drawdown": _maximum_drawdown(daily["net_return"]),
            }
        )
        daily_parts.append(daily)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_parts, ignore_index=True),
        turnover_frame,
    )


def alignment_sensitivity(
    author: pd.DataFrame,
    labels: pd.DataFrame,
    offsets: Iterable[int],
) -> pd.DataFrame:
    dates = sorted(labels["TRADE_DT"].astype(str).unique().tolist())
    date_to_index = {date: idx for idx, date in enumerate(dates)}
    author_dates = author["TRADE_DT"].astype(str).unique().tolist()
    rows: list[dict[str, float | int]] = []
    label_values = labels[[*KEY_COLUMNS, "y1_raw"]].rename(columns={"TRADE_DT": "LABEL_DT"})
    for offset in offsets:
        mapping = {
            date: dates[index + offset]
            for date in author_dates
            if (index := date_to_index.get(date)) is not None
            and 0 <= index + offset < len(dates)
        }
        shifted = author.copy()
        shifted["LABEL_DT"] = shifted["TRADE_DT"].map(mapping)
        shifted = shifted.loc[shifted["LABEL_DT"].notna()]
        merged = shifted.merge(
            label_values,
            on=["LABEL_DT", "S_INFO_WINDCODE"],
            how="inner",
            validate="one_to_one",
        )
        summary = summarize_ic(daily_spearman(merged, "author_factor"))
        rows.append({"label_date_offset": int(offset), **summary, "n_observations": int(len(merged))})
    return pd.DataFrame(rows)


def moving_block_bootstrap_mean_ci(
    values: pd.Series,
    block_length: int = 11,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return math.nan, math.nan
    if clean.size <= block_length:
        return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))
    rng = np.random.default_rng(seed)
    max_start = clean.size - block_length
    n_blocks = int(math.ceil(clean.size / block_length))
    estimates = np.empty(n_bootstrap, dtype=float)
    offsets = np.arange(block_length)
    for idx in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled_indices = (starts[:, None] + offsets[None, :]).reshape(-1)[: clean.size]
        estimates[idx] = clean[sampled_indices].mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _period_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[frame["TRADE_DT"].astype(str).between(str(start), str(end))]


def _metrics_rows(
    frame: pd.DataFrame,
    factor_columns: Iterable[str],
    periods: dict[str, tuple[str, str]],
    n_groups: int,
    rebalance_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.Series]]:
    rows: list[dict[str, float | int | str]] = []
    group_parts: list[pd.DataFrame] = []
    daily_map: dict[tuple[str, str], pd.Series] = {}
    for period_name, (start, end) in periods.items():
        period = _period_frame(frame, start, end)
        for factor_col in factor_columns:
            summary, daily_ic, groups = summarize_factor(
                period,
                factor_col,
                orientation=1.0,
                n_groups=n_groups,
                rebalance_every=rebalance_every,
            )
            rows.append(
                {
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "model": factor_col,
                    "model_label": MODEL_LABELS.get(factor_col, factor_col),
                    **summary,
                }
            )
            daily_map[(period_name, factor_col)] = daily_ic
            if len(groups):
                item = groups.copy()
                item.insert(0, "model", factor_col)
                item.insert(0, "period", period_name)
                group_parts.append(item)
    group_frame = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    return pd.DataFrame(rows), group_frame, daily_map


def _year_metrics(
    frame: pd.DataFrame,
    factor_columns: Iterable[str],
    n_groups: int,
    rebalance_every: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    work = frame.copy()
    work["year"] = work["TRADE_DT"].astype(str).str[:4]
    for year, year_frame in work.groupby("year", sort=True):
        for factor_col in factor_columns:
            summary, _, _ = summarize_factor(
                year_frame,
                factor_col,
                orientation=1.0,
                n_groups=n_groups,
                rebalance_every=rebalance_every,
            )
            rows.append(
                {
                    "year": year,
                    "model": factor_col,
                    "model_label": MODEL_LABELS.get(factor_col, factor_col),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _common_rank_tables(
    common: pd.DataFrame,
    factor_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = pd.DataFrame({"TRADE_DT": common["TRADE_DT"].astype(str)})
    ranks["return_rank"] = common.groupby("TRADE_DT", sort=False)["y1_raw"].rank(
        method="average", pct=True
    )
    for factor_col in factor_columns:
        ranks[factor_col] = common.groupby("TRADE_DT", sort=False)[factor_col].rank(
            method="average", pct=True
        )

    daily = pd.DataFrame(index=sorted(common["TRADE_DT"].astype(str).unique().tolist()))
    for factor_col in factor_columns:
        daily[factor_col] = _daily_pearson_from_ranked(
            ranks["TRADE_DT"], ranks[factor_col], ranks["return_rank"]
        )
    daily.index.name = "TRADE_DT"

    correlation_rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(factor_columns, 2):
        values = _daily_pearson_from_ranked(ranks["TRADE_DT"], ranks[left], ranks[right]).dropna()
        correlation_rows.append(
            {
                "left": left,
                "right": right,
                "left_label": MODEL_LABELS.get(left, left),
                "right_label": MODEL_LABELS.get(right, right),
                "n_dates": int(len(values)),
                "mean_daily_spearman": float(values.mean()),
                "median_daily_spearman": float(values.median()),
                "p10_daily_spearman": float(values.quantile(0.10)),
                "p90_daily_spearman": float(values.quantile(0.90)),
            }
        )
    return daily.reset_index(), pd.DataFrame(correlation_rows)


def _ensemble_metrics(
    common: pd.DataFrame,
    local_columns: list[str],
    periods: dict[str, tuple[str, str]],
    n_groups: int,
    rebalance_every: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    author_rank = common.groupby("TRADE_DT", sort=False)["author_factor"].rank(
        method="average", pct=True
    )
    for local_col in local_columns:
        local_rank = common.groupby("TRADE_DT", sort=False)[local_col].rank(
            method="average", pct=True
        )
        factor_col = f"ensemble_author_{local_col}"
        work = common[[*KEY_COLUMNS, "y1_raw"]].copy()
        work[factor_col] = (author_rank + local_rank) / 2.0
        for period_name, (start, end) in periods.items():
            period = _period_frame(work, start, end)
            summary, _, _ = summarize_factor(
                period,
                factor_col,
                orientation=1.0,
                n_groups=n_groups,
                rebalance_every=rebalance_every,
            )
            rows.append(
                {
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "model": factor_col,
                    "components": f"author_factor+{local_col}",
                    "model_label": f"作者因子 + {MODEL_LABELS.get(local_col, local_col)}",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_outputs(
    pairwise_frames: dict[str, pd.DataFrame],
    periods: dict[str, tuple[str, str]],
    n_groups: int,
    rebalance_every: int,
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, float | int | str]] = []
    ensemble_rows: list[dict[str, float | int | str]] = []
    paired_rows: list[dict[str, float | int | str]] = []

    for model, frame in pairwise_frames.items():
        metrics, groups, _ = _metrics_rows(
            frame,
            ["author_factor", model],
            periods,
            n_groups=n_groups,
            rebalance_every=rebalance_every,
        )
        metrics.insert(0, "comparison_model", model)
        metrics.insert(1, "comparison_model_label", MODEL_LABELS.get(model, model))
        metric_parts.append(metrics)
        if len(groups):
            groups.insert(0, "comparison_model", model)
            group_parts.append(groups)

        author_ic = daily_spearman(frame, "author_factor")
        local_ic = daily_spearman(frame, model)
        daily = pd.concat(
            [author_ic.rename("author_rankic"), local_ic.rename("local_rankic")],
            axis=1,
        ).dropna()
        daily["rankic_difference"] = daily["author_rankic"] - daily["local_rankic"]
        daily.insert(0, "comparison_model", model)
        daily_parts.append(daily.reset_index(names="TRADE_DT"))

        factor_rank = frame.groupby("TRADE_DT", sort=False)["author_factor"].rank(
            method="average", pct=True
        )
        local_rank = frame.groupby("TRADE_DT", sort=False)[model].rank(method="average", pct=True)
        factor_correlation = _daily_pearson_from_ranked(
            frame["TRADE_DT"], factor_rank, local_rank
        ).dropna()
        correlation_rows.append(
            {
                "comparison_model": model,
                "comparison_model_label": MODEL_LABELS.get(model, model),
                "n_dates": int(len(factor_correlation)),
                "mean_daily_spearman": float(factor_correlation.mean()),
                "median_daily_spearman": float(factor_correlation.median()),
                "p10_daily_spearman": float(factor_correlation.quantile(0.10)),
                "p90_daily_spearman": float(factor_correlation.quantile(0.90)),
            }
        )

        ensemble_col = f"ensemble_author_{model}"
        ensemble = frame[[*KEY_COLUMNS, "y1_raw"]].copy()
        ensemble[ensemble_col] = (factor_rank + local_rank) / 2.0
        for period_name, (start, end) in periods.items():
            summary, _, _ = summarize_factor(
                _period_frame(ensemble, start, end),
                ensemble_col,
                orientation=1.0,
                n_groups=n_groups,
                rebalance_every=rebalance_every,
            )
            ensemble_rows.append(
                {
                    "comparison_model": model,
                    "comparison_model_label": MODEL_LABELS.get(model, model),
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "model": ensemble_col,
                    "components": f"author_factor+{model}",
                    "model_label": f"作者因子 + {MODEL_LABELS.get(model, model)}",
                    **summary,
                }
            )

        difference = daily["rankic_difference"]
        ci_low, ci_high = moving_block_bootstrap_mean_ci(
            difference,
            block_length=11,
            n_bootstrap=bootstrap_samples,
            seed=42,
        )
        paired_rows.append(
            {
                "comparison": f"author_factor_minus_{model}",
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "n_dates": int(len(daily)),
                "mean_rankic_difference": float(difference.mean()),
                "median_rankic_difference": float(difference.median()),
                "author_daily_win_rate": float((difference > 0).mean()),
                "block_bootstrap_ci_low": ci_low,
                "block_bootstrap_ci_high": ci_high,
                "block_length": 11,
                "bootstrap_samples": int(bootstrap_samples),
            }
        )

    metrics_frame = pd.concat(metric_parts, ignore_index=True)
    groups_frame = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    daily_frame = pd.concat(daily_parts, ignore_index=True)
    return (
        metrics_frame,
        groups_frame,
        daily_frame,
        pd.DataFrame(correlation_rows),
        pd.DataFrame(ensemble_rows),
        pd.DataFrame(paired_rows),
    )


def _summarize_top_overlap(
    daily: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for period_name, (start, end) in periods.items():
        period = _period_frame(daily, start, end)
        for (model, fraction), group in period.groupby(
            ["comparison_model", "top_fraction"], sort=True
        ):
            rows.append(
                {
                    "comparison_model": model,
                    "comparison_model_label": MODEL_LABELS.get(model, model),
                    "period": period_name,
                    "start": start,
                    "end": end,
                    "top_fraction": float(fraction),
                    "n_dates": int(len(group)),
                    "mean_stocks": float(group["n_stocks"].mean()),
                    "mean_top_count": float(group["left_top_count"].mean()),
                    "mean_intersection_count": float(group["intersection_count"].mean()),
                    "mean_overlap_rate": float(group["overlap_rate"].mean()),
                    "median_overlap_rate": float(group["overlap_rate"].median()),
                    "p10_overlap_rate": float(group["overlap_rate"].quantile(0.10)),
                    "p90_overlap_rate": float(group["overlap_rate"].quantile(0.90)),
                    "mean_jaccard": float(group["jaccard"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _summarize_disagreement_returns(
    daily: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, float | int | str]] = []
    pair_rows: list[dict[str, float | int | str]] = []
    for period_name, (start, end) in periods.items():
        period = _period_frame(daily, start, end)
        keys = ["comparison_model", "top_fraction", "return_horizon"]
        for (model, fraction, return_col), comparison in period.groupby(keys, sort=True):
            horizon = RETURN_HORIZONS[str(return_col)]
            for selection_group, group in comparison.groupby("selection_group", sort=True):
                group_rows.append(
                    {
                        "comparison_model": model,
                        "comparison_model_label": MODEL_LABELS.get(model, model),
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "top_fraction": float(fraction),
                        "return_horizon": return_col,
                        "selection_group": selection_group,
                        "n_dates": int(len(group)),
                        "mean_group_stocks": float(group["n_stocks"].mean()),
                        "mean_return": float(group["group_return"].mean()),
                        "median_return": float(group["group_return"].median()),
                        "mean_excess_return": float(group["excess_return"].mean()),
                        "annualized_return_approx": float(group["group_return"].mean())
                        * 252.0
                        / horizon,
                        "annualized_excess_approx": float(group["excess_return"].mean())
                        * 252.0
                        / horizon,
                    }
                )
            pivot = comparison.pivot_table(
                index="TRADE_DT",
                columns="selection_group",
                values="group_return",
                aggfunc="first",
            )
            if {"author_only", "local_only"}.issubset(pivot.columns):
                paired = pivot[["author_only", "local_only"]].dropna()
                difference = paired["local_only"] - paired["author_only"]
                ci_low, ci_high = moving_block_bootstrap_mean_ci(
                    difference,
                    block_length=horizon,
                    n_bootstrap=bootstrap_samples,
                    seed=42,
                )
                pair_rows.append(
                    {
                        "comparison_model": model,
                        "comparison_model_label": MODEL_LABELS.get(model, model),
                        "period": period_name,
                        "start": start,
                        "end": end,
                        "top_fraction": float(fraction),
                        "return_horizon": return_col,
                        "n_dates": int(len(paired)),
                        "author_only_mean_return": float(paired["author_only"].mean()),
                        "local_only_mean_return": float(paired["local_only"].mean()),
                        "local_minus_author_mean": float(difference.mean()),
                        "local_minus_author_annualized_approx": float(difference.mean())
                        * 252.0
                        / horizon,
                        "local_daily_win_rate": float((difference > 0.0).mean()),
                        "block_bootstrap_ci_low": ci_low,
                        "block_bootstrap_ci_high": ci_high,
                        "block_length": horizon,
                        "bootstrap_samples": int(bootstrap_samples),
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(pair_rows)


def _extended_pairwise_outputs(
    pairwise_frames: dict[str, pd.DataFrame],
    periods: dict[str, tuple[str, str]],
    daily_returns: pd.DataFrame,
    top_fractions: Iterable[float],
    rebalance_every: int,
    cost_bps_values: Iterable[float],
    bootstrap_samples: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    overlap_parts: list[pd.DataFrame] = []
    disagreement_parts: list[pd.DataFrame] = []
    residual_metric_parts: list[pd.DataFrame] = []
    residual_daily_parts: list[pd.DataFrame] = []
    portfolio_metric_parts: list[pd.DataFrame] = []
    portfolio_daily_parts: list[pd.DataFrame] = []
    turnover_parts: list[pd.DataFrame] = []

    returns = (
        _as_string_keys(daily_returns[[*KEY_COLUMNS, "daily_return"]])
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .sort_values(KEY_COLUMNS)
    )
    return_series = returns.set_index(KEY_COLUMNS)["daily_return"].astype(float)
    market_dates = sorted(returns["TRADE_DT"].unique().tolist())

    for model, frame in pairwise_frames.items():
        for fraction in top_fractions:
            overlap = top_overlap_daily(frame, "author_factor", model, fraction)
            overlap.insert(0, "comparison_model", model)
            overlap_parts.append(overlap)

            disagreement = disagreement_group_returns(frame, model, fraction)
            disagreement.insert(0, "comparison_model", model)
            disagreement_parts.append(disagreement)

            for factor_col in ["author_factor", model]:
                metrics, daily, turnover = top_portfolio_backtest(
                    frame,
                    factor_col,
                    daily_return_series=return_series,
                    market_dates=market_dates,
                    top_fraction=fraction,
                    rebalance_every=rebalance_every,
                    cost_bps_values=cost_bps_values,
                )
                if len(metrics):
                    metrics.insert(0, "model", factor_col)
                    metrics.insert(0, "model_label", MODEL_LABELS.get(factor_col, factor_col))
                    metrics.insert(0, "comparison_model", model)
                    portfolio_metric_parts.append(metrics)
                if len(daily):
                    daily.insert(0, "model", factor_col)
                    daily.insert(0, "comparison_model", model)
                    daily["top_fraction"] = float(fraction)
                    portfolio_daily_parts.append(daily)
                if len(turnover):
                    turnover.insert(0, "model", factor_col)
                    turnover.insert(0, "comparison_model", model)
                    turnover["top_fraction"] = float(fraction)
                    turnover_parts.append(turnover)

        residual_metrics, residual_daily = residual_rankic_outputs(
            frame,
            model,
            periods,
            bootstrap_samples=bootstrap_samples,
        )
        residual_metrics.insert(0, "comparison_model", model)
        residual_metric_parts.append(residual_metrics)
        if len(residual_daily):
            residual_daily.insert(0, "comparison_model", model)
            residual_daily_parts.append(residual_daily)

    overlap_daily = pd.concat(overlap_parts, ignore_index=True)
    overlap_summary = _summarize_top_overlap(overlap_daily, periods)
    disagreement_daily = pd.concat(disagreement_parts, ignore_index=True)
    disagreement_summary, disagreement_pairs = _summarize_disagreement_returns(
        disagreement_daily,
        periods,
        bootstrap_samples=bootstrap_samples,
    )
    return (
        overlap_daily,
        overlap_summary,
        disagreement_daily,
        disagreement_summary,
        disagreement_pairs,
        pd.concat(residual_metric_parts, ignore_index=True),
        pd.concat(residual_daily_parts, ignore_index=True),
        pd.concat(portfolio_metric_parts, ignore_index=True),
        pd.concat(portfolio_daily_parts, ignore_index=True),
        pd.concat(turnover_parts, ignore_index=True),
    )


def _author_inventory(author: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        author.groupby("TRADE_DT", sort=True)["author_factor"]
        .agg(n_stocks="size", mean="mean", std="std", min="min", max="max", unique_values="nunique")
        .reset_index()
    )
    values = author["author_factor"].astype(float)
    inventory = pd.DataFrame(
        [
            {
                "file_count": int(author["TRADE_DT"].nunique()),
                "row_count": int(len(author)),
                "date_min": author["TRADE_DT"].min(),
                "date_max": author["TRADE_DT"].max(),
                "mean_daily_stocks": float(daily["n_stocks"].mean()),
                "min_daily_stocks": int(daily["n_stocks"].min()),
                "max_daily_stocks": int(daily["n_stocks"].max()),
                "factor_mean": float(values.mean()),
                "factor_std": float(values.std(ddof=1)),
                "factor_min": float(values.min()),
                "factor_p01": float(values.quantile(0.01)),
                "factor_median": float(values.median()),
                "factor_p99": float(values.quantile(0.99)),
                "factor_max": float(values.max()),
            }
        ]
    )
    return inventory, daily


def _format_pct(value: object) -> str:
    number = float(value)
    return "-" if not np.isfinite(number) else f"{number * 100:.2f}%"


def _format_number(value: object, digits: int = 3) -> str:
    number = float(value)
    return "-" if not np.isfinite(number) else f"{number:.{digits}f}"


def _format_date_id(value: object) -> str:
    text = str(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def write_report(
    output_dir: Path,
    inventory: pd.DataFrame,
    author_metrics: pd.DataFrame,
    pairwise_metrics: pd.DataFrame,
    pairwise_correlations: pd.DataFrame,
    pairwise_ensembles: pd.DataFrame,
    paired_differences: pd.DataFrame,
    coverage: pd.DataFrame,
    common_metrics: pd.DataFrame,
    all_common_correlations: pd.DataFrame,
    alignment: pd.DataFrame,
    autocorrelation: pd.DataFrame,
) -> None:
    author_pdf = author_metrics.loc[author_metrics["period"] == "pdf_available_2019_20240930"].iloc[0]
    author_post = author_metrics.loc[author_metrics["period"] == "post_pdf_20241001_20260422"].iloc[0]
    pair_all = pairwise_metrics.loc[
        pairwise_metrics["period"] == "all_pairwise_2019_20260408"
    ].copy()
    pair_post = pairwise_metrics.loc[
        pairwise_metrics["period"] == "post_pdf_pairwise_20241001_20260408"
    ].copy()
    local_pair_all = pair_all.loc[pair_all["model"] == pair_all["comparison_model"]].copy()
    best_local = local_pair_all.sort_values("rankic", ascending=False).iloc[0]
    best_local_author = pair_all.loc[
        (pair_all["comparison_model"] == best_local["comparison_model"])
        & (pair_all["model"] == "author_factor")
    ].iloc[0]
    best_local_difference = paired_differences.loc[
        paired_differences["model"] == best_local["comparison_model"]
    ].iloc[0]
    local_pair_post = pair_post.loc[pair_post["model"] == pair_post["comparison_model"]].copy()
    best_post_local = local_pair_post.sort_values("rankic", ascending=False).iloc[0]
    best_post_local_author = pair_post.loc[
        (pair_post["comparison_model"] == best_post_local["comparison_model"])
        & (pair_post["model"] == "author_factor")
    ].iloc[0]

    common_all = common_metrics.loc[common_metrics["period"] == "all_common_2019_20260408"].copy()
    common_all = common_all.sort_values("rankic", ascending=False)
    author_common = common_all.loc[common_all["model"] == "author_factor"].iloc[0]
    best_all_common_local = common_all.loc[common_all["model"] != "author_factor"].iloc[0]
    mean_author_corr = float(pairwise_correlations["mean_daily_spearman"].mean())

    ensemble_all = pairwise_ensembles.loc[
        pairwise_ensembles["period"] == "all_pairwise_2019_20260408"
    ].copy()
    author_all_reference = pair_all.loc[pair_all["model"] == "author_factor", [
        "comparison_model",
        "rankic",
    ]].rename(columns={"rankic": "author_rankic"})
    ensemble_all = ensemble_all.merge(author_all_reference, on="comparison_model", how="left")
    ensemble_all["rankic_increment"] = ensemble_all["rankic"] - ensemble_all["author_rankic"]
    best_full_ensemble = ensemble_all.sort_values("rankic_increment", ascending=False).iloc[0]

    ensemble_post = pairwise_ensembles.loc[
        pairwise_ensembles["period"] == "post_pdf_pairwise_20241001_20260408"
    ].copy()
    author_post_reference = pair_post.loc[pair_post["model"] == "author_factor", [
        "comparison_model",
        "rankic",
    ]].rename(columns={"rankic": "author_rankic"})
    ensemble_post = ensemble_post.merge(author_post_reference, on="comparison_model", how="left")
    ensemble_post["rankic_increment"] = ensemble_post["rankic"] - ensemble_post["author_rankic"]
    best_post_ensemble = ensemble_post.sort_values("rankic_increment", ascending=False).iloc[0]

    tuned = {"h32_tuned", "h48_tuned", "h64_tuned"}
    tuned_correlations = all_common_correlations.loc[
        all_common_correlations["left"].isin(tuned)
        & all_common_correlations["right"].isin(tuned)
    ]
    mean_tuned_correlation = float(tuned_correlations["mean_daily_spearman"].mean())
    same_date = alignment.loc[alignment["label_date_offset"] == 0].iloc[0]
    best_alignment = alignment.sort_values("rankic", ascending=False).iloc[0]
    lag5 = autocorrelation.loc[autocorrelation["lag"] == 5].iloc[0]
    mean_pairwise_stocks = float(coverage["pairwise_mean_stocks"].mean())
    all_common_stocks = float(coverage["all_common_mean_stocks"].iloc[0])

    if float(best_full_ensemble["rankic_increment"]) > 0:
        full_ensemble_text = (
            f"全阶段增量最高的是 {best_full_ensemble['model_label']}，RankIC 增加 "
            f"{_format_pct(best_full_ensemble['rankic_increment'])}。"
        )
    else:
        full_ensemble_text = (
            f"全阶段表现最接近作者单因子的是 {best_full_ensemble['model_label']}，RankIC 仍低 "
            f"{_format_pct(abs(float(best_full_ensemble['rankic_increment'])))}。"
        )

    if float(best_post_ensemble["rankic_increment"]) > 0:
        post_ensemble_text = (
            f"在 2024-10-01 之后，{best_post_ensemble['model_label']} 的 RankIC 比对应作者单因子高 "
            f"{_format_pct(best_post_ensemble['rankic_increment'])}。"
        )
    else:
        post_ensemble_text = (
            f"在 2024-10-01 之后，各个一比一组合仍未超过对应作者单因子，最小差距为 "
            f"{_format_pct(abs(float(best_post_ensemble['rankic_increment'])))}。"
        )

    lines = [
        "# 作者 ABCM 因子与本地模型比较",
        "",
        "更新日期：2026-07-23",
        "",
        "## 1. 结论",
        "",
        (
            f"作者因子文件覆盖 {_format_date_id(inventory.iloc[0]['date_min'])} 至 "
            f"{_format_date_id(inventory.iloc[0]['date_max'])}，"
            f"共 {int(inventory.iloc[0]['file_count']):,} 个交易日、{int(inventory.iloc[0]['row_count']):,} 条记录。"
            "字段和股票代码格式一致，未发现缺失值、无穷值或重复股票。"
        ),
        "",
        (
            "在可与 PDF 回测区间重合的 2019-01-02 至 2024-09-30，作者因子按本地收益标签复算得到 "
            f"RankIC（因子排序与未来收益排序的相关系数）{_format_pct(author_pdf['rankic'])}、"
            f"ICIR（RankIC 均值与波动率之比）{_format_number(author_pdf['icir'])}、"
            f"RankIC 胜率 {_format_pct(author_pdf['win_rate'])}。"
            f"PDF 对 2017-01-01 至 2024-09-30 报告的对应值为 12.69%、0.960 和 86.63%。"
            "作者文件缺少 2017 至 2018 年，且本地股票池、标签和组合口径与 PDF 不完全一致，因此这组数值用于交叉验证，不能视为对 PDF 表格的严格重算。"
        ),
        "",
        (
            f"同一期作者因子的 Top 5% 11 日超额年化近似值为 "
            f"{_format_pct(author_pdf['top_annualized_approx'])}；PDF 报告的 Top 5% 正式组合年化收益为 34.51%。"
            "PDF 使用中证全指、正式组合净值和换手控制；本地数值使用重叠 11 日标签超额收益进行线性年化，"
            "未扣交易成本。两项收益指标采用不同算法，只作量级参考，不计算直接差值。"
        ),
        "",
        (
            f"2024-10-01 至 2026-04-22，作者因子仍有 RankIC {_format_pct(author_post['rankic'])}、"
            f"ICIR {_format_number(author_post['icir'])}、胜率 {_format_pct(author_post['win_rate'])}。"
            "表现没有在 PDF 截止日后明显消失。作者模型是否滚动更新、训练截止日和股票过滤规则尚不明确，"
            "这段结果暂不能定义为严格样本外检验。"
        ),
        "",
        (
            f"按作者因子与每个模型分别取公共股票，平均每天约 {mean_pairwise_stocks:.0f} 只股票。"
            f"本地 RankIC 最高的是 {best_local['model_label']}，为 {_format_pct(best_local['rankic'])}；"
            f"同一股票池上的作者因子为 {_format_pct(best_local_author['rankic'])}，差值 "
            f"{_format_pct(best_local_author['rankic'] - best_local['rankic'])}。"
            f"11 日区块自助法给出的差值区间为 {_format_pct(best_local_difference['block_bootstrap_ci_low'])} 至 "
            f"{_format_pct(best_local_difference['block_bootstrap_ci_high'])}，当前样本下差距较稳定。"
        ),
        "",
        (
            f"作者因子与各本地信号的平均每日截面秩相关约为 {_format_pct(mean_author_corr)}，"
            f"本地 h32、h48、h64 三个容量候选之间的平均相关为 {_format_pct(mean_tuned_correlation)}。"
            "三个容量候选生成的排序高度相似，扩大这一范围内的模型容量没有产生明显不同的信号。"
        ),
        "",
        (
            f"2024-10-01 至 2026-04-08，本地表现最高的是 {best_post_local['model_label']}，RankIC 为 "
            f"{_format_pct(best_post_local['rankic'])}；同股票池作者因子为 "
            f"{_format_pct(best_post_local_author['rankic'])}，差距缩小到 "
            f"{_format_pct(best_post_local_author['rankic'] - best_post_local['rankic'])}。"
            "近期结果说明本地调权训练具有一定效果，但这一时间段也参与了现有候选比较，仍需后续滚动样本验证。"
        ),
        "",
        (
            f"将作者与本地信号按截面排名一比一合成后，{full_ensemble_text}{post_ensemble_text}"
            "这说明本地信号可能在近期提供少量补充，但固定一比一权重不适合作为全阶段方案；"
            "后续需要在独立时间段上选择较小权重或做残差信号训练。"
        ),
        "",
        "## 2. 成对比较",
        "",
        "成对公共股票池指同一交易日中作者因子与该模型均有输出的股票。",
        "`h32`、`h48`、`h64` 中的数字表示门控循环单元网络隐藏层维度；`h32 PDF 损失`采用 PDF 披露的一比一均方误差（MSE）与决定系数（R²）权重。",
        "",
        "| 本地模型 | 平均股票数 | 作者 RankIC | 本地 RankIC | RankIC 差值 | 作者 Top 5% 11 日超额年化近似值 | 本地 Top 5% 11 日超额年化近似值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for coverage_row in coverage.itertuples(index=False):
        model = coverage_row.model
        author_row = pair_all.loc[
            (pair_all["comparison_model"] == model) & (pair_all["model"] == "author_factor")
        ].iloc[0]
        local_row = pair_all.loc[
            (pair_all["comparison_model"] == model) & (pair_all["model"] == model)
        ].iloc[0]
        lines.append(
            f"| {MODEL_LABELS.get(model, model)} | {coverage_row.pairwise_mean_stocks:.0f} | "
            f"{_format_pct(author_row['rankic'])} | {_format_pct(local_row['rankic'])} | "
            f"{_format_pct(author_row['rankic'] - local_row['rankic'])} | "
            f"{_format_pct(author_row['top_annualized_approx'])} | "
            f"{_format_pct(local_row['top_annualized_approx'])} |"
        )
    lines.extend(
        [
            "",
            "表内作者与本地结果使用相同日期、相同股票池和相同公式，可以直接比较。"
            "本地结果只使用 2019-01-02 之后且作者因子可对齐的股票；既有五折总表覆盖 2005 年以后全部验证期，"
            "日期和股票池不同，两组数值不用于判断模型性能变化。",
        ]
    )

    lines.extend(
        [
            "",
            "## 3. 稳定性与口径限制",
            "",
            (
                f"把所有模型同时取交集后，股票池平均仅有 {all_common_stocks:.0f} 只。"
                f"该敏感性口径下作者 RankIC 为 {_format_pct(author_common['rankic'])}，"
                f"本地最高为 {best_all_common_local['model_label']} 的 {_format_pct(best_all_common_local['rankic'])}。"
                "这个结果方向一致，但窄股票池不作为主比较口径。"
            ),
            "",
            (
                f"作者因子 5 个交易日截面自相关为 {_format_pct(lag5['mean_autocorrelation'])}。"
                "这说明信号有一定持续性，同时仍会发生较明显的排序变化。"
            ),
            "",
            (
                f"同日标签对齐的 RankIC 为 {_format_pct(same_date['rankic'])}；偏移测试中最高值出现在 "
                f"{int(best_alignment['label_date_offset']):+d} 个交易日，RankIC 为 {_format_pct(best_alignment['rankic'])}。"
                "11 日收益窗口高度重叠，且因子自身有自相关，因此偏移结果不能单独证明文件日期定义。"
                "正式比较仍采用文件日期与本地因子形成日同日对齐。"
            ),
            "",
            "当前比较仍缺少作者模型的训练截止日、行业和市值中性化明细、中证全指成分及权重、停牌和涨跌停过滤、正式调仓净值、交易成本和换手约束。",
            "因此可以确认作者因子在本地数据上具有较强且持续的选股相关性，也可以确认本地 ABCM 尚未达到作者已导出因子的水平；暂时无法把差距归因于单一模型结构。",
            "",
            "## 4. 结果文件",
            "",
            "- `author_period_metrics.csv`：作者因子分阶段结果。",
            "- `author_year_metrics.csv`：作者因子分年度结果。",
            "- `pairwise_universe_metrics.csv`：作者与每个模型分别取公共股票后的比较。",
            "- `common_universe_metrics.csv`：所有模型同时取公共股票后的敏感性结果。",
            "- `factor_rank_correlation.csv`：作者与各本地因子的成对每日截面秩相关。",
            "- `ensemble_metrics.csv`：成对股票池上的一比一排序组合结果。",
            "- `alignment_sensitivity.csv`：标签日期偏移敏感性。",
            "- `paired_rankic_difference.csv`：作者与本地模型的配对 RankIC 差异及区块自助法区间。",
            "- `top_overlap_summary.csv`：Top 5% 和 Top 10% 持仓重合率。",
            "- `disagreement_group_summary.csv`：一致与分歧股票组的未来收益。",
            "- `residual_rankic_metrics.csv`：作者与本地因子相互剔除后的残差 RankIC。",
            "- `portfolio_backtest_metrics.csv`：简化 Top 组合的换手率、成本后复合年化绝对收益和最大回撤。",
            "- `analysis_manifest.json`：输入路径和主要参数。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare author-provided ABCM factors with local models.")
    parser.add_argument(
        "--author-dir",
        default="/mnt/sda1/datasets/quantitative_analysis/data",
    )
    parser.add_argument(
        "--cache-path",
        default="/dev/shm/abcm_cache/abcm1_f52_l60_y11_y21_lag1_clip1.pkl",
    )
    parser.add_argument(
        "--manifest-path",
        default=(
            "/mnt/sda1/datasets/quantitative_analysis/abcm_outputs/"
            "global_candidate_comparison_f52_full_epoch_20260717/selected_model_manifest.csv"
        ),
    )
    parser.add_argument(
        "--lightgbm-summary-path",
        default=(
            "/mnt/sda1/datasets/quantitative_analysis/abcm_outputs/"
            "baselines/lightgbm_f52_cv_summary_20260705.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/sda1/datasets/quantitative_analysis/abcm_outputs/"
            "author_factor_comparison_20260718"
        ),
    )
    parser.add_argument(
        "--candidates",
        default="h32_tuned,h48_tuned,h64_tuned,h32_pdf_loss",
    )
    parser.add_argument("--extended-models", default="h48_tuned")
    parser.add_argument("--n-groups", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--top-fractions", default="0.05,0.10")
    parser.add_argument("--cost-bps", default="0,10,20")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    extended_models = [
        item.strip() for item in args.extended_models.split(",") if item.strip()
    ]
    missing_extended = sorted(set(extended_models) - set(candidates))
    if missing_extended:
        raise ValueError(f"Extended models must also be candidates: {missing_extended}")
    top_fractions = [float(item.strip()) for item in args.top_fractions.split(",") if item.strip()]
    cost_bps_values = [float(item.strip()) for item in args.cost_bps.split(",") if item.strip()]

    author = load_author_factors(args.author_dir)
    inventory, daily_inventory = _author_inventory(author)
    labels = load_labels(args.cache_path, min_date=author["TRADE_DT"].min())
    author_labeled = author.merge(labels, on=KEY_COLUMNS, how="left", validate="one_to_one")

    inventory["y1_coverage"] = float(author_labeled["y1_raw"].notna().mean())
    inventory["y2_coverage"] = float(author_labeled["y2_raw"].notna().mean())
    inventory["y1_labeled_end"] = author_labeled.loc[
        author_labeled["y1_raw"].notna(), "TRADE_DT"
    ].max()
    inventory["y2_labeled_end"] = author_labeled.loc[
        author_labeled["y2_raw"].notna(), "TRADE_DT"
    ].max()

    author_periods = {
        "pdf_available_2019_20240930": (author["TRADE_DT"].min(), "20240930"),
        "post_pdf_20241001_20260422": ("20241001", "20260422"),
        "all_labeled_2019_20260422": (author["TRADE_DT"].min(), "20260422"),
    }
    author_metrics, author_groups, author_daily_map = _metrics_rows(
        author_labeled,
        ["author_factor"],
        author_periods,
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
    )
    author_year = _year_metrics(
        author_labeled.loc[author_labeled["y1_raw"].notna()],
        ["author_factor"],
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
    )
    author_daily = pd.DataFrame(
        {"author_factor": author_daily_map[("all_labeled_2019_20260422", "author_factor")]}
    ).reset_index(names="TRADE_DT")

    alignment = alignment_sensitivity(author, labels, offsets=range(-2, 3))
    autocorr_rows: list[dict[str, float | int]] = []
    for lag in [1, 5, 10, 20]:
        values, mean_overlap = factor_autocorrelation(author, "author_factor", lag=lag)
        autocorr_rows.append(
            {
                "lag": lag,
                "n_dates": int(len(values)),
                "mean_autocorrelation": float(values.mean()),
                "median_autocorrelation": float(values.median()),
                "mean_overlap_stocks": mean_overlap,
            }
        )
    autocorrelation = pd.DataFrame(autocorr_rows)

    runs = local_run_map(args.manifest_path, args.lightgbm_summary_path, candidates)
    common, coverage, pairwise_frames = build_common_frame(
        author_labeled,
        runs,
        min_date=author["TRADE_DT"].min(),
    )
    pairwise_periods = {
        "pdf_available_pairwise_2019_20240930": (common["TRADE_DT"].min(), "20240930"),
        "post_pdf_pairwise_20241001_20260408": ("20241001", "20260408"),
        "all_pairwise_2019_20260408": (common["TRADE_DT"].min(), common["TRADE_DT"].max()),
    }
    (
        pairwise_metrics,
        pairwise_groups,
        pairwise_daily,
        pairwise_correlations,
        pairwise_ensembles,
        paired,
    ) = _pairwise_outputs(
        pairwise_frames,
        pairwise_periods,
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
        bootstrap_samples=args.bootstrap_samples,
    )
    (
        top_overlap_daily_frame,
        top_overlap_summary,
        disagreement_daily,
        disagreement_summary,
        disagreement_pairs,
        residual_metrics,
        residual_daily,
        portfolio_metrics,
        portfolio_daily,
        portfolio_turnover,
    ) = _extended_pairwise_outputs(
        {model: pairwise_frames[model] for model in extended_models},
        pairwise_periods,
        daily_returns=labels[[*KEY_COLUMNS, "daily_return"]],
        top_fractions=top_fractions,
        rebalance_every=args.rebalance_every,
        cost_bps_values=cost_bps_values,
        bootstrap_samples=args.bootstrap_samples,
    )
    del pairwise_frames

    factor_columns = ["author_factor", *candidates, "lightgbm"]
    common_periods = {
        "pdf_available_common_2019_20240930": (common["TRADE_DT"].min(), "20240930"),
        "post_pdf_common_20241001_20260408": ("20241001", "20260408"),
        "all_common_2019_20260408": (common["TRADE_DT"].min(), common["TRADE_DT"].max()),
    }
    common_metrics, common_groups, _ = _metrics_rows(
        common,
        factor_columns,
        common_periods,
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
    )
    common_year = _year_metrics(
        common,
        factor_columns,
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
    )
    common_daily, all_common_correlations = _common_rank_tables(common, factor_columns)
    all_common_ensembles = _ensemble_metrics(
        common,
        [*candidates, "lightgbm"],
        common_periods,
        n_groups=args.n_groups,
        rebalance_every=args.rebalance_every,
    )

    inventory.to_csv(output_dir / "author_inventory.csv", index=False)
    daily_inventory.to_csv(output_dir / "author_daily_inventory.csv", index=False)
    author_metrics.to_csv(output_dir / "author_period_metrics.csv", index=False)
    author_year.to_csv(output_dir / "author_year_metrics.csv", index=False)
    author_daily.to_csv(output_dir / "author_daily_rankic.csv", index=False)
    author_groups.to_csv(output_dir / "author_group_returns.csv", index=False)
    alignment.to_csv(output_dir / "alignment_sensitivity.csv", index=False)
    autocorrelation.to_csv(output_dir / "author_autocorrelation.csv", index=False)
    coverage.to_csv(output_dir / "coverage_summary.csv", index=False)
    pairwise_metrics.to_csv(output_dir / "pairwise_universe_metrics.csv", index=False)
    pairwise_groups.to_csv(output_dir / "pairwise_universe_group_returns.csv", index=False)
    pairwise_daily.to_csv(output_dir / "pairwise_universe_daily_rankic.csv", index=False)
    pairwise_correlations.to_csv(output_dir / "factor_rank_correlation.csv", index=False)
    pairwise_ensembles.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_rankic_difference.csv", index=False)
    top_overlap_daily_frame.to_csv(output_dir / "top_overlap_daily.csv", index=False)
    top_overlap_summary.to_csv(output_dir / "top_overlap_summary.csv", index=False)
    disagreement_daily.to_csv(output_dir / "disagreement_group_daily_returns.csv", index=False)
    disagreement_summary.to_csv(output_dir / "disagreement_group_summary.csv", index=False)
    disagreement_pairs.to_csv(output_dir / "disagreement_pair_comparison.csv", index=False)
    residual_metrics.to_csv(output_dir / "residual_rankic_metrics.csv", index=False)
    residual_daily.to_csv(output_dir / "residual_rankic_daily.csv", index=False)
    portfolio_metrics.to_csv(output_dir / "portfolio_backtest_metrics.csv", index=False)
    portfolio_daily.to_csv(output_dir / "portfolio_backtest_daily.csv", index=False)
    portfolio_turnover.to_csv(output_dir / "portfolio_turnover_events.csv", index=False)
    common_metrics.to_csv(output_dir / "common_universe_metrics.csv", index=False)
    common_year.to_csv(output_dir / "common_universe_year_metrics.csv", index=False)
    common_daily.to_csv(output_dir / "common_universe_daily_rankic.csv", index=False)
    common_groups.to_csv(output_dir / "common_universe_group_returns.csv", index=False)
    all_common_correlations.to_csv(
        output_dir / "all_common_factor_rank_correlation.csv",
        index=False,
    )
    all_common_ensembles.to_csv(output_dir / "all_common_ensemble_metrics.csv", index=False)

    manifest = {
        "author_dir": str(Path(args.author_dir).resolve()),
        "cache_path": str(Path(args.cache_path).resolve()),
        "selected_model_manifest": str(Path(args.manifest_path).resolve()),
        "lightgbm_summary": str(Path(args.lightgbm_summary_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "candidates": candidates,
        "extended_models": extended_models,
        "n_groups": int(args.n_groups),
        "rebalance_every": int(args.rebalance_every),
        "bootstrap_samples": int(args.bootstrap_samples),
        "top_fractions": top_fractions,
        "cost_bps": cost_bps_values,
        "pdf_alpha_metrics": PDF_ALPHA_METRICS,
        "author_same_date_alignment": True,
        "top_annualization": "mean 11-day top excess multiplied by 252/11",
        "portfolio_backtest": (
            "equal-weight top portfolio, signal sampled every rebalance_every dates, "
            "executed at the next trading-day close, close-to-close price returns, "
            "missing holding returns filled with zero, final signal held for one rebalance interval"
        ),
        "portfolio_daily_return": "S_DQ_PCTCHANGE / 100",
        "portfolio_turnover": "0.5 * L1 distance between consecutive target weights",
        "transaction_cost": "cost_bps applied per 100% target turnover",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(
        output_dir,
        inventory,
        author_metrics,
        pairwise_metrics,
        pairwise_correlations,
        pairwise_ensembles,
        paired,
        coverage,
        common_metrics,
        all_common_correlations,
        alignment,
        autocorrelation,
    )

    print(
        pairwise_metrics.loc[
            pairwise_metrics["period"] == "all_pairwise_2019_20260408"
        ].to_string(index=False)
    )
    print(f"wrote_output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
