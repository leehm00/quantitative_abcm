from __future__ import annotations

import argparse
import csv
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.dataset import ABCMWindowSampler
from abcm.evaluation import evaluate_factor_frame, write_evaluation_outputs
from abcm.features import FEATURE_COLUMNS
from abcm.pipeline import (
    available_paired_training_dates,
    available_training_dates,
    load_config,
    load_or_prepare_abcm_frame_from_files,
)
from abcm.splits import select_validation_fold


@dataclass(frozen=True)
class LightGBMSamples:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _date_to_segment(frame: pd.DataFrame) -> dict[str, object]:
    if "segment_id" not in frame.columns:
        return {}
    date_segments = (
        frame[["TRADE_DT", "segment_id"]]
        .drop_duplicates()
        .assign(TRADE_DT=lambda data: data["TRADE_DT"].astype(str))
    )
    counts = date_segments.groupby("TRADE_DT")["segment_id"].nunique()
    ambiguous = counts[counts > 1].index.tolist()
    if ambiguous:
        raise ValueError(f"Dates appear in multiple segments: {ambiguous[:5]}")
    return dict(zip(date_segments["TRADE_DT"], date_segments["segment_id"]))


def build_lightgbm_samples(
    frame: pd.DataFrame,
    dates: Iterable[str],
    feature_columns: list[str] | None = None,
    lookback: int = 60,
    stock_limit: int = 512,
    seed: int = 42,
    sampler: ABCMWindowSampler | None = None,
) -> LightGBMSamples:
    features = feature_columns or FEATURE_COLUMNS
    sampler = sampler or ABCMWindowSampler(frame, features, lookback=lookback)
    segment_by_date = _date_to_segment(frame)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    for idx, date in enumerate(str(item) for item in dates):
        batch = sampler.sample_for_date(date, stock_limit=stock_limit, seed=seed + idx)
        x_current = batch.x[0, :, -1, :].astype(np.float32, copy=False)
        y_current = batch.y1[0].astype(np.float32, copy=False)
        x_parts.append(x_current)
        y_parts.append(y_current)
        for row_idx, code in enumerate(batch.codes[0]):
            row: dict[str, object] = {
                "TRADE_DT": batch.dates[0],
                "S_INFO_WINDCODE": code,
                "y1_raw": float(batch.y1[0, row_idx]),
                "y2_raw": float(batch.y2[0, row_idx]),
            }
            if segment_by_date:
                row["segment_id"] = segment_by_date[batch.dates[0]]
            meta_rows.append(row)
    if not x_parts:
        raise ValueError("No dates supplied for LightGBM samples")
    return LightGBMSamples(
        x=np.concatenate(x_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        meta=pd.DataFrame(meta_rows),
    )


def build_tabular_lightgbm_samples(
    frame: pd.DataFrame,
    dates: Iterable[str],
    feature_columns: list[str] | None = None,
    stock_limit: int = 512,
    seed: int = 42,
) -> LightGBMSamples:
    features = feature_columns or FEATURE_COLUMNS
    date_list = [str(item) for item in dates]
    date_order = {date: idx for idx, date in enumerate(date_list)}
    meta_cols = ["TRADE_DT", "S_INFO_WINDCODE", "y1_raw", "y2_raw"]
    if "segment_id" in frame.columns:
        meta_cols.insert(2, "segment_id")
    selected_cols = [*meta_cols, *features]
    work = frame.loc[frame["TRADE_DT"].astype(str).isin(date_order), selected_cols].copy()
    work["TRADE_DT"] = work["TRADE_DT"].astype(str)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[*features, "y1_raw", "y2_raw"])
    work["_date_order"] = work["TRADE_DT"].map(date_order)
    work = work.sort_values(["_date_order", "S_INFO_WINDCODE"]).reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
    for idx, (_, group) in enumerate(work.groupby("TRADE_DT", sort=False)):
        if len(group) > stock_limit:
            group = group.sample(n=stock_limit, random_state=seed + idx)
        chunks.append(group.sort_values(["_date_order", "S_INFO_WINDCODE"]))
    if not chunks:
        raise ValueError("No rows available for tabular LightGBM samples")
    sampled = pd.concat(chunks, ignore_index=True).drop(columns=["_date_order"])
    return LightGBMSamples(
        x=sampled[features].to_numpy(dtype=np.float32),
        y=sampled["y1_raw"].to_numpy(dtype=np.float32),
        meta=sampled[meta_cols].reset_index(drop=True),
    )


def factor_frame_from_predictions(
    samples: LightGBMSamples,
    predictions: np.ndarray,
    beta_dim: int = 12,
) -> pd.DataFrame:
    preds = np.asarray(predictions, dtype=float).reshape(-1)
    if len(preds) != len(samples.meta):
        raise ValueError(f"Prediction count {len(preds)} does not match sample count {len(samples.meta)}")
    out = samples.meta.copy()
    out["alpha_0"] = preds
    for idx in range(beta_dim):
        out[f"beta_{idx}"] = 0.0
    ordered = ["TRADE_DT", "S_INFO_WINDCODE"]
    if "segment_id" in out.columns:
        ordered.append("segment_id")
    ordered.extend(["alpha_0", *[f"beta_{idx}" for idx in range(beta_dim)], "y1_raw", "y2_raw"])
    return out[ordered]


def _select_export_dates(valid_dates: list[str], export_valid_dates: int) -> list[str]:
    if export_valid_dates < 0:
        return list(valid_dates)
    if export_valid_dates == 0:
        return []
    return list(valid_dates[:export_valid_dates])


def _eligible_dates(
    frame: pd.DataFrame,
    lookback: int,
    min_stocks: int,
    turnover_lag: int,
    stock_limit: int,
) -> list[str]:
    if turnover_lag > 0:
        return available_paired_training_dates(
            frame,
            feature_columns=FEATURE_COLUMNS,
            lookback=lookback,
            min_stocks=min_stocks,
            turnover_lag=turnover_lag,
            stock_limit=stock_limit,
        )
    return available_training_dates(frame, lookback=lookback, min_stocks=min_stocks)


def _validation_metrics(predictions: np.ndarray, samples: LightGBMSamples) -> dict[str, object]:
    y = samples.y.astype(float)
    pred = np.asarray(predictions, dtype=float).reshape(-1)
    mse = float(np.mean((pred - y) ** 2))
    corr = ""
    if len(pred) > 1 and np.std(pred) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(pred, y)[0, 1])
    n_dates = int(samples.meta["TRADE_DT"].nunique()) if not samples.meta.empty else 0
    return {
        "loss": mse,
        "mse": mse,
        "r2_residual": "",
        "corr": corr,
        "turnover": "",
        "n_stocks": float(len(samples.meta) / n_dates) if n_dates else 0.0,
        "n_dates": float(n_dates),
    }


def _write_single_row_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _dump_simple_yaml(cfg: dict) -> str:
    lines: list[str] = []
    for section, values in cfg.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if value in {None, ""}:
                continue
            if isinstance(value, str):
                lines.append(f'  {key}: "{value}"')
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _run_config(
    base_config: dict,
    args: argparse.Namespace,
    fold: int,
    output_dir: Path,
    stock_limit: int,
    max_files: int | None,
    prepared_frame_cache: str | None,
) -> dict:
    cfg = {
        "data": deepcopy(base_config.get("data", {})),
        "model": {},
        "train": deepcopy(base_config.get("train", {})),
    }
    cfg["data"]["stock_limit"] = stock_limit
    if max_files is not None:
        cfg["data"]["max_files"] = max_files
    if prepared_frame_cache is not None:
        cfg["data"]["prepared_frame_cache"] = prepared_frame_cache
    cfg["model"].update(
        {
            "type": "lightgbm",
            "sample_mode": args.sample_mode,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
        }
    )
    cfg["train"]["learning_rate"] = args.learning_rate
    cfg["train"]["max_steps"] = args.n_estimators
    cfg["train"]["date_batch_size"] = 0
    cfg["train"]["validation_fold"] = fold
    cfg["train"]["output_dir"] = str(output_dir.parent)
    return cfg


def _run_name(args: argparse.Namespace, fold: int, stock_limit: int, label_clip_abs: object) -> str:
    clip = "" if label_clip_abs in {None, ""} else f"_clip{label_clip_abs:g}"
    return (
        f"lightgbm_{args.sample_mode}_lr{args.learning_rate:g}_leaves{args.num_leaves}_est{args.n_estimators}"
        f"_n{stock_limit}_vf{fold}{clip}"
    )


def _fit_model(args: argparse.Namespace, train_samples: LightGBMSamples, valid_samples: LightGBMSamples):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.num_threads,
        verbosity=args.verbosity,
    )
    callbacks = []
    if args.log_period > 0:
        callbacks.append(lgb.log_evaluation(period=args.log_period))
    if args.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=args.log_period > 0))
    model.fit(
        train_samples.x,
        train_samples.y,
        eval_set=[(valid_samples.x, valid_samples.y)],
        eval_names=["valid"],
        eval_metric="l2",
        callbacks=callbacks,
    )
    return model


def _write_training_log(model, output_dir: Path) -> None:
    result = getattr(model, "evals_result_", {})
    valid_l2 = result.get("valid", {}).get("l2", [])
    if not valid_l2:
        return
    with (output_dir / "training_log.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "valid_l2"])
        writer.writeheader()
        for idx, value in enumerate(valid_l2, start=1):
            writer.writerow({"iteration": idx, "valid_l2": value})


def _write_feature_importance(model, output_dir: Path) -> None:
    booster = model.booster_
    rows = []
    for name, split, gain in zip(
        FEATURE_COLUMNS,
        booster.feature_importance(importance_type="split"),
        booster.feature_importance(importance_type="gain"),
    ):
        rows.append({"feature": name, "split": int(split), "gain": float(gain)})
    with (output_dir / "feature_importance.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "split", "gain"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_folds(value: str | None, default_fold: int) -> list[int]:
    if value in {None, ""}:
        return [default_fold]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LightGBM baseline with ABCM data splits and evaluation outputs.")
    parser.add_argument("--config", default="configs/abcm1_daily_label_clip1.yaml")
    parser.add_argument("--output-root", default="outputs/baselines/lightgbm")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--prepared-frame-cache", default=None)
    parser.add_argument("--validation-folds", default=None, help="Comma-separated fold ids, e.g. 0,1,2,3 or -1.")
    parser.add_argument("--export-valid-dates", type=int, default=-1)
    parser.add_argument("--sample-mode", choices=["tabular", "sampler"], default="tabular")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-period", type=int, default=50)
    parser.add_argument("--verbosity", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    data_root = Path(data_cfg.get("root", "data/testdata"))
    max_files = args.max_files if args.max_files is not None else int(data_cfg.get("max_files", 2))
    stock_limit = args.stock_limit if args.stock_limit is not None else int(data_cfg.get("stock_limit", 512))
    cache_path = args.prepared_frame_cache if args.prepared_frame_cache is not None else data_cfg.get("prepared_frame_cache")
    label_clip_abs = _optional_float(data_cfg.get("label_clip_abs"))

    frame = load_or_prepare_abcm_frame_from_files(
        data_root,
        max_files=max_files,
        cache_path=cache_path,
        feature_columns=FEATURE_COLUMNS,
        y1_horizon=int(data_cfg.get("y1_horizon", 11)),
        y2_horizon=int(data_cfg.get("y2_horizon", 21)),
        entry_lag=int(data_cfg.get("entry_lag", 1)),
        label_clip_abs=label_clip_abs,
    )

    lookback = int(data_cfg.get("lookback", 60))
    min_stocks = int(train_cfg.get("min_stocks", 64))
    turnover_lag = int(train_cfg.get("turnover_lag", 5))
    dates = _eligible_dates(frame, lookback, min_stocks, turnover_lag, stock_limit)
    if not dates:
        raise RuntimeError("No eligible dates after preprocessing")

    cv_folds = int(train_cfg.get("cv_folds", 5))
    default_fold = int(train_cfg.get("validation_fold", -1))
    folds = _parse_folds(args.validation_folds, default_fold)
    sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=lookback) if args.sample_mode == "sampler" else None
    output_root = Path(args.output_root)
    run_dirs: list[Path] = []
    for fold in folds:
        fold_arg = None if fold < 0 else fold
        train_dates, valid_dates = select_validation_fold(dates, n_folds=cv_folds, fold_id=fold_arg)
        export_dates = _select_export_dates(valid_dates, args.export_valid_dates)
        if not train_dates or not export_dates:
            raise RuntimeError(f"Fold {fold} has train_dates={len(train_dates)} export_dates={len(export_dates)}")

        run_name = _run_name(args, fold, stock_limit, label_clip_abs)
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / run_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = _run_config(config, args, fold, output_dir, stock_limit, max_files, cache_path)
        (output_dir / "config.yaml").write_text(_dump_simple_yaml(run_config))
        with (output_dir / "split_summary.txt").open("w") as fh:
            fh.write(f"cv_folds={cv_folds}\n")
            fh.write(f"validation_fold={fold}\n")
            fh.write(f"train_dates={len(train_dates)}\n")
            fh.write(f"valid_dates={len(valid_dates)}\n")
            fh.write(f"valid_start={valid_dates[0] if valid_dates else ''}\n")
            fh.write(f"valid_end={valid_dates[-1] if valid_dates else ''}\n")

        print(f"fold={fold} train_dates={len(train_dates)} valid_dates={len(valid_dates)}")
        if args.sample_mode == "sampler":
            train_samples = build_lightgbm_samples(
                frame,
                train_dates,
                feature_columns=FEATURE_COLUMNS,
                lookback=lookback,
                stock_limit=stock_limit,
                seed=args.seed,
                sampler=sampler,
            )
            valid_samples = build_lightgbm_samples(
                frame,
                valid_dates,
                feature_columns=FEATURE_COLUMNS,
                lookback=lookback,
                stock_limit=stock_limit,
                seed=args.seed + 10_000,
                sampler=sampler,
            )
        else:
            train_samples = build_tabular_lightgbm_samples(
                frame,
                train_dates,
                feature_columns=FEATURE_COLUMNS,
                stock_limit=stock_limit,
                seed=args.seed,
            )
            valid_samples = build_tabular_lightgbm_samples(
                frame,
                valid_dates,
                feature_columns=FEATURE_COLUMNS,
                stock_limit=stock_limit,
                seed=args.seed + 10_000,
            )
        export_samples = valid_samples
        if export_dates != valid_dates:
            if args.sample_mode == "sampler":
                export_samples = build_lightgbm_samples(
                    frame,
                    export_dates,
                    feature_columns=FEATURE_COLUMNS,
                    lookback=lookback,
                    stock_limit=stock_limit,
                    seed=args.seed + 20_000,
                    sampler=sampler,
                )
            else:
                export_samples = build_tabular_lightgbm_samples(
                    frame,
                    export_dates,
                    feature_columns=FEATURE_COLUMNS,
                    stock_limit=stock_limit,
                    seed=args.seed + 20_000,
                )

        model = _fit_model(args, train_samples, valid_samples)
        valid_pred = model.predict(valid_samples.x)
        export_pred = valid_pred if export_samples is valid_samples else model.predict(export_samples.x)
        factors = factor_frame_from_predictions(export_samples, export_pred)
        factors.to_csv(output_dir / "factors.csv", index=False)
        _write_single_row_csv(output_dir / "validation_metrics.csv", _validation_metrics(valid_pred, valid_samples))
        write_evaluation_outputs(evaluate_factor_frame(factors), output_dir)
        model.booster_.save_model(output_dir / "model.txt")
        with (output_dir / "model.pkl").open("wb") as fh:
            pickle.dump(model, fh)
        _write_training_log(model, output_dir)
        _write_feature_importance(model, output_dir)
        print(f"run_dir={output_dir}")
        run_dirs.append(output_dir)

    print("run_dirs=" + ",".join(str(path) for path in run_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
