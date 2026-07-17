from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.dataset import ABCMWindowSampler
from abcm.features import FEATURE_COLUMNS
from abcm.model import ABCM
from abcm.pipeline import (
    available_paired_training_dates,
    available_training_dates,
    evaluate_loss_for_dates,
    export_factors_for_dates,
    load_or_prepare_abcm_frame_from_files,
    load_config,
    train_one_batch,
)
from abcm.splits import select_validation_fold


def select_export_dates(valid_dates: list[str], export_valid_dates: int) -> list[str]:
    if export_valid_dates < 0:
        return list(valid_dates)
    if export_valid_dates == 0:
        return []
    return list(valid_dates[:export_valid_dates])


def loss_weights_from_config(train_cfg: dict) -> dict[str, float]:
    return {
        "lambda_mse": float(train_cfg.get("lambda_mse", 1.0)),
        "lambda_r2": float(train_cfg.get("lambda_r2", 1.0)),
        "lambda_alpha_corr": float(train_cfg.get("lambda_alpha_corr", 0.0)),
        "lambda_corr": float(train_cfg.get("lambda_corr", 0.01)),
        "lambda_to": float(train_cfg.get("lambda_to", 0.01)),
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def estimate_epoch_steps(train_date_count: int, date_batch_size: int) -> int:
    if train_date_count <= 0:
        return 0
    return int(math.ceil(train_date_count / max(1, date_batch_size)))


def iter_training_date_batches(
    train_dates: list[str],
    max_steps: int,
    date_batch_size: int,
    seed: int,
    shuffle_each_epoch: bool = True,
):
    if not train_dates:
        raise ValueError("train_dates must not be empty")
    batch_size = max(1, int(date_batch_size))
    epoch_steps = estimate_epoch_steps(len(train_dates), batch_size)
    emitted = 0
    epoch = 0
    while emitted < max_steps:
        ordered_dates = list(train_dates)
        if shuffle_each_epoch:
            random.Random(int(seed) + epoch).shuffle(ordered_dates)
        for epoch_step, start in enumerate(range(0, len(ordered_dates), batch_size)):
            if emitted >= max_steps:
                return
            yield epoch, epoch_step, ordered_dates[start : start + batch_size]
            emitted += 1
        if epoch_steps == 0:
            return
        epoch += 1


def count_model_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return int(sum(param.numel() for param in parameters))


def build_training_run_summary(
    *,
    train_start_utc: str,
    train_end_utc: str,
    total_train_seconds: float,
    step_seconds: list[float],
    train_date_count: int,
    valid_date_count: int,
    max_steps: int,
    date_batch_size: int,
    model_parameter_count: int,
    trainable_parameter_count: int,
    device: str,
    cuda_metadata: dict | None = None,
    processed_train_date_slots: int | None = None,
    shuffle_train_dates: bool = False,
) -> dict:
    mean_step_seconds = float(sum(step_seconds) / len(step_seconds)) if step_seconds else 0.0
    estimated_steps = estimate_epoch_steps(train_date_count=train_date_count, date_batch_size=date_batch_size)
    processed_slots = (
        int(processed_train_date_slots)
        if processed_train_date_slots is not None
        else int(len(step_seconds) * date_batch_size)
    )
    summary = {
        "train_start_utc": train_start_utc,
        "train_end_utc": train_end_utc,
        "total_train_seconds": float(total_train_seconds),
        "completed_steps": int(len(step_seconds)),
        "max_steps": int(max_steps),
        "train_date_count": int(train_date_count),
        "valid_date_count": int(valid_date_count),
        "date_batch_size": int(date_batch_size),
        "estimated_epoch_steps": int(estimated_steps),
        "processed_train_date_slots": processed_slots,
        "completed_epoch_equivalents": float(processed_slots / train_date_count) if train_date_count else 0.0,
        "shuffle_train_dates": bool(shuffle_train_dates),
        "mean_step_seconds": mean_step_seconds,
        "estimated_epoch_seconds": float(mean_step_seconds * estimated_steps),
        "model_parameter_count": int(model_parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "device": device,
    }
    summary.update(cuda_metadata or {})
    return summary


def _cuda_device_index(device: str | torch.device) -> int | None:
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError):
        return None
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    if torch_device.index is not None:
        return int(torch_device.index)
    return int(torch.cuda.current_device())


def reset_cuda_peak_stats(device: str | torch.device) -> None:
    device_index = _cuda_device_index(device)
    if device_index is not None:
        torch.cuda.reset_peak_memory_stats(device_index)


def synchronize_cuda(device: str | torch.device) -> None:
    device_index = _cuda_device_index(device)
    if device_index is not None:
        torch.cuda.synchronize(device_index)


def collect_cuda_metadata(device: str | torch.device) -> dict:
    metadata: dict[str, object] = {"cuda_available": bool(torch.cuda.is_available())}
    device_index = _cuda_device_index(device)
    if device_index is None:
        return metadata
    metadata.update(
        {
            "cuda_device_index": device_index,
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        }
    )
    return metadata


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    run_start_utc = utc_timestamp()
    run_started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train an ABCM1 daily model.")
    parser.add_argument("--config", default="configs/abcm1_daily.yaml")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)

    data_root = Path(data_cfg.get("root", "data/testdata"))
    max_files = args.max_files if args.max_files is not None else int(data_cfg.get("max_files", 2))
    frame = load_or_prepare_abcm_frame_from_files(
        data_root,
        max_files=max_files,
        cache_path=data_cfg.get("prepared_frame_cache"),
        feature_columns=FEATURE_COLUMNS,
        y1_horizon=int(data_cfg.get("y1_horizon", 11)),
        y2_horizon=int(data_cfg.get("y2_horizon", 21)),
        entry_lag=int(data_cfg.get("entry_lag", 1)),
        label_clip_abs=data_cfg.get("label_clip_abs"),
        label_transform=data_cfg.get("label_transform"),
    )

    lookback = int(data_cfg.get("lookback", 60))
    min_stocks = int(train_cfg.get("min_stocks", 64))
    turnover_lag = int(train_cfg.get("turnover_lag", 5))
    stock_limit = args.stock_limit if args.stock_limit is not None else int(data_cfg.get("stock_limit", 512))
    if turnover_lag > 0:
        dates = available_paired_training_dates(
            frame,
            feature_columns=FEATURE_COLUMNS,
            lookback=lookback,
            min_stocks=min_stocks,
            turnover_lag=turnover_lag,
            stock_limit=stock_limit,
        )
    else:
        dates = available_training_dates(frame, lookback=lookback, min_stocks=min_stocks)
    if not dates:
        raise RuntimeError("No eligible training dates after preprocessing")
    cv_folds = int(train_cfg.get("cv_folds", 5))
    validation_fold = int(train_cfg.get("validation_fold", -1))
    fold_arg = None if validation_fold < 0 else validation_fold
    train_dates, valid_dates = select_validation_fold(dates, n_folds=cv_folds, fold_id=fold_arg)
    if not train_dates:
        raise RuntimeError("No training dates after validation fold selection")

    device = args.device or train_cfg.get("device", "cpu")
    print(f"device={device}")
    model = ABCM(input_dim=len(FEATURE_COLUMNS), **model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=lookback)
    loss_weights = loss_weights_from_config(train_cfg)

    max_steps = args.max_steps if args.max_steps is not None else int(train_cfg.get("max_steps", 5))
    date_batch_size = max(1, int(train_cfg.get("date_batch_size", 1)))
    shuffle_train_dates = bool(train_cfg.get("shuffle_train_dates", True))
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(train_cfg.get("output_dir", "outputs/abcm1_daily")) / run_id
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")

    model_parameter_count = count_model_parameters(model)
    trainable_parameter_count = count_model_parameters(model, trainable_only=True)
    reset_cuda_peak_stats(device)
    train_start_utc = utc_timestamp()
    train_started_at = time.perf_counter()
    log_rows = []
    processed_train_date_slots = 0
    training_batches = iter_training_date_batches(
        train_dates,
        max_steps=max_steps,
        date_batch_size=date_batch_size,
        seed=seed,
        shuffle_each_epoch=shuffle_train_dates,
    )
    for step, (epoch, epoch_step, step_dates) in enumerate(training_batches):
        synchronize_cuda(device)
        step_started_at = time.perf_counter()
        metrics = train_one_batch(
            model,
            frame,
            feature_columns=FEATURE_COLUMNS,
            dates=step_dates,
            lookback=lookback,
            stock_limit=stock_limit,
            optimizer=optimizer,
            device=device,
            turnover_lag=turnover_lag,
            sampler=sampler,
            loss_weights=loss_weights,
        )
        synchronize_cuda(device)
        step_seconds = time.perf_counter() - step_started_at
        metrics["step"] = step
        metrics["epoch"] = epoch
        metrics["epoch_step"] = epoch_step
        metrics["step_seconds"] = step_seconds
        metrics["cumulative_train_seconds"] = time.perf_counter() - train_started_at
        metrics["dates_per_second"] = float(metrics["n_dates"]) / step_seconds if step_seconds > 0 else 0.0
        metrics["date_stock_rows_per_second"] = (
            float(metrics["n_dates"]) * float(metrics["n_stocks"]) / step_seconds if step_seconds > 0 else 0.0
        )
        processed_train_date_slots += int(metrics["n_dates"])
        log_rows.append(metrics)
        print(
            f"step={step} date={step_dates[0]} n_dates={int(metrics['n_dates'])} loss={metrics['loss']:.6f} "
            f"mse={metrics['mse']:.6f} r2_residual={metrics['r2_residual']:.6f} n={int(metrics['n_stocks'])} "
            f"step_seconds={metrics['step_seconds']:.3f}"
        )
    train_end_utc = utc_timestamp()
    total_train_seconds = time.perf_counter() - train_started_at
    run_summary = build_training_run_summary(
        train_start_utc=train_start_utc,
        train_end_utc=train_end_utc,
        total_train_seconds=total_train_seconds,
        step_seconds=[float(row["step_seconds"]) for row in log_rows],
        train_date_count=len(train_dates),
        valid_date_count=len(valid_dates),
        max_steps=max_steps,
        date_batch_size=date_batch_size,
        model_parameter_count=model_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        device=str(device),
        cuda_metadata=collect_cuda_metadata(device),
        processed_train_date_slots=processed_train_date_slots,
        shuffle_train_dates=shuffle_train_dates,
    )
    write_json(output_dir / "run_summary.json", run_summary)

    torch.save({"model_state_dict": model.state_dict(), "model_config": model_cfg}, output_dir / "checkpoints" / "best.pt")
    if log_rows:
        with (output_dir / "training_log.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
    with (output_dir / "split_summary.txt").open("w") as fh:
        fh.write(f"cv_folds={cv_folds}\n")
        fh.write(f"validation_fold={validation_fold}\n")
        fh.write(f"train_dates={len(train_dates)}\n")
        fh.write(f"valid_dates={len(valid_dates)}\n")
        fh.write(f"valid_start={valid_dates[0] if valid_dates else ''}\n")
        fh.write(f"valid_end={valid_dates[-1] if valid_dates else ''}\n")
    export_valid_dates = int(train_cfg.get("export_valid_dates", 1))
    export_dates = select_export_dates(valid_dates, export_valid_dates)
    if export_dates:
        validation_metrics = evaluate_loss_for_dates(
            model,
            frame,
            dates=export_dates,
            feature_columns=FEATURE_COLUMNS,
            lookback=lookback,
            stock_limit=stock_limit,
            device=device,
            turnover_lag=turnover_lag,
            sampler=sampler,
            loss_weights=loss_weights,
        )
        with (output_dir / "validation_metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(validation_metrics.keys()))
            writer.writeheader()
            writer.writerow(validation_metrics)
        factors = export_factors_for_dates(
            model,
            frame,
            dates=export_dates,
            feature_columns=FEATURE_COLUMNS,
            lookback=lookback,
            stock_limit=stock_limit,
            device=device,
            sampler=sampler,
        )
        factors.to_csv(output_dir / "factors.csv", index=False)
    run_summary["run_start_utc"] = run_start_utc
    run_summary["run_end_utc"] = utc_timestamp()
    run_summary["total_run_seconds"] = float(time.perf_counter() - run_started_at)
    write_json(output_dir / "run_summary.json", run_summary)
    print(f"run_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
