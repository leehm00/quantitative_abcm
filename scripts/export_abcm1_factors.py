from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.dataset import ABCMWindowSampler
from abcm.evaluation import evaluate_factor_frame, write_evaluation_outputs
from abcm.features import FEATURE_COLUMNS
from abcm.model import ABCM
from abcm.pipeline import (
    available_paired_training_dates,
    available_training_dates,
    export_factors_for_dates,
    load_or_prepare_abcm_frame_from_files,
    load_config,
)
from abcm.splits import select_validation_fold


def select_split_dates(
    train_dates: list[str],
    valid_dates: list[str],
    split: str,
    max_dates: int,
) -> list[str]:
    if split == "train":
        selected = list(train_dates)
    elif split == "valid":
        selected = list(valid_dates)
    elif split == "all":
        selected = list(train_dates) + list(valid_dates)
    else:
        raise ValueError(f"Unsupported split: {split}")
    if max_dates >= 0:
        return selected[:max_dates]
    return selected


def load_trained_model(
    checkpoint_path: str | Path,
    input_dim: int,
    fallback_model_config: dict,
    device: str | torch.device,
) -> ABCM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint.get("model_config", fallback_model_config)
    model = ABCM(input_dim=input_dim, **model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_from_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    split: str = "valid",
    max_dates: int = -1,
    device: str | None = None,
    evaluate: bool = True,
) -> Path:
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    data_root = Path(data_cfg.get("root", "data/testdata"))
    max_files = int(data_cfg.get("max_files", 2))
    frame = load_or_prepare_abcm_frame_from_files(
        data_root,
        max_files=max_files,
        cache_path=data_cfg.get("prepared_frame_cache"),
        feature_columns=FEATURE_COLUMNS,
        y1_horizon=int(data_cfg.get("y1_horizon", 11)),
        y2_horizon=int(data_cfg.get("y2_horizon", 21)),
        entry_lag=int(data_cfg.get("entry_lag", 1)),
        label_clip_abs=data_cfg.get("label_clip_abs"),
    )

    lookback = int(data_cfg.get("lookback", 60))
    min_stocks = int(train_cfg.get("min_stocks", 64))
    turnover_lag = int(train_cfg.get("turnover_lag", 5))
    stock_limit = int(data_cfg.get("stock_limit", 512))
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
    cv_folds = int(train_cfg.get("cv_folds", 5))
    validation_fold = int(train_cfg.get("validation_fold", -1))
    fold_arg = None if validation_fold < 0 else validation_fold
    train_dates, valid_dates = select_validation_fold(dates, n_folds=cv_folds, fold_id=fold_arg)
    export_dates = select_split_dates(train_dates, valid_dates, split=split, max_dates=max_dates)
    if not export_dates:
        raise RuntimeError("No dates selected for export")

    run_device = device or train_cfg.get("device", "cpu")
    model = load_trained_model(checkpoint_path, input_dim=len(FEATURE_COLUMNS), fallback_model_config=model_cfg, device=run_device)
    sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=lookback)
    factors = export_factors_for_dates(
        model,
        frame,
        dates=export_dates,
        feature_columns=FEATURE_COLUMNS,
        lookback=lookback,
        stock_limit=stock_limit,
        device=run_device,
        sampler=sampler,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(output_path, index=False)
    if evaluate:
        result = evaluate_factor_frame(factors)
        write_evaluation_outputs(result, output_path.parent)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ABCM1 factors from a saved checkpoint.")
    parser.add_argument("--config", required=True, help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoints/best.pt.")
    parser.add_argument("--output-csv", required=True, help="Destination factors CSV path.")
    parser.add_argument("--split", choices=["train", "valid", "all"], default="valid")
    parser.add_argument("--max-dates", type=int, default=-1, help="-1 exports all selected split dates.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    output = export_from_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        split=args.split,
        max_dates=args.max_dates,
        device=args.device,
        evaluate=not args.no_evaluate,
    )
    df = pd.read_csv(output, usecols=["TRADE_DT"])
    print(f"factors_csv={output}")
    print(f"rows={len(df)}")
    print(f"dates={df['TRADE_DT'].astype(str).nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
