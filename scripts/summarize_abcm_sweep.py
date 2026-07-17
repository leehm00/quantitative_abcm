from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.pipeline import load_config
from scripts.sweep_abcm1 import _read_evaluation_metrics, _read_single_row_csv, _safe_float, _sort_rows_by_return


FIELDNAMES = [
    "name",
    "run_dir",
    "hidden_dim",
    "learning_rate",
    "stock_limit",
    "max_steps",
    "date_batch_size",
    "lambda_mse",
    "lambda_r2",
    "lambda_alpha_corr",
    "lambda_corr",
    "lambda_to",
    "validation_fold",
    "label_transform",
    "dropout",
    "weight_decay",
    "validation_loss",
    "validation_mse",
    "validation_r2_residual",
    "validation_turnover",
    "alpha_orientation",
    "alpha_rankic",
    "alpha_oriented_rankic",
    "alpha_abs_rankic",
    "alpha_icir",
    "alpha_oriented_icir",
    "alpha_win_rate",
    "alpha_oriented_win_rate",
    "alpha_hit_rate",
    "alpha_top_excess_mean",
    "alpha_top_excess_median",
    "alpha_top_excess_trimmed_mean",
    "alpha_top_excess_max",
    "alpha_top_excess_abs_gt_1_count",
    "alpha_long_short_mean",
    "alpha_long_short_median",
    "alpha_long_short_trimmed_mean",
    "alpha_long_short_max",
    "alpha_long_short_abs_gt_1_count",
    "alpha_top_excess_positive_rate",
    "alpha_long_short_positive_rate",
]


def _hidden_dim_from_name(name: str) -> int | str:
    first = name.split("_", 1)[0]
    if first.startswith("h") and first[1:].isdigit():
        return int(first[1:])
    return ""


def _validation_fold_from_config(run_dir: Path) -> int | str:
    values = _training_values_from_config(run_dir)
    return values.get("validation_fold", "")


def _training_values_from_config(run_dir: Path) -> dict[str, float | int | str]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        config = load_config(config_path)
    except Exception:
        return {}
    train = config.get("train", {})
    data = config.get("data", {})
    model = config.get("model", {})
    result: dict[str, float | int | str] = {}
    for key in ["learning_rate", "lambda_mse", "lambda_r2", "lambda_alpha_corr", "lambda_corr", "lambda_to", "weight_decay"]:
        value = train.get(key)
        result[key] = "" if value in {None, ""} else float(value)
    dropout = model.get("dropout")
    result["dropout"] = "" if dropout in {None, ""} else float(dropout)
    for key in ["max_steps", "date_batch_size", "validation_fold"]:
        value = train.get(key)
        result[key] = "" if value in {None, ""} else int(value)
    stock_limit = data.get("stock_limit")
    result["stock_limit"] = "" if stock_limit in {None, ""} else int(stock_limit)
    label_transform = data.get("label_transform")
    result["label_transform"] = "" if label_transform in {None, ""} else str(label_transform)
    return result


def _completed_run_dirs(sweep_root: Path) -> list[Path]:
    dirs = []
    for alpha_path in sweep_root.glob("*/20*/metrics_alpha.csv"):
        run_dir = alpha_path.parent
        required = [
            run_dir / "validation_metrics.csv",
            run_dir / "prediction_accuracy.csv",
            run_dir / "alpha_long_short.csv",
        ]
        if all(path.exists() for path in required):
            dirs.append(run_dir)
    return sorted(dirs)


def summarize_completed_runs(sweep_root: str | Path) -> list[dict]:
    root = Path(sweep_root)
    rows: list[dict] = []
    for run_dir in _completed_run_dirs(root):
        validation = _read_single_row_csv(run_dir / "validation_metrics.csv")
        metrics = _read_evaluation_metrics(run_dir)
        name = run_dir.parent.name
        config_values = _training_values_from_config(run_dir)
        rows.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "hidden_dim": _hidden_dim_from_name(name),
                "learning_rate": config_values.get("learning_rate", ""),
                "stock_limit": config_values.get("stock_limit", ""),
                "max_steps": config_values.get("max_steps", ""),
                "date_batch_size": config_values.get("date_batch_size", ""),
                "lambda_mse": config_values.get("lambda_mse", ""),
                "lambda_r2": config_values.get("lambda_r2", ""),
                "lambda_alpha_corr": config_values.get("lambda_alpha_corr", ""),
                "lambda_corr": config_values.get("lambda_corr", ""),
                "lambda_to": config_values.get("lambda_to", ""),
                "validation_fold": config_values.get("validation_fold", ""),
                "label_transform": config_values.get("label_transform", ""),
                "dropout": config_values.get("dropout", ""),
                "weight_decay": config_values.get("weight_decay", ""),
                "validation_loss": _safe_float(validation.get("loss")),
                "validation_mse": _safe_float(validation.get("mse")),
                "validation_r2_residual": _safe_float(validation.get("r2_residual")),
                "validation_turnover": _safe_float(validation.get("turnover")),
                **metrics,
            }
        )
    return _sort_rows_by_return(rows)


def write_summary(sweep_root: str | Path, output_csv: str | Path) -> list[dict]:
    rows = summarize_completed_runs(sweep_root)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize completed ABCM sweep runs by validation return metrics.")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    output_csv = Path(args.output_csv) if args.output_csv else sweep_root / "partial_completed_return_summary.csv"
    rows = write_summary(sweep_root, output_csv)
    print(f"rows={len(rows)}")
    print(f"summary_csv={output_csv}")
    for row in rows[:10]:
        print(
            f"{row['name']} top_excess={row['alpha_top_excess_mean']} "
            f"long_short={row['alpha_long_short_mean']} "
            f"oriented_rankic={row['alpha_oriented_rankic']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
