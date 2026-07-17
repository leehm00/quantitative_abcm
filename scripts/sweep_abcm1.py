from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.pipeline import load_config
from abcm.sweep import assign_devices, build_sweep_runs, default_sweep_output_dir


def _write_config(
    base_config: dict,
    path: Path,
    run,
    output_root: Path,
    max_files: int | None,
    device: str,
    export_valid_dates: int | None = None,
    prepared_frame_cache: str | None = None,
) -> None:
    cfg = {
        "data": dict(base_config.get("data", {})),
        "model": dict(base_config.get("model", {})),
        "train": dict(base_config.get("train", {})),
    }
    cfg["model"]["hidden_dim"] = run.hidden_dim
    cfg["model"]["gru_layers"] = run.gru_layers
    cfg["train"]["learning_rate"] = run.learning_rate
    cfg["train"]["max_steps"] = run.max_steps
    cfg["train"]["date_batch_size"] = run.date_batch_size
    cfg["train"]["lambda_mse"] = run.lambda_mse
    cfg["train"]["lambda_r2"] = run.lambda_r2
    cfg["train"]["lambda_alpha_corr"] = run.lambda_alpha_corr
    cfg["train"]["lambda_corr"] = run.lambda_corr
    cfg["train"]["lambda_to"] = run.lambda_to
    cfg["train"]["validation_fold"] = run.validation_fold
    cfg["train"]["output_dir"] = str(output_root / run.name)
    cfg["train"]["device"] = device
    if run.dropout is not None:
        cfg["model"]["dropout"] = run.dropout
    if run.weight_decay is not None:
        cfg["train"]["weight_decay"] = run.weight_decay
    if export_valid_dates is not None:
        cfg["train"]["export_valid_dates"] = export_valid_dates
    cfg["data"]["stock_limit"] = run.stock_limit
    if max_files is not None:
        cfg["data"]["max_files"] = max_files
    if prepared_frame_cache is not None:
        cfg["data"]["prepared_frame_cache"] = prepared_frame_cache
    if getattr(run, "label_transform", None) is not None:
        cfg["data"]["label_transform"] = run.label_transform
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_simple_yaml(cfg))


def _dump_simple_yaml(cfg: dict) -> str:
    lines = []
    for section, values in cfg.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if isinstance(value, str):
                lines.append(f'  {key}: "{value}"')
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def _run_command(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)
    return completed.stdout + completed.stderr


def _parse_run_dir(output: str) -> Path:
    match = re.search(r"run_dir=(.+)", output)
    if match is None:
        raise RuntimeError(f"Could not parse run_dir from output:\n{output}")
    return Path(match.group(1).strip())


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0] if rows else {}


def _safe_float(value: str | float | int | None) -> float | str:
    if value in {None, ""}:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def _csv_column_values(path: Path, column: str) -> list[float]:
    if not path.exists():
        return []
    values = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            value = _safe_float(row.get(column))
            if value != "":
                values.append(float(value))
    return values


def _mean_values(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def _mean_csv_column(path: Path, column: str) -> float | str:
    return _mean_values(_csv_column_values(path, column))


def _median_values(values: list[float]) -> float | str:
    if not values:
        return ""
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _quantile_values(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(float(q), 0.0), 1.0)
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _trimmed_mean_values(values: list[float], lower_q: float = 0.05, upper_q: float = 0.95) -> float | str:
    if not values:
        return ""
    lower = _quantile_values(values, lower_q)
    upper = _quantile_values(values, upper_q)
    if lower == "" or upper == "":
        return ""
    trimmed = [value for value in values if float(lower) <= value <= float(upper)]
    return _mean_values(trimmed or values)


def _max_values(values: list[float]) -> float | str:
    if not values:
        return ""
    return max(values)


def _abs_gt_count_values(values: list[float], threshold: float = 1.0) -> int | str:
    if not values:
        return ""
    return sum(1 for value in values if abs(value) > threshold)


def _positive_rate_csv_column(path: Path, column: str) -> float | str:
    values = _csv_column_values(path, column)
    if not values:
        return ""
    return sum(1 for value in values if value > 0) / len(values)


def _return_distribution_metrics(path: Path, column: str, prefix: str) -> dict[str, float | int | str]:
    values = _csv_column_values(path, column)
    return {
        f"{prefix}_median": _median_values(values),
        f"{prefix}_trimmed_mean": _trimmed_mean_values(values),
        f"{prefix}_max": _max_values(values),
        f"{prefix}_abs_gt_1_count": _abs_gt_count_values(values),
    }


def _read_evaluation_metrics(run_dir: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        "alpha_orientation": "",
        "alpha_rankic": "",
        "alpha_oriented_rankic": "",
        "alpha_abs_rankic": "",
        "alpha_icir": "",
        "alpha_oriented_icir": "",
        "alpha_win_rate": "",
        "alpha_oriented_win_rate": "",
        "alpha_hit_rate": "",
        "alpha_top_excess_mean": "",
        "alpha_long_short_mean": "",
        "alpha_top_excess_positive_rate": "",
        "alpha_long_short_positive_rate": "",
    }
    alpha_path = run_dir / "metrics_alpha.csv"
    if alpha_path.exists():
        row = _read_single_row_csv(alpha_path)
        metrics["alpha_rankic"] = _safe_float(row.get("rankic"))
        metrics["alpha_abs_rankic"] = _safe_float(row.get("abs_rankic"))
        metrics["alpha_icir"] = _safe_float(row.get("icir"))
        metrics["alpha_win_rate"] = _safe_float(row.get("win_rate"))
    accuracy_path = run_dir / "prediction_accuracy.csv"
    if accuracy_path.exists():
        with accuracy_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("factor") == "alpha_0":
                    metrics["alpha_orientation"] = _safe_float(row.get("orientation"))
                    metrics["alpha_hit_rate"] = _safe_float(row.get("cross_sectional_hit_rate"))
                    break
    orientation = metrics["alpha_orientation"]
    if orientation != "":
        if metrics["alpha_rankic"] != "":
            metrics["alpha_oriented_rankic"] = float(orientation) * float(metrics["alpha_rankic"])
        if metrics["alpha_icir"] != "":
            metrics["alpha_oriented_icir"] = float(orientation) * float(metrics["alpha_icir"])
        if metrics["alpha_win_rate"] != "":
            raw_win_rate = float(metrics["alpha_win_rate"])
            metrics["alpha_oriented_win_rate"] = raw_win_rate if float(orientation) >= 0 else 1.0 - raw_win_rate
    long_short_path = run_dir / "alpha_long_short.csv"
    metrics["alpha_top_excess_mean"] = _mean_csv_column(long_short_path, "top_excess_return")
    metrics["alpha_long_short_mean"] = _mean_csv_column(long_short_path, "long_short_return")
    metrics["alpha_top_excess_positive_rate"] = _positive_rate_csv_column(long_short_path, "top_excess_return")
    metrics["alpha_long_short_positive_rate"] = _positive_rate_csv_column(long_short_path, "long_short_return")
    metrics.update(_return_distribution_metrics(long_short_path, "top_excess_return", "alpha_top_excess"))
    metrics.update(_return_distribution_metrics(long_short_path, "long_short_return", "alpha_long_short"))
    return metrics


def _sort_value(row: dict, key: str, default: float) -> float:
    value = _safe_float(row.get(key))
    if value == "":
        return default
    return float(value)


def _sort_rows_by_return(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            _sort_value(row, "alpha_top_excess_mean", float("-inf")),
            _sort_value(row, "alpha_long_short_mean", float("-inf")),
            _sort_value(row, "alpha_top_excess_positive_rate", float("-inf")),
            _sort_value(row, "alpha_oriented_rankic", _sort_value(row, "alpha_rankic", float("-inf"))),
            -_sort_value(row, "validation_loss", float("inf")),
        ),
        reverse=True,
    )


def _run_one_sweep(
    base_config: dict,
    run,
    device: str,
    output_root: Path,
    max_files: int | None,
    export_valid_dates: int | None,
    prepared_frame_cache: str | None,
) -> dict:
    config_path = output_root / "_configs" / f"{run.name}.yaml"
    _write_config(
        base_config,
        config_path,
        run,
        output_root,
        max_files,
        device,
        export_valid_dates=export_valid_dates,
        prepared_frame_cache=prepared_frame_cache,
    )
    log_path = output_root / "_logs" / f"{run.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_output = _run_command(
        [sys.executable, "scripts/train_abcm1.py", "--config", str(config_path)],
        ROOT,
    )
    run_dir = _parse_run_dir(train_output)
    eval_output = _run_command(
        [sys.executable, "scripts/evaluate_abcm1.py", "--factors-csv", str(run_dir / "factors.csv")],
        ROOT,
    )
    log_path.write_text(train_output + "\n" + eval_output)
    validation = _read_single_row_csv(run_dir / "validation_metrics.csv")
    beta_rankic = ""
    beta_abs_rankic = ""
    beta_metrics = run_dir / "metrics_beta.csv"
    if beta_metrics.exists():
        with beta_metrics.open(newline="") as fh:
            beta_rows = list(csv.DictReader(fh))
        if beta_rows:
            beta_rankic = sum(abs(float(row["rankic"])) for row in beta_rows) / len(beta_rows)
            beta_abs_rankic = sum(float(row["abs_rankic"]) for row in beta_rows) / len(beta_rows)
    return {
        "name": run.name,
        "run_dir": str(run_dir),
        "device": device,
        "hidden_dim": run.hidden_dim,
        "gru_layers": run.gru_layers,
        "learning_rate": run.learning_rate,
        "stock_limit": run.stock_limit,
        "max_steps": run.max_steps,
        "date_batch_size": run.date_batch_size,
        "lambda_mse": run.lambda_mse,
        "lambda_r2": run.lambda_r2,
        "lambda_alpha_corr": run.lambda_alpha_corr,
        "lambda_corr": run.lambda_corr,
        "lambda_to": run.lambda_to,
        "validation_fold": run.validation_fold,
        "dropout": "" if run.dropout is None else run.dropout,
        "weight_decay": "" if run.weight_decay is None else run.weight_decay,
        "label_transform": "" if run.label_transform is None else run.label_transform,
        "validation_loss": validation.get("loss", ""),
        "validation_r2_residual": validation.get("r2_residual", ""),
        "validation_turnover": validation.get("turnover", ""),
        "mean_abs_beta_rankic": beta_rankic,
        "mean_beta_abs_rankic": beta_abs_rankic,
        **_read_evaluation_metrics(run_dir),
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ABCM1 hyperparameter sweep.")
    parser.add_argument("--base-config", default="configs/abcm1_daily.yaml")
    parser.add_argument("--output-root", default=str(default_sweep_output_dir()))
    parser.add_argument("--hidden-dims", default="128,256")
    parser.add_argument("--gru-layers", default="2")
    parser.add_argument("--learning-rates", default="0.001,0.0005")
    parser.add_argument("--stock-limits", default="512")
    parser.add_argument("--max-steps", default="50")
    parser.add_argument("--date-batch-sizes", default="1")
    parser.add_argument("--lambda-mses", default="1.0")
    parser.add_argument("--lambda-r2s", default="1.0")
    parser.add_argument("--lambda-alpha-corrs", default="0.0")
    parser.add_argument("--lambda-corrs", default="0.01")
    parser.add_argument("--lambda-tos", default="0.01")
    parser.add_argument("--validation-folds", default="-1")
    parser.add_argument("--dropouts", default="")
    parser.add_argument("--weight-decays", default="")
    parser.add_argument("--label-transforms", default="")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--export-valid-dates", type=int, default=None)
    parser.add_argument("--prepared-frame-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2")
    parser.add_argument("--parallel", type=int, default=1)
    args = parser.parse_args()

    grid = {
        "hidden_dim": [int(x) for x in args.hidden_dims.split(",") if x],
        "gru_layers": [int(x) for x in args.gru_layers.split(",") if x],
        "learning_rate": [float(x) for x in args.learning_rates.split(",") if x],
        "stock_limit": [int(x) for x in args.stock_limits.split(",") if x],
        "max_steps": [int(x) for x in args.max_steps.split(",") if x],
        "date_batch_size": [int(x) for x in args.date_batch_sizes.split(",") if x],
        "lambda_mse": [float(x) for x in args.lambda_mses.split(",") if x],
        "lambda_r2": [float(x) for x in args.lambda_r2s.split(",") if x],
        "lambda_alpha_corr": [float(x) for x in args.lambda_alpha_corrs.split(",") if x],
        "lambda_corr": [float(x) for x in args.lambda_corrs.split(",") if x],
        "lambda_to": [float(x) for x in args.lambda_tos.split(",") if x],
        "validation_fold": [int(x) for x in args.validation_folds.split(",") if x],
    }
    if args.dropouts:
        grid["dropout"] = [float(x) for x in args.dropouts.split(",") if x]
    if args.weight_decays:
        grid["weight_decay"] = [float(x) for x in args.weight_decays.split(",") if x]
    if args.label_transforms:
        grid["label_transform"] = [x.strip() for x in args.label_transforms.split(",") if x.strip()]
    runs = build_sweep_runs(grid)
    base_config = load_config(args.base_config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in (args.devices or args.device).split(",") if item.strip()]
    assigned = assign_devices(runs, devices)
    parallel = max(1, min(int(args.parallel), len(assigned)))
    summary_rows = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        future_to_run = {
            executor.submit(
                _run_one_sweep,
                base_config,
                run,
                device,
                output_root,
                args.max_files,
                args.export_valid_dates,
                args.prepared_frame_cache,
            ): (run, device)
            for run, device in assigned
        }
        for future in as_completed(future_to_run):
            run, device = future_to_run[future]
            row = future.result()
            summary_rows.append(row)
            print(f"finished {run.name} on {device}: run_dir={row['run_dir']} log={row['log_path']}")
    summary_rows.sort(key=lambda row: row["name"])
    leaderboard = output_root / "sweep_leaderboard.csv"
    with leaderboard.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return_leaderboard = output_root / "sweep_leaderboard_by_return.csv"
    with return_leaderboard.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(_sort_rows_by_return(summary_rows))
    print(f"sweep_leaderboard={leaderboard}")
    print(f"sweep_leaderboard_by_return={return_leaderboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
