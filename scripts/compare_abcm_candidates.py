from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_FILENAMES = {
    "partial_completed_return_summary.csv",
    "sweep_leaderboard_by_return.csv",
    "capacity_best_by_hidden.csv",
}

DATA_FIELDS = [
    "data_root",
    "max_files",
    "lookback",
    "y1_horizon",
    "y2_horizon",
    "entry_lag",
    "label_clip_abs",
    "label_transform",
]

CONFIG_FIELDS = [
    "model_type",
    "sample_mode",
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
    "dropout",
    "weight_decay",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
]

INTEGER_FIELDS = {
    "hidden_dim",
    "stock_limit",
    "max_steps",
    "date_batch_size",
    "validation_fold",
    "max_files",
    "lookback",
    "y1_horizon",
    "y2_horizon",
    "entry_lag",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "n_runs",
    "alpha_top_excess_abs_gt_1_count",
    "alpha_long_short_abs_gt_1_count",
}

FLOAT_FIELDS = {
    "learning_rate",
    "lambda_mse",
    "lambda_r2",
    "lambda_alpha_corr",
    "lambda_corr",
    "lambda_to",
    "dropout",
    "weight_decay",
    "label_clip_abs",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
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
    "alpha_long_short_mean",
    "alpha_long_short_median",
    "alpha_long_short_trimmed_mean",
    "alpha_long_short_max",
    "alpha_top_excess_positive_rate",
    "alpha_long_short_positive_rate",
    "mean_abs_beta_rankic",
    "mean_beta_abs_rankic",
}

LEADERBOARD_FIELDS = [
    "name",
    "run_dir",
    "source_summary",
    *DATA_FIELDS,
    *CONFIG_FIELDS,
    "validation_fold",
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
    "rough_top_excess_ann_11d",
    "rough_long_short_ann_11d",
    "rough_top_excess_ann_5d",
    "rough_long_short_ann_5d",
]

GROUPED_FIELDS = [
    "config_key",
    *DATA_FIELDS,
    *CONFIG_FIELDS,
    "n_runs",
    "validation_folds",
    "validation_fold_scope",
    "top_excess_mean",
    "top_excess_std",
    "top_excess_min",
    "top_excess_max",
    "top_excess_trimmed_mean",
    "top_excess_trimmed_std",
    "top_excess_trimmed_min",
    "top_excess_trimmed_max",
    "top_excess_abs_gt_1_count",
    "long_short_mean",
    "long_short_std",
    "long_short_min",
    "long_short_max",
    "long_short_trimmed_mean",
    "long_short_trimmed_std",
    "long_short_trimmed_min",
    "long_short_trimmed_max",
    "long_short_abs_gt_1_count",
    "positive_top_excess_rate",
    "positive_long_short_rate",
    "oriented_rankic_mean",
    "hit_rate_mean",
    "validation_loss_mean",
    "best_run_name",
    "best_run_dir",
    "run_dirs",
]


def _safe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: object) -> int | str:
    number = _safe_float(value)
    if number is None:
        return ""
    return int(number)


def _coerce_value(key: str, value: object) -> object:
    if key in INTEGER_FIELDS:
        return _safe_int(value)
    if key in FLOAT_FIELDS or key.startswith("rough_"):
        number = _safe_float(value)
        return "" if number is None else number
    return "" if value is None else value


def _coerce_row(row: dict[str, object], source_summary: Path) -> dict[str, object]:
    out = {key: _coerce_value(key, value) for key, value in row.items()}
    out["source_summary"] = str(source_summary)
    _add_config_metadata(out)
    top_excess = _safe_float(out.get("alpha_top_excess_mean"))
    long_short = _safe_float(out.get("alpha_long_short_mean"))
    out["rough_top_excess_ann_11d"] = "" if top_excess is None else top_excess * 252.0 / 11.0
    out["rough_long_short_ann_11d"] = "" if long_short is None else long_short * 252.0 / 11.0
    out["rough_top_excess_ann_5d"] = "" if top_excess is None else top_excess * 252.0 / 5.0
    out["rough_long_short_ann_5d"] = "" if long_short is None else long_short * 252.0 / 5.0
    return out


def _candidate_config_paths(row: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    run_dir = row.get("run_dir")
    if run_dir not in {"", None}:
        paths.append(Path(str(run_dir)) / "config.yaml")

    source_summary = row.get("source_summary")
    name = row.get("name")
    if source_summary not in {"", None} and name not in {"", None}:
        paths.append(Path(str(source_summary)).parent / "_configs" / f"{name}.yaml")

    return paths


def _load_config(path: Path) -> dict[str, object]:
    from abcm.pipeline import load_config

    loaded = load_config(path)
    return loaded if isinstance(loaded, dict) else {}


def _add_config_metadata(row: dict[str, object]) -> None:
    config_path = next((path for path in _candidate_config_paths(row) if path.exists()), None)
    if config_path is None:
        return

    config = _load_config(config_path)
    data = config.get("data", {})
    model = config.get("model", {})
    if not isinstance(data, dict):
        data = {}
    if not isinstance(model, dict):
        model = {}

    metadata = {
        "data_root": data.get("root", ""),
        "max_files": data.get("max_files", ""),
        "lookback": data.get("lookback", ""),
        "y1_horizon": data.get("y1_horizon", ""),
        "y2_horizon": data.get("y2_horizon", ""),
        "entry_lag": data.get("entry_lag", ""),
        "label_clip_abs": data.get("label_clip_abs", ""),
        "label_transform": data.get("label_transform", ""),
        "model_type": model.get("type", ""),
        "sample_mode": model.get("sample_mode", ""),
        "num_leaves": model.get("num_leaves", ""),
        "max_depth": model.get("max_depth", ""),
        "min_child_samples": model.get("min_child_samples", ""),
        "subsample": model.get("subsample", ""),
        "colsample_bytree": model.get("colsample_bytree", ""),
        "reg_alpha": model.get("reg_alpha", ""),
        "reg_lambda": model.get("reg_lambda", ""),
    }
    for field, value in metadata.items():
        row[field] = _coerce_value(field, value)


def _summary_paths(search_roots: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for root_item in search_roots:
        root = Path(root_item)
        if root.is_file() and root.name in SUMMARY_FILENAMES:
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*.csv") if path.name in SUMMARY_FILENAMES)
    return sorted(set(paths))


def _nonempty_count(row: dict[str, object]) -> int:
    return sum(1 for value in row.values() if value not in {"", None})


def _sort_value(row: dict[str, object], key: str, default: float) -> float:
    number = _safe_float(row.get(key))
    return default if number is None else number


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            _sort_value(row, "alpha_top_excess_mean", float("-inf")),
            _sort_value(row, "alpha_long_short_mean", float("-inf")),
            _sort_value(row, "alpha_top_excess_positive_rate", float("-inf")),
            _sort_value(row, "alpha_oriented_rankic", float("-inf")),
            _sort_value(row, "alpha_hit_rate", float("-inf")),
            -_sort_value(row, "validation_loss", float("inf")),
        ),
        reverse=True,
    )


def collect_candidate_rows(search_roots: Iterable[str | Path]) -> list[dict[str, object]]:
    deduped: dict[str, dict[str, object]] = {}
    for summary_path in _summary_paths(search_roots):
        with summary_path.open(newline="") as fh:
            for raw_row in csv.DictReader(fh):
                row = _coerce_row(dict(raw_row), summary_path)
                key = str(row.get("run_dir") or f"{summary_path}:{row.get('name', '')}")
                previous = deduped.get(key)
                if previous is None or _nonempty_count(row) > _nonempty_count(previous):
                    deduped[key] = row
    return _sort_rows(list(deduped.values()))


def _config_key(row: dict[str, object]) -> tuple[object, ...]:
    return (*tuple(row.get(field, "") for field in (*DATA_FIELDS, *CONFIG_FIELDS)), _validation_fold_scope(row))


def _validation_fold_scope(row: dict[str, object]) -> str:
    fold = row.get("validation_fold", "")
    if fold in {"", None}:
        return "unknown"
    try:
        fold_value = int(float(fold))
    except (TypeError, ValueError):
        return "unknown"
    return "screening" if fold_value < 0 else "cv"


def _mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return sum(values) / len(values)


def _std(values: list[float]) -> float | str:
    if not values:
        return ""
    mean = float(_mean(values))
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _min(values: list[float]) -> float | str:
    return "" if not values else min(values)


def _max(values: list[float]) -> float | str:
    return "" if not values else max(values)


def _sum(values: list[float]) -> float | int | str:
    if not values:
        return ""
    result = sum(values)
    return int(result) if float(result).is_integer() else result


def _metric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = _safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _fold_values(rows: list[dict[str, object]]) -> str:
    folds = []
    for row in rows:
        fold = row.get("validation_fold", "")
        if fold not in {"", None}:
            folds.append(str(fold))
    return ",".join(sorted(set(folds), key=lambda item: int(float(item))))


def _config_key_text(row: dict[str, object]) -> str:
    parts = [f"{field}={row.get(field, '')}" for field in (*DATA_FIELDS, *CONFIG_FIELDS)]
    parts.append(f"validation_fold_scope={_validation_fold_scope(row)}")
    return "|".join(parts)


def aggregate_by_config(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_config_key(row)].append(row)

    grouped_rows: list[dict[str, object]] = []
    for group_rows in groups.values():
        ordered = _sort_rows(group_rows)
        best = ordered[0]
        top_values = _metric_values(group_rows, "alpha_top_excess_mean")
        top_trimmed_values = _metric_values(group_rows, "alpha_top_excess_trimmed_mean")
        top_extreme_values = _metric_values(group_rows, "alpha_top_excess_abs_gt_1_count")
        long_short_values = _metric_values(group_rows, "alpha_long_short_mean")
        long_short_trimmed_values = _metric_values(group_rows, "alpha_long_short_trimmed_mean")
        long_short_extreme_values = _metric_values(group_rows, "alpha_long_short_abs_gt_1_count")
        rankic_values = _metric_values(group_rows, "alpha_oriented_rankic")
        hit_values = _metric_values(group_rows, "alpha_hit_rate")
        loss_values = _metric_values(group_rows, "validation_loss")
        grouped_rows.append(
            {
                "config_key": _config_key_text(best),
                **{field: best.get(field, "") for field in DATA_FIELDS},
                **{field: best.get(field, "") for field in CONFIG_FIELDS},
                "n_runs": len(group_rows),
                "validation_folds": _fold_values(group_rows),
                "validation_fold_scope": _validation_fold_scope(best),
                "top_excess_mean": _mean(top_values),
                "top_excess_std": _std(top_values),
                "top_excess_min": _min(top_values),
                "top_excess_max": _max(top_values),
                "top_excess_trimmed_mean": _mean(top_trimmed_values),
                "top_excess_trimmed_std": _std(top_trimmed_values),
                "top_excess_trimmed_min": _min(top_trimmed_values),
                "top_excess_trimmed_max": _max(top_trimmed_values),
                "top_excess_abs_gt_1_count": _sum(top_extreme_values),
                "long_short_mean": _mean(long_short_values),
                "long_short_std": _std(long_short_values),
                "long_short_min": _min(long_short_values),
                "long_short_max": _max(long_short_values),
                "long_short_trimmed_mean": _mean(long_short_trimmed_values),
                "long_short_trimmed_std": _std(long_short_trimmed_values),
                "long_short_trimmed_min": _min(long_short_trimmed_values),
                "long_short_trimmed_max": _max(long_short_trimmed_values),
                "long_short_abs_gt_1_count": _sum(long_short_extreme_values),
                "positive_top_excess_rate": "" if not top_values else sum(value > 0 for value in top_values) / len(top_values),
                "positive_long_short_rate": ""
                if not long_short_values
                else sum(value > 0 for value in long_short_values) / len(long_short_values),
                "oriented_rankic_mean": _mean(rankic_values),
                "hit_rate_mean": _mean(hit_values),
                "validation_loss_mean": _mean(loss_values),
                "best_run_name": best.get("name", ""),
                "best_run_dir": best.get("run_dir", ""),
                "run_dirs": ";".join(str(row.get("run_dir", "")) for row in ordered if row.get("run_dir", "")),
            }
        )
    return sorted(
        grouped_rows,
        key=lambda row: (
            _sort_value(row, "top_excess_trimmed_mean", float("-inf")),
            _sort_value(row, "top_excess_mean", float("-inf")),
            _sort_value(row, "top_excess_min", float("-inf")),
            _sort_value(row, "long_short_trimmed_mean", float("-inf")),
            _sort_value(row, "long_short_mean", float("-inf")),
            _sort_value(row, "oriented_rankic_mean", float("-inf")),
        ),
        reverse=True,
    )


def _fieldnames(rows: list[dict[str, object]], preferred: list[str]) -> list[str]:
    seen = set(preferred)
    extras = sorted({key for row in rows for key in row.keys()} - seen)
    return preferred + extras


def _write_rows(path: Path, rows: list[dict[str, object]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows, preferred_fields)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_clipped_label(row: dict[str, object]) -> bool:
    return _safe_float(row.get("label_clip_abs")) == 1.0


def _metric_is_zero_or_missing(row: dict[str, object], key: str) -> bool:
    value = _safe_float(row.get(key))
    return value in {None, 0.0}


def _stable_cv_clip_rows(grouped: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in grouped
        if _is_clipped_label(row)
        and row.get("validation_fold_scope") == "cv"
        and _safe_int(row.get("n_runs")) >= 4
        and _metric_is_zero_or_missing(row, "top_excess_abs_gt_1_count")
        and _metric_is_zero_or_missing(row, "long_short_abs_gt_1_count")
    ]


def _screening_clip_rows(grouped: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in grouped
        if _is_clipped_label(row)
        and row.get("validation_fold_scope") == "screening"
        and _metric_is_zero_or_missing(row, "top_excess_abs_gt_1_count")
        and _metric_is_zero_or_missing(row, "long_short_abs_gt_1_count")
    ]


def write_candidate_comparison(
    search_roots: Iterable[str | Path],
    output_dir: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = collect_candidate_rows(search_roots)
    grouped = aggregate_by_config(rows)
    root = Path(output_dir)
    _write_rows(root / "global_candidate_leaderboard.csv", rows, LEADERBOARD_FIELDS)
    _write_rows(root / "global_candidate_by_config.csv", grouped, GROUPED_FIELDS)
    _write_rows(root / "stable_cv_clip_leaderboard.csv", _stable_cv_clip_rows(grouped), GROUPED_FIELDS)
    _write_rows(root / "screening_clip_leaderboard.csv", _screening_clip_rows(grouped), GROUPED_FIELDS)
    return rows, grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a global ABCM candidate comparison from completed sweep summaries.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Sweep output root or summary CSV. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/global_candidate_comparison",
    )
    args = parser.parse_args()

    search_roots = args.search_root or ["outputs"]
    rows, grouped = write_candidate_comparison(search_roots, args.output_dir)
    print(f"candidate_rows={len(rows)}")
    print(f"grouped_configs={len(grouped)}")
    print(f"leaderboard={Path(args.output_dir) / 'global_candidate_leaderboard.csv'}")
    print(f"by_config={Path(args.output_dir) / 'global_candidate_by_config.csv'}")
    for row in rows[:10]:
        print(
            f"{row.get('name', '')} top_excess={row.get('alpha_top_excess_mean', '')} "
            f"long_short={row.get('alpha_long_short_mean', '')} "
            f"rankic={row.get('alpha_oriented_rankic', row.get('alpha_rankic', ''))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
