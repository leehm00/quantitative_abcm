from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.pipeline import load_config


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else math.nan


def finite_std(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=0)) if array.size else math.nan


def trimmed_mean(values, lower_q: float = 0.05, upper_q: float = 0.95) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan
    lower, upper = np.quantile(array, [lower_q, upper_q])
    trimmed = array[(array >= lower) & (array <= upper)]
    return float((trimmed if trimmed.size else array).mean())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, root = value.partition("=")
    if not separator or not name or not root:
        raise argparse.ArgumentTypeError("candidate must use NAME=/path/to/sweep_root")
    return name, Path(root)


def discover_run_dirs(root: Path) -> list[Path]:
    run_dirs = []
    for summary_path in root.rglob("run_summary.json"):
        run_dir = summary_path.parent
        required = [
            "config.yaml",
            "split_summary.txt",
            "metrics_alpha.csv",
            "metrics_beta.csv",
            "prediction_accuracy.csv",
            "alpha_long_short.csv",
            "rolling_rsquare.csv",
            "checkpoints/best.pt",
        ]
        if all((run_dir / item).exists() for item in required):
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def summarize_candidate(
    candidate: str,
    root: Path,
    annualization_days: int,
    holding_period: int,
) -> tuple[dict, list[dict], list[dict]]:
    fold_rows: list[dict] = []
    manifest_rows: list[dict] = []
    beta_frames: list[pd.DataFrame] = []
    model_hashes: list[str] = []
    config: dict | None = None

    for run_dir in discover_run_dirs(root):
        config = load_config(run_dir / "config.yaml")
        train_cfg = config.get("train", {})
        model_cfg = config.get("model", {})
        split = parse_key_values(run_dir / "split_summary.txt")
        run_summary = json.loads((run_dir / "run_summary.json").read_text())
        alpha = pd.read_csv(run_dir / "metrics_alpha.csv").iloc[0]
        accuracy = pd.read_csv(run_dir / "prediction_accuracy.csv")
        accuracy = accuracy.loc[accuracy["factor"] == "alpha_0"].iloc[0]
        returns = pd.read_csv(run_dir / "alpha_long_short.csv")
        beta = pd.read_csv(run_dir / "metrics_beta.csv")
        rolling = pd.read_csv(run_dir / "rolling_rsquare.csv")
        beta_frames.append(beta)

        orientation = float(accuracy["orientation"])
        raw_win_rate = float(alpha["win_rate"])
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        state_hash = model_state_sha256(checkpoint_path)
        model_hashes.append(state_hash)

        fold_rows.append(
            {
                "candidate": candidate,
                "validation_fold": int(train_cfg.get("validation_fold", -1)),
                "valid_start": split.get("valid_start", ""),
                "valid_end": split.get("valid_end", ""),
                "run_dir": str(run_dir),
                "completed_steps": int(run_summary["completed_steps"]),
                "completed_epoch_equivalents": float(run_summary["completed_epoch_equivalents"]),
                "shuffle_train_dates": bool(run_summary["shuffle_train_dates"]),
                "rankic": orientation * float(alpha["rankic"]),
                "icir": orientation * float(alpha["icir"]),
                "rankic_win_rate": raw_win_rate if orientation >= 0 else 1.0 - raw_win_rate,
                "hit_rate": float(accuracy["cross_sectional_hit_rate"]),
                "top_excess_mean": finite_mean(returns["top_excess_return"]),
                "top_excess_trimmed_mean": trimmed_mean(returns["top_excess_return"]),
                "long_short_mean": finite_mean(returns["long_short_return"]),
                "rolling_rsquare_mean": finite_mean(rolling["rolling_rsquare"]),
            }
        )
        manifest_rows.append(
            {
                "candidate": candidate,
                "validation_fold": int(train_cfg.get("validation_fold", -1)),
                "run_dir": str(run_dir),
                "config_path": str(run_dir / "config.yaml"),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "model_state_sha256": state_hash,
                "completed_epoch_equivalents": float(run_summary["completed_epoch_equivalents"]),
            }
        )

    if config is None or not fold_rows:
        raise RuntimeError(f"No completed runs found under {root}")

    fold_rows.sort(key=lambda row: int(row["validation_fold"]))
    manifest_rows.sort(key=lambda row: int(row["validation_fold"]))
    beta_all = pd.concat(beta_frames, ignore_index=True)
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    annualization = float(annualization_days) / float(holding_period)
    latest_fold = max(fold_rows, key=lambda row: int(row["validation_fold"]))

    summary = {
        "candidate": candidate,
        "n_folds": len(fold_rows),
        "validation_folds": ",".join(str(row["validation_fold"]) for row in fold_rows),
        "hidden_dim": int(model_cfg.get("hidden_dim", 0)),
        "learning_rate": float(train_cfg.get("learning_rate", 0.0)),
        "max_steps": int(train_cfg.get("max_steps", 0)),
        "date_batch_size": int(train_cfg.get("date_batch_size", 0)),
        "lambda_mse": float(train_cfg.get("lambda_mse", 1.0)),
        "lambda_r2": float(train_cfg.get("lambda_r2", 1.0)),
        "lambda_alpha_corr": float(train_cfg.get("lambda_alpha_corr", 0.0)),
        "min_completed_epoch_equivalents": min(float(row["completed_epoch_equivalents"]) for row in fold_rows),
        "all_shuffle_train_dates": all(bool(row["shuffle_train_dates"]) for row in fold_rows),
        "unique_model_states": len(set(model_hashes)),
        "rankic_mean": finite_mean(row["rankic"] for row in fold_rows),
        "rankic_std": finite_std(row["rankic"] for row in fold_rows),
        "icir_mean": finite_mean(row["icir"] for row in fold_rows),
        "rankic_win_rate_mean": finite_mean(row["rankic_win_rate"] for row in fold_rows),
        "hit_rate_mean": finite_mean(row["hit_rate"] for row in fold_rows),
        "top_excess_mean": finite_mean(row["top_excess_mean"] for row in fold_rows),
        "top_excess_std": finite_std(row["top_excess_mean"] for row in fold_rows),
        "top_excess_trimmed_mean": finite_mean(row["top_excess_trimmed_mean"] for row in fold_rows),
        "top_excess_trimmed_std": finite_std(row["top_excess_trimmed_mean"] for row in fold_rows),
        "long_short_mean": finite_mean(row["long_short_mean"] for row in fold_rows),
        "long_short_std": finite_std(row["long_short_mean"] for row in fold_rows),
        "top_annualized_approx": annualization * finite_mean(row["top_excess_mean"] for row in fold_rows),
        "top_annualized_std": annualization * finite_std(row["top_excess_mean"] for row in fold_rows),
        "top_trimmed_annualized_approx": annualization
        * finite_mean(row["top_excess_trimmed_mean"] for row in fold_rows),
        "long_short_annualized_approx": annualization * finite_mean(row["long_short_mean"] for row in fold_rows),
        "latest_fold": int(latest_fold["validation_fold"]),
        "latest_fold_rankic": float(latest_fold["rankic"]),
        "latest_fold_top_annualized_approx": annualization * float(latest_fold["top_excess_mean"]),
        "beta_rankic_mean": finite_mean(beta_all["rankic"]),
        "beta_abs_signed_rankic_mean": finite_mean(beta_all["rankic"].abs()),
        "beta_abs_rankic_mean": finite_mean(beta_all["abs_rankic"]),
        "beta_abs_icir_mean": finite_mean(beta_all["icir"].abs()),
        "beta_win_rate_mean": finite_mean(beta_all["win_rate"]),
        "beta_autocorrelation_mean": finite_mean(beta_all["autocorrelation"]),
        "beta_abs_signed_rankic_le_3pct_rate": finite_mean(beta_all["rankic"].abs() <= 0.03),
        "beta_abs_icir_le_0_2_rate": finite_mean(beta_all["icir"].abs() <= 0.2),
        "beta_win_rate_le_0_6_rate": finite_mean(beta_all["win_rate"] <= 0.6),
        "beta_abs_rankic_ge_0_08_rate": finite_mean(beta_all["abs_rankic"] >= 0.08),
        "beta_autocorrelation_ge_0_7_rate": finite_mean(beta_all["autocorrelation"] >= 0.7),
        "rolling_rsquare_mean": finite_mean(row["rolling_rsquare_mean"] for row in fold_rows),
        "sweep_root": str(root),
    }
    return summary, fold_rows, manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize selected full-epoch ABCM candidates.")
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annualization-days", type=int, default=252)
    parser.add_argument("--holding-period", type=int, default=11)
    args = parser.parse_args()

    summaries: list[dict] = []
    folds: list[dict] = []
    manifests: list[dict] = []
    for candidate, root in args.candidate:
        summary, fold_rows, manifest_rows = summarize_candidate(
            candidate,
            root,
            annualization_days=args.annualization_days,
            holding_period=args.holding_period,
        )
        summaries.append(summary)
        folds.extend(fold_rows)
        manifests.extend(manifest_rows)

    output_dir = Path(args.output_dir)
    write_rows(output_dir / "selected_candidate_summary.csv", summaries)
    write_rows(output_dir / "selected_fold_metrics.csv", folds)
    write_rows(output_dir / "selected_model_manifest.csv", manifests)
    print(f"candidates={len(summaries)} folds={len(folds)}")
    print(f"candidate_summary={output_dir / 'selected_candidate_summary.csv'}")
    print(f"fold_metrics={output_dir / 'selected_fold_metrics.csv'}")
    print(f"model_manifest={output_dir / 'selected_model_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
