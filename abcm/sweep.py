from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepRun:
    name: str
    hidden_dim: int
    gru_layers: int
    learning_rate: float
    stock_limit: int
    max_steps: int
    date_batch_size: int = 1
    lambda_mse: float = 1.0
    lambda_r2: float = 1.0
    lambda_alpha_corr: float = 0.0
    lambda_corr: float = 0.01
    lambda_to: float = 0.01
    validation_fold: int = -1
    dropout: float | None = None
    weight_decay: float | None = None
    label_transform: str | None = None


def default_sweep_output_dir() -> Path:
    return Path("outputs")


def build_sweep_runs(grid: dict[str, list[Any]]) -> list[SweepRun]:
    required = ["hidden_dim", "gru_layers", "learning_rate", "stock_limit", "max_steps"]
    missing = [key for key in required if key not in grid]
    if missing:
        raise KeyError(f"Missing sweep grid keys: {missing}")
    date_batch_sizes = grid.get("date_batch_size", [1])
    lambda_mses = grid.get("lambda_mse", [1.0])
    lambda_r2s = grid.get("lambda_r2", [1.0])
    lambda_alpha_corrs = grid.get("lambda_alpha_corr", [0.0])
    lambda_corrs = grid.get("lambda_corr", [0.01])
    lambda_tos = grid.get("lambda_to", [0.01])
    validation_folds = grid.get("validation_fold", [-1])
    dropouts = grid.get("dropout", [None])
    weight_decays = grid.get("weight_decay", [None])
    label_transforms = grid.get("label_transform", [None])
    runs = []
    for (
        hidden_dim,
        gru_layers,
        learning_rate,
        stock_limit,
        max_steps,
        date_batch_size,
        lambda_mse,
        lambda_r2,
        lambda_alpha_corr,
        lambda_corr,
        lambda_to,
        validation_fold,
        dropout,
        weight_decay,
        label_transform,
    ) in itertools.product(
        grid["hidden_dim"],
        grid["gru_layers"],
        grid["learning_rate"],
        grid["stock_limit"],
        grid["max_steps"],
        date_batch_sizes,
        lambda_mses,
        lambda_r2s,
        lambda_alpha_corrs,
        lambda_corrs,
        lambda_tos,
        validation_folds,
        dropouts,
        weight_decays,
        label_transforms,
    ):
        name = (
            f"h{hidden_dim}_g{gru_layers}_lr{learning_rate}"
            f"_n{stock_limit}_s{max_steps}_db{date_batch_size}"
        )
        if int(validation_fold) >= 0:
            name += f"_vf{int(validation_fold)}"
        if (
            float(lambda_mse),
            float(lambda_r2),
            float(lambda_alpha_corr),
            float(lambda_corr),
            float(lambda_to),
        ) != (1.0, 1.0, 0.0, 0.01, 0.01):
            name += (
                f"_mse{float(lambda_mse):g}"
                f"_r2{float(lambda_r2):g}"
                f"_acorr{float(lambda_alpha_corr):g}"
                f"_corr{float(lambda_corr):g}"
                f"_to{float(lambda_to):g}"
            )
        if dropout is not None:
            name += f"_do{float(dropout):g}"
        if weight_decay is not None:
            name += f"_wd{float(weight_decay):g}"
        if label_transform not in {None, ""}:
            name += f"_label{str(label_transform)}"
        runs.append(
            SweepRun(
                name=name,
                hidden_dim=int(hidden_dim),
                gru_layers=int(gru_layers),
                learning_rate=float(learning_rate),
                stock_limit=int(stock_limit),
                max_steps=int(max_steps),
                date_batch_size=int(date_batch_size),
                lambda_mse=float(lambda_mse),
                lambda_r2=float(lambda_r2),
                lambda_alpha_corr=float(lambda_alpha_corr),
                lambda_corr=float(lambda_corr),
                lambda_to=float(lambda_to),
                validation_fold=int(validation_fold),
                dropout=None if dropout is None else float(dropout),
                weight_decay=None if weight_decay is None else float(weight_decay),
                label_transform=None if label_transform in {None, ""} else str(label_transform),
            )
        )
    return runs


def assign_devices(runs: list[SweepRun], devices: list[str]) -> list[tuple[SweepRun, str]]:
    if not devices:
        raise ValueError("At least one device is required")
    return [(run, devices[idx % len(devices)]) for idx, run in enumerate(runs)]
