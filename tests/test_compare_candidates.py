import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_abcm_candidates import (
    aggregate_by_config,
    collect_candidate_rows,
    write_candidate_comparison,
)


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
    "lambda_corr",
    "lambda_to",
    "validation_fold",
    "dropout",
    "weight_decay",
    "validation_loss",
    "alpha_oriented_rankic",
    "alpha_hit_rate",
    "alpha_top_excess_mean",
    "alpha_top_excess_trimmed_mean",
    "alpha_top_excess_abs_gt_1_count",
    "alpha_long_short_mean",
    "alpha_long_short_trimmed_mean",
    "alpha_long_short_abs_gt_1_count",
    "alpha_top_excess_positive_rate",
    "alpha_long_short_positive_rate",
]


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_config(
    run_dir: Path,
    label_clip_abs: float | None,
    num_leaves: int | None = None,
    label_transform: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    clip_line = "" if label_clip_abs is None else f"  label_clip_abs: {label_clip_abs}\n"
    transform_line = "" if label_transform is None else f"  label_transform: \"{label_transform}\"\n"
    leaves_line = "" if num_leaves is None else f"  num_leaves: {num_leaves}\n"
    (run_dir / "config.yaml").write_text(
        "data:\n"
        "  root: \"data/testdata\"\n"
        "  max_files: 49\n"
        "  lookback: 60\n"
        "  y1_horizon: 11\n"
        "  y2_horizon: 21\n"
        "  entry_lag: 1\n"
        f"{clip_line}"
        f"{transform_line}"
        "  stock_limit: 1536\n"
        "\n"
        "model:\n"
        "  type: \"lightgbm\"\n"
        "  hidden_dim: 32\n"
        f"{leaves_line}"
        "\n"
        "train:\n"
        "  learning_rate: 0.0003\n"
        "  max_steps: 800\n"
    )


def _row(
    name: str,
    run_dir: str,
    fold: int,
    top_excess: float,
    long_short: float,
    top_excess_trimmed: float | None = None,
    long_short_trimmed: float | None = None,
    top_excess_extreme_count: int = 0,
    long_short_extreme_count: int = 0,
) -> dict[str, object]:
    return {
        "name": name,
        "run_dir": run_dir,
        "hidden_dim": 32,
        "learning_rate": 0.0003,
        "stock_limit": 1536,
        "max_steps": 800,
        "date_batch_size": 2,
        "lambda_mse": 5.0,
        "lambda_r2": 0.5,
        "lambda_corr": 0.01,
        "lambda_to": 0.01,
        "validation_fold": fold,
        "dropout": 0.3,
        "weight_decay": 0.001,
        "validation_loss": 0.4,
        "alpha_oriented_rankic": 0.05,
        "alpha_hit_rate": 0.52,
        "alpha_top_excess_mean": top_excess,
        "alpha_top_excess_trimmed_mean": top_excess if top_excess_trimmed is None else top_excess_trimmed,
        "alpha_top_excess_abs_gt_1_count": top_excess_extreme_count,
        "alpha_long_short_mean": long_short,
        "alpha_long_short_trimmed_mean": long_short if long_short_trimmed is None else long_short_trimmed,
        "alpha_long_short_abs_gt_1_count": long_short_extreme_count,
        "alpha_top_excess_positive_rate": 0.6,
        "alpha_long_short_positive_rate": 0.62,
    }


class CompareCandidatesTests(unittest.TestCase):
    def test_collect_candidate_rows_scans_summaries_and_deduplicates_run_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = _row("h32_g2_lr0.0003_n1536_s800_db2_vf2_mse5_r20.5_corr0.01_to0.01", "/runs/a", 2, 0.02, 0.03)
            better = _row("h64_g2_lr0.0002_n1536_s1500_db2_mse5_r20.5_corr0.01_to0.01", "/runs/b", -1, 0.04, 0.02)
            _write_summary(root / "sweep_a" / "partial_completed_return_summary.csv", [duplicate, better])
            _write_summary(root / "sweep_a" / "sweep_leaderboard_by_return.csv", [duplicate])

            rows = collect_candidate_rows([root])

        self.assertEqual([row["run_dir"] for row in rows], ["/runs/b", "/runs/a"])
        self.assertEqual(rows[0]["rough_top_excess_ann_11d"], 0.04 * 252.0 / 11.0)
        self.assertEqual(rows[0]["rough_long_short_ann_11d"], 0.02 * 252.0 / 11.0)

    def test_aggregate_by_config_summarizes_cross_fold_return_stability(self):
        rows = [
            _row("h32_g2_lr0.0003_n1536_s800_db2_vf0_mse5_r20.5_corr0.01_to0.01", "/runs/f0", 0, 0.01, 0.03),
            _row("h32_g2_lr0.0003_n1536_s800_db2_vf1_mse5_r20.5_corr0.01_to0.01", "/runs/f1", 1, 0.03, 0.05),
            _row("h32_g2_lr0.0003_n1536_s800_db2_vf2_mse5_r20.5_corr0.01_to0.01", "/runs/f2", 2, -0.02, 0.01),
        ]

        grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["n_runs"], 3)
        self.assertEqual(grouped[0]["validation_folds"], "0,1,2")
        self.assertAlmostEqual(grouped[0]["top_excess_mean"], (0.01 + 0.03 - 0.02) / 3.0)
        self.assertEqual(grouped[0]["top_excess_min"], -0.02)
        self.assertAlmostEqual(grouped[0]["long_short_mean"], 0.03)
        self.assertEqual(grouped[0]["positive_top_excess_rate"], 2 / 3)

    def test_aggregate_by_config_keeps_screening_fold_separate_from_cv_folds(self):
        rows = [
            _row("same_config_screen", "/runs/screen", -1, 0.004, 0.016),
            _row("same_config_f0", "/runs/f0", 0, 0.02, 0.03),
            _row("same_config_f1", "/runs/f1", 1, 0.03, 0.04),
            _row("same_config_f2", "/runs/f2", 2, 0.04, 0.05),
            _row("same_config_f3", "/runs/f3", 3, 0.05, 0.06),
        ]

        grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 2)
        by_folds = {row["validation_folds"]: row for row in grouped}
        self.assertEqual(by_folds["-1"]["n_runs"], 1)
        self.assertEqual(by_folds["0,1,2,3"]["n_runs"], 4)
        self.assertAlmostEqual(by_folds["0,1,2,3"]["top_excess_mean"], 0.035)

    def test_aggregate_by_config_summarizes_trimmed_returns_and_extreme_counts(self):
        rows = [
            _row(
                "h32_g2_lr0.0003_n1536_s800_db2_vf0_mse5_r20.5_corr0.01_to0.01",
                "/runs/f0",
                0,
                0.01,
                0.03,
                top_excess_trimmed=0.008,
                long_short_trimmed=0.025,
                top_excess_extreme_count=0,
                long_short_extreme_count=1,
            ),
            _row(
                "h32_g2_lr0.0003_n1536_s800_db2_vf1_mse5_r20.5_corr0.01_to0.01",
                "/runs/f1",
                1,
                0.03,
                0.05,
                top_excess_trimmed=0.02,
                long_short_trimmed=0.04,
                top_excess_extreme_count=2,
                long_short_extreme_count=3,
            ),
        ]

        grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 1)
        self.assertAlmostEqual(grouped[0]["top_excess_trimmed_mean"], 0.014)
        self.assertAlmostEqual(grouped[0]["long_short_trimmed_mean"], 0.0325)
        self.assertEqual(grouped[0]["top_excess_abs_gt_1_count"], 2)
        self.assertEqual(grouped[0]["long_short_abs_gt_1_count"], 4)

    def test_aggregate_by_config_keeps_raw_and_clipped_labels_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_run = root / "runs" / "raw" / "20260703_010000"
            clipped_run = root / "runs" / "clipped" / "20260703_020000"
            _write_run_config(raw_run, None)
            _write_run_config(clipped_run, 1.0)
            _write_summary(
                root / "sweep_a" / "partial_completed_return_summary.csv",
                [
                    _row("same_config_raw", str(raw_run), 0, 0.01, 0.02),
                    _row("same_config_clipped", str(clipped_run), 0, 0.03, 0.04),
                ],
            )

            rows = collect_candidate_rows([root])
            grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 2)
        self.assertEqual({row["label_clip_abs"] for row in grouped}, {"", 1.0})

    def test_aggregate_by_config_keeps_label_transforms_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_target_run = root / "runs" / "raw_target" / "20260703_010000"
            rank_target_run = root / "runs" / "rank_target" / "20260703_020000"
            _write_run_config(raw_target_run, 1.0, label_transform=None)
            _write_run_config(rank_target_run, 1.0, label_transform="rank_centered")
            _write_summary(
                root / "sweep_a" / "partial_completed_return_summary.csv",
                [
                    _row("same_config_raw_target", str(raw_target_run), 0, 0.01, 0.02),
                    _row("same_config_rank_target", str(rank_target_run), 0, 0.03, 0.04),
                ],
            )

            rows = collect_candidate_rows([root])
            grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 2)
        self.assertEqual({row["label_transform"] for row in grouped}, {"", "rank_centered"})

    def test_aggregate_by_config_keeps_lightgbm_model_params_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaves31_run = root / "runs" / "leaves31" / "20260703_010000"
            leaves63_run = root / "runs" / "leaves63" / "20260703_020000"
            _write_run_config(leaves31_run, 1.0, num_leaves=31)
            _write_run_config(leaves63_run, 1.0, num_leaves=63)
            _write_summary(
                root / "sweep_a" / "partial_completed_return_summary.csv",
                [
                    _row("lightgbm_same_lr_leaves31", str(leaves31_run), 0, 0.01, 0.02),
                    _row("lightgbm_same_lr_leaves63", str(leaves63_run), 0, 0.03, 0.04),
                ],
            )

            rows = collect_candidate_rows([root])
            grouped = aggregate_by_config(rows)

        self.assertEqual(len(grouped), 2)
        self.assertEqual({row["num_leaves"] for row in grouped}, {31, 63})

    def test_write_candidate_comparison_creates_leaderboard_and_grouped_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_summary(
                root / "sweep_a" / "partial_completed_return_summary.csv",
                [
                    _row("h32_g2_lr0.0003_n1536_s800_db2_vf0_mse5_r20.5_corr0.01_to0.01", "/runs/f0", 0, 0.01, 0.03),
                    _row("h32_g2_lr0.0003_n1536_s800_db2_vf1_mse5_r20.5_corr0.01_to0.01", "/runs/f1", 1, 0.03, 0.05),
                ],
            )
            output_dir = root / "comparison"

            leaderboard, grouped = write_candidate_comparison([root], output_dir)

            with (output_dir / "global_candidate_leaderboard.csv").open(newline="") as fh:
                written_leaderboard = list(csv.DictReader(fh))
            with (output_dir / "global_candidate_by_config.csv").open(newline="") as fh:
                written_grouped = list(csv.DictReader(fh))

        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(written_leaderboard[0]["run_dir"], "/runs/f1")
        self.assertEqual(written_grouped[0]["n_runs"], "2")

    def test_write_candidate_comparison_creates_filtered_clipped_leaderboards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stable_rows = [
                _row("stable_f0", "/runs/stable_f0", 0, 0.01, 0.02, top_excess_trimmed=0.009),
                _row("stable_f1", "/runs/stable_f1", 1, 0.02, 0.03, top_excess_trimmed=0.018),
                _row("stable_f2", "/runs/stable_f2", 2, 0.03, 0.04, top_excess_trimmed=0.027),
                _row("stable_f3", "/runs/stable_f3", 3, 0.04, 0.05, top_excess_trimmed=0.036),
            ]
            screening_row = _row("screening", "/runs/screening", -1, 0.05, 0.06)
            extreme_rows = [
                _row("extreme_f0", "/runs/extreme_f0", 0, 0.10, 0.02, top_excess_extreme_count=1),
                _row("extreme_f1", "/runs/extreme_f1", 1, 0.10, 0.02, top_excess_extreme_count=1),
                _row("extreme_f2", "/runs/extreme_f2", 2, 0.10, 0.02, top_excess_extreme_count=1),
                _row("extreme_f3", "/runs/extreme_f3", 3, 0.10, 0.02, top_excess_extreme_count=1),
            ]
            for row in extreme_rows:
                row["learning_rate"] = 0.0002
            for idx, row in enumerate([*stable_rows, screening_row, *extreme_rows]):
                run_dir = root / f"run_{idx}" / "20260703_010000"
                _write_run_config(run_dir, 1.0)
                row["run_dir"] = str(run_dir)
            _write_summary(root / "sweep_a" / "partial_completed_return_summary.csv", [*stable_rows, screening_row, *extreme_rows])
            output_dir = root / "comparison"

            write_candidate_comparison([root], output_dir)

            with (output_dir / "stable_cv_clip_leaderboard.csv").open(newline="") as fh:
                stable = list(csv.DictReader(fh))
            with (output_dir / "screening_clip_leaderboard.csv").open(newline="") as fh:
                screening = list(csv.DictReader(fh))

        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0]["validation_fold_scope"], "cv")
        self.assertEqual(stable[0]["validation_folds"], "0,1,2,3")
        self.assertEqual(stable[0]["top_excess_abs_gt_1_count"], "0")
        self.assertAlmostEqual(float(stable[0]["top_excess_trimmed_mean"]), 0.0225)
        self.assertEqual(len(screening), 1)
        self.assertEqual(screening[0]["validation_fold_scope"], "screening")
        self.assertEqual(screening[0]["validation_folds"], "-1")


if __name__ == "__main__":
    unittest.main()
