import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_abcm_sweep import summarize_completed_runs, write_summary


def _write_run(
    root: Path,
    name: str,
    top_excess: float,
    long_short: float,
    rankic: float,
    long_short_rows: list[tuple[float, float]] | None = None,
) -> Path:
    run_dir = root / name / "20260703_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "data:\n"
        "  stock_limit: 1536\n"
        "train:\n"
        "  learning_rate: 0.0002\n"
        "  max_steps: 800\n"
        "  date_batch_size: 2\n"
        "  lambda_mse: 5.0\n"
        "  lambda_r2: 0.5\n"
        "  lambda_corr: 0.01\n"
        "  lambda_to: 0.01\n"
        "  validation_fold: 3\n"
        "  weight_decay: 0.001\n"
        "model:\n"
        "  dropout: 0.3\n"
    )
    (run_dir / "validation_metrics.csv").write_text(
        "loss,mse,r2_residual,corr,turnover,n_stocks,n_dates\n"
        "0.4,0.01,0.78,0.1,0.01,100,2\n"
    )
    (run_dir / "metrics_alpha.csv").write_text(
        "factor,rankic,abs_rankic,icir,win_rate,autocorrelation\n"
        f"alpha_0,{rankic},0.1,0.6,0.7,0.9\n"
    )
    (run_dir / "prediction_accuracy.csv").write_text(
        "factor,orientation,rankic,cross_sectional_hit_rate,mean_daily_hit_rate,n_dates,n_observations\n"
        f"alpha_0,1.0,{rankic},0.53,0.53,2,20\n"
    )
    rows = long_short_rows or [(top_excess, long_short)]
    lines = [
        "TRADE_DT,orientation,top_group,bottom_group,top_return,bottom_return,universe_return,top_excess_return,bottom_excess_return,long_short_return\n"
    ]
    for idx, (row_top_excess, row_long_short) in enumerate(rows, start=1):
        lines.append(
            f"202001{idx:02d},1.0,19,0,0.03,0.01,0.02,{row_top_excess},-0.01,{row_long_short}\n"
        )
    (run_dir / "alpha_long_short.csv").write_text("".join(lines))
    return run_dir


class SummarizeSweepScriptTests(unittest.TestCase):
    def test_summarize_completed_runs_sorts_by_return_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "h384_g2_lr0.0002_n1536_s2500_db2", 0.001, 0.002, 0.04)
            _write_run(root, "h384_g2_lr0.00015_n1536_s2500_db2", 0.004, 0.003, 0.05)
            incomplete = root / "h768_g2_lr0.00015_n1536_s2500_db2" / "20260703_000001"
            incomplete.mkdir(parents=True)
            (incomplete / "validation_metrics.csv").write_text("loss\n0.5\n")

            rows = summarize_completed_runs(root)

        self.assertEqual([row["name"] for row in rows], ["h384_g2_lr0.00015_n1536_s2500_db2", "h384_g2_lr0.0002_n1536_s2500_db2"])
        self.assertEqual(rows[0]["alpha_top_excess_mean"], 0.004)
        self.assertEqual(rows[0]["validation_fold"], 3)
        self.assertEqual(rows[0]["stock_limit"], 1536)
        self.assertEqual(rows[0]["max_steps"], 800)
        self.assertEqual(rows[0]["date_batch_size"], 2)
        self.assertEqual(rows[0]["learning_rate"], 0.0002)
        self.assertEqual(rows[0]["lambda_mse"], 5.0)
        self.assertEqual(rows[0]["dropout"], 0.3)
        self.assertEqual(rows[0]["weight_decay"], 0.001)

    def test_summarize_completed_runs_adds_robust_return_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(
                root,
                "h32_g2_lr0.0005_n1536_s800_db2",
                0.004,
                0.003,
                0.05,
                long_short_rows=[(0.01, 0.02)] * 10 + [(0.02, 0.02)] * 9 + [(10.0, 12.0)],
            )

            rows = summarize_completed_runs(root)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["alpha_top_excess_mean"], 0.514)
        self.assertAlmostEqual(rows[0]["alpha_top_excess_median"], 0.015)
        self.assertAlmostEqual(rows[0]["alpha_top_excess_trimmed_mean"], 0.014736842105263158)
        self.assertEqual(rows[0]["alpha_top_excess_max"], 10.0)
        self.assertEqual(rows[0]["alpha_top_excess_abs_gt_1_count"], 1)
        self.assertAlmostEqual(rows[0]["alpha_long_short_median"], 0.02)
        self.assertAlmostEqual(rows[0]["alpha_long_short_trimmed_mean"], 0.02)
        self.assertEqual(rows[0]["alpha_long_short_abs_gt_1_count"], 1)

    def test_summarize_completed_runs_includes_non_h_model_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "lightgbm_lr0.05_leaves31", 0.005, 0.004, 0.06)

            rows = summarize_completed_runs(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "lightgbm_lr0.05_leaves31")
        self.assertEqual(rows[0]["hidden_dim"], "")

    def test_write_summary_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "h384_g2_lr0.00015_n1536_s2500_db2", 0.004, 0.003, 0.05)
            output = root / "summary.csv"

            rows = write_summary(root, output)

            with output.open(newline="") as fh:
                written = list(csv.DictReader(fh))

        self.assertEqual(len(rows), 1)
        self.assertEqual(written[0]["name"], "h384_g2_lr0.00015_n1536_s2500_db2")
        self.assertEqual(written[0]["validation_fold"], "3")
        self.assertEqual(written[0]["max_steps"], "800")
        self.assertEqual(written[0]["lambda_mse"], "5.0")
        self.assertEqual(written[0]["dropout"], "0.3")
        self.assertEqual(written[0]["weight_decay"], "0.001")


if __name__ == "__main__":
    unittest.main()
