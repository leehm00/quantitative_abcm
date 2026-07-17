import unittest
import tempfile
from pathlib import Path

from abcm.sweep import assign_devices, build_sweep_runs, default_sweep_output_dir
from scripts.sweep_abcm1 import _read_evaluation_metrics, _sort_rows_by_return, _write_config


class SweepTests(unittest.TestCase):
    def test_build_sweep_runs_expands_grid_with_names(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [128, 256],
                "gru_layers": [2],
                "learning_rate": [0.001, 0.0005],
                "stock_limit": [512],
                "max_steps": [20],
            }
        )

        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0].hidden_dim, 128)
        self.assertIn("h128", runs[0].name)
        self.assertIn("lr0.001", runs[0].name)

    def test_default_sweep_output_dir_uses_repository_output_directory(self):
        self.assertEqual(str(default_sweep_output_dir()), "outputs")

    def test_assign_devices_round_robins_runs(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [128, 256, 384],
                "gru_layers": [2],
                "learning_rate": [0.001],
                "stock_limit": [512],
                "max_steps": [20],
            }
        )

        assigned = assign_devices(runs, ["cuda:0", "cuda:1"])

        self.assertEqual([device for _, device in assigned], ["cuda:0", "cuda:1", "cuda:0"])

    def test_build_sweep_runs_includes_date_batch_size(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
                "date_batch_size": [2],
            }
        )

        self.assertEqual(runs[0].date_batch_size, 2)
        self.assertIn("db2", runs[0].name)

    def test_build_sweep_runs_includes_loss_weights(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
                "lambda_mse": [1.0, 5.0],
                "lambda_r2": [1.0, 0.5],
                "lambda_alpha_corr": [0.0, 0.2],
            }
        )

        self.assertEqual(len(runs), 8)
        self.assertEqual(runs[-1].lambda_mse, 5.0)
        self.assertEqual(runs[-1].lambda_r2, 0.5)
        self.assertEqual(runs[-1].lambda_alpha_corr, 0.2)
        self.assertIn("mse5", runs[-1].name)
        self.assertIn("r20.5", runs[-1].name)
        self.assertIn("acorr0.2", runs[-1].name)

    def test_build_sweep_runs_includes_validation_fold(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
                "validation_fold": [3],
            }
        )

        self.assertEqual(runs[0].validation_fold, 3)
        self.assertIn("vf3", runs[0].name)

    def test_build_sweep_runs_includes_regularization(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [64],
                "gru_layers": [2],
                "learning_rate": [0.0002],
                "stock_limit": [1536],
                "max_steps": [800],
                "dropout": [0.1, 0.3],
                "weight_decay": [0.0001, 0.001],
            }
        )

        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[-1].dropout, 0.3)
        self.assertEqual(runs[-1].weight_decay, 0.001)
        self.assertIn("do0.3", runs[-1].name)
        self.assertIn("wd0.001", runs[-1].name)

    def test_build_sweep_runs_includes_label_transform(self):
        runs = build_sweep_runs(
            {
                "hidden_dim": [32],
                "gru_layers": [2],
                "learning_rate": [0.0002],
                "stock_limit": [1536],
                "max_steps": [400],
                "label_transform": ["rank_centered"],
            }
        )

        self.assertEqual(runs[0].label_transform, "rank_centered")
        self.assertIn("labelrank_centered", runs[0].name)

    def test_write_config_can_override_export_valid_dates(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {"export_valid_dates": 10}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            _write_config(base_config, path, run, Path(tmp), max_files=2, device="cpu", export_valid_dates=-1)
            text = path.read_text()

        self.assertIn("export_valid_dates: -1", text)

    def test_write_config_can_set_prepared_frame_cache(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            cache_path = "outputs/cache/prepared.pkl"
            _write_config(
                base_config,
                path,
                run,
                Path(tmp),
                max_files=49,
                device="cpu",
                prepared_frame_cache=cache_path,
            )
            text = path.read_text()

        self.assertIn(f'prepared_frame_cache: "{cache_path}"', text)

    def test_write_config_includes_loss_weights(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
                "lambda_mse": [5.0],
                "lambda_r2": [0.5],
                "lambda_alpha_corr": [0.2],
                "lambda_corr": [0.02],
                "lambda_to": [0.03],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            _write_config(base_config, path, run, Path(tmp), max_files=2, device="cpu")
            text = path.read_text()

        self.assertIn("lambda_mse: 5.0", text)
        self.assertIn("lambda_r2: 0.5", text)
        self.assertIn("lambda_alpha_corr: 0.2", text)
        self.assertIn("lambda_corr: 0.02", text)
        self.assertIn("lambda_to: 0.03", text)

    def test_write_config_includes_validation_fold(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [384],
                "gru_layers": [2],
                "learning_rate": [0.0005],
                "stock_limit": [1024],
                "max_steps": [100],
                "validation_fold": [3],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            _write_config(base_config, path, run, Path(tmp), max_files=2, device="cpu")
            text = path.read_text()

        self.assertIn("validation_fold: 3", text)

    def test_write_config_includes_regularization(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [64],
                "gru_layers": [2],
                "learning_rate": [0.0002],
                "stock_limit": [1536],
                "max_steps": [800],
                "dropout": [0.3],
                "weight_decay": [0.001],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            _write_config(base_config, path, run, Path(tmp), max_files=2, device="cpu")
            text = path.read_text()

        self.assertIn("dropout: 0.3", text)
        self.assertIn("weight_decay: 0.001", text)

    def test_write_config_includes_label_transform(self):
        run = build_sweep_runs(
            {
                "hidden_dim": [32],
                "gru_layers": [2],
                "learning_rate": [0.0002],
                "stock_limit": [1536],
                "max_steps": [400],
                "label_transform": ["rank_centered"],
            }
        )[0]
        base_config = {"data": {}, "model": {}, "train": {}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            _write_config(base_config, path, run, Path(tmp), max_files=2, device="cpu")
            text = path.read_text()

        self.assertIn('label_transform: "rank_centered"', text)

    def test_read_evaluation_metrics_extracts_alpha_return_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics_alpha.csv").write_text(
                "factor,rankic,abs_rankic,icir,win_rate,autocorrelation\n"
                "alpha_0,0.12,0.13,0.9,0.8,0.7\n"
            )
            (run_dir / "prediction_accuracy.csv").write_text(
                "factor,orientation,rankic,cross_sectional_hit_rate,mean_daily_hit_rate,n_dates,n_observations\n"
                "alpha_0,1.0,0.12,0.55,0.56,2,20\n"
            )
            (run_dir / "alpha_group_returns.csv").write_text(
                "group,orientation,mean_return,mean_excess_return,positive_rate,mean_n_stocks,n_periods\n"
                "0,1.0,0.01,-0.01,0.4,10,2\n"
                "19,1.0,0.03,0.02,0.6,10,2\n"
            )
            (run_dir / "alpha_long_short.csv").write_text(
                "TRADE_DT,orientation,top_group,bottom_group,top_return,bottom_return,universe_return,top_excess_return,bottom_excess_return,long_short_return\n"
                "20200101,1.0,19,0,0.03,0.01,0.02,0.01,-0.01,0.02\n"
                "20200108,1.0,19,0,0.01,0.02,0.015,-0.005,0.005,-0.01\n"
            )

            metrics = _read_evaluation_metrics(run_dir)

        self.assertEqual(metrics["alpha_rankic"], 0.12)
        self.assertEqual(metrics["alpha_orientation"], 1.0)
        self.assertEqual(metrics["alpha_oriented_rankic"], 0.12)
        self.assertEqual(metrics["alpha_oriented_icir"], 0.9)
        self.assertEqual(metrics["alpha_oriented_win_rate"], 0.8)
        self.assertEqual(metrics["alpha_hit_rate"], 0.55)
        self.assertEqual(metrics["alpha_top_excess_mean"], 0.0025)
        self.assertEqual(metrics["alpha_long_short_mean"], 0.005)
        self.assertEqual(metrics["alpha_long_short_positive_rate"], 0.5)

    def test_read_evaluation_metrics_orients_negative_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics_alpha.csv").write_text(
                "factor,rankic,abs_rankic,icir,win_rate,autocorrelation\n"
                "alpha_0,-0.08,0.08,-0.7,0.25,0.7\n"
            )
            (run_dir / "prediction_accuracy.csv").write_text(
                "factor,orientation,rankic,cross_sectional_hit_rate,mean_daily_hit_rate,n_dates,n_observations\n"
                "alpha_0,-1.0,-0.08,0.57,0.57,2,20\n"
            )

            metrics = _read_evaluation_metrics(run_dir)

        self.assertEqual(metrics["alpha_orientation"], -1.0)
        self.assertEqual(metrics["alpha_oriented_rankic"], 0.08)
        self.assertEqual(metrics["alpha_oriented_icir"], 0.7)
        self.assertEqual(metrics["alpha_oriented_win_rate"], 0.75)

    def test_sort_rows_by_return_prioritizes_validation_returns(self):
        rows = [
            {
                "name": "low_return_low_loss",
                "alpha_top_excess_mean": "0.001",
                "alpha_long_short_mean": "0.001",
                "alpha_top_excess_positive_rate": "0.55",
                "alpha_rankic": "0.05",
                "validation_loss": "0.1",
            },
            {
                "name": "high_return_higher_loss",
                "alpha_top_excess_mean": "0.004",
                "alpha_long_short_mean": "0.002",
                "alpha_top_excess_positive_rate": "0.6",
                "alpha_rankic": "0.04",
                "validation_loss": "0.9",
            },
        ]

        sorted_rows = _sort_rows_by_return(rows)

        self.assertEqual(sorted_rows[0]["name"], "high_return_higher_loss")

    def test_sort_rows_by_return_uses_oriented_rankic_as_tiebreaker(self):
        rows = [
            {
                "name": "positive_raw_rankic",
                "alpha_top_excess_mean": "0.004",
                "alpha_long_short_mean": "0.002",
                "alpha_top_excess_positive_rate": "0.6",
                "alpha_orientation": "1.0",
                "alpha_rankic": "0.03",
                "alpha_oriented_rankic": "0.03",
                "validation_loss": "0.4",
            },
            {
                "name": "negative_raw_rankic_better_after_flip",
                "alpha_top_excess_mean": "0.004",
                "alpha_long_short_mean": "0.002",
                "alpha_top_excess_positive_rate": "0.6",
                "alpha_orientation": "-1.0",
                "alpha_rankic": "-0.07",
                "alpha_oriented_rankic": "0.07",
                "validation_loss": "0.5",
            },
        ]

        sorted_rows = _sort_rows_by_return(rows)

        self.assertEqual(sorted_rows[0]["name"], "negative_raw_rankic_better_after_flip")


if __name__ == "__main__":
    unittest.main()
