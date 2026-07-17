import tempfile
import unittest

import numpy as np
import pandas as pd

from abcm.features import FEATURE_COLUMNS
from abcm.model import ABCM
from abcm.dataset import ABCMWindowSampler
from abcm.pipeline import (
    available_paired_training_dates,
    evaluate_loss_for_dates,
    export_factors_for_dates,
    load_or_prepare_abcm_frame,
    load_or_prepare_abcm_frame_from_files,
    prepare_abcm_frame,
    train_one_batch,
)
from scripts.train_abcm1 import (
    build_training_run_summary,
    estimate_epoch_steps,
    iter_training_date_batches,
    loss_weights_from_config,
    select_export_dates,
)


def _synthetic_raw_frame(n_dates=12, n_codes=4):
    rows = []
    dates = pd.bdate_range("2020-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    for code_idx in range(n_codes):
        code = f"00000{code_idx + 1}.SZ"
        for date_idx, date in enumerate(dates):
            close = 10.0 + code_idx + date_idx * 0.1
            rows.append(
                {
                    "S_INFO_WINDCODE": code,
                    "TRADE_DT": date,
                    "S_DQ_PRECLOSE": close - 0.1,
                    "S_DQ_OPEN": close - 0.05,
                    "S_DQ_HIGH": close + 0.1,
                    "S_DQ_LOW": close - 0.2,
                    "S_DQ_CLOSE": close,
                    "S_DQ_VOLUME": 1000.0 + date_idx,
                    "S_DQ_AMOUNT": 10000.0 + 10 * date_idx,
                    "S_DQ_AVGPRICE": close,
                    "S_DQ_PCTCHANGE": 0.1,
                    "S_DQ_ADJFACTOR": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _prepared_frame_with_gap(n_codes=4):
    rows = []
    segments = [
        (0, [f"202001{day:02d}" for day in range(1, 9)]),
        (1, [f"202101{day:02d}" for day in range(1, 9)]),
    ]
    for segment_id, dates in segments:
        for code_idx in range(n_codes):
            code = f"00000{code_idx + 1}.SZ"
            for date in dates:
                row = {
                    "segment_id": segment_id,
                    "TRADE_DT": date,
                    "S_INFO_WINDCODE": code,
                    "y1_raw": 0.01,
                    "y2_raw": 0.02,
                }
                row.update({feature: 0.0 for feature in FEATURE_COLUMNS})
                rows.append(row)
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def test_loss_weights_from_config_includes_alpha_corr_weight(self):
        weights = loss_weights_from_config(
            {
                "lambda_mse": 5.0,
                "lambda_r2": 0.5,
                "lambda_corr": 0.02,
                "lambda_to": 0.03,
                "lambda_alpha_corr": 0.2,
            }
        )

        self.assertEqual(weights["lambda_alpha_corr"], 0.2)

    def test_select_export_dates_supports_all_and_none(self):
        valid_dates = ["20200117", "20200120", "20200121"]

        self.assertEqual(select_export_dates(valid_dates, -1), valid_dates)
        self.assertEqual(select_export_dates(valid_dates, 2), valid_dates[:2])
        self.assertEqual(select_export_dates(valid_dates, 0), [])

    def test_training_run_summary_estimates_epoch_time(self):
        summary = build_training_run_summary(
            train_start_utc="2026-07-05T10:00:00Z",
            train_end_utc="2026-07-05T10:02:00Z",
            total_train_seconds=120.0,
            step_seconds=[0.5, 0.75, 1.0],
            train_date_count=101,
            valid_date_count=20,
            max_steps=3,
            date_batch_size=2,
            model_parameter_count=1234,
            trainable_parameter_count=1200,
            device="cuda:0",
            cuda_metadata={"cuda_device_name": "A40", "cuda_peak_memory_allocated_bytes": 4096},
        )

        self.assertEqual(estimate_epoch_steps(train_date_count=101, date_batch_size=2), 51)
        self.assertEqual(summary["estimated_epoch_steps"], 51)
        self.assertAlmostEqual(summary["mean_step_seconds"], 0.75)
        self.assertAlmostEqual(summary["estimated_epoch_seconds"], 38.25)
        self.assertEqual(summary["model_parameter_count"], 1234)
        self.assertEqual(summary["cuda_device_name"], "A40")

    def test_training_date_batches_cover_one_epoch_without_dropping_last_date(self):
        train_dates = [f"202001{day:02d}" for day in range(1, 8)]

        batches = list(
            iter_training_date_batches(
                train_dates,
                max_steps=estimate_epoch_steps(len(train_dates), date_batch_size=3),
                date_batch_size=3,
                seed=42,
            )
        )
        emitted_dates = [date for _, _, dates in batches for date in dates]

        self.assertEqual(len(batches), 3)
        self.assertEqual([len(dates) for _, _, dates in batches], [3, 3, 1])
        self.assertCountEqual(emitted_dates, train_dates)
        self.assertEqual(len(emitted_dates), len(set(emitted_dates)))

    def test_training_date_batches_are_deterministic_and_change_between_epochs(self):
        train_dates = [f"202001{day:02d}" for day in range(1, 11)]
        epoch_steps = estimate_epoch_steps(len(train_dates), date_batch_size=2)

        first = list(
            iter_training_date_batches(
                train_dates,
                max_steps=epoch_steps * 2,
                date_batch_size=2,
                seed=42,
            )
        )
        second = list(
            iter_training_date_batches(
                train_dates,
                max_steps=epoch_steps * 2,
                date_batch_size=2,
                seed=42,
            )
        )
        first_epoch = [date for epoch, _, dates in first if epoch == 0 for date in dates]
        second_epoch = [date for epoch, _, dates in first if epoch == 1 for date in dates]

        self.assertEqual(first, second)
        self.assertCountEqual(first_epoch, train_dates)
        self.assertCountEqual(second_epoch, train_dates)
        self.assertNotEqual(first_epoch, second_epoch)

    def test_training_run_summary_records_actual_date_coverage(self):
        summary = build_training_run_summary(
            train_start_utc="2026-07-17T10:00:00Z",
            train_end_utc="2026-07-17T10:01:00Z",
            total_train_seconds=60.0,
            step_seconds=[0.5, 0.5, 0.5],
            train_date_count=5,
            valid_date_count=2,
            max_steps=3,
            date_batch_size=2,
            model_parameter_count=100,
            trainable_parameter_count=100,
            device="cpu",
            processed_train_date_slots=5,
            shuffle_train_dates=True,
        )

        self.assertEqual(summary["processed_train_date_slots"], 5)
        self.assertEqual(summary["completed_epoch_equivalents"], 1.0)
        self.assertTrue(summary["shuffle_train_dates"])

    def test_prepare_frame_and_train_one_batch(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        self.assertIn("segment_id", frame.columns)
        self.assertIn("y1_raw", frame.columns)
        self.assertFalse(np.isinf(frame[FEATURE_COLUMNS].to_numpy()).any())

        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)
        result = train_one_batch(
            model,
            frame,
            feature_columns=FEATURE_COLUMNS,
            date="20200117",
            lookback=5,
            stock_limit=4,
            turnover_lag=5,
        )

        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["n_stocks"], 0)
        self.assertTrue(np.isfinite(result["turnover"]))
        self.assertGreater(result["turnover"], 0.0)

    def test_load_or_prepare_abcm_frame_uses_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = f"{tmp}/prepared.pkl"
            first = load_or_prepare_abcm_frame(
                _synthetic_raw_frame(n_dates=12),
                cache_path=cache_path,
                feature_columns=FEATURE_COLUMNS,
                y1_horizon=2,
                y2_horizon=3,
                entry_lag=1,
            )
            second = load_or_prepare_abcm_frame(
                _synthetic_raw_frame(n_dates=14),
                cache_path=cache_path,
                feature_columns=FEATURE_COLUMNS,
                y1_horizon=2,
                y2_horizon=3,
                entry_lag=1,
            )

        self.assertEqual(len(first), len(second))
        self.assertEqual(first["TRADE_DT"].nunique(), second["TRADE_DT"].nunique())

    def test_load_or_prepare_abcm_frame_from_files_skips_raw_when_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = f"{tmp}/prepared.pkl"
            expected = prepare_abcm_frame(
                _synthetic_raw_frame(n_dates=12),
                feature_columns=FEATURE_COLUMNS,
                y1_horizon=2,
                y2_horizon=3,
                entry_lag=1,
            )
            expected.to_pickle(cache_path)

            loaded = load_or_prepare_abcm_frame_from_files(
                f"{tmp}/missing_raw_dir",
                max_files=49,
                cache_path=cache_path,
                feature_columns=FEATURE_COLUMNS,
                y1_horizon=2,
                y2_horizon=3,
                entry_lag=1,
            )

        pd.testing.assert_frame_equal(expected, loaded)

    def test_train_one_batch_accepts_reused_sampler(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=5)
        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)

        result = train_one_batch(
            model,
            frame,
            feature_columns=FEATURE_COLUMNS,
            date="20200117",
            lookback=5,
            stock_limit=4,
            turnover_lag=5,
            sampler=sampler,
        )

        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["n_stocks"], 0)

    def test_label_transform_uses_train_targets_but_exports_raw_returns(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
            label_transform="rank_centered",
        )
        date = "20200117"
        sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=5)

        batch = sampler.sample_for_date(date, stock_limit=4, seed=42)
        frame_slice = (
            frame.loc[frame["TRADE_DT"].astype(str) == date]
            .set_index("S_INFO_WINDCODE")
            .loc[batch.codes[0]]
        )

        np.testing.assert_allclose(batch.y1[0], frame_slice["y1_train"].to_numpy(dtype=np.float32))
        self.assertFalse(np.allclose(frame_slice["y1_raw"], frame_slice["y1_train"]))

        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)
        factors = export_factors_for_dates(
            model,
            frame,
            dates=[date],
            feature_columns=FEATURE_COLUMNS,
            lookback=5,
            stock_limit=4,
            device="cpu",
            sampler=sampler,
        ).set_index("S_INFO_WINDCODE").loc[batch.codes[0]]

        np.testing.assert_allclose(factors["y1_raw"].to_numpy(dtype=float), frame_slice["y1_raw"].to_numpy(dtype=float))

    def test_train_one_batch_accepts_multiple_dates(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=5)
        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)

        result = train_one_batch(
            model,
            frame,
            feature_columns=FEATURE_COLUMNS,
            dates=["20200117", "20200120"],
            lookback=5,
            stock_limit=4,
            turnover_lag=5,
            sampler=sampler,
        )

        self.assertEqual(result["n_dates"], 2)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(np.isfinite(result["turnover"]))

    def test_train_one_batch_accepts_loss_weights(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=5)
        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)

        result = train_one_batch(
            model,
            frame,
            feature_columns=FEATURE_COLUMNS,
            date="20200117",
            lookback=5,
            stock_limit=4,
            turnover_lag=5,
            sampler=sampler,
            loss_weights={"lambda_mse": 2.0, "lambda_r2": 0.5},
        )

        self.assertTrue(np.isfinite(result["loss"]))
        self.assertGreater(result["n_stocks"], 0)

    def test_sampler_returns_lagged_same_stock_batch_for_turnover(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=16),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        sampler = ABCMWindowSampler(frame, FEATURE_COLUMNS, lookback=5)

        batch = sampler.sample_pair_for_date("20200117", lag_periods=5, stock_limit=4, seed=7)

        self.assertEqual(batch.current.dates, ["20200117"])
        self.assertEqual(batch.previous.dates, ["20200110"])
        self.assertEqual(batch.current.codes, batch.previous.codes)
        self.assertEqual(batch.current.x.shape, batch.previous.x.shape)

    def test_export_factors_for_dates_has_expected_columns(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)

        factors = export_factors_for_dates(
            model,
            frame,
            dates=["20200109"],
            feature_columns=FEATURE_COLUMNS,
            lookback=5,
            stock_limit=4,
        )

        expected = {"segment_id", "TRADE_DT", "S_INFO_WINDCODE", "alpha_0", "y1_raw", "y2_raw"}
        expected.update({f"beta_{idx}" for idx in range(12)})
        self.assertTrue(expected.issubset(factors.columns))
        self.assertGreater(len(factors), 0)
        self.assertEqual(factors["segment_id"].nunique(), 1)

    def test_evaluate_loss_for_dates_returns_mean_without_training(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )
        model = ABCM(input_dim=len(FEATURE_COLUMNS), hidden_dim=8, gru_layers=1)
        before = [param.detach().clone() for param in model.parameters()]

        result = evaluate_loss_for_dates(
            model,
            frame,
            dates=["20200117", "20200120"],
            feature_columns=FEATURE_COLUMNS,
            lookback=5,
            stock_limit=4,
            turnover_lag=5,
        )

        self.assertEqual(result["n_dates"], 2)
        self.assertTrue(np.isfinite(result["loss"]))
        after = list(model.parameters())
        for old, new in zip(before, after):
            self.assertTrue(np.allclose(old.numpy(), new.detach().numpy()))

    def test_available_paired_training_dates_only_returns_pairable_dates(self):
        frame = prepare_abcm_frame(
            _synthetic_raw_frame(n_dates=20),
            feature_columns=FEATURE_COLUMNS,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )

        dates = available_paired_training_dates(
            frame,
            feature_columns=FEATURE_COLUMNS,
            lookback=5,
            min_stocks=4,
            turnover_lag=5,
            stock_limit=4,
        )

        self.assertIn("20200117", dates)
        self.assertNotIn("20200109", dates)

    def test_available_paired_training_dates_does_not_cross_gap_segments(self):
        frame = _prepared_frame_with_gap()

        dates = available_paired_training_dates(
            frame,
            feature_columns=FEATURE_COLUMNS,
            lookback=5,
            min_stocks=4,
            turnover_lag=2,
            stock_limit=4,
        )

        self.assertNotIn("20210101", dates)
        self.assertNotIn("20210105", dates)
        self.assertIn("20210107", dates)


if __name__ == "__main__":
    unittest.main()
