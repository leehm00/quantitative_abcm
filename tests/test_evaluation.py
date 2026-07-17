import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from abcm.evaluation import (
    alpha_group_return_tables,
    evaluate_factor_frame,
    prediction_accuracy_table,
    write_evaluation_outputs,
)


def _factor_frame():
    rows = []
    dates = [f"202001{day:02d}" for day in range(1, 9)]
    for date_idx, date in enumerate(dates):
        for code_idx in range(20):
            code = f"C{code_idx:03d}"
            value = code_idx + 1 + date_idx * 0.1
            row = {
                "TRADE_DT": date,
                "S_INFO_WINDCODE": code,
                "alpha_0": value,
                "y1_raw": value * 0.01,
                "y2_raw": value * 0.02,
            }
            for beta_idx in range(12):
                row[f"beta_{beta_idx}"] = value + beta_idx * 0.01
            rows.append(row)
    return pd.DataFrame(rows)


def _segmented_factor_frame():
    rows = []
    segments = [
        (0, ["20200101", "20200102", "20200103"]),
        (1, ["20210101", "20210102", "20210103"]),
    ]
    for segment_id, dates in segments:
        for date_idx, date in enumerate(dates):
            for code_idx in range(20):
                code = f"C{code_idx:03d}"
                value = code_idx + 1 + date_idx * 0.1
                row = {
                    "segment_id": segment_id,
                    "TRADE_DT": date,
                    "S_INFO_WINDCODE": code,
                    "alpha_0": value,
                    "y1_raw": value * 0.01,
                    "y2_raw": value * 0.02,
                }
                for beta_idx in range(12):
                    row[f"beta_{beta_idx}"] = value + beta_idx * 0.01
                rows.append(row)
    return pd.DataFrame(rows)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_factor_frame_returns_expected_tables(self):
        result = evaluate_factor_frame(_factor_frame(), rolling_window=3, autocorr_lag=2)

        self.assertEqual(len(result.beta_metrics), 12)
        self.assertEqual(len(result.alpha_metrics), 1)
        self.assertIn("r_square", result.daily_rsquare.columns)
        self.assertIn("rolling_rsquare", result.rolling_rsquare.columns)
        self.assertTrue(np.isfinite(result.daily_rsquare["r_square"]).all())

    def test_write_evaluation_outputs_creates_csv_files(self):
        result = evaluate_factor_frame(_factor_frame(), rolling_window=3, autocorr_lag=2)

        with tempfile.TemporaryDirectory() as tmp:
            write_evaluation_outputs(result, tmp)
            names = sorted(path.name for path in Path(tmp).iterdir())

        self.assertEqual(
            names,
            [
                "alpha_group_returns.csv",
                "alpha_long_short.csv",
                "metrics_alpha.csv",
                "metrics_beta.csv",
                "prediction_accuracy.csv",
                "rolling_rsquare.csv",
                "rsquare_daily.csv",
            ],
        )

    def test_rolling_rsquare_resets_for_each_segment(self):
        result = evaluate_factor_frame(_segmented_factor_frame(), rolling_window=3, autocorr_lag=1)
        rolling = result.rolling_rsquare.set_index("TRADE_DT")
        daily = result.daily_rsquare.set_index("TRADE_DT")

        self.assertIn("segment_id", result.daily_rsquare.columns)
        self.assertIn("segment_id", result.rolling_rsquare.columns)
        self.assertEqual(int(result.daily_rsquare.loc[result.daily_rsquare["TRADE_DT"] == "20210101", "segment_id"].iloc[0]), 1)
        self.assertAlmostEqual(
            float(rolling.loc["20210101", "rolling_rsquare"]),
            float(daily.loc["20210101", "r_square"]),
        )

    def test_prediction_accuracy_orients_factor_by_rank_ic(self):
        df = pd.DataFrame(
            {
                "TRADE_DT": ["20200101"] * 4 + ["20200102"] * 4,
                "S_INFO_WINDCODE": ["A", "B", "C", "D"] * 2,
                "alpha_0": [4.0, 3.0, 2.0, 1.0] * 2,
                "y1_raw": [1.0, 2.0, 3.0, 4.0] * 2,
            }
        )

        out = prediction_accuracy_table(df, ["alpha_0"])

        self.assertEqual(float(out.loc[0, "orientation"]), -1.0)
        self.assertAlmostEqual(float(out.loc[0, "cross_sectional_hit_rate"]), 1.0)
        self.assertAlmostEqual(float(out.loc[0, "mean_daily_hit_rate"]), 1.0)

    def test_alpha_group_return_tables_report_long_short(self):
        rows = []
        for date in ["20200101", "20200102"]:
            for idx in range(8):
                rows.append(
                    {
                        "TRADE_DT": date,
                        "S_INFO_WINDCODE": f"C{idx:03d}",
                        "alpha_0": float(idx),
                        "y1_raw": float(idx) / 100.0,
                    }
                )
        df = pd.DataFrame(rows)

        group_returns, long_short = alpha_group_return_tables(
            df,
            factor_col="alpha_0",
            n_groups=4,
            rebalance_every=1,
        )

        self.assertEqual(group_returns["group"].tolist(), [0, 1, 2, 3])
        self.assertGreater(
            float(group_returns.loc[group_returns["group"] == 3, "mean_return"].iloc[0]),
            float(group_returns.loc[group_returns["group"] == 0, "mean_return"].iloc[0]),
        )
        self.assertTrue((long_short["long_short_return"] > 0).all())


if __name__ == "__main__":
    unittest.main()
