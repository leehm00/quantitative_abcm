import unittest

import numpy as np
import pandas as pd

from abcm.features import FEATURE_COLUMNS
from scripts.train_lightgbm_baseline import (
    build_lightgbm_samples,
    build_tabular_lightgbm_samples,
    factor_frame_from_predictions,
)


def _frame() -> pd.DataFrame:
    rows = []
    dates = ["20200101", "20200102", "20200103"]
    codes = ["A", "B", "C"]
    for date_idx, date in enumerate(dates):
        for code_idx, code in enumerate(codes):
            row = {
                "TRADE_DT": date,
                "S_INFO_WINDCODE": code,
                "segment_id": 0,
                "y1_raw": float(date_idx + code_idx) / 100.0,
                "y2_raw": float(date_idx + code_idx) / 50.0,
            }
            for feature_idx, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = float(date_idx * 100 + code_idx * 10 + feature_idx)
            rows.append(row)
    return pd.DataFrame(rows)


class LightGBMBaselineTests(unittest.TestCase):
    def test_build_lightgbm_samples_uses_current_window_features_and_labels(self):
        samples = build_lightgbm_samples(
            _frame(),
            dates=["20200103"],
            feature_columns=FEATURE_COLUMNS,
            lookback=2,
            stock_limit=3,
            seed=7,
        )

        expected = (
            _frame()
            .loc[_frame()["TRADE_DT"] == "20200103", FEATURE_COLUMNS]
            .to_numpy(dtype=np.float32)
        )
        self.assertEqual(samples.x.shape, (3, len(FEATURE_COLUMNS)))
        np.testing.assert_allclose(samples.x, expected)
        np.testing.assert_allclose(samples.y, np.array([0.02, 0.03, 0.04], dtype=np.float32))
        self.assertEqual(samples.meta["S_INFO_WINDCODE"].tolist(), ["A", "B", "C"])

    def test_factor_frame_from_predictions_exports_alpha_beta_and_labels(self):
        samples = build_lightgbm_samples(
            _frame(),
            dates=["20200103"],
            feature_columns=FEATURE_COLUMNS,
            lookback=2,
            stock_limit=3,
            seed=7,
        )
        factors = factor_frame_from_predictions(samples, np.array([0.1, 0.2, 0.3]))

        self.assertEqual(factors["alpha_0"].tolist(), [0.1, 0.2, 0.3])
        self.assertEqual(factors["TRADE_DT"].tolist(), ["20200103", "20200103", "20200103"])
        self.assertEqual(factors["segment_id"].tolist(), [0, 0, 0])
        self.assertTrue(all(f"beta_{idx}" in factors.columns for idx in range(12)))
        self.assertTrue((factors[[f"beta_{idx}" for idx in range(12)]] == 0.0).all().all())
        np.testing.assert_allclose(factors["y1_raw"].to_numpy(), np.array([0.02, 0.03, 0.04]))
        np.testing.assert_allclose(factors["y2_raw"].to_numpy(), np.array([0.04, 0.06, 0.08]))

    def test_build_tabular_lightgbm_samples_limits_each_date(self):
        samples = build_tabular_lightgbm_samples(
            _frame(),
            dates=["20200102", "20200103"],
            feature_columns=FEATURE_COLUMNS,
            stock_limit=2,
            seed=11,
        )

        self.assertEqual(samples.x.shape, (4, len(FEATURE_COLUMNS)))
        self.assertEqual(samples.y.shape, (4,))
        counts = samples.meta.groupby("TRADE_DT").size().to_dict()
        self.assertEqual(counts, {"20200102": 2, "20200103": 2})
        self.assertTrue(set(samples.meta.columns) >= {"TRADE_DT", "S_INFO_WINDCODE", "segment_id", "y1_raw", "y2_raw"})


if __name__ == "__main__":
    unittest.main()
