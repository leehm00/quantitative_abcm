import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from abcm.data import add_adjusted_prices, list_data_files, summarize_data_files
from abcm.features import (
    add_forward_return_labels,
    add_price_volume_features,
    cross_sectional_mad_zscore,
)


class DataAndFeatureTests(unittest.TestCase):
    def test_list_data_files_sorts_by_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["testdata_10.pkl", "testdata_2.pkl", "testdata_0.pkl"]:
                (root / name).write_bytes(b"x")

            names = [path.name for path in list_data_files(root)]

        self.assertEqual(names, ["testdata_0.pkl", "testdata_2.pkl", "testdata_10.pkl"])

    def test_summarize_data_files_reports_missing_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for number in [0, 1, 4]:
                df = pd.DataFrame(
                    {
                        "S_INFO_WINDCODE": ["000001.SZ"],
                        "TRADE_DT": [f"2020010{number + 1}"],
                    }
                )
                df.to_pickle(root / f"testdata_{number}.pkl")

            inventory = summarize_data_files(root)

        self.assertEqual(inventory.file_count, 3)
        self.assertEqual(inventory.missing_file_numbers, [2, 3])
        self.assertEqual(inventory.date_min, "20200101")
        self.assertEqual(inventory.date_max, "20200105")

    def test_add_adjusted_prices_uses_adjustment_factor(self):
        df = pd.DataFrame(
            {
                "S_DQ_PRECLOSE": [9.0],
                "S_DQ_OPEN": [10.0],
                "S_DQ_HIGH": [12.0],
                "S_DQ_LOW": [8.0],
                "S_DQ_CLOSE": [11.0],
                "S_DQ_AVGPRICE": [10.5],
                "S_DQ_ADJFACTOR": [2.0],
            }
        )

        out = add_adjusted_prices(df)

        self.assertEqual(out.loc[0, "adj_preclose"], 18.0)
        self.assertEqual(out.loc[0, "adj_close"], 22.0)
        self.assertEqual(out.loc[0, "adj_vwap"], 21.0)

    def test_feature_and_label_construction_does_not_cross_segments(self):
        rows = []
        for segment_id, dates, start_price in [
            (0, ["20200101", "20200102", "20200103", "20200106"], 10.0),
            (1, ["20210101", "20210104", "20210105", "20210106"], 20.0),
        ]:
            for offset, date in enumerate(dates):
                close = start_price + offset
                rows.append(
                    {
                        "segment_id": segment_id,
                        "S_INFO_WINDCODE": "000001.SZ",
                        "TRADE_DT": date,
                        "S_DQ_PRECLOSE": close - 0.5,
                        "S_DQ_OPEN": close - 0.25,
                        "S_DQ_HIGH": close + 0.5,
                        "S_DQ_LOW": close - 0.5,
                        "S_DQ_CLOSE": close,
                        "S_DQ_VOLUME": 1000 + offset,
                        "S_DQ_AMOUNT": 10000 + offset,
                        "S_DQ_AVGPRICE": close,
                        "S_DQ_ADJFACTOR": 1.0,
                    }
                )
        df = add_adjusted_prices(pd.DataFrame(rows))

        featured = add_price_volume_features(df)
        labeled = add_forward_return_labels(
            featured,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
        )

        last_row_segment_0 = labeled[labeled["TRADE_DT"] == "20200106"].iloc[0]
        first_row_segment_1 = labeled[labeled["TRADE_DT"] == "20210101"].iloc[0]
        self.assertTrue(np.isnan(last_row_segment_0["y1_raw"]))
        self.assertAlmostEqual(first_row_segment_1["y1_raw"], 22.0 / 21.0 - 1.0)
        self.assertIn("ret_1d", labeled.columns)
        self.assertIn("vol_20d", labeled.columns)

    def test_forward_return_labels_can_clip_extreme_returns(self):
        rows = []
        for offset, close in enumerate([1.0, 1.0, 1003.0, 905.0], start=1):
            rows.append(
                {
                    "segment_id": 0,
                    "S_INFO_WINDCODE": "833427.BJ",
                    "TRADE_DT": f"2020010{offset}",
                    "adj_close": close,
                }
            )
        df = pd.DataFrame(rows)

        labeled = add_forward_return_labels(
            df,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
            label_clip_abs=1.0,
        )

        first = labeled[labeled["TRADE_DT"] == "20200101"].iloc[0]
        self.assertEqual(first["y1_raw"], 1.0)
        self.assertEqual(first["y2_raw"], 1.0)

    def test_forward_return_labels_can_add_rank_centered_training_targets(self):
        rows = []
        closes_by_code = {
            "000001.SZ": [10.0, 10.0, 11.0, 11.0],
            "000002.SZ": [10.0, 10.0, 12.0, 12.0],
            "000003.SZ": [10.0, 10.0, 13.0, 13.0],
        }
        for code, closes in closes_by_code.items():
            for offset, close in enumerate(closes, start=1):
                rows.append(
                    {
                        "segment_id": 0,
                        "S_INFO_WINDCODE": code,
                        "TRADE_DT": f"2020010{offset}",
                        "adj_close": close,
                    }
                )
        df = pd.DataFrame(rows)

        labeled = add_forward_return_labels(
            df,
            y1_horizon=2,
            y2_horizon=3,
            entry_lag=1,
            label_transform="rank_centered",
        )

        first_date = labeled[labeled["TRADE_DT"] == "20200101"].sort_values("S_INFO_WINDCODE")
        self.assertEqual(first_date["y1_raw"].round(6).tolist(), [0.1, 0.2, 0.3])
        self.assertEqual(first_date["y1_train"].round(6).tolist(), [-1.0, 0.0, 1.0])

    def test_cross_sectional_mad_zscore_fills_and_scales_by_date(self):
        df = pd.DataFrame(
            {
                "TRADE_DT": ["20200101"] * 4,
                "feature": [1.0, 2.0, 3.0, np.nan],
            }
        )

        out = cross_sectional_mad_zscore(df, ["feature"])

        self.assertFalse(out["feature"].isna().any())
        self.assertAlmostEqual(float(out["feature"].mean()), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
