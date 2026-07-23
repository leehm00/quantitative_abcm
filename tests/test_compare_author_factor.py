from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_author_factor import (
    alignment_sensitivity,
    build_common_frame,
    cross_sectional_rank_residual,
    daily_spearman,
    disagreement_group_returns,
    group_return_series,
    load_labels,
    moving_block_bootstrap_mean_ci,
    residual_rankic_outputs,
    summarize_factor,
    top_overlap_daily,
    top_portfolio_backtest,
    trimmed_mean,
)


class CompareAuthorFactorTest(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for date_idx, date in enumerate(["20260102", "20260105", "20260106"]):
            for stock_idx in range(20):
                value = float(stock_idx)
                rows.append(
                    {
                        "TRADE_DT": date,
                        "S_INFO_WINDCODE": f"{stock_idx:06d}.SZ",
                        "factor": value,
                        "y1_raw": value * 0.01 + date_idx * 0.001,
                    }
                )
        self.frame = pd.DataFrame(rows)

    def test_daily_spearman_detects_perfect_ordering(self) -> None:
        result = daily_spearman(self.frame, "factor")
        np.testing.assert_allclose(result.to_numpy(), np.ones(3))

    def test_load_labels_uses_raw_percentage_change_for_daily_return(self) -> None:
        frame = pd.DataFrame(
            {
                "TRADE_DT": ["20260102", "20260105"],
                "S_INFO_WINDCODE": ["000001.SZ", "000001.SZ"],
                "S_DQ_PCTCHANGE": [2.5, -1.25],
                "y1_raw": [0.1, 0.2],
                "y2_raw": [0.2, 0.3],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pkl"
            frame.to_pickle(path)
            result = load_labels(path, min_date="20260101")
        np.testing.assert_allclose(result["daily_return"], [0.025, -0.0125])

    def test_group_returns_use_top_and_bottom_groups(self) -> None:
        result = group_return_series(
            self.frame,
            "factor",
            n_groups=4,
            rebalance_every=1,
        )
        self.assertEqual(len(result), 3)
        self.assertTrue((result["top_excess"] > 0).all())
        self.assertTrue((result["long_short"] > 0).all())

    def test_summarize_factor_reports_positive_metrics(self) -> None:
        summary, daily, groups = summarize_factor(
            self.frame,
            "factor",
            n_groups=4,
            rebalance_every=1,
        )
        self.assertEqual(summary["rankic"], 1.0)
        self.assertEqual(summary["win_rate"], 1.0)
        self.assertGreater(summary["top_annualized_approx"], 0.0)
        self.assertEqual(len(daily), 3)
        self.assertEqual(len(groups), 3)

    def test_trimmed_mean_removes_both_tails(self) -> None:
        values = list(range(100)) + [-1000, 1000]
        result = trimmed_mean(values, fraction=0.01)
        self.assertAlmostEqual(result, 49.5)

    def test_block_bootstrap_interval_contains_constant_mean(self) -> None:
        low, high = moving_block_bootstrap_mean_ci(
            pd.Series(np.full(50, 0.1)),
            block_length=5,
            n_bootstrap=100,
            seed=7,
        )
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.1)

    def test_alignment_sensitivity_skips_dates_without_labels(self) -> None:
        author = pd.DataFrame(
            {
                "TRADE_DT": ["20260102", "20260105", "20260106"],
                "S_INFO_WINDCODE": ["000001.SZ"] * 3,
                "author_factor": [1.0, 2.0, 3.0],
            }
        )
        labels = pd.DataFrame(
            {
                "TRADE_DT": ["20260102", "20260105"],
                "S_INFO_WINDCODE": ["000001.SZ"] * 2,
                "y1_raw": [0.1, 0.2],
            }
        )
        result = alignment_sensitivity(author, labels, offsets=[0])
        self.assertEqual(result.loc[0, "n_observations"], 2)

    def test_top_overlap_reports_intersection_and_jaccard(self) -> None:
        frame = pd.DataFrame(
            {
                "TRADE_DT": ["20260102"] * 10,
                "S_INFO_WINDCODE": [f"{idx:06d}.SZ" for idx in range(10)],
                "author_factor": list(range(10)),
                "local_factor": [*range(7), 9, 7, 8],
            }
        )
        result = top_overlap_daily(
            frame,
            "author_factor",
            "local_factor",
            top_fraction=0.2,
        )
        self.assertEqual(int(result.loc[0, "intersection_count"]), 1)
        self.assertAlmostEqual(float(result.loc[0, "overlap_rate"]), 0.5)
        self.assertAlmostEqual(float(result.loc[0, "jaccard"]), 1.0 / 3.0)

    def test_disagreement_groups_keep_author_and_local_only_stocks_separate(self) -> None:
        frame = pd.DataFrame(
            {
                "TRADE_DT": ["20260102"] * 10,
                "S_INFO_WINDCODE": [f"{idx:06d}.SZ" for idx in range(10)],
                "author_factor": list(range(10)),
                "local_factor": [*range(7), 9, 7, 8],
                "y1_raw": np.arange(10, dtype=float) / 100.0,
                "y2_raw": np.arange(10, dtype=float) / 50.0,
            }
        )
        result = disagreement_group_returns(frame, "local_factor", top_fraction=0.2)
        y1 = result.loc[result["return_horizon"] == "y1_raw"].set_index("selection_group")
        self.assertAlmostEqual(float(y1.loc["both_top", "group_return"]), 0.09)
        self.assertAlmostEqual(float(y1.loc["author_only", "group_return"]), 0.08)
        self.assertAlmostEqual(float(y1.loc["local_only", "group_return"]), 0.07)

    def test_rank_residual_is_orthogonal_to_control_rank(self) -> None:
        frame = pd.DataFrame(
            {
                "TRADE_DT": ["20260102"] * 20,
                "target": np.sin(np.arange(20, dtype=float)),
                "control": np.arange(20, dtype=float),
            }
        )
        residual = cross_sectional_rank_residual(frame, "target", "control")
        control_rank = frame["control"].rank(method="average", pct=True)
        self.assertAlmostEqual(float(np.corrcoef(residual, control_rank)[0, 1]), 0.0, places=12)

    def test_residual_rankic_outputs_include_daily_rows(self) -> None:
        frame = self.frame.rename(columns={"factor": "author_factor"}).copy()
        frame["local_factor"] = frame["author_factor"] + np.sin(
            np.arange(len(frame), dtype=float)
        )
        frame["y2_raw"] = frame["y1_raw"] * 1.5
        metrics, daily = residual_rankic_outputs(
            frame,
            "local_factor",
            periods={"all": ("20260102", "20260106")},
            bootstrap_samples=10,
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(set(metrics["residual"]), {"local_residual", "author_residual"})
        self.assertEqual(set(daily["return_horizon"]), {"y1_raw", "y2_raw"})
        self.assertIn("TRADE_DT", daily.columns)

    def test_portfolio_backtest_applies_turnover_cost_and_drawdown(self) -> None:
        market_dates = [f"2026010{day}" for day in range(1, 9)]
        stocks = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        factor_rows = []
        for signal_idx, signal_date in enumerate(["20260101", "20260103", "20260105"]):
            values = [4.0, 3.0, 2.0, 1.0] if signal_idx % 2 == 0 else [1.0, 2.0, 3.0, 4.0]
            for stock, value in zip(stocks, values):
                factor_rows.append(
                    {
                        "TRADE_DT": signal_date,
                        "S_INFO_WINDCODE": stock,
                        "factor": value,
                    }
                )
        return_index = pd.MultiIndex.from_product(
            [market_dates, stocks],
            names=["TRADE_DT", "S_INFO_WINDCODE"],
        )
        returns = pd.Series(0.001, index=return_index, name="ret_1d")
        metrics, daily, turnover = top_portfolio_backtest(
            pd.DataFrame(factor_rows),
            "factor",
            daily_return_series=returns,
            market_dates=market_dates,
            top_fraction=0.5,
            rebalance_every=1,
            cost_bps_values=[0.0, 100.0],
        )
        by_cost = metrics.set_index("cost_bps")
        self.assertEqual(len(turnover), 3)
        self.assertAlmostEqual(float(by_cost.loc[0.0, "mean_target_turnover"]), 1.0)
        self.assertGreater(
            float(by_cost.loc[0.0, "net_annualized_return"]),
            float(by_cost.loc[100.0, "net_annualized_return"]),
        )
        self.assertGreater(float(by_cost.loc[100.0, "net_max_drawdown"]), 0.0)
        self.assertEqual(set(daily["cost_bps"]), {0.0, 100.0})
        self.assertEqual(str(by_cost.loc[0.0, "end"]), "20260107")

    def test_common_frame_keeps_pairwise_universes_separate(self) -> None:
        author = pd.DataFrame(
            {
                "TRADE_DT": ["20260102"] * 3,
                "S_INFO_WINDCODE": ["000001.SZ", "000002.SZ", "000003.SZ"],
                "author_factor": [1.0, 2.0, 3.0],
                "y1_raw": [0.1, 0.2, 0.3],
                "y2_raw": [0.2, 0.3, 0.4],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {}
            for model, stocks in {
                "model_a": ["000001.SZ", "000002.SZ"],
                "model_b": ["000002.SZ", "000003.SZ"],
            }.items():
                run_dir = root / model
                run_dir.mkdir()
                pd.DataFrame(
                    {
                        "TRADE_DT": ["20260102"] * 2,
                        "S_INFO_WINDCODE": stocks,
                        "alpha_0": [0.1, 0.2],
                        "y1_raw": [author.set_index("S_INFO_WINDCODE").loc[stock, "y1_raw"] for stock in stocks],
                        "y2_raw": [author.set_index("S_INFO_WINDCODE").loc[stock, "y2_raw"] for stock in stocks],
                    }
                ).to_csv(run_dir / "factors.csv", index=False)
                runs[model] = [str(run_dir)]

            common, coverage, pairwise = build_common_frame(author, runs, min_date="20260101")

        self.assertEqual(len(pairwise["model_a"]), 2)
        self.assertEqual(len(pairwise["model_b"]), 2)
        self.assertEqual(len(common), 1)
        self.assertEqual(coverage.set_index("model").loc["model_a", "pairwise_rows"], 2)


if __name__ == "__main__":
    unittest.main()
