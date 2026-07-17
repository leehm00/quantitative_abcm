import unittest
import warnings

import numpy as np
import pandas as pd
import torch

from abcm.losses import abc_loss
from abcm.metrics import factor_autocorrelation, rank_ic, r_square_cross_section
from abcm.model import ABCM


class ModelLossMetricTests(unittest.TestCase):
    def test_abcm_forward_shapes(self):
        model = ABCM(input_dim=12, hidden_dim=16, alpha_dim=1, beta_dim=12, gru_layers=1)
        x = torch.randn(2, 5, 7, 12)

        factors, alpha, beta = model(x)

        self.assertEqual(tuple(alpha.shape), (2, 5, 1))
        self.assertEqual(tuple(beta.shape), (2, 5, 12))
        self.assertEqual(tuple(factors.shape), (2, 5, 13))

    def test_abc_loss_returns_finite_components(self):
        torch.manual_seed(42)
        factors = torch.randn(2, 8, 13)
        alpha = factors[:, :, :1]
        beta = factors[:, :, 1:]
        y1 = torch.randn(2, 8)
        y2 = torch.randn(2, 8)
        beta_prev = beta + 0.05 * torch.randn_like(beta)

        result = abc_loss(factors, alpha, beta, y1, y2, beta_prev=beta_prev)

        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(torch.isfinite(result.r2_residual))
        self.assertGreaterEqual(float(result.r2_residual), 0.0)

    def test_abc_loss_applies_mse_and_r2_weights(self):
        torch.manual_seed(7)
        factors = torch.randn(1, 10, 13)
        alpha = factors[:, :, :1]
        beta = factors[:, :, 1:]
        y1 = torch.randn(1, 10)
        y2 = torch.randn(1, 10)

        result = abc_loss(
            factors,
            alpha,
            beta,
            y1,
            y2,
            lambda_mse=2.0,
            lambda_r2=0.5,
            lambda_corr=0.0,
            lambda_to=0.0,
        )

        expected = 2.0 * result.mse + 0.5 * result.r2_residual
        self.assertTrue(torch.allclose(result.total, expected))

    def test_abc_loss_alpha_corr_rewards_same_direction_returns(self):
        y1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        y2 = torch.ones_like(y1)
        beta = torch.zeros(1, 5, 12)
        good_alpha = y1.unsqueeze(-1)
        bad_alpha = torch.flip(y1, dims=[1]).unsqueeze(-1)
        good_factors = torch.cat([good_alpha, beta], dim=-1)
        bad_factors = torch.cat([bad_alpha, beta], dim=-1)

        good = abc_loss(
            good_factors,
            good_alpha,
            beta,
            y1,
            y2,
            lambda_mse=0.0,
            lambda_r2=0.0,
            lambda_corr=0.0,
            lambda_to=0.0,
            lambda_alpha_corr=1.0,
        )
        bad = abc_loss(
            bad_factors,
            bad_alpha,
            beta,
            y1,
            y2,
            lambda_mse=0.0,
            lambda_r2=0.0,
            lambda_corr=0.0,
            lambda_to=0.0,
            lambda_alpha_corr=1.0,
        )

        self.assertLess(float(good.alpha_corr), 1e-6)
        self.assertGreater(float(bad.alpha_corr), 1.9)
        self.assertLess(float(good.total), float(bad.total))

    def test_rank_ic_and_rsquare_metrics(self):
        df = pd.DataFrame(
            {
                "TRADE_DT": ["20200101"] * 5,
                "factor": [1, 2, 3, 4, 5],
                "forward_return": [1, 2, 3, 4, 5],
            }
        )

        self.assertAlmostEqual(rank_ic(df, "factor", "forward_return").iloc[0], 1.0)
        x = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(r_square_cross_section(x, y), 1.0)

    def test_rank_ic_returns_nan_for_constant_factor_without_warning(self):
        df = pd.DataFrame(
            {
                "TRADE_DT": ["20200101"] * 5,
                "factor": [0, 0, 0, 0, 0],
                "forward_return": [1, 2, 3, 4, 5],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = rank_ic(df, "factor", "forward_return")

        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertEqual(caught, [])

    def test_factor_autocorrelation_matches_same_codes(self):
        df = pd.DataFrame(
            {
                "TRADE_DT": ["20200101", "20200101", "20200106", "20200106"],
                "S_INFO_WINDCODE": ["A", "B", "A", "B"],
                "beta_0": [1.0, 2.0, 2.0, 4.0],
            }
        )

        out = factor_autocorrelation(df, "beta_0", lag_periods=1)

        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]), 1.0)

    def test_factor_autocorrelation_resets_at_segment_boundaries(self):
        df = pd.DataFrame(
            {
                "segment_id": [0, 0, 0, 0, 1, 1, 1, 1],
                "TRADE_DT": [
                    "20200101",
                    "20200101",
                    "20200102",
                    "20200102",
                    "20210101",
                    "20210101",
                    "20210102",
                    "20210102",
                ],
                "S_INFO_WINDCODE": ["A", "B"] * 4,
                "beta_0": [1.0, 2.0, 2.0, 4.0, 10.0, 20.0, 20.0, 40.0],
            }
        )

        out = factor_autocorrelation(df, "beta_0", lag_periods=1)

        self.assertEqual(out.index.astype(str).tolist(), ["20200102", "20210102"])
        self.assertAlmostEqual(float(out.loc["20200102"]), 1.0)
        self.assertAlmostEqual(float(out.loc["20210102"]), 1.0)


if __name__ == "__main__":
    unittest.main()
