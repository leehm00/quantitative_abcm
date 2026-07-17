from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ABCLoss:
    total: torch.Tensor
    mse: torch.Tensor
    r2_residual: torch.Tensor
    alpha_corr: torch.Tensor
    corr: torch.Tensor
    turnover: torch.Tensor


def _r2_residual_loss(factors: torch.Tensor, returns: torch.Tensor, ridge_eps: float) -> torch.Tensor:
    losses = []
    eye = torch.eye(factors.size(-1), dtype=factors.dtype, device=factors.device)
    for batch_idx in range(factors.size(0)):
        x = factors[batch_idx]
        y = returns[batch_idx].unsqueeze(-1)
        valid = torch.isfinite(x).all(dim=1) & torch.isfinite(y.squeeze(-1))
        x = x[valid]
        y = y[valid]
        if x.size(0) <= x.size(1):
            losses.append(torch.zeros((), dtype=factors.dtype, device=factors.device))
            continue
        xtx = x.transpose(0, 1) @ x
        xty = x.transpose(0, 1) @ y
        coef = torch.linalg.solve(xtx + ridge_eps * eye, xty)
        y_hat = x @ coef
        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum(y**2)
        losses.append(ss_res / (ss_tot + 1e-8))
    return torch.stack(losses).mean()


def _corr_penalty(factors: torch.Tensor) -> torch.Tensor:
    penalties = []
    n_factors = factors.size(-1)
    eye = torch.eye(n_factors, dtype=factors.dtype, device=factors.device)
    for batch_idx in range(factors.size(0)):
        x = factors[batch_idx]
        valid = torch.isfinite(x).all(dim=1)
        x = x[valid]
        if x.size(0) < 3:
            penalties.append(torch.zeros((), dtype=factors.dtype, device=factors.device))
            continue
        centered = x - x.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False).clamp_min(1e-6)
        z = centered / std
        corr = z.transpose(0, 1) @ z / z.size(0)
        penalties.append(((corr - eye) ** 2).mean())
    return torch.stack(penalties).mean()


def _alpha_corr_loss(alpha_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    losses = []
    for batch_idx in range(alpha_pred.size(0)):
        pred = alpha_pred[batch_idx]
        target = returns[batch_idx]
        valid = torch.isfinite(pred) & torch.isfinite(target)
        pred = pred[valid]
        target = target[valid]
        if pred.size(0) < 3:
            losses.append(torch.zeros((), dtype=alpha_pred.dtype, device=alpha_pred.device))
            continue
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        pred_std = pred_centered.std(unbiased=False)
        target_std = target_centered.std(unbiased=False)
        if pred_std <= 1e-6 or target_std <= 1e-6:
            losses.append(torch.zeros((), dtype=alpha_pred.dtype, device=alpha_pred.device))
            continue
        corr = torch.mean((pred_centered / pred_std) * (target_centered / target_std))
        losses.append(1.0 - corr.clamp(-1.0, 1.0))
    return torch.stack(losses).mean()


def abc_loss(
    factors: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    beta_prev: torch.Tensor | None = None,
    lambda_mse: float = 1.0,
    lambda_r2: float = 1.0,
    lambda_alpha_corr: float = 0.0,
    lambda_corr: float = 0.01,
    lambda_to: float = 0.01,
    ridge_eps: float = 1e-4,
) -> ABCLoss:
    alpha_pred = alpha.squeeze(-1) if alpha.size(-1) == 1 else alpha.mean(dim=-1)
    mse = F.mse_loss(alpha_pred, y1)
    r2_residual = _r2_residual_loss(factors, y2, ridge_eps)
    alpha_corr = _alpha_corr_loss(alpha_pred, y1)
    corr = _corr_penalty(factors)
    turnover = torch.zeros((), dtype=factors.dtype, device=factors.device)
    if beta_prev is not None:
        turnover = F.mse_loss(beta, beta_prev)
    total = (
        lambda_mse * mse
        + lambda_r2 * r2_residual
        + lambda_alpha_corr * alpha_corr
        + lambda_corr * corr
        + lambda_to * turnover
    )
    return ABCLoss(total=total, mse=mse, r2_residual=r2_residual, alpha_corr=alpha_corr, corr=corr, turnover=turnover)
