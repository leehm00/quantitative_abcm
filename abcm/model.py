from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, beta_dim: int):
        super().__init__()
        self.beta_latent = nn.Linear(hidden_dim, beta_dim, bias=False)
        self.w1 = nn.Linear(hidden_dim, beta_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, beta_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, beta_dim, bias=False)
        self.w_out = nn.Linear(beta_dim, beta_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        latent = self.beta_latent(hidden)
        m1 = self.w1(hidden)
        m2 = self.w2(hidden)
        m3 = self.w3(hidden)
        adjacency = F.relu(torch.bmm(m1, m2.transpose(1, 2)))
        n_stocks = adjacency.size(1)
        keep = ~torch.eye(n_stocks, dtype=torch.bool, device=adjacency.device).unsqueeze(0)
        masked = adjacency.masked_fill(~keep, float("-inf"))
        weights = torch.softmax(masked, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights * keep.to(weights.dtype)
        aggregated = torch.bmm(weights, m3)
        return self.w_out(latent + aggregated)


class ABCM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        alpha_dim: int = 1,
        beta_dim: int = 12,
        gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha_dim = alpha_dim
        self.beta_dim = beta_dim
        effective_dropout = dropout if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, alpha_dim),
        )
        self.beta_head = ASTGNNLayer(hidden_dim, beta_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected x with shape (date_batch, stock_batch, lookback, feature_dim)")
        batch, n_stocks, lookback, feature_dim = x.shape
        flat = x.reshape(batch * n_stocks, lookback, feature_dim)
        gru_out, _ = self.gru(flat)
        hidden = gru_out[:, -1, :].reshape(batch, n_stocks, self.hidden_dim)
        alpha = self.alpha_head(hidden)
        beta = self.beta_head(hidden)
        factors = torch.cat([alpha, beta], dim=-1)
        return factors, alpha, beta
