"""Network definition for the proposed Improved-model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMultiTimeEncoding(nn.Module):
    """mTAN-style continuous encoding with learnable frequencies and phases."""

    def __init__(self, d_time: int = 16):
        super().__init__()
        if d_time < 2:
            raise ValueError("d_time must be at least 2.")
        seed_periods = torch.tensor(
            [7.0, 14.0, 30.0, 60.0, 90.0, 182.621, 365.2425],
            dtype=torch.float32,
        )
        periods = seed_periods.repeat(math.ceil((d_time - 1) / len(seed_periods)))
        periods = periods[: d_time - 1]
        self.frequency = nn.Parameter(2.0 * math.pi / periods)
        self.phase = nn.Parameter(torch.zeros(d_time - 1))
        self.linear_weight = nn.Parameter(torch.ones(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time is measured in days.  The linear coordinate is bounded while
        # the periodic coordinates retain physically meaningful day periods.
        linear = time.unsqueeze(-1) / 365.2425
        linear = linear * self.linear_weight + self.linear_bias
        periodic = torch.sin(
            time.unsqueeze(-1) * self.frequency + self.phase
        )
        return torch.cat([linear, periodic], dim=-1)


class MultiScaleDecompositionEncoder(nn.Module):
    """Separate local, medium and slow components before temporal attention."""

    def __init__(self, input_size: int, d_model: int, dropout: float):
        super().__init__()
        self.variable_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.GELU(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid(),
        )
        self.local_conv = nn.Conv1d(
            input_size, d_model, kernel_size=3, padding=2, dilation=2,
        )
        self.weekly_conv = nn.Conv1d(
            input_size, d_model, kernel_size=3, padding=7, dilation=7,
        )
        self.medium_projection = nn.Linear(input_size, d_model)
        self.trend_projection = nn.Linear(input_size, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
        # [B,T,C] -> channel-wise moving average, preserving length.
        left = (kernel - 1) // 2
        right = kernel // 2
        values = F.pad(x.transpose(1, 2), (left, right), mode="replicate")
        return F.avg_pool1d(values, kernel_size=kernel, stride=1).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.variable_gate(x.mean(dim=1)).unsqueeze(1)
        x = x * (0.5 + gate)

        smooth7 = self._moving_average(x, 7)
        smooth30 = self._moving_average(x, 30)
        local = x - smooth7
        medium = smooth7 - smooth30

        local_t = local.transpose(1, 2)
        local_short = self.local_conv(local_t).transpose(1, 2)
        local_weekly = self.weekly_conv(local_t).transpose(1, 2)
        medium_token = self.medium_projection(medium)
        trend_token = self.trend_projection(smooth30)
        return self.fusion(torch.cat(
            [local_short, local_weekly, medium_token, trend_token], dim=-1,
        ))


class ImprovedModel(nn.Module):
    """Proposed hybrid model for 90-to-90 and 90-to-365 prediction.

    The last future feature is a leakage-free standardized seasonal baseline.
    The neural part predicts only a horizon-gated correction.  Its final layer
    is zero-initialized, so the untrained network exactly equals that baseline.
    """

    def __init__(
        self,
        input_size: int,
        future_feature_size: int,
        pred_len: int,
        d_model: int = 48,
        d_time: int = 16,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 96,
        dropout: float = 0.15,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if future_feature_size < 2:
            raise ValueError("future features must include calendar and baseline.")

        self.pred_len = pred_len
        self.future_feature_size = future_feature_size
        self.decomposition = MultiScaleDecompositionEncoder(
            input_size, d_model, dropout,
        )
        self.time_encoding = LearnableMultiTimeEncoding(d_time=d_time)
        self.history_time_projection = nn.Linear(d_time, d_model)
        self.future_projection = nn.Sequential(
            nn.Linear(future_feature_size - 1 + d_time, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.decoder_norm = nn.LayerNorm(d_model)

        # Horizon-adaptive gate learns how much short-memory correction is
        # credible as the forecast gets farther from the observed context.
        self.horizon_gate = nn.Sequential(
            nn.Linear(d_model + 2, max(12, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(12, d_model // 2), 1),
            nn.Sigmoid(),
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(self, x: torch.Tensor, future_time: torch.Tensor) -> torch.Tensor:
        if future_time.shape[1:] != (self.pred_len, self.future_feature_size):
            raise ValueError(
                f"Expected future shape [B,{self.pred_len},"
                f"{self.future_feature_size}], got {tuple(future_time.shape)}."
            )
        batch, history_len, _ = x.shape
        device, dtype = x.device, x.dtype

        history_tokens = self.decomposition(x)
        history_days = torch.arange(
            1 - history_len, 1, device=device, dtype=dtype,
        ).unsqueeze(0).expand(batch, -1)
        history_tokens = history_tokens + self.history_time_projection(
            self.time_encoding(history_days)
        )
        memory = self.encoder(history_tokens)

        future_days = torch.arange(
            1, self.pred_len + 1, device=device, dtype=dtype,
        ).unsqueeze(0).expand(batch, -1)
        future_known = future_time[:, :, :-1]
        baseline = future_time[:, :, -1]
        query = self.future_projection(torch.cat(
            [future_known, self.time_encoding(future_days)], dim=-1,
        ))
        attended, _ = self.cross_attention(query, memory, memory)
        decoded = self.decoder_norm(query + attended)
        decoded = self.decoder_norm(decoded + self.decoder_ffn(decoded))

        horizon_fraction = future_days.unsqueeze(-1) / float(self.pred_len)
        log_fraction = torch.log1p(future_days).unsqueeze(-1) / math.log1p(
            self.pred_len
        )
        gate = self.horizon_gate(torch.cat(
            [decoded, horizon_fraction, log_fraction], dim=-1,
        )).squeeze(-1)
        correction = self.correction_head(decoded).squeeze(-1)
        return baseline + gate * correction
