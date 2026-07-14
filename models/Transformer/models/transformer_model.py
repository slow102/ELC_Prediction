import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    """Fixed position encoding that works for both short and long horizons."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(
            position * div_term[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding "
                f"length {self.encoding.size(1)}."
            )
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class WeeklyPatchEmbedding(nn.Module):
    """
    Convert a daily multivariate sequence into short history patches.

    Seven-day patches are the default because household electricity has a
    strong weekly cycle. Left padding keeps the most recent day aligned with
    the end of the final patch.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        patch_len: int = 7,
        patch_stride: int = 7,
    ):
        super().__init__()

        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive.")

        self.input_size = input_size
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.projection = nn.Linear(input_size * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                "Expected history input [batch, time, input_size], got "
                f"{tuple(x.shape)}."
            )

        time_steps = x.size(1)
        if time_steps < self.patch_len:
            left_pad = self.patch_len - time_steps
        else:
            remainder = (time_steps - self.patch_len) % self.patch_stride
            left_pad = (self.patch_stride - remainder) % self.patch_stride

        if left_pad > 0:
            # Replicate the oldest observed day instead of inserting an
            # artificial zero in standardized feature space.
            x = F.pad(
                x.transpose(1, 2),
                (left_pad, 0),
                mode="replicate",
            ).transpose(1, 2)

        # unfold -> [batch, num_patches, features, patch_len]
        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.patch_stride,
        )
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.flatten(start_dim=2)

        return self.norm(self.projection(patches))


class PowerPatchTransformer(nn.Module):
    """
    Calendar-aware encoder-decoder Transformer for direct multi-step forecasts.

    Historical multivariate observations are compressed into weekly patch
    tokens. Future calendar variables form decoder queries and cross-attend to
    the encoded history. The decoder never receives future observed power or
    future observed weather.

    A RevIN-style per-window normalization is available as an ablation option.
    Its statistics come only from the observed history of the current sample.

    The network predicts every horizon directly by default. An optional weekly
    seasonal-naive residual connection is available as an ablation setting.

    Inputs:
        x: [batch_size, input_len, input_size], standardized history
        future_time: [batch_size, pred_len, future_feature_size]

    Output:
        prediction: [batch_size, pred_len], standardized target scale
    """

    def __init__(
        self,
        input_size: int,
        future_feature_size: int,
        pred_len: int,
        target_feature_index: int,
        target_x_mean: float,
        target_x_scale: float,
        target_y_mean: float,
        target_y_scale: float,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        patch_len: int = 7,
        patch_stride: int = 7,
        use_weekly_residual: bool = False,
        use_revin: bool = False,
        revin_eps: float = 1e-5,
        max_history_len: int = 512,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}."
            )
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if not 0 <= target_feature_index < input_size:
            raise ValueError("target_feature_index is outside the input features.")
        if target_x_scale <= 0 or target_y_scale <= 0:
            raise ValueError("Scaler standard deviations must be positive.")

        self.pred_len = pred_len
        self.future_feature_size = future_feature_size
        self.target_feature_index = target_feature_index
        self.use_weekly_residual = use_weekly_residual
        self.use_revin = use_revin
        self.revin_eps = revin_eps

        self.patch_embedding = WeeklyPatchEmbedding(
            input_size=input_size,
            d_model=d_model,
            patch_len=patch_len,
            patch_stride=patch_stride,
        )
        self.history_summary_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.history_position = SinusoidalPositionEncoding(
            d_model=d_model,
            max_len=max_history_len,
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
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.future_projection = nn.Sequential(
            nn.Linear(future_feature_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.future_position = SinusoidalPositionEncoding(
            d_model=d_model,
            max_len=pred_len,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        hidden_size = max(16, d_model // 2)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.register_buffer(
            "target_x_mean",
            torch.tensor(float(target_x_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "target_x_scale",
            torch.tensor(float(target_x_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "target_y_mean",
            torch.tensor(float(target_y_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "target_y_scale",
            torch.tensor(float(target_y_scale), dtype=torch.float32),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.history_summary_token, mean=0.0, std=0.02)

        # With the seasonal residual connection enabled, training starts from
        # the weekly-naive forecast and gradually learns corrections.
        if self.use_weekly_residual:
            final_layer = self.residual_head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def _weekly_baseline(self, x: torch.Tensor) -> torch.Tensor:
        target_x_scaled = x[:, :, self.target_feature_index]
        target_original = (
            target_x_scaled * self.target_x_scale + self.target_x_mean
        )

        seasonal_period = min(7, target_original.size(1))
        last_period = target_original[:, -seasonal_period:]
        repeats = math.ceil(self.pred_len / seasonal_period)
        baseline_original = last_period.repeat(1, repeats)[:, : self.pred_len]

        return (baseline_original - self.target_y_mean) / self.target_y_scale

    def _target_context_stats(self, x: torch.Tensor):
        """Return observed-history target mean/std on the y-scaler scale."""
        target_x_scaled = x[:, :, self.target_feature_index]
        target_original = (
            target_x_scaled * self.target_x_scale + self.target_x_mean
        )
        target_y_scaled = (
            target_original - self.target_y_mean
        ) / self.target_y_scale

        context_mean = target_y_scaled.mean(dim=1, keepdim=True)
        context_std = torch.sqrt(
            target_y_scaled.var(dim=1, keepdim=True, unbiased=False)
            + self.revin_eps
        )
        return context_mean, context_std

    def _normalize_history(self, x: torch.Tensor) -> torch.Tensor:
        """RevIN-style normalization using observed history only."""
        history_mean = x.mean(dim=1, keepdim=True)
        history_std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps
        )
        return (x - history_mean) / history_std

    def forward(
        self,
        x: torch.Tensor,
        future_time: torch.Tensor,
    ) -> torch.Tensor:
        if future_time.ndim != 3:
            raise ValueError(
                "Expected future_time [batch, pred_len, features], got "
                f"{tuple(future_time.shape)}."
            )
        if future_time.size(1) != self.pred_len:
            raise ValueError(
                f"Expected {self.pred_len} future steps, "
                f"got {future_time.size(1)}."
            )
        if future_time.size(2) != self.future_feature_size:
            raise ValueError(
                f"Expected {self.future_feature_size} future features, "
                f"got {future_time.size(2)}."
            )

        encoder_x = self._normalize_history(x) if self.use_revin else x
        history_tokens = self.patch_embedding(encoder_x)
        summary_token = (
            history_tokens.mean(dim=1, keepdim=True)
            + self.history_summary_token.expand(x.size(0), -1, -1)
        )
        history_tokens = torch.cat([summary_token, history_tokens], dim=1)
        history_tokens = self.history_position(history_tokens)
        memory = self.encoder(history_tokens)

        future_queries = self.future_projection(future_time)
        future_queries = self.future_position(future_queries)

        # No causal mask is required: every decoder input is a known-in-advance
        # calendar feature, never a future target observation.
        decoded = self.decoder(tgt=future_queries, memory=memory)
        model_output = self.residual_head(decoded).squeeze(-1)

        if self.use_revin:
            context_mean, context_std = self._target_context_stats(x)
            model_output = model_output * context_std
        else:
            context_mean = None

        if self.use_weekly_residual:
            return self._weekly_baseline(x) + model_output
        if context_mean is not None:
            return context_mean + model_output
        return model_output
