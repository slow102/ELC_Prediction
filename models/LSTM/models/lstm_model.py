import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Encoder-decoder LSTM for multi-step daily power prediction.

    Input:
        x: [batch_size, input_len, input_size]
        future_time: [batch_size, pred_len, future_feature_size]

    Output:
        out: [batch_size, pred_len]
    """
    def __init__(
        self,
        input_size: int,
        future_feature_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        pred_len: int = 90,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=future_feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.pred_len = pred_len

    def forward(
        self,
        x: torch.Tensor,
        future_time: torch.Tensor,
    ) -> torch.Tensor:
        if future_time.size(1) != self.pred_len:
            raise ValueError(
                f"Expected {self.pred_len} future steps, "
                f"got {future_time.size(1)}."
            )

        _, encoder_state = self.encoder(x)
        decoder_out, _ = self.decoder(future_time, encoder_state)
        return self.output_head(decoder_out).squeeze(-1)
