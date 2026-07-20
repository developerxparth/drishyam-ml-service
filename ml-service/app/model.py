"""
model.py
--------
The LSTM itself. Deliberately small and univariate (rainfall only) —
appropriate for a monthly, single-state series with maybe 200-300 data
points. This is the "AI engine" referenced on your PPT's tech-stack slide.

Input:  a window of SEQ_LEN consecutive monthly rainfall values (normalized)
Output: the predicted next month's rainfall value (normalized)
"""

import torch
import torch.nn as nn

SEQ_LEN = 12  # one year of monthly history predicts the next month


class RainfallLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, SEQ_LEN, 1)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]          # (batch, hidden_size)
        return self.head(last_step)        # (batch, 1)


def make_sequences(values, seq_len=SEQ_LEN):
    """
    Turns a flat list of monthly rainfall values into (X, y) training
    pairs: each X is a window of seq_len months, y is the month right
    after it.
    """
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return X, y


def normalize(values, vmin, vmax):
    span = (vmax - vmin) or 1.0
    return [(v - vmin) / span for v in values]


def denormalize(value, vmin, vmax):
    span = (vmax - vmin) or 1.0
    return value * span + vmin


def load_model(path: str, hidden_size: int = 32) -> RainfallLSTM:
    model = RainfallLSTM(hidden_size=hidden_size)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
