"""Small neural references for the guide's temporal-regression route.

Every model accepts the same oldest-first ``(batch, history, features)`` tensor
and returns one nonnegative value per example. Preprocessing and inversion of
the target scale belong to the runner. In particular, a Softplus head requires
a nonnegative target representation, such as positive multiplicative scaling;
mean-centering the target would conflict with that output constraint.

These are project architectures, not replicas of the wide/dilated TCN or the
attention architectures described in the documentary literature profiles.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _WindowModel(nn.Module):
    """Shared input contract; no model silently reshapes or adds information."""

    family: str

    def __init__(self, history_days: int, width: int, dropout: float,
                 input_features: int) -> None:
        super().__init__()
        self.history_days = _positive_integer(history_days, "history_days")
        self.width = _positive_integer(width, "width")
        self.input_features = _positive_integer(input_features, "input_features")
        self.dropout_probability = float(dropout)
        if not math.isfinite(self.dropout_probability) or not 0 <= self.dropout_probability < 1:
            raise ValueError("dropout must be finite and in [0, 1)")

    def _check_input(self, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")
        expected = (self.history_days, self.input_features)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"input must have shape (batch, {expected[0]}, {expected[1]})")
        if values.shape[0] < 1:
            raise ValueError("input batch must not be empty")
        if not torch.is_floating_point(values):
            raise TypeError("input must have a floating-point dtype")


class SmallMLP(_WindowModel):
    """One hidden layer applied to the flattened, ordered covariate window."""

    family = "MLP"

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.history_days * self.input_features, self.width),
            nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.width, 1),
            nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        return self.network(values).squeeze(-1)


class SingleLayerLSTM(_WindowModel):
    """One unidirectional recurrent layer and a last-state regression head."""

    family = "LSTM"

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        # A one-layer LSTM has no inter-layer dropout. Apply it to the last
        # hidden state instead; PyTorch supplies input and recurrent biases.
        self.encoder = nn.LSTM(
            input_size=self.input_features, hidden_size=self.width,
            num_layers=1, bias=True, batch_first=True, bidirectional=False,
        )
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Linear(self.width, 1)
        self.output = nn.Softplus()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        _, (hidden, _) = self.encoder(values)
        return self.output(self.head(self.dropout(hidden[-1]))).squeeze(-1)


class _CausalConv1d(nn.Module):
    """Ordinary convolution with padding exclusively on the historical side."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size,
                              stride=1, padding=0, dilation=1, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class ShallowCausalCNN1D(_WindowModel):
    """Two shallow causal convolutions, temporal mean pooling and scalar head.

    There are no dilations, residual blocks, attention layers or hidden input
    sources. ``encode`` exposes the time-indexed features for a causality check;
    pooling uses the entire already-observed input window for its final forecast.
    """

    family = "CNN1D"
    kernel_size = 3

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        self.conv1 = _CausalConv1d(self.input_features, self.width, self.kernel_size)
        self.conv2 = _CausalConv1d(self.width, self.width, self.kernel_size)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Linear(self.width, 1)
        self.output = nn.Softplus()

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        features = values.transpose(1, 2)
        features = self.dropout(F.relu(self.conv1(features)))
        return self.dropout(F.relu(self.conv2(features)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.encode(values)
        return self.output(self.head(features.mean(dim=-1))).squeeze(-1)


def build_model(family: str, history_days: int, width: int,
                dropout: float = .1, input_features: int = 5) -> nn.Module:
    """Construct one family without selecting a seed, dtype, device or scaler."""
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    builders = {"MLP": SmallMLP, "LSTM": SingleLayerLSTM,
                "CNN1D": ShallowCausalCNN1D}
    try:
        builder = builders[family.strip().upper()]
    except KeyError as exc:
        raise ValueError(f"unknown model family {family!r}; choose MLP, LSTM or CNN1D") from exc
    return builder(history_days, width, dropout, input_features)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count trainable and frozen parameters, excluding nonparameter buffers."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "non_trainable": non_trainable,
            "total": trainable + non_trainable}


def architecture_spec(model: nn.Module) -> dict[str, Any]:
    """Return JSON-compatible architecture metadata without constructing a model."""
    if not isinstance(model, _WindowModel):
        raise TypeError("architecture_spec expects a model built by build_model")
    result: dict[str, Any] = {
        "family": model.family, "history_days": model.history_days,
        "input_features": model.input_features, "input_layout": "batch,day,feature; oldest first",
        "width": model.width, "dropout": model.dropout_probability,
        "output_activation": "Softplus", "output_shape": "batch",
        "output_units": "nonnegative scaled target; inverse scale belongs to runner",
        "parameters": count_parameters(model),
    }
    if isinstance(model, SmallMLP):
        result.update({"hidden_layers": 1, "hidden_activation": "ReLU",
                       "input_view": "flattened ordered window"})
    elif isinstance(model, SingleLayerLSTM):
        result.update({"recurrent_layers": 1, "bidirectional": False,
                       "readout": "last hidden state", "bias_convention": "two biases per gate",
                       "dropout_location": "last hidden state before linear head"})
    else:
        result.update({"convolution_layers": 2, "kernel_size": 3,
                       "dilation": 1, "padding": "left only, two values per layer",
                       "hidden_activation": "ReLU", "pooling": "mean over observed input window",
                       "residual_connections": False, "encoder_layout": "batch,channel,day"})
    return result
