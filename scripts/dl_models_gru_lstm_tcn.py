"""Portable public adapter for the LSTM, GRU and TCN benchmark.

Published after the recorded local fits: this file has a different source hash
from the frozen training implementation. The architecture/parameter contracts
are equivalent; original training provenance belongs to the archived manifests.
Every model consumes an oldest-first (batch, days, five OHLCV features) tensor
and emits one nonnegative scaled-target value. Scaling belongs to the runner.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dl_models import (
    SingleLayerLSTM,
    _WindowModel,
    architecture_spec as _historical_architecture_spec,
    count_parameters,
)

__all__ = ["SingleLayerLSTM", "SingleLayerGRU", "CompactResidualTCN",
           "build_model", "architecture_spec", "count_parameters"]
FAMILIES = ("LSTM", "GRU", "TCN")


class SingleLayerGRU(_WindowModel):
    """One unidirectional GRU and a last-hidden-state Softplus regression head."""

    family = "GRU"

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        self.encoder = nn.GRU(input_size=self.input_features, hidden_size=self.width,
            num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Linear(self.width, 1)
        self.output = nn.Softplus()

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        return self.encoder(values)[0].transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        _, hidden = self.encoder(values)
        return self.output(self.head(self.dropout(hidden[-1]))).squeeze(-1)


class CausalDilatedConv1d(nn.Module):
    """Length-preserving convolution with historical-side padding only."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size,
                              stride=1, padding=0, dilation=dilation, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalDilatedConv1d(input_channels, output_channels, kernel_size, dilation)
        self.conv2 = CausalDilatedConv1d(output_channels, output_channels, kernel_size, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (nn.Identity() if input_channels == output_channels else
                         nn.Conv1d(input_channels, output_channels, 1, bias=True))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.dropout1(F.relu(self.conv1(values)))
        features = self.dropout2(F.relu(self.conv2(features)))
        return F.relu(features + self.residual(values))


class CompactResidualTCN(_WindowModel):
    """Three residual blocks, two kernel-3 convolutions each, dilations 1/2/4.

    The fixed 29-day receptive field covers D7/D14/D21/D28. The scalar head uses
    the last observed time position. No weight/time normalization is introduced.
    This compact project architecture is not a fitted-paper replica.
    """

    family = "TCN"
    kernel_size = 3
    dilations = (1, 2, 4)

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        blocks, channels = [], self.input_features
        for dilation in self.dilations:
            blocks.append(ResidualTemporalBlock(channels, self.width,
                self.kernel_size, dilation, self.dropout_probability))
            channels = self.width
        self.encoder = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Linear(self.width, 1)
        self.output = nn.Softplus()

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        self._check_input(values)
        return self.encoder(values.transpose(1, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.head(self.dropout(self.encode(values)[:, :, -1]))).squeeze(-1)


def build_model(family: str, history_days: int, width: int,
                dropout: float = .1, input_features: int = 5) -> nn.Module:
    """Build only a member of the current LSTM/GRU/TCN benchmark."""
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    builders = {"LSTM": SingleLayerLSTM, "GRU": SingleLayerGRU, "TCN": CompactResidualTCN}
    try:
        builder = builders[family.strip().upper()]
    except KeyError as exc:
        raise ValueError(f"unknown model family {family!r}; choose LSTM, GRU or TCN") from exc
    return builder(history_days, width, dropout, input_features)


def architecture_spec(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, SingleLayerLSTM):
        return _historical_architecture_spec(model)
    if not isinstance(model, (SingleLayerGRU, CompactResidualTCN)):
        raise TypeError("Expected a model constructed by this adapter")
    result = {"family": model.family, "history_days": model.history_days,
        "input_features": model.input_features, "input_layout": "batch,day,feature; oldest first",
        "width": model.width, "dropout": model.dropout_probability,
        "output_activation": "Softplus", "output_shape": "batch",
        "output_units": "nonnegative scaled target; inverse scale belongs to runner",
        "parameters": count_parameters(model),
        "implementation_scope": "Compact project adaptation; not a fitted-paper replica"}
    if isinstance(model, SingleLayerGRU):
        result.update({"recurrent_layers": 1, "bidirectional": False,
            "control_gates": ["reset", "update"], "candidate_projection": "new",
            "bias_convention": "PyTorch input and recurrent biases per gate",
            "readout": "last hidden state",
            "dropout_location": "last hidden state before linear head"})
    else:
        result.update({"residual_blocks": 3, "convolution_layers": 6,
            "convolutions_per_block": 2, "kernel_size": 3,
            "block_dilations": [1, 2, 4], "residual_connections": True,
            "first_block_projection": "pointwise 1x1 when input channels differ",
            "receptive_field_days": 29,
            "padding": "left only: (kernel_size-1)*dilation per convolution",
            "hidden_activation": "ReLU", "readout": "last observed time position",
            "normalization": "none", "weight_normalization": False,
            "dropout_location": "after each temporal convolution and before scalar head",
            "encoder_layout": "batch,channel,day"})
    return result
