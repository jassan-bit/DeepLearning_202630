"""Local, unpublished GRU/TCN candidates with the historical LSTM as control.

This module does not change the project repository.  Inputs, positive target
head, and checkpoint fields follow ``scripts/dl_models.py``.  GRU/TCN are
explicit compact project adaptations, not reproductions of a paper's fitted
network.  No architectural or optimizer search occurs in this module.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


HISTORICAL_MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "DeepLearning_202630" / "scripts" / "dl_models.py"
)
if "dl_models" in sys.modules and Path(sys.modules["dl_models"].__file__).resolve() == HISTORICAL_MODEL_FILE:
    _historical = sys.modules["dl_models"]
else:
    _name = "_historical_dl_models_for_local_comparison"
    if _name not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_name, HISTORICAL_MODEL_FILE)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Cannot read the historical models: {HISTORICAL_MODEL_FILE}")
        _historical = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _historical
        _spec.loader.exec_module(_historical)
    else:
        _historical = sys.modules[_name]


SingleLayerLSTM = _historical.SingleLayerLSTM
count_parameters = _historical.count_parameters
_WindowModel = _historical._WindowModel


class SingleLayerGRU(_WindowModel):
    """One unidirectional GRU layer; last hidden state and Softplus scalar head."""

    family = "GRU"

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        self.encoder = nn.GRU(
            input_size=self.input_features, hidden_size=self.width,
            num_layers=1, bias=True, batch_first=True, bidirectional=False,
        )
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
    """Stride-one convolution, preserving length with historical-side padding."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size,
                              stride=1, padding=0, dilation=dilation, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class ResidualTemporalBlock(nn.Module):
    """Two equally dilated causal convolutions plus identity/pointwise residual."""

    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalDilatedConv1d(input_channels, output_channels,
                                        kernel_size, dilation)
        self.conv2 = CausalDilatedConv1d(output_channels, output_channels,
                                        kernel_size, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = (nn.Identity() if input_channels == output_channels else
                         nn.Conv1d(input_channels, output_channels, 1, bias=True))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.dropout1(F.relu(self.conv1(values)))
        features = self.dropout2(F.relu(self.conv2(features)))
        return F.relu(features + self.residual(values))


class CompactResidualTCN(_WindowModel):
    """Three temporal residual blocks; dilations 1/2/4 and receptive field 29.

    A fixed architecture covers every planned D7/D14/D21/D28 window, without
    increasing its depth for a particular D. Only padding precedes the observed
    window. No normalization across time, attention, future padding, target
    lags, or exogenous features are introduced. The readout is the last time
    position, followed by the shared positive scalar head.
    """

    family = "TCN"
    kernel_size = 3
    dilations = (1, 2, 4)

    def __init__(self, history_days: int, width: int, dropout: float = .1,
                 input_features: int = 5) -> None:
        super().__init__(history_days, width, dropout, input_features)
        blocks = []
        channels = self.input_features
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
        last = self.encode(values)[:, :, -1]
        return self.output(self.head(self.dropout(last))).squeeze(-1)


def build_model(family: str, history_days: int, width: int,
                dropout: float = .1, input_features: int = 5) -> nn.Module:
    """Historical API, with GRU/TCN additions and unchanged old-model builders."""
    if not isinstance(family, str):
        raise TypeError("family must be a string")
    key = family.strip().upper()
    builders = {"GRU": SingleLayerGRU, "TCN": CompactResidualTCN}
    if key in builders:
        return builders[key](history_days, width, dropout, input_features)
    return _historical.build_model(family, history_days, width, dropout, input_features)


def architecture_spec(model: nn.Module) -> dict[str, Any]:
    if not isinstance(model, (SingleLayerGRU, CompactResidualTCN)):
        return _historical.architecture_spec(model)
    result = {
        "family": model.family, "history_days": model.history_days,
        "input_features": model.input_features,
        "input_layout": "batch,day,feature; oldest first",
        "width": model.width, "dropout": model.dropout_probability,
        "output_activation": "Softplus", "output_shape": "batch",
        "output_units": "nonnegative scaled target; inverse scale belongs to runner",
        "parameters": count_parameters(model),
        "implementation_scope": "Compact project adaptation; not a fitted-paper replica",
    }
    if isinstance(model, SingleLayerGRU):
        result.update({"recurrent_layers": 1, "bidirectional": False,
            "gates": ["reset", "update", "new"],
            "bias_convention": "PyTorch input and recurrent biases per gate",
            "readout": "last hidden state",
            "dropout_location": "last hidden state before linear head"})
    else:
        result.update({"residual_blocks": len(model.dilations),
            "convolution_layers": 2 * len(model.dilations),
            "convolutions_per_block": 2, "kernel_size": model.kernel_size,
            "block_dilations": list(model.dilations),
            "residual_connections": True,
            "first_block_projection": "pointwise 1x1 when input channels differ",
            "receptive_field_days": model.receptive_field,
            "padding": "left only: (kernel_size-1)*dilation per convolution",
            "hidden_activation": "ReLU", "readout": "last observed time position",
            "normalization": "none", "weight_normalization": False,
            "dropout_location": "after each temporal convolution and before scalar head",
            "encoder_layout": "batch,channel,day"})
    return result


def install_into(fitting_module: Any) -> None:
    """Redirect an imported historical fitting helper, confined to this process.

    No file or historical protocol is modified. The caller must supply a new
    local protocol/output directory before training and use ``architecture_spec``
    from this module when writing new model metadata.
    """
    fitting_module.build_model = build_model

