"""Development candidate from the mathematical specification of E3.

The backbone uses PyTorch primitives only. The candidate mechanism combines a
fixed causal-reference discrepancy penalty in attention logits with radial
saturation of values. Content attention, CNN and LSTM are established building
blocks, not claimed inventions. Literature novelty, student ownership/defence
and statistical superiority remain unverified.

Input: already transformed/standardized OHLCV, shape (batch, days, 5), oldest
first. Output: positive sigma7 / training_target_scale, shape (batch,). The
training fold's target scale must be applied outside this module.

The certified Lipschitz bound concerns the latent head H -> prediction only.
It does not certify the complete map from raw OHLCV through preprocessing,
CNN and LSTM. The bound assumes fixed inference parameters and fixed P.

Implementation was written with AI assistance from the equations in the E3
design note. No third-party model implementation or pretrained model is used.
The author must understand and defend the delivered algorithm and derivation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


ABLATIONS = ("full", "without_penalty", "without_saturation", "without_content")


@dataclass(frozen=True)
class CandidateConfig:
    """Immutable software configuration, not a preregistered experiment design."""

    history_days: int = 7
    input_features: int = 5
    cnn_channels: int = 8
    hidden_size: int = 12
    kernel_size: int = 3
    reference_lookback: int = 3
    radius: float = 1.0
    epsilon: float = 1e-4
    query_bound: float = 1.0
    readout_bound: float = 4.0
    penalty_bound: float = 1.0
    initial_penalty: float = 0.25
    temperature_min: float = 0.25
    initial_temperature: float = 1.0
    ablation: str = "full"

    def __post_init__(self):
        for name in ("history_days", "input_features", "cnn_channels", "hidden_size",
                     "kernel_size", "reference_lookback"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_features != 5:
            raise ValueError("The E3 input contract has five OHLCV channels")
        for name in ("radius", "epsilon", "query_bound", "readout_bound", "penalty_bound",
                     "initial_penalty", "temperature_min", "initial_temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.initial_penalty < self.penalty_bound:
            raise ValueError("initial_penalty must be strictly below penalty_bound")
        if not self.initial_temperature > self.temperature_min:
            raise ValueError("initial_temperature must exceed temperature_min")
        if self.ablation not in ABLATIONS:
            raise ValueError("Unknown isolated ablation")


def validate_input(values: torch.Tensor, history_days: int = 7,
                   input_features: int = 5) -> None:
    """External serving guard, also required before invoking a traced artifact.

    This validates tensor shape/type/finiteness, not physical OHLCV ranges or
    out-of-distribution distance. The input must already have fold-frozen
    preprocessing applied; this function never estimates a scaler.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if (values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != history_days
            or values.shape[2] != input_features):
        raise ValueError(f"Expected nonempty batch x {history_days} x {input_features}")
    if not values.is_floating_point():
        raise TypeError("Input must have a floating dtype")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Input must be finite")


def causal_reference_matrix(days: int, lookback: int) -> torch.Tensor:
    """P[j,k] averages observed states max(0,j-lookback+1)..j, independent of H."""
    if type(days) is not int or type(lookback) is not int or days < 1 or lookback < 1:
        raise ValueError("days and lookback must be positive integers")
    matrix = torch.zeros(days, days)
    for day in range(days):
        first = max(0, day - lookback + 1)
        matrix[day, first:day + 1] = 1.0 / (day - first + 1)
    return matrix


def _bounded_vector(raw: torch.Tensor, bound: float) -> torch.Tensor:
    """Smooth norm bound: ||bound*raw/sqrt(1+||raw||²)|| < bound."""
    return bound * raw / torch.sqrt(1.0 + raw.square().sum())


class CausalEncoder(nn.Module):
    """Causal convolution and unidirectional LSTM; each state uses its prefix."""

    def __init__(self, config: CandidateConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(config.input_features, config.cnn_channels,
                              config.kernel_size, padding=0)
        self.recurrent = nn.LSTM(config.cnn_channels, config.hidden_size,
                                 batch_first=True, bidirectional=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            validate_input(values, self.config.history_days, self.config.input_features)
        channels = values.transpose(1, 2)
        channels = F.pad(channels, (self.config.kernel_size - 1, 0))
        features = torch.tanh(self.conv(channels)).transpose(1, 2)
        return self.recurrent(features)[0]


class BoundedDiscordanceAttention(nn.Module):
    """Latent head implementing the stated equations without a model library.

    Three isolated functional ablations remove penalty, radial saturation or
    content logits. They investigate three components; they do not establish
    three novel contributions. Only penalty+saturation are candidate changes.
    """

    def __init__(self, config: CandidateConfig):
        super().__init__()
        self.config = config
        self.use_penalty = config.ablation != "without_penalty"
        self.use_saturation = config.ablation != "without_saturation"
        self.use_content = config.ablation != "without_content"
        self.register_buffer("P", causal_reference_matrix(config.history_days,
                                                          config.reference_lookback))
        if self.use_content:
            self.query_raw = nn.Parameter(torch.randn(config.hidden_size) * 0.1)
        else:
            self.register_parameter("query_raw", None)
        self.readout_raw = nn.Parameter(torch.randn(config.hidden_size) * 0.2)
        self.bias = nn.Parameter(torch.zeros(()))
        if self.use_penalty:
            ratio = config.initial_penalty / config.penalty_bound
            self.penalty_raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        else:
            self.register_parameter("penalty_raw", None)
        gap = config.initial_temperature - config.temperature_min
        raw_temperature = gap if gap > 20 else math.log(math.expm1(gap))
        self.temperature_raw = nn.Parameter(torch.tensor(raw_temperature))

    def effective_parameters(self) -> dict[str, torch.Tensor]:
        """Constraints are structural for every optimizer step, not post-hoc claims."""
        q = (_bounded_vector(self.query_raw, self.config.query_bound) if self.use_content
             else self.readout_raw.new_zeros(self.config.hidden_size))
        w = _bounded_vector(self.readout_raw, self.config.readout_bound)
        penalty = (self.config.penalty_bound * torch.sigmoid(self.penalty_raw)
                   if self.use_penalty else self.bias.new_zeros(()))
        temperature = self.config.temperature_min + F.softplus(self.temperature_raw)
        return {"q": q, "w": w, "penalty": penalty, "temperature": temperature}

    def attention_components(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.jit.is_tracing():
            if (not isinstance(states, torch.Tensor) or states.ndim != 3
                    or states.shape[0] < 1 or states.shape[1] != self.config.history_days
                    or states.shape[2] != self.config.hidden_size):
                raise ValueError("Expected nonempty batch x history_days x hidden_size")
            if not states.is_floating_point() or not bool(torch.isfinite(states).all()):
                raise ValueError("Latent states must be floating and finite")
        pars = self.effective_parameters()
        reference = torch.matmul(self.P, states)
        distance = torch.sqrt((states - reference).square().sum(dim=-1)
                              + self.config.epsilon ** 2)
        content = torch.matmul(states, pars["q"])
        logits = content - pars["penalty"] * distance
        weights = torch.softmax(logits / pars["temperature"], dim=-1)
        values = (self.config.radius * states /
                  torch.sqrt(self.config.radius ** 2 + states.square().sum(dim=-1, keepdim=True))
                  if self.use_saturation else states)
        context = (weights.unsqueeze(-1) * values).sum(dim=1)
        return {"reference": reference, "distance": distance, "content": content,
                "weights": weights, "values": values, "context": context}

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        context = self.attention_components(states)["context"]
        return F.softplus(torch.matmul(context, self.effective_parameters()["w"]) + self.bias)

    @torch.no_grad()
    def latent_lipschitz_bound(self) -> float:
        """Upper bound for this head on latent H, under fixed inference weights.

        Identity values (without_saturation) do not satisfy this global bound.
        P must remain fixed and input independent. Bias changes do not affect it.
        Multiplying output by target_scale multiplies this constant as well.
        """
        if not self.use_saturation:
            raise ValueError("The stated global bound requires radial saturation")
        n = self.config.history_days
        operator = torch.eye(n, dtype=self.P.dtype, device=self.P.device) - self.P
        norm = float(torch.linalg.matrix_norm(operator, ord=2))
        query_bound = self.config.query_bound if self.use_content else 0.0
        penalty_bound = self.config.penalty_bound if self.use_penalty else 0.0
        return self.config.readout_bound * (
            1.0 + self.config.radius * math.sqrt(n) / (2.0 * self.config.temperature_min)
            * (query_bound + penalty_bound * norm))


class OriginalModel(nn.Module):
    """Candidate CNN -> LSTM -> bounded discordance head, for development only."""

    def __init__(self, config: CandidateConfig | None = None, *, seed: int = 0):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        self.config = CandidateConfig() if config is None else config
        if not isinstance(self.config, CandidateConfig):
            raise TypeError("config must be CandidateConfig")
        self.initialization_seed = seed
        # CPU initialization only. default_generator.manual_seed avoids changing
        # CUDA generators, unlike torch.manual_seed. fork_rng restores CPU state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = CausalEncoder(self.config)
            self.mechanism = BoundedDiscordanceAttention(self.config)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mechanism(self.encode(values))

    def configuration(self) -> dict:
        return {"config": asdict(self.config), "initialization_seed": self.initialization_seed,
                "status": "development_candidate_not_preregistered",
                "output": "sigma7/training_target_scale",
                "novelty_verified": False, "statistical_superiority_verified": False}
