"""New D7 spectral/multiscale NAS families, not author-result replications.

No loaders, optimizers or historical model mutations are supplied here. TimesNet
keeps period-2D Inception mixing; TimeMixer keeps moving-average decomposition,
bottom-up season mixing, top-down trend mixing and scale-specific FMM.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .e2_nas_common import NASConfig, validate_input, moment_features, block_dropout
except ImportError:
    from e2_nas_common import NASConfig, validate_input, moment_features, block_dropout


def _choice(value, allowed, name):
    if type(value) is not int or value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}")
    return value


class _OddKernelInception(nn.Module):
    """Independent source-style odd kernels, fused by exact linear identity.

    This retains every kernel parameter and its gradient. Zero-padding and
    averaging weights before convolution is identical to averaging the outputs.
    No kernels or widths are dropped to make the NAS cheaper.
    """
    def __init__(self, in_channels, out_channels, num_kernels):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 2*i+1, padding=i)
            for i in range(num_kernels)
        ])
        self.padding = num_kernels-1
        for kernel in self.kernels:
            nn.init.kaiming_normal_(kernel.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(kernel.bias)

    def effective_parameters(self):
        size = 2*self.padding+1
        weight = None
        bias = None
        for kernel in self.kernels:
            pad = (size-kernel.kernel_size[0])//2
            padded = F.pad(kernel.weight, (pad, pad, pad, pad))
            weight = padded if weight is None else weight+padded
            bias = kernel.bias if bias is None else bias+kernel.bias
        return weight/len(self.kernels), bias/len(self.kernels)

    def forward(self, values):
        weight, bias = self.effective_parameters()
        return F.conv2d(values, weight, bias, padding=self.padding)


class _NASPeriodBlock(nn.Module):
    frequency_indices = (1, 2, 3, 4)
    periods = (8, 4, 2, 2)

    def __init__(self, width, top_k, num_kernels):
        super().__init__()
        self.width, self.top_k = width, top_k
        self.inception_in = _OddKernelInception(width, 2*width, num_kernels)
        self.inception_out = _OddKernelInception(2*width, width, num_kernels)
        self.norm = nn.LayerNorm(width)

    def period_weights(self, values):
        # Every example selects its own frequencies. Other examples in a batch
        # cannot change its top-k set. Stable ordering gives deterministic ties.
        amplitude = torch.fft.rfft(values, dim=1).abs().mean(-1)[:, 1:5]
        selected = torch.argsort(amplitude, dim=1, descending=True, stable=True)[:, :self.top_k]
        mask = torch.zeros_like(amplitude, dtype=torch.bool).scatter(1, selected, True)
        return amplitude.masked_fill(~mask, -torch.inf).softmax(1)

    def forward(self, values):
        if values.shape[1:] != (8, self.width):
            raise ValueError("TimesBlock requires eight learned tokens at its declared width")
        weights = self.period_weights(values)
        first = self.inception_in.effective_parameters()
        second = self.inception_out.effective_parameters()
        by_period = {}
        for period in (8, 4, 2):
            grid = values.reshape(len(values), 8//period, period, self.width).permute(0, 3, 1, 2)
            mixed = F.conv2d(grid, first[0], first[1], padding=self.inception_in.padding)
            mixed = F.gelu(mixed)
            mixed = F.conv2d(mixed, second[0], second[1], padding=self.inception_out.padding)
            by_period[period] = mixed.permute(0, 2, 3, 1).reshape(len(values), 8, self.width)
        branches = torch.stack([by_period[p] for p in self.periods], dim=-1)
        return self.norm(values+(branches*weights[:, None, None, :]).sum(-1))


def _sinusoidal(length, width):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    scales = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)*(-math.log(10000.0)/width))
    values = torch.zeros(length, width)
    values[:, 0::2] = torch.sin(positions*scales)
    values[:, 1::2] = torch.cos(positions*scales)
    return values[None]


class TimesNetNAS(nn.Module):
    """Three variable-width TimesBlocks for the explicit TN-D7-NAS-v1 family."""
    def __init__(self, config: NASConfig, *, top_k=4, num_kernels=6):
        super().__init__()
        top_k = _choice(top_k, (2, 3, 4), "top_k")
        num_kernels = _choice(num_kernels, (2, 4, 6), "num_kernels")
        self.config = config
        self.top_k, self.num_kernels = top_k, num_kernels
        # Fixed source-interface dimensions are explicit: source D32 embedding,
        # then the registered channel projection into each searched block.
        self.token_embedding = nn.Conv1d(5, 32, 3, padding=1, padding_mode="circular", bias=False)
        self.register_buffer("position_embedding", _sinusoidal(7, 32))
        self.temporal_projection = nn.Linear(7, 8)
        incoming = (32,)+tuple(config.widths[:-1])
        self.channel_projections = nn.ModuleList(nn.Linear(a, b) for a, b in zip(incoming, config.widths))
        self.blocks = nn.ModuleList(_NASPeriodBlock(w, top_k, num_kernels) for w in config.widths)
        self.dropouts = nn.ModuleList(block_dropout(config, i) for i in range(3))
        self.readout = nn.Linear(config.widths[-1]+10, 1)

    def latent_features(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False)+1e-5)
        values = self.token_embedding(((x-mean)/std).transpose(1, 2)).transpose(1, 2)
        values = values+self.position_embedding.to(dtype=values.dtype)
        values = self.temporal_projection(values.transpose(1, 2)).transpose(1, 2)
        for projection, block, dropout in zip(self.channel_projections, self.blocks, self.dropouts):
            values = dropout(block(projection(values)))
        return values[:, -1]

    def forward(self, x):
        validate_input(x)
        latent = self.latent_features(x)
        return F.softplus(self.readout(torch.cat((latent, moment_features(x)), -1))).squeeze(-1)


class _ScaleNormalize(nn.Module):
    """Per-window source-style mean/variance and five learned channel affines."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(5))
        self.bias = nn.Parameter(torch.zeros(5))

    def forward(self, x):
        mean = x.mean(1, keepdim=True).detach()
        scale = torch.sqrt(x.var(1, keepdim=True, unbiased=False)+1e-5).detach()
        return ((x-mean)/scale)*self.weight+self.bias


def _moving_average(values, window):
    # values N,T,D. Replicate edge values; padding never adds observations.
    half = (window-1)//2
    return F.avg_pool1d(F.pad(values.transpose(1, 2), (half, half), mode="replicate"),
                        window, stride=1).transpose(1, 2)


class _NASPDMBlock(nn.Module):
    """Channel-independent PDM with the source's two directional scale mixers."""
    lengths = (7, 3, 1)

    def __init__(self, width, moving_average):
        super().__init__()
        self.width, self.moving_average = width, moving_average
        self.season_down = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))
            for a, b in zip(self.lengths[:-1], self.lengths[1:])
        )
        self.trend_up = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, b))
            for a, b in ((1, 3), (3, 7))
        )
        self.feature_ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def decompose(self, levels):
        trends = [_moving_average(values, self.moving_average) for values in levels]
        seasons = [values-trend for values, trend in zip(levels, trends)]
        return seasons, trends

    def mix_seasons(self, seasons):
        high = seasons[0].transpose(1, 2)
        result = [high.transpose(1, 2)]
        for i, layer in enumerate(self.season_down):
            high = seasons[i+1].transpose(1, 2)+layer(high)
            result.append(high.transpose(1, 2))
        return result

    def mix_trends(self, trends):
        low = trends[-1].transpose(1, 2)
        result = [low.transpose(1, 2)]
        for i, layer in enumerate(self.trend_up):
            low = trends[-2-i].transpose(1, 2)+layer(low)
            result.append(low.transpose(1, 2))
        return list(reversed(result))

    def forward(self, levels):
        if [x.shape[1] for x in levels] != list(self.lengths):
            raise ValueError("PDM requires the three registered scales 7/3/1")
        seasons, trends = self.decompose(levels)
        seasons, trends = self.mix_seasons(seasons), self.mix_trends(trends)
        return [original+self.feature_ff(season+trend)
                for original, season, trend in zip(levels, seasons, trends)]


class TimeMixerNAS(nn.Module):
    """Three PDM groups with variable channels and scale-specific FMM latents."""
    scale_lengths = (7, 3, 1)

    def __init__(self, config: NASConfig, *, moving_average=7):
        super().__init__()
        moving_average = _choice(moving_average, (3, 7, 25), "moving_average")
        self.config, self.moving_average = config, moving_average
        self.normalizers = nn.ModuleList(_ScaleNormalize() for _ in self.scale_lengths)
        self.token_embedding = nn.Conv1d(1, 16, 3, padding=1, padding_mode="circular", bias=False)
        incoming = (16,)+tuple(config.widths[:-1])
        self.channel_projections = nn.ModuleList(nn.Linear(a, b) for a, b in zip(incoming, config.widths))
        self.blocks = nn.ModuleList(_NASPDMBlock(w, moving_average) for w in config.widths)
        self.dropouts = nn.ModuleList(block_dropout(config, i) for i in range(3))
        self.fmm = nn.ModuleList(nn.Linear(length, 1) for length in self.scale_lengths)
        self.readout = nn.Linear(5*config.widths[-1]+10, 1)

    def input_scales(self, x):
        fine = x.transpose(1, 2)
        medium = F.avg_pool1d(fine, 2, stride=2)
        coarse = F.avg_pool1d(medium, 2, stride=2)
        return [values.transpose(1, 2) for values in (fine, medium, coarse)]

    def latent_features(self, x):
        batch = len(x)
        values = []
        for normalizer, level in zip(self.normalizers, self.input_scales(x)):
            level = normalizer(level).transpose(1, 2).reshape(batch*5, level.shape[1], 1)
            values.append(self.token_embedding(level.transpose(1, 2)).transpose(1, 2))
        for projection, block, dropout in zip(self.channel_projections, self.blocks, self.dropouts):
            values = [projection(level) for level in values]
            values = [dropout(level) for level in block(values)]
        return sum(layer(level.transpose(1, 2)).squeeze(-1).reshape(batch, 5, self.config.widths[-1])
                   for layer, level in zip(self.fmm, values)).reshape(batch, -1)

    def forward(self, x):
        validate_input(x)
        return F.softplus(self.readout(torch.cat((self.latent_features(x), moment_features(x)), -1))).squeeze(-1)
