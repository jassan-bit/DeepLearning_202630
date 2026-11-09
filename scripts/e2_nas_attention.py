"""New D7 NAS attention families, separate from frozen recipe implementations.

Only architecture construction and tensor operations are implemented here. No
market loader, search, optimizer or downloaded source/weights is used. PatchTST
is the own three-width variant specified by E2 section 27, not an assertion of
numerical equivalence with the published fixed-width PatchTST implementation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .e2_nas_common import NASConfig, validate_input, moment_features, block_dropout
except ImportError:
    from e2_nas_common import NASConfig, validate_input, moment_features, block_dropout


class WindowRevIN(nn.Module):
    """Normalize each real seven-day window before replication padding.

Means and population deviations are detached, as in the source RevIN operation.
The affine parameters belong to the five OHLCV variables. The scalar volatility
readout is never denormalized as if it were one of the five input variables.
No statistics are retained across calls or estimated across batch examples.
"""
    def __init__(self, channels: int = 5, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.epsilon).detach()
        return ((x - mean) / scale) * self.weight + self.bias


class PatchEncoderBlock(nn.Module):
    """One post-norm PatchTST block; no attention scores cross block boundaries."""
    def __init__(self, width, config, index):
        super().__init__()
        self.width = width
        self.heads = width // 16
        self.feedforward_width = 8 * width
        self.attention = nn.MultiheadAttention(width, self.heads, dropout=0.0, batch_first=True)
        self.attention_dropout = block_dropout(config, index)
        self.attention_norm = nn.BatchNorm1d(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 8 * width), nn.GELU(), nn.Linear(8 * width, width))
        self.feedforward_dropout = block_dropout(config, index)
        self.feedforward_norm = nn.BatchNorm1d(width)
        self.residual_attention = False

    @staticmethod
    def _norm(norm, x):
        # The source applies BatchNorm to feature channels over batch and patch
        # positions. B=1 is valid: five variables each contribute two patches.
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        value, _ = self.attention(x, x, x, need_weights=False)
        x = self._norm(self.attention_norm, x + self.attention_dropout(value))
        return self._norm(self.feedforward_norm, x + self.feedforward_dropout(self.feedforward(x)))


class PatchTSTNAS(nn.Module):
    """RevIN -> two replicated D7 patches -> three variable-width encoders.

Channels share the encoder; mixing the five variables happens only in the
joint scalar readout. The ten extra features are moments of the original
scaled D7 input, not of the normalized or padded sequence.
"""
    recipe_id = "PT-D7-NAS-v1"

    def __init__(self, config: NASConfig):
        super().__init__()
        if not isinstance(config, NASConfig):
            raise TypeError("An explicit NASConfig is required")
        self.config = config
        self.revin = WindowRevIN()
        w1, w2, w3 = config.widths
        self.patch_length = 16
        self.patch_stride = 8
        self.patch_count = 2
        self.left_padding, self.right_padding = 9, 8
        self.patch_projection = nn.Linear(16, w1)
        self.position = nn.Parameter(torch.empty(2, w1))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.blocks = nn.ModuleList(PatchEncoderBlock(w, config, i) for i, w in enumerate(config.widths))
        self.projections = nn.ModuleList((nn.Linear(w1, w2), nn.Linear(w2, w3)))
        self.head = nn.Linear(5 * 2 * w3 + 10, 1)

    def normalize_and_patch(self, x):
        validate_input(x)
        normalized = self.revin(x)
        padded = F.pad(normalized.transpose(1, 2), (self.left_padding, self.right_padding), mode="replicate")
        # [B, variables, two patches, patch length16].
        return padded.unfold(-1, self.patch_length, self.patch_stride)

    def encode(self, x):
        patches = self.normalize_and_patch(x)
        batch = x.shape[0]
        z = self.patch_projection(patches.reshape(batch * 5, 2, 16)) + self.position
        z = self.blocks[0](z)
        for projection, block in zip(self.projections, self.blocks[1:]):
            z = block(projection(z))
        # Source-style readout ordering: variable, latent feature, patch.
        return z.reshape(batch, 5, 2, self.config.widths[-1]).transpose(2, 3).reshape(batch, -1)

    def forward(self, x):
        validate_input(x)
        latent = self.encode(x)
        return F.softplus(self.head(torch.cat((latent, moment_features(x)), dim=1))).squeeze(-1)


def _validate_prefix(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] < 1
            or x.shape[-1] != 5 or not 1 <= x.shape[1] <= 7):
        raise ValueError("Expected a nonempty transformed [B,1..7,5] prefix")
    if (not torch.is_floating_point(x)
            or (x.device.type != "meta" and not bool(torch.isfinite(x).all()))):
        raise ValueError("Prefix inputs must be finite floating-point values")


class HybridNAS(nn.Module):
    """Causal CNN -> reset unidirectional LSTM -> additive temporal attention.

Prefix causality concerns the already transformed D7 tensor; the common
relative_window transform uses the last known close across the whole window.
The final attention reads all seven observed states, never a future target.
"""
    recipe_id = "HIB-D7-NAS-v1"

    def __init__(self, config: NASConfig, *, kernel=3):
        super().__init__()
        if not isinstance(config, NASConfig):
            raise TypeError("An explicit NASConfig is required")
        if type(kernel) is not int or kernel not in (2, 3, 5):
            raise ValueError("Hybrid kernel must be one of the frozen HPO values 2,3,5")
        self.config, self.kernel = config, kernel
        w1, w2, w3 = config.widths
        self.cnn = nn.Conv1d(5, w1, kernel_size=kernel)
        self.cnn_dropout = block_dropout(config, 0)
        self.lstm = nn.LSTM(w1, w2, num_layers=1, batch_first=True, bidirectional=False)
        self.sequence_dropout = block_dropout(config, 1)
        self.attention_projection = nn.Linear(w2, w3)
        self.attention_score = nn.Linear(w3, 1, bias=False)
        self.context_dropout = block_dropout(config, 2)
        self.head = nn.Linear(w2, 1)

    def encode_steps(self, x):
        _validate_prefix(x)
        z = F.relu(self.cnn(F.pad(x.transpose(1, 2), (self.kernel - 1, 0)))).transpose(1, 2)
        z = self.cnn_dropout(z)
        # No state is accepted or saved: hidden/cell are fresh zeros each call.
        z, _ = self.lstm(z)
        return self.sequence_dropout(z)

    def forward_with_attention(self, x):
        validate_input(x)
        z = self.encode_steps(x)
        score = self.attention_score(torch.tanh(self.attention_projection(z))).squeeze(-1)
        weights = score.softmax(dim=1)
        context = (z * weights.unsqueeze(-1)).sum(dim=1)
        prediction = F.softplus(self.head(self.context_dropout(context))).squeeze(-1)
        return prediction, weights

    def forward(self, x):
        return self.forward_with_attention(x)[0]
