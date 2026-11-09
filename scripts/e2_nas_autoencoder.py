"""Independent SAE/TiDE NAS variants for the versioned E2 search design.

These are new three-group architectures, not replacements for historical
recipe bodies or replications of published results.  Inputs are already
fold-scaled D7 windows.  Outputs are positive predictions of sigma7/a; the
caller alone fits a on training labels and restores original units.

No data loaders, fitting loops, downloaded weights, or market access occur.
TiDE follows the residual encoder/decoder idea of Das et al. (2023); the
fixed feature encoder and scalar bridge are documented in section 27.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .e2_nas_common import NASConfig, block_dropout, moment_features, validate_input
except ImportError:
    from e2_nas_common import NASConfig, block_dropout, moment_features, validate_input


def _check_sequence(x: Tensor, channels: int) -> None:
    if (not isinstance(x, Tensor) or x.ndim != 3 or x.shape[0] < 1
            or tuple(x.shape[1:]) != (7, channels)):
        raise ValueError(f'Expected nonempty [B,7,{channels}] windows')
    if not x.is_floating_point() or (x.device.type != 'meta' and not bool(torch.isfinite(x).all())):
        raise ValueError('Expected finite floating-point windows')


def _lstm(incoming: int, hidden: int) -> nn.LSTM:
    """Fresh native PyTorch uniform-initialized LSTM, no retained state.

    This is a declared new-NAS initialization policy, not the orthogonal/
    forget-bias initialization of the historical source recipe.
    """
    return nn.LSTM(incoming, hidden, batch_first=True)


def _dense(incoming: int, outgoing: int) -> nn.Linear:
    layer = nn.Linear(incoming, outgoing)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _last(encoder: nn.LSTM, x: Tensor) -> Tensor:
    return encoder(x)[1][0][-1]


def _repeat7(x: Tensor) -> Tensor:
    return x.unsqueeze(1).expand(-1, 7, -1)


class SAENAS(nn.Module):
    """Two transferred LSTM encoders followed by the third dense NAS group."""

    def __init__(self, config: NASConfig):
        super().__init__()
        self.config = config
        w1, w2, w3 = config.widths
        self.encoder1, self.encoder2 = _lstm(5, w1), _lstm(w1, w2)
        self.dropout1, self.dropout2 = block_dropout(config, 0), block_dropout(config, 1)
        self.dense = _dense(w2, w3)
        self.dropout3 = block_dropout(config, 2)
        self.readout = _dense(w3, 1)

    def latent_features(self, x: Tensor) -> Tensor:
        validate_input(x)
        h1 = self.dropout1(_last(self.encoder1, x))
        h2 = self.dropout2(_last(self.encoder2, _repeat7(h1)))
        return self.dropout3(F.relu(self.dense(h2)))

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(self.readout(self.latent_features(x))).squeeze(-1)


class SAEAutoencoderNAS(nn.Module):
    """One auxiliary group: LSTM encoder, repeated latent and mirror decoder.

    The decoder has width w_i and projects to the original five channels.
    The group's bit controls dropout on the encoded vector and on decoder
    outputs immediately before the five-channel tanh reconstruction.
    """

    def __init__(self, config: NASConfig, phase: int):
        super().__init__()
        if type(phase) is not int or phase not in (1, 2):
            raise ValueError('Auxiliary phase must be 1 or 2')
        self.config, self.phase = config, phase
        incoming = 5 if phase == 1 else config.widths[0]
        hidden = config.widths[phase - 1]
        self.encoder, self.decoder = _lstm(incoming, hidden), _lstm(hidden, hidden)
        self.encoder_dropout = block_dropout(config, phase - 1)
        self.decoder_dropout = block_dropout(config, phase - 1)
        self.reconstruction = _dense(hidden, 5)

    def encode(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.encoder.input_size)
        return self.encoder_dropout(_last(self.encoder, x))

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.encode(x)
        decoded = self.decoder(_repeat7(hidden))[0]
        return torch.tanh(self.reconstruction(self.decoder_dropout(decoded)))


class SAEPretrainingNAS:
    """Fold-local staging only; the caller fits each phase with a fresh optimizer.

    train_x must contain training windows only.  The supplied fold identifier
    records caller provenance and does not itself certify chronological splits.
    Both auxiliary targets are the original train_x/s, while encoder inputs
    remain unscaled.  The stages are kept outside the inference predictor.
    """

    def __init__(self, config: NASConfig, train_x: Tensor, *, fold_id: str):
        validate_input(train_x)
        if not isinstance(fold_id, str) or not fold_id.strip():
            raise ValueError('A nonempty training-fold identifier is required')
        self.config, self.fold_id = config, fold_id
        self._train_x = train_x.detach().clone()
        self.scale = max(1.0, float(self._train_x.abs().max()))
        self.ae1 = SAEAutoencoderNAS(config, 1).to(device=train_x.device, dtype=train_x.dtype)
        self.ae2 = SAEAutoencoderNAS(config, 2).to(device=train_x.device, dtype=train_x.dtype)
        self.phase, self._phase2_x = 'ae1', None

    def reconstruction_target(self, x: Tensor) -> Tensor:
        validate_input(x)
        return x / self.scale

    def phase1_data(self) -> tuple[Tensor, Tensor]:
        if self.phase != 'ae1':
            raise RuntimeError('AE1 data are only exposed in phase one')
        return self._train_x.clone(), self.reconstruction_target(self._train_x).clone()

    def _frozen_phase1_inputs(self, x: Tensor) -> Tensor:
        validate_input(x)
        self.ae1.eval()
        self.ae1.requires_grad_(False)
        with torch.no_grad():
            return _repeat7(self.ae1.encode(x)).detach().clone()

    def prepare_phase2(self) -> tuple[Tensor, Tensor]:
        if self.phase != 'ae1':
            raise RuntimeError('Prepare phase two once, after fitting AE1')
        self._phase2_x = self._frozen_phase1_inputs(self._train_x)
        self.phase = 'ae2'
        return self.phase2_data()

    def phase2_data(self) -> tuple[Tensor, Tensor]:
        if self.phase != 'ae2':
            raise RuntimeError('Prepare phase two before requesting its data')
        return self._phase2_x.clone(), self.reconstruction_target(self._train_x).clone()

    def phase2_inputs(self, x: Tensor) -> Tensor:
        """Transform other admissible inputs without refitting scale or AE1."""
        if self.phase != 'ae2':
            raise RuntimeError('Prepare phase two before encoding other windows')
        return self._frozen_phase1_inputs(x)

    def transfer_predictor(self) -> SAENAS:
        if self.phase != 'ae2':
            raise RuntimeError('Prepare and fit both auxiliary phases before transfer')
        predictor = SAENAS(self.config).to(device=self._train_x.device, dtype=self._train_x.dtype)
        predictor.encoder1 = deepcopy(self.ae1.encoder)
        predictor.encoder2 = deepcopy(self.ae2.encoder)
        predictor.requires_grad_(True)
        predictor.train()
        return predictor


def build_auxiliary_stages(config: NASConfig, train_x: Tensor, *, fold_id: str) -> SAEPretrainingNAS:
    return SAEPretrainingNAS(config, train_x, fold_id=fold_id)


class _DenseResidual(nn.Module):
    """Two Dense maps plus projection residual; no hidden stochastic dropout."""

    def __init__(self, incoming: int, hidden: int, outgoing: int):
        super().__init__()
        self.linear_a = _dense(incoming, hidden)
        self.linear_b = _dense(hidden, outgoing)
        self.linear_res = _dense(incoming, outgoing)
        self.norm = nn.LayerNorm(outgoing, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.linear_b(F.relu(self.linear_a(x))) + self.linear_res(x))


class TiDENAS(nn.Module):
    """Three residual groups with independent encoder/decoder/temporal widths."""

    def __init__(self, config: NASConfig):
        super().__init__()
        self.config = config
        w1, w2, w3 = config.widths
        self.affine_weight = nn.Parameter(torch.ones(5))
        self.affine_bias = nn.Parameter(torch.zeros(5))
        self.series_embedding = nn.Embedding(5, 16)
        nn.init.uniform_(self.series_embedding.weight, -0.05, 0.05)
        self.feature_encoder = _DenseResidual(5, 64, 4)
        self.encoder = _DenseResidual(55, w1, w1)
        self.decoder = _DenseResidual(w1, w2, 8)
        self.temporal_decoder = _DenseResidual(12, w3, 8)
        self.dropout1, self.dropout2 = block_dropout(config, 0), block_dropout(config, 1)
        self.dropout3 = block_dropout(config, 2)
        self.residual_projection = _dense(7, 1)
        self.readout = _dense(50, 1)

    @staticmethod
    def future_covariate_sentinel(x: Tensor) -> Tensor:
        validate_input(x)
        return x.new_zeros((len(x), 1, 5))

    def latent_features(self, x: Tensor) -> Tensor:
        validate_input(x)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.where(std == 0, torch.ones_like(std), std)
        normalized = normalized * self.affine_weight + self.affine_bias
        past = normalized.transpose(1, 2)
        past_cov = self.feature_encoder(x).reshape(len(x), -1)
        future_cov = self.feature_encoder(self.future_covariate_sentinel(x)).squeeze(1)
        ids = torch.arange(5, device=x.device)
        identities = self.series_embedding(ids).unsqueeze(0).expand(len(x), -1, -1)
        joined = torch.cat((past, past_cov[:, None].expand(-1, 5, -1),
                            future_cov[:, None].expand(-1, 5, -1), identities), dim=-1)
        encoded = self.dropout1(self.encoder(joined))
        decoded = self.dropout2(self.decoder(encoded))
        temporal_input = torch.cat((decoded, future_cov[:, None].expand(-1, 5, -1)), dim=-1)
        temporal = self.dropout3(self.temporal_decoder(temporal_input))
        return temporal + self.residual_projection(past)

    def forward(self, x: Tensor) -> Tensor:
        latent = self.latent_features(x).reshape(len(x), 40)
        joined = torch.cat((latent, moment_features(x)), dim=-1)
        return F.softplus(self.readout(joined)).squeeze(-1)
