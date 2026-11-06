# Copyright 2026 The Google Research Authors (TSMixer/TiDE reference algorithms).
# Adaptations for the E2 Bitcoin D7 scalar task are documented below.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Defined-backbone D7 adapters, separate from the frozen P0 experiment.

TSMixer and TiDE are explicit PyTorch ports of the pinned Google implementations.
TimeMixer, TimesNet and PatchTST instantiate the pinned authors' modules supplied
by the caller. No module fetches code or reads observations on import or build.
All task heads predict one positive sigma7/y_scale, never OHLCV denormalization.
These are task adaptations, not exact replicas of the original forecasting task.
"""
from __future__ import annotations

from hashlib import sha256
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


MODELS = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST")
GOOGLE_COMMIT = "d3a4bd8f0eceddda553e47085033b537935b283f"
TIMEMIXER_COMMIT = "e24610583b36fdd8c76cc17a8df4e65759a5f460"
TSL_COMMIT = "4e938a1767106324dd753b2a44832bf870a0252e"
PATCH_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
SOURCE_FILES = {
    "TSMixer": {"tsmixer/tsmixer_basic/models/tsmixer.py": "84a7451f86b40b989f7ab7aa36368f24f39610ad62d7ce43e474a30a4e006159"},
    "TiDE": {"tide/models.py": "ad8fd8d498809deeed9c847a169cada42134cbd0351b03e750706132a5d151c7"},
    "TimeMixer": {"models/TimeMixer.py": "817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be"},
    "TimesNet": {"models/TimesNet.py": "f64c4bed1fd7347090044a0163bd4209c9f9ec1c1b19ceff47842df36b64bba7"},
    "PatchTST": {
        "models/PatchTST.py": "49d8bb865e1226d6338842f4b33c4bc6992cefb048dc2a4a5a8c410ca708586b",
        "layers/PatchTST_backbone.py": "df67173153787c2356bdfb6491159cd754332ef7382986efe879e1fbea8ebf26",
        "layers/PatchTST_layers.py": "21c06c70a90c60ee2a269b5c600c702834dea22cdfd72915e6b0f8b4a28db3f6",
        "layers/RevIN.py": "e64c0ccded9228b347134e7368420d3fb10f70c75145b5ff2d8bdd8c3af59df6",
    },
}
PRESET_EVIDENCE = {
    "TSMixer": {"paper": "TSMixer complete PDF Table8 ETTh1; six blocks/H512, including reported search-range inconsistency"},
    "TiDE": {
        "paper": "TiDE Table8 ETTh1; paper source structure, not official H96 recipe",
        "tide/train.py": "3edd19eed01f92ef50cadca73fbe8a793c701d34edb5ea376c6bcc55e92f0854",
        "tide/scripts/etth1.sh": "55e085f1513af3a4a0ef8f7244512ee78beba5ab09540192b3cd3bb37441981c",
    },
    "TimeMixer": {"scripts/long_term_forecast/ETT_script/TimeMixer_ETTh1_unify.sh": "13086d899b992b67cdd4409d26f686c1ac7e15e9ad4fd03dd53a2e0728554fc4"},
    "TimesNet": {
        "paper": "TimesNet long-term Table7 dimension rule, d_model32 for C5",
        "run.py": "db6dfa215520ba193fe7095380f9005949a958efa9526975410cb5d97afe3fc4",
        "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh": "8490506e32ef9ee8f3a62a4286b35be552a3035d243d87104bd02e9cc381922e",
    },
    "PatchTST": {"PatchTST_supervised/scripts/PatchTST/etth1.sh": "556c87cd7dd795094ff6cb73465e84c90686017e1535e0ba23ed43683b9b0897"},
}


def _check(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (7, 5) or len(x) < 1:
        raise ValueError("Expected nonempty [batch,7,5] D7 OHLCV windows")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("D7 inputs must be finite floating-point values")


def _moments(x):
    return torch.cat((x.mean(1), x.std(1, unbiased=False)), dim=-1)


def _keras_dense_initialization(module):
    """Match the reference Keras Dense defaults, not torch Linear defaults."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _dropout(name, candidate):
    if not isinstance(candidate, dict):
        raise TypeError("Provide the registered candidate as a dictionary")
    if candidate.get("model", name) != name:
        raise ValueError("Candidate model and requested backbone differ")
    p = candidate.get("parameters", candidate)
    if any(k in p for k in ("width", "depth", "d_model", "d_ff", "e_layers", "blocks")):
        raise ValueError("This defined architecture cannot be silently resized by a candidate")
    value = float(p.get("dropout", .1))
    if not math.isfinite(value) or not 0 <= value < 1.0:
        raise ValueError("Dropout must be finite in [0,1); the registered candidate registry controls the admitted search/source points")
    return value


def _root(name, roots, optional=False):
    value = (roots or {}).get(name)
    if value is None:
        if optional:
            return None
        raise ValueError("Explicit pinned source_roots['" + name + "'] is required")
    root = Path(value).resolve()
    if name == "PatchTST" and (root / "PatchTST_supervised").is_dir():
        root = root / "PatchTST_supervised"
    for relative, expected in SOURCE_FILES[name].items():
        path = root / relative
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned author source missing or changed: " + name + "/" + relative)
    return root


def _author_model(name, root, args):
    # The official repositories use colliding top-level namespaces. References
    # held by an already-built module remain bound to their own module globals.
    for key in list(sys.modules):
        if key in ("models", "layers", "utils") or key.startswith(("models.", "layers.", "utils.")):
            del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module("models." + name).Model(args)
    finally:
        sys.path.remove(str(root))


def _args(dropout, **overrides):
    d = dict(task_name="long_term_forecast", seq_len=7, label_len=0, pred_len=1,
             enc_in=5, dec_in=5, c_out=5, d_model=16, d_ff=32, e_layers=2,
             d_layers=1, n_heads=4, factor=3, activation="gelu", dropout=dropout,
             embed="timeF", freq="d", output_attention=False, channel_independence=1,
             decomp_method="moving_avg", moving_avg=25, use_norm=1,
             down_sampling_layers=2, down_sampling_window=2, down_sampling_method="avg",
             use_future_temporal_feature=0, top_k=4, num_kernels=6)
    d.update(overrides)
    return SimpleNamespace(**d)


class _ScalarAdapter(nn.Module):
    def _finish(self, latent, x):
        return F.softplus(self.readout(torch.cat((latent.reshape(len(x), -1), _moments(x)), dim=-1))).squeeze(-1)


class KerasAxisBatchNorm(nn.Module):
    """Keras BatchNormalization(axis=[-2,-1]) including its running variance.

    Keras momentum=.99 and epsilon=.001 are retained; statistics are over the
    batch alone for each time×channel feature. A singleton training batch is
    defined, unlike torch BatchNorm1d's training restriction for one value.
    """
    def __init__(self, shape=(7, 5), momentum=.99, epsilon=.001):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.momentum, self.epsilon = momentum, epsilon

    def forward(self, x):
        if self.training:
            mean, variance = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean.detach(), alpha=1-self.momentum)
                self.running_var.mul_(self.momentum).add_(variance.detach(), alpha=1-self.momentum)
        else:
            mean, variance = self.running_mean, self.running_var
        return (x-mean) * torch.rsqrt(variance+self.epsilon) * self.weight + self.bias


class _MixerBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.temporal_norm = KerasAxisBatchNorm()
        self.temporal = nn.Linear(7, 7)
        self.feature_norm = KerasAxisBatchNorm()
        self.feature_1 = nn.Linear(5, 512)
        self.feature_2 = nn.Linear(512, 5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(F.relu(self.temporal(self.temporal_norm(x).transpose(1, 2))).transpose(1, 2))
        return x + self.dropout(self.feature_2(self.dropout(F.relu(self.feature_1(self.feature_norm(x))))))


class TSMixerAdapter(_ScalarAdapter):
    def __init__(self, dropout):
        super().__init__()
        self.blocks = nn.Sequential(*[_MixerBlock(dropout) for _ in range(6)])
        _keras_dense_initialization(self.blocks)
        self.readout = nn.Linear(35 + 10, 1)

    def forward(self, x):
        _check(x)
        return self._finish(self.blocks(x), x)


class _DenseResidual(nn.Module):
    def __init__(self, incoming, hidden, outgoing, dropout):
        super().__init__()
        self.linear_a = nn.Linear(incoming, hidden)
        self.linear_b = nn.Linear(hidden, outgoing)
        self.linear_res = nn.Linear(incoming, outgoing)
        self.dropout = nn.Dropout(dropout)
        # TensorFlow LayerNormalization default epsilon=1e-3 is preserved.
        self.norm = nn.LayerNorm(outgoing, eps=.001)

    def forward(self, x):
        return self.norm(self.dropout(self.linear_b(F.relu(self.linear_a(x)))) + self.linear_res(x))


class TiDEAdapter(_ScalarAdapter):
    """Google TiDE with the Table8 ETTh1 H336/H720 structure and scalar bridge.

    Official num_layers=2 means hidden_dims=[256,256], hence one residual
    block (two Dense transformations) in each encoder/decoder. The temporal
    decoder emits latent8 rather than normalizing a scalar with LayerNorm(1),
    which would collapse to a constant. The final sigma head follows latent8.
    No historical or future calendar values are introduced.
    """
    def __init__(self, dropout):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(5))
        self.affine_bias = nn.Parameter(torch.zeros(5))
        self.series_embedding = nn.Embedding(5, 16)
        self.feature_encoder = _DenseResidual(5, 64, 4, dropout)
        self.encoder = _DenseResidual(7 + 7*4 + 4 + 16, 256, 256, dropout)
        self.decoder = _DenseResidual(256, 256, 8, dropout)
        self.temporal_decoder = _DenseResidual(8 + 4, 128, 8, dropout)
        self.residual_projection = nn.Linear(7, 1)
        _keras_dense_initialization(self)
        nn.init.uniform_(self.series_embedding.weight, -.05, .05)
        self.readout = nn.Linear(5*8 + 10, 1)

    def latent_features(self, x):
        mean, std = x.mean(1, keepdim=True), x.std(1, unbiased=False, keepdim=True)
        normalized = (x-mean) / torch.where(std == 0, torch.ones_like(std), std)
        normalized = normalized * self.affine_weight + self.affine_bias
        past_cov = self.feature_encoder(x).reshape(len(x), -1)
        # Absent future covariates have a fixed zero sentinel, not OHLCV truth.
        future_cov = self.feature_encoder(x.new_zeros((len(x), 1, 5))).squeeze(1)
        channel_ids = torch.arange(5, device=x.device)
        emb = self.series_embedding(channel_ids).unsqueeze(0).expand(len(x), -1, -1)
        past = normalized.transpose(1, 2)
        joined = torch.cat((past, past_cov[:, None].expand(-1, 5, -1),
                            future_cov[:, None].expand(-1, 5, -1), emb), dim=-1)
        decoded = self.decoder(self.encoder(joined))
        temporal = self.temporal_decoder(torch.cat((decoded, future_cov[:, None].expand(-1, 5, -1)), dim=-1))
        return temporal + self.residual_projection(past)

    def forward(self, x):
        _check(x)
        return self._finish(self.latent_features(x), x)


class TimeMixerAdapter(_ScalarAdapter):
    def __init__(self, root, dropout):
        super().__init__()
        self.backbone = _author_model("TimeMixer", root, _args(dropout))
        del self.backbone.projection_layer
        if hasattr(self.backbone.enc_embedding, "temporal_embedding"):
            del self.backbone.enc_embedding.temporal_embedding
        for block in self.backbone.pdm_blocks:
            del block.layer_norm  # Registered upstream but not called upstream.
        self.readout = nn.Linear(5*16 + 10, 1)

    def latent_features(self, x):
        b = len(x)
        levels, _ = self.backbone._Model__multi_scale_process_inputs(x, None)
        if [level.shape[1] for level in levels] != [7, 3, 1]:
            raise RuntimeError("Registered three-level D7 pooling changed")
        encoded = []
        for i, values in enumerate(levels):
            normalized = self.backbone.normalize_layers[i](values, "norm")
            normalized = normalized.permute(0, 2, 1).reshape(b*5, values.shape[1], 1)
            encoded.append(self.backbone.enc_embedding(normalized, None))
        for block in self.backbone.pdm_blocks:
            encoded = block(encoded)
        return sum(self.backbone.predict_layers[i](z.transpose(1, 2)).squeeze(-1).reshape(b, 5, 16)
                   for i, z in enumerate(encoded))

    def forward(self, x):
        _check(x)
        return self._finish(self.latent_features(x), x)


class _EquivalentInception(nn.Module):
    """Exact linear identity for the author's average of odd-kernel convolutions.

    Six independently parameterized source kernels remain registered/trainable.
    Center-padding their weights and averaging before the convolution equals
    averaging their outputs. This changes implementation cost, not the function
    class, parameter count or gradient paths. Equality is tested in float64.
    """
    def __init__(self, source):
        super().__init__()
        self.kernels = source.kernels
        self.padding = max(k.kernel_size[0] for k in self.kernels) // 2
        for k in self.kernels:
            if (k.stride != (1, 1) or k.dilation != (1, 1) or k.groups != 1
                    or k.kernel_size[0] != k.kernel_size[1]
                    or k.padding != (k.kernel_size[0]//2,)*2 or k.padding_mode != "zeros"):
                raise ValueError("Inception kernel fusion requires the pinned odd-kernel source convention")

    def effective_parameters(self):
        largest = 2*self.padding+1
        weights = []
        for k in self.kernels:
            pad = (largest-k.kernel_size[0])//2
            weights.append(F.pad(k.weight, (pad, pad, pad, pad)))
        weight = torch.stack(weights).mean(0)
        biases = [k.bias for k in self.kernels]
        if any(b is None for b in biases):
            if not all(b is None for b in biases):
                raise ValueError("Mixed bias conventions are not part of the pinned Inception source")
            bias = None
        else:
            bias = torch.stack(biases).mean(0)
        return weight, bias

    def prepared_forward(self, x, parameters):
        return F.conv2d(x, parameters[0], parameters[1], padding=self.padding)

    def forward(self, x):
        return self.prepared_forward(x, self.effective_parameters())


class _CanonicalTimesBlock(nn.Module):
    """All four non-DC L8 frequencies; no batch-averaged period selection.

    The source extends D7 to eight learned tokens before the FFT. Source k=5
    would include DC/divide by zero. k=4 includes every admissible frequency,
    so its set is fixed and the exact source weighted aggregation can be
    evaluated in canonical frequency order, independently for every example.
    """
    def __init__(self, original):
        super().__init__()
        self.conv = nn.Sequential(_EquivalentInception(original.conv[0]),
                                  original.conv[1], _EquivalentInception(original.conv[2]))
        self.frequency_indices = (1, 2, 3, 4)
        self.periods = (8, 4, 2, 2)

    def forward(self, x):
        b, length, channels = x.shape
        if length != 8:
            raise ValueError("TimesNet D7+H1 latent calendar must have length8")
        amplitude = torch.fft.rfft(x, dim=1).abs().mean(dim=-1)[:, 1:5]
        weights = amplitude.softmax(dim=1)
        prepared_1 = self.conv[0].effective_parameters()
        prepared_2 = self.conv[2].effective_parameters()
        results = {}
        # Frequencies3 and4 share the source floor(8/frequency) period2.
        for period in sorted(set(self.periods), reverse=True):
            padded_length = math.ceil(length/period)*period
            values = F.pad(x, (0, 0, 0, padded_length-length))
            grid = values.reshape(b, padded_length//period, period, channels).permute(0, 3, 1, 2).contiguous()
            mixed = self.conv[0].prepared_forward(grid, prepared_1)
            mixed = self.conv[1](mixed)
            mixed = self.conv[2].prepared_forward(mixed, prepared_2)
            mixed = mixed.permute(0, 2, 3, 1).reshape(b, padded_length, channels)
            results[period] = mixed[:, :length]
        branches = torch.stack([results[p] for p in self.periods], dim=-1)
        return x + (branches * weights[:, None, None]).sum(-1)


class TimesNetAdapter(_ScalarAdapter):
    def __init__(self, root, dropout):
        super().__init__()
        # Paper Table7 d_model rule gives32 for C5. FF32 and six Inception
        # kernels are resolved by pinned official ETTh1 script/run.py, with
        # this combined provenance explicit rather than an invented paper row.
        self.backbone = _author_model("TimesNet", root, _args(dropout, d_model=32, d_ff=32))
        self.backbone.model = nn.ModuleList([_CanonicalTimesBlock(block) for block in self.backbone.model])
        del self.backbone.projection
        if hasattr(self.backbone.enc_embedding, "temporal_embedding"):
            del self.backbone.enc_embedding.temporal_embedding
        self.readout = nn.Linear(32 + 10, 1)

    def latent_features(self, x):
        mean = x.mean(1, keepdim=True).detach()
        std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        encoded = self.backbone.enc_embedding((x-mean)/std, None)
        encoded = self.backbone.predict_linear(encoded.transpose(1, 2)).transpose(1, 2)
        for block in self.backbone.model:
            encoded = self.backbone.layer_norm(block(encoded))
        return encoded[:, -1]

    def forward(self, x):
        _check(x)
        return self._finish(self.latent_features(x), x)


class PatchTSTAdapter(_ScalarAdapter):
    def __init__(self, root, dropout):
        super().__init__()
        args = _args(dropout, seq_len=16, e_layers=3, d_ff=128, fc_dropout=dropout,
                     head_dropout=dropout, individual=False, patch_len=16, stride=8,
                     padding_patch="end", revin=True, affine=True, subtract_last=False,
                     decomposition=False, kernel_size=25)
        author = _author_model("PatchTST", root, args)
        self.backbone = author.model
        del self.backbone.head
        if self.backbone.backbone.patch_num != 2:
            raise RuntimeError("Patch16/stride8 padded-history representation must produce two patches")
        self.readout = nn.Linear(5*16*2 + 10, 1)

    def latent_features(self, x):
        # Statistics use the seven actual days, before deterministic padding.
        normalized = self.backbone.revin_layer(x, "norm").transpose(1, 2)
        padded = F.pad(normalized, (9, 0), mode="replicate")
        padded = self.backbone.padding_patch_layer(padded)
        patches = padded.unfold(-1, 16, 8).permute(0, 1, 3, 2)
        return self.backbone.backbone(patches)

    def forward(self, x):
        _check(x)
        return self._finish(self.latent_features(x), x)


def build_model(name, *, candidate, source_roots=None):
    if name not in MODELS:
        raise ValueError("This module supports only the five named defined backbones")
    dropout = _dropout(name, candidate)
    root = _root(name, source_roots, optional=name in ("TSMixer", "TiDE"))
    if name == "TSMixer":
        model = TSMixerAdapter(dropout)
        structural = dict(mixer_blocks=6, feature_hidden=512, activation="ReLU",
                          normalization="Keras BatchNorm axis(time,channel), pre-norm, momentum.99,epsilon.001",
                          backbone_dense_initialization="Keras GlorotUniform weights, zero biases")
        changes = ["Input length7/channels5", "Original temporal OHLCV output head replaced by Linear45,1+Softplus"]
    elif name == "TiDE":
        model = TiDEAdapter(dropout)
        structural = dict(hidden_size=256, source_num_layers_flag=2, actual_encoder_residual_blocks=1,
                          actual_decoder_residual_blocks=1, decoder_output_dim=8,
                          temporal_decoder_hidden=128, temporal_covariate_encoder_hidden=64,
                          temporal_width=4, layer_norm=True, RevIN=True,
                          learned_series_embedding=16, source_reference_horizons=[336, 720],
                          backbone_dense_initialization="Keras GlorotUniform weights, zero biases",
                          series_embedding_initialization="Keras uniform[-.05,.05]")
        changes = ["Five OHLCV channel identities; past covariates reuse sameD7, absent future covariates fixedzero",
                   "Temporal decoder output latent8 with LN8 replaces source scalaroutput1/LN1 to avoid constant collapse",
                   "Global residual7to1 broadcast to latent8; Linear50,1+Softplus; no OHLCV inverse normalization"]
    elif name == "TimeMixer":
        model = TimeMixerAdapter(root, dropout)
        structural = dict(d_model=16, d_ff=32, e_layers=2, down_sampling_layers=2,
                          pool_window=2, scales=[7, 3, 1], moving_avg=25, channel_independence=1)
        changes = ["Published three pooling levels would produce length0; maximum validD7 pooling depth2 used",
                   "Retain PDM and scale-specific FMM latent projections; Linear90,1+Softplus, no OHLCV denorm"]
    elif name == "TimesNet":
        model = TimesNetAdapter(root, dropout)
        structural = dict(d_model=32, d_ff=32, e_layers=2, num_kernels=6,
                          learned_temporal_extension=[7, 8], top_k=4, period_floor_rule=True,
                          frequency_indices=[1, 2, 3, 4], periods=[8, 4, 2, 2],
                          Inception_implementation="Exact centered-kernel fusion; all six original parameterizations retained")
        changes = ["Paper C5d_model32; FF32 from pinnedETTh1script and num_kernels6 from pinnedrun.py",
                   "Source learns D7to8 before FFT: topk5 bounded to allfour nonDC frequencies",
                   "Canonical allfrequency aggregation removes source batch-dependent topk and DC division",
                   "Full Inception2D blocks retained; final learned-token latent32 plus sameD7moments->Linear42,1+Softplus"]
    else:
        model = PatchTSTAdapter(root, dropout)
        structural = dict(d_model=16, d_ff=128, e_layers=3, n_heads=4, patch_len=16,
                          stride=8, padded_context=16, actual_observed_context=7,
                          original_end_padding=8, patches=2, residual_attention=True,
                          batchnorm=True, revin_affine=True)
        changes = ["Normalize only observedD7; left edge replication9 to fit publishedpatch16, then original rightpadding8",
                   "Original channel-independent residual-attention encoder retained, flatten160+moments10->Linear170,1+Softplus",
                   "Replicated edges are deterministic functions, not additional historical/future observations"]
    commits = {"TSMixer": GOOGLE_COMMIT, "TiDE": GOOGLE_COMMIT, "TimeMixer": TIMEMIXER_COMMIT,
               "TimesNet": TSL_COMMIT, "PatchTST": PATCH_COMMIT}
    model.architecture_metadata = {
        "model": name, "history_days": 7, "channels": 5, "horizon": 1,
        "target": "sigma7 / train-only y_scale", "dropout": dropout,
        "source_commit": commits[name], "source_files": SOURCE_FILES[name],
        "structural_preset_evidence": PRESET_EVIDENCE[name],
        "source_files_verified_at_build": root is not None,
        "adapter_module": "scripts/e2_final_models_mlp.py",
        "adapter_module_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "structural_parameters": structural, "task_adaptations": changes,
        "target_lags_added": False, "calendar_covariates_added": False,
        "future_values_consumed": False, "original_OHLCV_denormalization_used": False,
        "exact_original_task_replica": False,
        "registered_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return model
