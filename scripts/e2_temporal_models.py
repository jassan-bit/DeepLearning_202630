"""Explicit D7 OHLCV-to-volatility adapters for a registered development pilot.

Published author modules provide the encoders. This file defines the task bridge:
no calendar input, no OHLCV forecasting/denormalization head, and one positive
scalar in units Y / mean(Y_train). These are adaptations, not paper replicas.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

MODELS = ("TimeMixer", "iTransformer", "PatchTST", "GRU")
TOP_MODELS = MODELS[:3]
AUTHOR_COMMITS = {"TimeMixer": "e24610583b36fdd8c76cc17a8df4e65759a5f460",
                  "iTransformer": "4e938a1767106324dd753b2a44832bf870a0252e",
                  "PatchTST": "4e938a1767106324dd753b2a44832bf870a0252e"}


def _author(name, source_root, args):
    root = Path(source_root).resolve()
    if not (root / "models" / (name + ".py")).is_file():
        raise ValueError("Missing fixed author source for " + name)
    for key in list(sys.modules):
        if key in ("models", "layers", "utils") or key.startswith(("models.", "layers.", "utils.")):
            del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("models." + name)
        return module.Model(args, patch_len=2, stride=1) if name == "PatchTST" else module.Model(args)
    finally:
        sys.path.remove(str(root))


def _args(dropout):
    return SimpleNamespace(task_name="long_term_forecast", seq_len=7, label_len=0,
        pred_len=1, enc_in=5, dec_in=5, c_out=5, e_layers=2, d_layers=1,
        d_model=16, d_ff=32, n_heads=4, factor=3, activation="gelu",
        dropout=dropout, embed="timeF", freq="d", output_attention=False,
        channel_independence=1, decomp_method="moving_avg", moving_avg=25,
        use_norm=1, down_sampling_layers=1, down_sampling_window=2,
        down_sampling_method="avg", use_future_temporal_feature=0, top_k=2)


class D7VolatilityAdapter(nn.Module):
    """A published encoder plus the same 90-to-1 positive task readout."""
    def __init__(self, name, source_root, dropout=.1):
        super().__init__()
        if name not in TOP_MODELS or dropout != .1:
            raise ValueError("Only the fixed pilot architecture/dropout is registered")
        self.family = name
        self.backbone = _author(name, source_root, _args(dropout))
        self.head = nn.Linear(5 * 16 + 10, 1)
        self.output = nn.Softplus()
        self.dropout_probability = dropout
        self._remove_unused_author_outputs()

    def _remove_unused_author_outputs(self):
        # These heads would emit covariates, not volatility; do not retain them
        # as apparently active parameters in this task adapter.
        if self.family == "TimeMixer":
            del self.backbone.projection_layer
            # Historical calendar marks are never passed under the D7 contract.
            if hasattr(self.backbone.enc_embedding, "temporal_embedding"):
                del self.backbone.enc_embedding.temporal_embedding
            # The pinned author's PDM registers but never uses this LayerNorm.
            for block in self.backbone.pdm_blocks:
                if hasattr(block, "layer_norm"):
                    del block.layer_norm
        elif self.family == "iTransformer":
            del self.backbone.projection
        else:
            del self.backbone.head

    @staticmethod
    def _check(x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (7, 5) or not len(x):
            raise ValueError("Expected nonempty [batch,7days,5OHLCV] input")
        if not torch.is_floating_point(x) or not torch.isfinite(x).all():
            raise ValueError("Adapter inputs must be finite floating-point values")

    def latent_features(self, x):
        self._check(x)
        b = len(x)
        if self.family == "TimeMixer":
            levels, _ = self.backbone._Model__multi_scale_process_inputs(x, None)
            encoded = []
            for i, values in enumerate(levels):
                normalized = self.backbone.normalize_layers[i](values, "norm")
                normalized = normalized.permute(0, 2, 1).contiguous().reshape(b * 5, values.shape[1], 1)
                encoded.append(self.backbone.enc_embedding(normalized, None))
            for block in self.backbone.pdm_blocks:
                encoded = block(encoded)
            # Keep the author's scale-specific temporal forecast projections
            # and additive FMM, but fuse LATENTS before the old covariate head.
            fused = [self.backbone.predict_layers[i](z.permute(0, 2, 1))
                     .squeeze(-1).reshape(b, 5, 16) for i, z in enumerate(encoded)]
            return torch.stack(fused, dim=0).sum(dim=0)
        mean = x.mean(dim=1, keepdim=True).detach()
        standard = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normalized = (x - mean) / standard
        if self.family == "iTransformer":
            embedded = self.backbone.enc_embedding(normalized, None)
            encoded, _ = self.backbone.encoder(embedded, attn_mask=None)
            return encoded
        embedded, variables = self.backbone.patch_embedding(normalized.permute(0, 2, 1))
        encoded, _ = self.backbone.encoder(embedded)
        # Right replication padding yields seven patch tokens with P2/S1.
        return encoded.mean(dim=1).reshape(b, variables, 16)

    def forward(self, x):
        latent = self.latent_features(x)
        # Derived moments preserve amplitude removed by instance normalization;
        # they use the SAME observed window, not new channels or target lags.
        moments = torch.cat((x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=1)
        joined = torch.cat((latent.reshape(len(x), -1), moments), dim=1)
        if joined.shape[1] != 90:
            raise RuntimeError("Latent/moment task readout contract changed")
        return self.output(self.head(joined)).squeeze(-1)


def build_model(name, source_root=None, *, dropout=.1):
    if name == "GRU":
        from dl_models_gru_lstm_tcn import build_model as inherited
        model = inherited("GRU", history_days=7, width=16, dropout=.1)
        model._e2_registered_role = "fresh fit of inherited GRU/D7-c03 recipe"
        return model
    return D7VolatilityAdapter(name, source_root, dropout)


def count_parameters(model):
    return {"registered": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "active_in_last_backward": sum(p.numel() for p in model.parameters() if p.grad is not None)}


def architecture_spec(model):
    if not isinstance(model, D7VolatilityAdapter):
        from dl_models_gru_lstm_tcn import architecture_spec as inherited
        return {**inherited(model), "role": getattr(model, "_e2_registered_role", "inherited reference")}
    return {"model": model.family, "history_days": 7, "input_channels": 5,
            "target": "sigma7 / mean(sigma7_training)", "forecast_horizon_days": 1,
            "d_model": 16, "d_ff": 32, "encoder_layers": 2,
            "attention_heads_if_applicable": 4 if model.family != "TimeMixer" else None,
            "dropout": .1, "author_source_commit": AUTHOR_COMMITS[model.family],
            "latent_shape": [5, 16], "readout": "Linear(90,1) followed by Softplus",
            "moment_inputs": "Five means and five ddof0 standard deviations of the same scaled D7 window",
            "calendar_markers": False, "target_lags_added": False,
            "original_covariate_projection_and_denormalization_used": False,
            "TimeMixer_scales_if_applicable": [7, 3] if model.family == "TimeMixer" else None,
            "TimeMixer_moving_average_if_applicable": 25 if model.family == "TimeMixer" else None,
            "TimeMixer_FMM_if_applicable": "Sum of per-scale latent temporal projections" if model.family == "TimeMixer" else None,
            "patch_length_stride_if_applicable": [2, 1] if model.family == "PatchTST" else None,
            "published_backbone_adapted_not_exact_paper_configuration": True,
            "NAS_final_applicability_not_decided_by_this_pilot": True,
            **count_parameters(model)}
