"""Fixed author PatchTST body with a declared seven-day scalar-task adapter.

The supervised encoder and original Flatten_Head come from the six hash-pinned
author files (including LICENSE). P3/S2 is the previously documented D7 geometry;
it is not the paper's long-context P16/S8 preset. Only the five normalized-space
channel predictions are combined into sigma7: OHLCV de-normalization is excluded.
No market data, search, optimizer, downloads or pretrained weights are used here.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .e2_current_models import _verify_sources
    from .e2_final_models_mlp import _author_model
except ImportError:
    from e2_current_models import _verify_sources
    from e2_final_models_mlp import _author_model


RECIPE_ID = "PT-D7-PUBLISHED-v1"
AUTHOR_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"


def _check_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 3
            or tuple(x.shape[1:]) != (7, 5) or len(x) < 1):
        raise ValueError("PatchTST requires nonempty [batch,7,5] observed D7 windows")
    if not torch.is_floating_point(x) or (x.device.type != "meta" and not bool(torch.isfinite(x).all())):
        raise ValueError("PatchTST D7 inputs must be finite floating-point values")


class PublishedPatchTSTD7(nn.Module):
    """Unchanged fixed-width author body and forecast head; scalar readout outside."""
    recipe_id = RECIPE_ID

    def __init__(self, *, dropout, source_roots):
        super().__init__()
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float))
                or not math.isfinite(dropout) or not 0 <= dropout < 1):
            raise ValueError("Provide finite PatchTST dropout in [0,1)")
        verified = _verify_sources("PatchTST", source_roots)
        if verified["commit"] != AUTHOR_COMMIT:
            raise ValueError("The published PatchTST author commit differs from the declared recipe")
        root = Path(source_roots["PatchTST"]).resolve()
        if (root / "PatchTST_supervised").is_dir():
            root = root / "PatchTST_supervised"
        config = SimpleNamespace(seq_len=7, pred_len=1, enc_in=5, e_layers=3,
            n_heads=4, d_model=16, d_ff=128, dropout=float(dropout),
            fc_dropout=float(dropout), head_dropout=0.0, individual=False,
            patch_len=3, stride=2, padding_patch="end", revin=True, affine=True,
            subtract_last=False, decomposition=False, kernel_size=25)
        author = _author_model("PatchTST", root, config)
        self.backbone = author.model
        if (self.backbone.backbone.patch_num != 4
                or not self.backbone.backbone.encoder.res_attention):
            raise ValueError("The author D7 geometry/residual-attention contract differs")
        self.readout = nn.Linear(5, 1)
        self.architecture_metadata = {
            "model": "PatchTST", "recipe_id": RECIPE_ID,
            "input_shape": [7, 5], "horizon": 1,
            "output": "positive sigma7 / training-only target scale",
            "author_repository": verified["repository"], "source_commit": AUTHOR_COMMIT,
            "source_verification": deepcopy(verified), "source_files_verified_at_build": True,
            "body": "original supervised TSTiEncoder and Flatten_Head",
            "d_model": 16, "d_ff": 128, "encoder_layers": 3, "n_heads": 4,
            "residual_attention": True, "normalization": "author BatchNorm, post-norm",
            "patch_len": 3, "stride": 2, "patch_count": 4,
            "left_padding": 0, "right_padding": 2,
            "padding": "author end replication of last observed value",
            "revin_statistics_scope": "seven actual observed days before padding",
            "source_flatten_head_preserved": True, "head_dropout": 0.0,
            "published_source_preset": {
                "evidence": "scripts/PatchTST/etth1.sh at the verified author commit",
                "seq_len": 336, "patch_len": 16, "stride": 8,
                "dropout": 0.3, "fc_dropout": 0.3, "head_dropout": 0.0,
            },
            "actual_dropout": {"encoder": float(dropout), "fc": float(dropout), "head": 0.0},
            "scalar_readout": "Linear(5,1) then Softplus before OHLCV denormalization",
            "same_window_moment_features": 0, "dropout": float(dropout),
            "task_adaptations": [
                "Observed context 7 instead of source ETTh1 336; fixed P3/S2 rather than P16/S8",
                "P3/S2 D7 geometry follows the prior documentary adaptation proposal",
                "Five OHLCV channel predictions combined into one scalar before source denormalization",
            ],
            "paper_replication_claimed": False, "exact_published_long_context_preset": False,
            "structural_architecture_search": False, "weights_pretrained": False,
            "target_lags_added": False, "calendar_covariates_added": False,
            "market_data_read": False,
        }

    def normalized_patches(self, x):
        """Follow the author norm/padding/unfold operations on seven actual days."""
        _check_input(x)
        normalized = self.backbone.revin_layer(x, "norm").transpose(1, 2)
        padded = self.backbone.padding_patch_layer(normalized)
        return padded.unfold(-1, self.backbone.patch_len, self.backbone.stride)

    def channel_predictions(self, x):
        """Return the five native head outputs, excluding input-channel denorm."""
        patches = self.normalized_patches(x).permute(0, 1, 3, 2)
        encoded = self.backbone.backbone(patches)
        return self.backbone.head(encoded).squeeze(-1)

    def forward(self, x):
        return F.softplus(self.readout(self.channel_predictions(x))).squeeze(-1)


def build_patchtst(*, dropout, source_roots):
    """Build fresh weights after verifying every declared PatchTST source file."""
    return PublishedPatchTSTD7(dropout=dropout, source_roots=source_roots)
