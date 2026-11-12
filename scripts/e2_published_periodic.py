"""Defined TimesNet/TimeMixer bodies with a separate D7 scalar task head.

The complete pinned author forecasting path is retained. Input-channel
normalization/denormalization remains inside that path; only the external
five-feature head predicts sigma7/a. The caller alone restores target units.
No market data, optimizer, architecture search, or historical receipt is read.
"""
# _NonDCTimesBlock adapts the MIT-licensed THUML TimesBlock forward.
# Copyright (c) 2021 THUML @ Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "book_e2/_static/current_implementation_20261108/source_design.json"
VERSION = "e2-published-periodic-v1"
RECIPE_IDS = {"TimesNet": "TN-D7-PUBLISHED-v1", "TimeMixer": "TM-D7-PUBLISHED-v1"}
_IMPORT_LOCK = RLock()


def _check_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 3
            or tuple(x.shape[1:]) != (7, 5) or x.shape[0] < 1
            or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
        raise ValueError("Supply nonempty finite floating-point [batch,7,5] D7 windows")


def _check_dropout(dropout):
    if (type(dropout) not in (int, float) or not math.isfinite(dropout)
            or not 0 <= dropout <= .5):
        raise ValueError("Dropout must be finite in [0,.5]")
    return float(dropout)


def verify_source_tree(name, source_roots):
    """Hash every declared author file, including license and dependencies."""
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if name not in RECIPE_IDS or not isinstance(source_roots, dict) or name not in source_roots:
        raise ValueError("An explicit pinned author source root is required for " + str(name))
    root = Path(source_roots[name]).resolve()
    rows = [row for row in manifest["author_sources"] if row["model"] == name]
    if not rows or not any(row["path"] == "LICENSE" for row in rows):
        raise ValueError("The author manifest must cover source files and license")
    for row in rows:
        path = root / row["path"]
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Pinned author source missing or changed: " + name + "/" + row["path"])
    return root, {**deepcopy(manifest["source_locations"][name]),
                  "verified_files": [{"path": row["path"], "sha256": row["sha256"]} for row in rows],
                  "verification_scope": "Every declared file for this author tree, not the whole repository"}


def _author_module(name, root):
    # Author repositories use the same top-level models/layers/utils names.
    # Already-constructed modules retain references to their original globals.
    with _IMPORT_LOCK:
        for key in list(sys.modules):
            if key in ("models", "layers", "utils") or key.startswith(("models.", "layers.", "utils.")):
                del sys.modules[key]
        sys.path.insert(0, str(root))
        try:
            return importlib.import_module("models." + name)
        finally:
            sys.path.remove(str(root))


def _args(name, dropout):
    values = dict(task_name="long_term_forecast", seq_len=7, label_len=0,
        pred_len=1, enc_in=5, dec_in=5, c_out=5, d_model=32, d_ff=32,
        e_layers=2, d_layers=1, n_heads=4, factor=3, activation="gelu",
        dropout=dropout, embed="timeF", freq="d", output_attention=False,
        channel_independence=1, decomp_method="moving_avg", moving_avg=25,
        use_norm=1, down_sampling_layers=1, down_sampling_window=2,
        down_sampling_method="avg", use_future_temporal_feature=0,
        top_k=3, num_kernels=6)
    if name == "TimeMixer":
        values["e_layers"] = 4
    return SimpleNamespace(**values)


def fft_for_period_non_dc(x, k):
    """Author batch-mean selection with DC excluded from the top-k domain.

    The paper selects positive frequencies. Upstream sets DC amplitude to zero
    but leaves that bin eligible in an all-zero tie, producing division by zero.
    This changes only that invalid domain; batch aggregation, topk, amplitudes,
    floor periods and branch weights retain the pinned code's operations.
    """
    xf = torch.fft.rfft(x, dim=1)
    amplitudes = xf.abs()
    frequency_list = amplitudes.mean(0).mean(-1)
    if type(k) is not int or not 1 <= k <= len(frequency_list) - 1:
        raise ValueError("top_k must fit the non-DC frequency domain")
    _, indices = torch.topk(frequency_list[1:], k)
    indices = (indices + 1).detach().cpu().numpy()
    return x.shape[1] // indices, amplitudes.mean(-1)[:, indices]


class _NonDCTimesBlock(nn.Module):
    """Pinned TimesBlock forward; only the FFT helper excludes DC explicitly."""
    def __init__(self, source):
        super().__init__()
        self.seq_len, self.pred_len, self.k = source.seq_len, source.pred_len, source.k
        self.conv = source.conv

    def forward(self, x):
        batch, length, channels = x.size()
        periods, weights = fft_for_period_non_dc(x, self.k)
        branches = []
        for period in periods:
            total = self.seq_len + self.pred_len
            padded = ((total // period) + 1) * period if total % period else total
            values = torch.cat((x, x.new_zeros((batch, padded - total, channels))), dim=1) if padded != total else x
            grid = values.reshape(batch, padded // period, period, channels).permute(0, 3, 1, 2).contiguous()
            values = self.conv(grid).permute(0, 2, 3, 1).reshape(batch, -1, channels)
            branches.append(values[:, :total])
        branches = torch.stack(branches, dim=-1)
        weights = F.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, length, channels, 1)
        return torch.sum(branches * weights, dim=-1) + x


def _freeze_unused_source_parameters(backbone, name):
    """Keep registered upstream modules; freeze only branches the source never calls."""
    prefixes = ["enc_embedding.temporal_embedding."]
    if name == "TimeMixer":
        prefixes += [f"pdm_blocks.{i}.layer_norm." for i in range(len(backbone.pdm_blocks))]
    frozen = []
    for key, parameter in backbone.named_parameters():
        if any(key.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(False)
            frozen.append(key)
    return frozen


class _PublishedPeriodic(nn.Module):
    def __init__(self, name, dropout, source_roots):
        super().__init__()
        dropout = _check_dropout(dropout)
        root, source = verify_source_tree(name, source_roots)
        module = _author_module(name, root)
        self.name = name
        self.backbone = module.Model(_args(name, dropout))
        if name == "TimesNet":
            self.backbone.model = nn.ModuleList(_NonDCTimesBlock(block) for block in self.backbone.model)
        frozen = _freeze_unused_source_parameters(self.backbone, name)
        self.readout = nn.Linear(5, 1)
        self.architecture_metadata = {
            "factory_version": VERSION, "model": name, "recipe_id": RECIPE_IDS[name],
            "input_shape": [7, 5], "output": "positive sigma7/a; caller restores target units",
            "body_output_shape": [1, 5], "source_forecast_complete": True,
            "task_head": "Linear(5,1)+Softplus outside the complete forecasting body",
            "same_window_moments": 0, "future_covariates": False,
            "architecture_search": False, "paper_replication_claimed": False,
            "historical_recipe_replaced": False, "source_verification": source,
            "frozen_unused_source_parameters": frozen,
            "frozen_parameter_reason": "Official branches not called with x_mark=None; TimeMixer PDM.layer_norm is registered but never called upstream",
            "fixed_body": {"layers": 2 if name == "TimesNet" else 4,
                           "d_model": 32, "d_ff": 32},
        }
        if name == "TimesNet":
            self.architecture_metadata.update(
                fixed_top_k=3, num_kernels=6, fft_effective_length=8,
                fft_selection="pinned batch-mean topk over non-DC bins; no per-example replacement",
                fft_dc_guard="Exclude bin0 from topk domain, including constant/tied spectra",
                inference_batch_size1_enforced=True,
                inference_policy="In eval, invoke the unchanged forecasting body separately for each window in caller order, preventing later windows from influencing FFT selection",
                period_rounding="floor matches pinned author code; paper equation uses ceil",
                preset_source="TimesNet v3 Appendix B k1..5; Table7 two layers and d_model rule; pinned run.py/ETTh1 d_ff32 and six kernels")
        else:
            self.architecture_metadata.update(
                scale_lengths=[7, 3], down_sampling_layers=1, moving_average=25,
                preset_source="TimeMixer v1 Table7 M4 short-term; pinned M4 Monthly script: four PDM, d_model32, d_ff32, M1, avg2",
                preset_source_url="https://github.com/kwuking/TimeMixer/blob/e24610583b36fdd8c76cc17a8df4e65759a5f460/scripts/short_term_forecast/M4/TimeMixer.sh")

    def body_features(self, x):
        _check_input(x)
        if self.name == "TimesNet" and not self.training and len(x) > 1:
            # Batch-mean FFT is unchanged inside the official body. At inference
            # each call receives one available window, so another (possibly
            # later) validation/test window cannot influence its period choice.
            values = torch.cat([self.backbone(x[i:i + 1], None, None, None)
                                for i in range(len(x))], dim=0)
        else:
            values = self.backbone(x, None, None, None)
        if values.shape != (len(x), 1, 5) or not bool(torch.isfinite(values).all()):
            raise RuntimeError("The pinned complete forecasting body must emit finite [batch,1,5]")
        return values[:, 0, :]

    def forward(self, x):
        return F.softplus(self.readout(self.body_features(x))).squeeze(-1)


class TimesNetPublished(_PublishedPeriodic):
    def __init__(self, dropout=.1, source_roots=None):
        super().__init__("TimesNet", dropout, source_roots)


class TimeMixerPublished(_PublishedPeriodic):
    def __init__(self, dropout=.1, source_roots=None):
        super().__init__("TimeMixer", dropout, source_roots)


def build_periodic(name, dropout=.1, source_roots=None):
    if name not in RECIPE_IDS:
        raise ValueError("Choose a defined TimesNet or TimeMixer body")
    return {"TimesNet": TimesNetPublished, "TimeMixer": TimeMixerPublished}[name](dropout, source_roots)
