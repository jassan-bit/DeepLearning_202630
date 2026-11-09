"""Shared contracts for the versioned section-27 NAS variants.

These are new research variants, not replacements of the historical recipes.
Inputs are already transformed/scaled D7 tensors; no data files are loaded.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

WIDTHS = (16, 32, 64, 128, 256)


@dataclass(frozen=True)
class NASConfig:
    widths: tuple[int, int, int]
    dropout_pattern: tuple[int, int, int]

    def __post_init__(self):
        for values, name in ((self.widths, "widths"), (self.dropout_pattern, "dropout_pattern")):
            if not isinstance(values, (tuple, list)) or len(values) != 3:
                raise ValueError(name + " must contain exactly three values")
        if any(type(w) is not int or w not in WIDTHS for w in self.widths):
            raise ValueError("Each width must be one of16,32,64,128,256")
        if any(type(b) is not int or b not in (0, 1) for b in self.dropout_pattern):
            raise ValueError("Dropout pattern must contain three integer bits")
        object.__setattr__(self, "widths", tuple(self.widths))
        object.__setattr__(self, "dropout_pattern", tuple(self.dropout_pattern))

    @property
    def bits(self):
        return self.dropout_pattern

    def p(self, index):
        if index not in (0, 1, 2):
            raise ValueError("NAS block index must be0,1,2")
        return .2 if self.dropout_pattern[index] else 0.

    @property
    def pattern_id(self):
        return "w" + "-".join(map(str, self.widths)) + ":b" + "".join(map(str, self.dropout_pattern))


def validate_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] < 1
            or tuple(x.shape[1:]) != (7, 5) or not torch.is_floating_point(x)):
        raise ValueError("Expected nonempty finite floating-point [batch,7,5]")
    # Meta tensors validate geometry only, never finiteness or data-dependent behavior.
    if x.device.type != "meta" and not bool(torch.isfinite(x).all()):
        raise ValueError("D7 inputs must contain only finite values")
    return x


def moment_features(x):
    validate_input(x)
    return torch.cat((x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=-1)


def block_dropout(config, index):
    if not isinstance(config, NASConfig):
        raise TypeError("Explicit NASConfig required")
    return nn.Dropout(config.p(index))

