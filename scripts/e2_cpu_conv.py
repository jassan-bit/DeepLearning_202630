"""CPU execution identity for the tiny spatial grids of the D7 NAS TimesNet.

With zero padding and an odd, same-size convolution, a kernel coefficient whose
offset exceeds H-1 or W-1 multiplies padding at every output position. A central
view can therefore omit those coefficients from the convolution. They remain
registered parameters, and slice backward supplies their zero gradients, so
AdamW decay and optimizer state still cover the complete original parameters.

The period block in this repository uses *functional fused* convolutions rather
than Conv2d.forward. Its replacement preserves that fusion, frequency weights,
branch order, GELU, residual and normalization. It first fuses the rectangle
covering the union of the three grids, then crops a view for each grid. Original
kernel order and the averaging denominator are retained. This is an algebraic
identity; different floating-point convolution kernels can introduce roundoff.

Only new TimesNetNAS instances are patched. No global class, model source file,
parameter tensor, state-dict key, architecture metadata, data or optimizer is
modified. Non-CPU and non-fp32/fp64 execution delegates to the original methods.
"""
from __future__ import annotations

from copy import deepcopy
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .e2_nas_spectral import TimesNetNAS, _NASPeriodBlock, _OddKernelInception
except ImportError:
    from e2_nas_spectral import TimesNetNAS, _NASPeriodBlock, _OddKernelInception

VERSION = "e2-timesnet-cpu-zero-crop-v1"
_REPORT = "_e2_cpu_crop_report"
_ORIGINAL = "_e2_cpu_crop_original_forward"


def _pair(value):
    if type(value) is int:
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(type(v) is int for v in value):
        return tuple(value)
    return None


def _cpu_float(value):
    return (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.dtype in (torch.float32, torch.float64))


def _target_kernel(kernel, height, width):
    return min(kernel[0], 2*height-1), min(kernel[1], 2*width-1)


def cropped_conv2d(values, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1):
    """Same zero-padded odd Conv2d, retaining original tensors and autograd.

    Unsupported geometry/device/dtype uses ordinary F.conv2d, including its
    original validation errors. No crop is taken if the whole kernel is useful.
    """
    ordinary = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    if (not _cpu_float(values) or not _cpu_float(weight) or values.ndim != 4
            or weight.ndim != 4 or _pair(stride) != (1, 1)
            or _pair(dilation) != (1, 1) or groups != 1
            or min(values.shape[-2:]) < 1):
        return F.conv2d(values, weight, bias, **ordinary)
    kernel = tuple(weight.shape[-2:])
    if (any(k % 2 != 1 for k in kernel)
            or _pair(padding) != tuple(k//2 for k in kernel)):
        return F.conv2d(values, weight, bias, **ordinary)
    target = _target_kernel(kernel, *values.shape[-2:])
    if target == kernel:
        return F.conv2d(values, weight, bias, **ordinary)
    offsets = tuple((old-new)//2 for old, new in zip(kernel, target))
    view = weight[..., offsets[0]:offsets[0]+target[0], offsets[1]:offsets[1]+target[1]]
    return F.conv2d(values, view.contiguous(), bias,
                    padding=tuple(k//2 for k in target))


def _fused_crop(inception, height, width):
    """Central crop of the original fused parameters, before large padding.

    Every original weight participates as a tensor/view in the graph. The
    averaging sum uses exactly the original kernel order and denominator.
    """
    size = 2*inception.padding+1
    target = _target_kernel((size, size), height, width)
    # Retain the source implementation when there is no algebraic reduction.
    if target == (size, size):
        weight, bias = inception.effective_parameters()
        return weight, bias, (inception.padding, inception.padding)
    weight = None
    bias = None
    for kernel in inception.kernels:
        value = kernel.weight
        current = tuple(value.shape[-2:])
        taken = tuple(min(a, b) for a, b in zip(current, target))
        offsets = tuple((a-b)//2 for a, b in zip(current, taken))
        value = value[..., offsets[0]:offsets[0]+taken[0], offsets[1]:offsets[1]+taken[1]]
        pad_h, pad_w = (target[0]-taken[0])//2, (target[1]-taken[1])//2
        value = F.pad(value, (pad_w, pad_w, pad_h, pad_h))
        weight = value if weight is None else weight+value
        bias = kernel.bias if bias is None else bias+kernel.bias
    return weight/len(inception.kernels), bias/len(inception.kernels), tuple(k//2 for k in target)


def _inception_cpu_forward(self, values):
    if not _cpu_float(values) or values.ndim != 4 or min(values.shape[-2:]) < 1:
        return getattr(self, _ORIGINAL)(values)
    weight, bias, padding = _fused_crop(self, *values.shape[-2:])
    return F.conv2d(values, weight, bias, padding=padding)


def _period_cpu_forward(self, values):
    if not _cpu_float(values):
        return getattr(self, _ORIGINAL)(values)
    if values.shape[1:] != (8, self.width):
        raise ValueError("TimesBlock requires eight learned tokens at its declared width")
    weights = self.period_weights(values)
    periods = (8, 4, 2)
    # All three period grids share one fused tensor/gradient path, as in source.
    max_height = max(8//period for period in periods)
    max_width = max(periods)
    first = _fused_crop(self.inception_in, max_height, max_width)
    second = _fused_crop(self.inception_out, max_height, max_width)
    by_period = {}
    for period in periods:
        grid = values.reshape(len(values), 8//period, period, self.width).permute(0, 3, 1, 2)
        mixed = cropped_conv2d(grid, first[0], first[1], padding=first[2])
        mixed = F.gelu(mixed)
        mixed = cropped_conv2d(mixed, second[0], second[1], padding=second[2])
        by_period[period] = mixed.permute(0, 2, 3, 1).reshape(len(values), 8, self.width)
    branches = torch.stack([by_period[period] for period in self.periods], dim=-1)
    return self.norm(values+(branches*weights[:, None, None, :]).sum(-1))


def _install(module, function):
    if hasattr(module, _ORIGINAL):
        raise ValueError("A child module already has a different execution decoration")
    setattr(module, _ORIGINAL, module.forward)
    module.forward = MethodType(function, module)


def enable_cpu_crop(model, name="TimesNet"):
    """Patch one known NAS TimesNet instance and return that same instance.

    Existing architecture_metadata is kept intact. The separate plain-Python
    execution_optimization report is not a buffer, parameter or state-dict key.
    Reapplying this version is idempotent. No market operation occurs here.
    """
    if name != "TimesNet" or type(model) is not TimesNetNAS:
        raise ValueError("CPU zero-crop applies only to the versioned TimesNetNAS family")
    if hasattr(model, _REPORT):
        if getattr(model, _REPORT).get("version") != VERSION:
            raise ValueError("A different CPU execution optimization is already installed")
        return model
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    keys = tuple(model.state_dict())
    modules = [(path, module) for path, module in model.named_modules()
               if type(module) in (_NASPeriodBlock, _OddKernelInception)]
    blocks = [(path, module) for path, module in modules if type(module) is _NASPeriodBlock]
    inceptions = [(path, module) for path, module in modules if type(module) is _OddKernelInception]
    if len(blocks) != 3 or len(inceptions) != 6:
        raise ValueError("The registered three-block TimesNet body changed")
    # Validate every patch before making any change.
    for _, module in modules:
        expected = _NASPeriodBlock.forward if type(module) is _NASPeriodBlock else _OddKernelInception.forward
        if (hasattr(module, _ORIGINAL) or getattr(module.forward, "__func__", None) is not expected):
            raise ValueError("TimesNet execution was already changed outside this optimizer")
    for _, module in blocks:
        _install(module, _period_cpu_forward)
    for _, module in inceptions:
        _install(module, _inception_cpu_forward)
    after = list(model.named_parameters())
    if (tuple(model.state_dict()) != keys or len(after) != len(parameters)
            or any(a != b or left is not right for (a, left), (b, right) in zip(parameters, after))):
        raise RuntimeError("CPU execution optimization changed the registered parameters")
    report = {
        "version": VERSION, "family": "TimesNet", "cpu_only": True,
        "supported_dtypes": ["float32", "float64"],
        "method": "central zero-padding crop before shared fusion and per-grid convolution",
        "registered_parameter_count": sum(p.numel() for _, p in parameters),
        "parameter_objects_preserved": True, "state_dict_keys_preserved": True,
        "architecture_metadata_preserved": True,
        "period_blocks": [path for path, _ in blocks],
        "inception_modules": [path for path, _ in inceptions],
        "original_weights_removed": 0,
        "all_original_parameters_receive_gradient_tensors": True,
        "floating_point_note": "Algebraically identical; backend roundoff can differ.",
        "market_training_executed_by_enable": False,
    }
    setattr(model, _REPORT, report)
    model.execution_optimization = deepcopy(report)
    return model


def execution_report(model):
    """Return a copy of this instance's static execution report."""
    if not hasattr(model, _REPORT):
        raise ValueError("CPU zero-crop is not installed on this model")
    return deepcopy(getattr(model, _REPORT))


def main():
    """Run the artificial parity suite and preserve an exclusive evidence file."""
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    import sys
    import unittest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Parity evidence must be written to a new path")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"scripts"))
    sys.modules.setdefault("e2_cpu_conv", sys.modules[__name__])
    suite = unittest.defaultTestLoader.discover(str(root/"tests"), pattern="test_e2_cpu_conv.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sources = ("scripts/e2_cpu_conv.py", "tests/test_e2_cpu_conv.py",
               "scripts/e2_nas_spectral.py", "scripts/e2_nas_factory.py",
               "scripts/e2_nas_common.py", "scripts/e2_staged_training.py")
    report = {
        "version": VERSION, "passed": result.wasSuccessful() and not result.skipped,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "market_data_read": False,
        "optimizer_work": "Artificial one-step AdamW parity only; no performance benchmark",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "command": ["python", "scripts/e2_cpu_conv.py", "--verify", "--output", str(args.output)],
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        "torch": str(torch.__version__), "dtypes_checked": ["float64", "float32"],
        "direct_fp64_geometries": 36,
        "tolerances": {"convolution_fp64": {"rtol": 2e-12, "atol": 2e-12},
                       "convolution_fp32": {"rtol": 5e-5, "atol": 2e-5},
                       "timesnet_fp64": {"rtol": 2e-10, "atol": 2e-11},
                       "timesnet_reference_fp32": {"rtol": 3e-4, "atol": 4e-5},
                       "adamw_state_fp64": {"rtol": 5e-10, "atol": 5e-11}},
        "largest_parameter_count_meta_checked": 225075755,
        "limits": ["Algebraic zero-padding identity does not imply bit-identical floating-point results.",
                   "This suite is artificial correctness evidence, not a measured runtime speedup.",
                   "CPU execution only; other devices/dtypes preserve the original forward path."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "tests_run": result.testsRun,
                      "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
