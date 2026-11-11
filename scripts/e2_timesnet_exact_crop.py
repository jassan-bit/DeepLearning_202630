"""Support-aware TimesNet convolution; same weights, architecture and objective.

On a zero-padded H by W grid, weights farther than H-1 or W-1 from the
kernel centre multiply zeros for every output. Cropping only those positions
before fusing the independent kernels preserves the function and gradients.
This is a versioned implementation change, not a published-body NAS exemption.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import itertools
import time

VERSION = "timesnet-support-crop-v1"
_ORIGINAL_FORWARD = None
_FAST_CASES = ()


def effective_parameters(layer, height, width):
    import torch.nn.functional as F
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("positive_grid_shape_required")
    radius_h = min(layer.padding, height-1)
    radius_w = min(layer.padding, width-1)
    weight = bias = None
    for kernel in layer.kernels:
        if (kernel.stride != (1, 1) or kernel.dilation != (1, 1)
                or kernel.groups != 1 or kernel.padding_mode != "zeros"
                or kernel.bias is None):
            raise ValueError("unsupported_convolution_contract")
        centre_h, centre_w = (k//2 for k in kernel.kernel_size)
        keep_h, keep_w = min(centre_h, radius_h), min(centre_w, radius_w)
        part = kernel.weight[:, :, centre_h-keep_h:centre_h+keep_h+1,
                             centre_w-keep_w:centre_w+keep_w+1]
        part = F.pad(part, (radius_w-keep_w, radius_w-keep_w,
                            radius_h-keep_h, radius_h-keep_h))
        weight = part if weight is None else weight+part
        bias = kernel.bias if bias is None else bias+kernel.bias
    n = len(layer.kernels)
    return weight/n, bias/n, (radius_h, radius_w)


def period_forward(block, values):
    import torch
    import torch.nn.functional as F
    if values.shape[1:] != (8, block.width):
        raise ValueError("registered_timesnet_grid_required")
    weights = block.period_weights(values)
    by_period = {}
    for period in (8, 4, 2):
        height, width = 8//period, period
        grid = values.reshape(len(values), height, width, block.width).permute(0, 3, 1, 2)
        first = effective_parameters(block.inception_in, height, width)
        second = effective_parameters(block.inception_out, height, width)
        mixed = F.conv2d(grid, first[0], first[1], padding=first[2])
        mixed = F.gelu(mixed)
        mixed = F.conv2d(mixed, second[0], second[1], padding=second[2])
        by_period[period] = mixed.permute(0, 2, 3, 1).reshape(len(values), 8, block.width)
    branches = torch.stack([by_period[p] for p in block.periods], dim=-1)
    return block.norm(values+(branches*weights[:, None, None, :]).sum(-1))


def dispatch(block, values):
    if (block.width, len(values)) in _FAST_CASES and len(block.inception_in.kernels) == 6:
        return period_forward(block, values)
    return _ORIGINAL_FORWARD(block, values)


def install(fast_cases):
    """Opt in inside a separately registered worker, never edit frozen A3 files."""
    import e2_nas_spectral as spectral
    global _ORIGINAL_FORWARD, _FAST_CASES
    if any(len(case)!=2 or type(case[0]) is not int or case[0] not in (16,32,64,128,256)
           or type(case[1]) is not int or case[1] not in (16,32,64) for case in fast_cases):
        raise ValueError("invalid_profiled_widths")
    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = spectral._NASPeriodBlock.forward
    _FAST_CASES = tuple(tuple(case) for case in fast_cases)
    spectral._NASPeriodBlock.forward = dispatch


def benchmark(*, device="cpu", repeats=5, widths=(16, 32, 64, 128, 256), batch_sizes=(16,32,64)):
    import torch
    import e2_nas_spectral as spectral
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("real_cuda_required_no_cpu_substitution")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = []
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    for width,batch in itertools.product(widths,batch_sizes):
        torch.manual_seed(731101+width)
        original = spectral._NASPeriodBlock(width, 4, 6).to(device)
        optimized = deepcopy(original)
        x = torch.randn(batch, 8, width, device=device)
        old = original(x)
        new = period_forward(optimized, x)
        torch.testing.assert_close(new, old, rtol=3e-5, atol=3e-6)
        loss_weights = torch.randn_like(old)
        (old*loss_weights).sum().backward()
        (new*loss_weights).sum().backward()
        for (name, a), (other, b) in zip(original.named_parameters(), optimized.named_parameters()):
            if name != other:
                raise ValueError("parameter_identity_changed")
            torch.testing.assert_close(b.grad, a.grad, rtol=3e-4, atol=3e-5)
        timings = {"original": [], "cropped": []}
        # Alternate order; include fusion, forward and backward, excluding AdamW.
        for repeat in range(repeats+1):
            names = ("original", "cropped") if repeat % 2 == 0 else ("cropped", "original")
            for name in names:
                model = original if name == "original" else optimized
                model.zero_grad(set_to_none=True)
                sync();start = time.perf_counter()
                output = model(x) if name == "original" else period_forward(model, x)
                (output*loss_weights).sum().backward()
                sync();elapsed = time.perf_counter()-start
                if repeat:
                    timings[name].append(elapsed)
        before, after = (statistics.median(timings[k]) for k in ("original", "cropped"))
        rows.append({"width": width, "batch": batch, "num_kernels": 6,
            "parameters": sum(p.numel() for p in original.parameters()),
            "original_seconds": before, "cropped_seconds": after,
            "speedup": before/after, "forward_and_gradient_parity": True})
        del original, optimized, old, new, x, model, output, loss_weights
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    fast = [[row["width"],row["batch"]] for row in rows if row["speedup"] >= 1.05]
    return {"version": VERSION, "device": device, "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
        "repeats": repeats, "synthetic_only": True, "market_data_read": False,
        "optimizer_included": False, "end_to_end_training_speedup_measured": False,
        "rows": rows, "fast_cases": fast,
        "cuda_gate_passed": device.startswith("cuda") and bool(fast)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("at least three timed repeats required")
    result = benchmark(device=args.device, repeats=args.repeats)
    if args.output:
        # Profiling receipts are immutable and live outside any campaign ledger.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
