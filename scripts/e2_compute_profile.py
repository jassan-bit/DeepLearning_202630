"""CPU forward arithmetic estimates and measured throughput, never model fitting.

Counts follow an explicit scalar-operation convention, not processor instructions.
Unknown floating operators make the total unavailable; the known subtotal is
retained. Native FFT algorithms are deliberately not assigned exact FLOP totals.
Measurement must run after weights close, with competing workers quiescent.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
import statistics
import time

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


CONVENTION = {
    "id": "E2-CPU-scalar-arithmetic-MAC2-v1",
    "scope": "Observed CPU eval-forward path; not hardware instructions, backward, preprocessing or optimizer cost",
    "dense_products": "Real MAC=2; complex MAC=8; convolution includes padded dense kernel products",
    "elementwise": "One operation per real output for elementary arithmetic, comparisons, or an explicitly supported nonlinear primitive; this unit-cost convention is not transcendental hardware latency",
    "reductions": "Sum/max/min/argmax/argmin: input minus output elements; mean adds one division per output",
    "variance": "Real centered population/corrected variance:4N operations including mean; standard deviation adds one sqrt per group",
    "normalization": "Real two-pass centered normalization:5N+2G before optional affine weight/bias (one operation per element each); batchnorm eval uses fixed running moments",
    "softmax": "Stable dense reference: maximum,shift,exp,sum,division =>5N-2G; log-softmax=>5N-G",
    "fused_recurrence": "Standard nonprojected LSTM/GRU/RNN are counted from their actual per-layer matrices. LSTM adds13H,GRU11H,RNN2H scalar operations per time/sample beyond MAC2 affine products and optional two biases; these are explicit reference equations, not native fused-kernel instructions",
    "complex": "Complex addition/subtraction=2; multiplication=6; division=11; abs=4; exp=5 real scalar operations",
    "excluded": "Memory moves/copies/allocations/views, integer or boolean indexing/shape work, RNG, Python scalar arithmetic, and comparisons without floating inputs",
    "FFT": "Native FFT algorithm unknown; always partial. A radix2 complex-FFT reference5Nlog2N is shown only for power-of-two dimensions; it is neither an exact native cost nor a hardware bound",
    "unknown": "Any unrecognized operator involving floating/complex tensors is recorded and prevents a complete total",
    "interpretation": "A complete total is an estimate under this convention; a partial subtotal is not a lower bound on actual processor instructions. Nonlinear unit primitives and equivalent decomposed graphs can differ in this graph-level count; measured latency/throughput determine physical execution efficiency",
}

_UNARY = {
    "abs", "neg", "reciprocal", "sqrt", "rsqrt", "exp", "expm1", "exp2", "log", "log1p", "log2", "log10",
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "sigmoid", "silu", "softplus", "gelu", "relu", "relu6",
    "leaky_relu", "elu", "selu", "hardtanh", "hardswish", "hardsigmoid", "erf", "erfc", "sign", "sgn",
    "floor", "ceil", "round", "trunc", "clamp", "clamp_min", "clamp_max", "isnan", "isinf", "isfinite",
}
_BINARY = {"mul", "div", "true_divide", "floor_divide", "pow", "maximum", "minimum", "fmax", "fmin", "remainder", "fmod", "atan2"}
_COMPARE = {"eq", "ne", "lt", "le", "gt", "ge"}
_ZERO = {
    "alias", "detach", "view", "_unsafe_view", "reshape", "reshape_as", "view_as", "permute", "transpose", "t",
    "squeeze", "unsqueeze", "expand", "expand_as", "as_strided", "select", "slice", "split", "split_with_sizes",
    "unsafe_split", "unsafe_split_with_sizes", "unbind", "chunk", "narrow", "repeat", "repeat_interleave", "tile",
    "cat", "stack", "clone", "copy", "copy_", "contiguous", "_to_copy", "to", "lift_fresh", "lift_fresh_copy",
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "zeros", "zeros_like", "new_zeros",
    "ones", "ones_like", "new_ones", "full", "full_like", "new_full", "zero_", "fill_", "set_", "resize_",
    "index", "index_put", "index_put_", "index_select", "gather", "scatter", "scatter_", "masked_fill", "masked_fill_",
    "where", "embedding", "constant_pad_nd", "reflection_pad1d", "reflection_pad2d", "replication_pad1d", "replication_pad2d",
    "roll", "flip", "real", "imag", "view_as_real", "view_as_complex", "conj", "_conj",
    "resolve_neg", "_local_scalar_dense", "item", "is_pinned", "is_same_size", "numel",
    "rand", "rand_like", "randn", "randn_like", "normal", "normal_", "uniform_", "bernoulli", "bernoulli_",
    "dropout", "feature_dropout", "feature_alpha_dropout", "alpha_dropout",
}


def _tensors(value):
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]


def _floating(tensor):
    return tensor.dtype.is_floating_point or tensor.dtype.is_complex


def _first_tensor(value):
    values = _tensors(value)
    return values[0] if values else None


def _argument(args, kwargs, position, name, default=None):
    return args[position] if len(args) > position else kwargs.get(name, default)


def _complex_factor(inputs, real=1, complex_value=None):
    leaves = tree_flatten(inputs)[0]
    if any((isinstance(x, torch.Tensor) and x.dtype.is_complex) or isinstance(x, complex) for x in leaves):
        if complex_value is None:
            raise NotImplementedError("Complex arithmetic for this operator is not defined by the convention")
        return complex_value
    return real


def _einsum_cost(args, kwargs, output):
    equation, operands = args[:2]
    equation = equation.replace(" ", "")
    left, arrow, right = equation.partition("->")
    if not arrow:
        raise NotImplementedError("Implicit einsum output/path not modeled")
    terms = left.split(",")
    ellipsis_rank = max(t.ndim-len(term.replace("...", "")) for term, t in zip(terms, operands))
    ellipsis = ["@"+str(i) for i in range(ellipsis_rank)]
    dimensions = {}
    labels_by_operand = []
    for term, tensor in zip(terms, operands):
        if "..." in term:
            before, after = term.split("...")
            rank = tensor.ndim-len(before)-len(after)
            labels = list(before)+ellipsis[ellipsis_rank-rank:]+list(after)
        else:
            labels = list(term)
        if len(labels) != tensor.ndim:
            raise NotImplementedError("Unmodeled einsum index structure")
        local = {}
        for label, extent in zip(labels, tensor.shape):
            extent = int(extent)
            if label in local and local[label] != extent:
                raise ValueError("Repeated einsum index has inconsistent dimensions")
            local[label] = extent
            previous = dimensions.get(label, 1)
            if previous not in (1, extent) and extent != 1:
                raise ValueError("Einsum dimensions do not broadcast")
            dimensions[label] = max(previous, extent)
        labels_by_operand.append(dict(local))
    labels_out = list(right.replace("...", "")) + (ellipsis if "..." in right else [])
    if len(operands) == 1:
        size = math.prod(labels_by_operand[0].values())
        return max(0, size-output.numel())*_complex_factor(operands, 1, 2)
    path = kwargs.get("path")
    if path is None:
        # Explicit left-to-right reference when no contraction path is supplied.
        path = [index for _ in range(len(operands)-1) for index in (0, 1)]
    if len(path) != 2*(len(operands)-1):
        raise NotImplementedError("Unknown einsum contraction path representation")
    cost = 0; active = list(labels_by_operand)
    for offset in range(0, len(path), 2):
        i, j = map(int, path[offset:offset+2])
        if i == j or min(i, j) < 0 or max(i, j) >= len(active):
            raise NotImplementedError("Invalid einsum contraction path")
        union = set(active[i]) | set(active[j])
        pair_dimensions = {label: max(active[i].get(label, 1), active[j].get(label, 1)) for label in union}
        other_labels = set().union(*(set(labels) for k, labels in enumerate(active) if k not in (i, j)))
        kept = union & (other_labels | set(labels_out))
        contracted = union-kept
        output_size = math.prod(pair_dimensions[label] for label in kept)
        if contracted:
            extent = math.prod(pair_dimensions[label] for label in contracted)
            cost += output_size*extent*_complex_factor(operands, 2, 8)
        else:
            cost += output_size*_complex_factor(operands, 1, 6)
        active = [labels for k, labels in enumerate(active) if k not in (i, j)] + [{label: pair_dimensions[label] for label in sorted(kept)}]
    return cost


def _fft_reference(name, args, kwargs, output):
    tensor = args[0]
    if name.startswith("fft_"):
        transform = name[len("fft_"):]
        if transform in ("fft", "ifft", "rfft", "irfft", "hfft", "ihfft"):
            dimensions = [int(_argument(args, kwargs, 2, "dim", -1)) % tensor.ndim]
            shape = list(tensor.shape)
            requested = _argument(args, kwargs, 1, "n")
            if transform in ("irfft", "ifft", "hfft"):
                shape[dimensions[0]] = int(output.shape[dimensions[0]])
            elif requested is not None:
                shape[dimensions[0]] = int(requested)
        else:
            requested = _argument(args, kwargs, 1, "s")
            dims = _argument(args, kwargs, 2, "dim")
            if dims is None:
                count = 2 if transform.endswith("2") else len(requested) if requested is not None else tensor.ndim
                dims = list(range(tensor.ndim-count, tensor.ndim))
            dimensions = [int(i) % tensor.ndim for i in dims]
            shape = list(output.shape if transform.startswith(("irfft", "ifft", "hfft")) else tensor.shape)
            if requested is not None:
                for axis, length in zip(dimensions, requested):
                    if int(length) != -1: shape[axis] = int(length)
    else:
        dimensions = [int(i) % tensor.ndim for i in args[1]]
        shape = list(output.shape if name == "_fft_c2r" else tensor.shape)
    lengths = [int(shape[d]) for d in dimensions]
    power2 = all(n > 0 and n & (n-1) == 0 for n in lengths)
    total_points = math.prod(shape)
    reference = 5*total_points*sum(math.log2(n) for n in lengths) if power2 else None
    return {"operator": "aten."+name, "transform_lengths": lengths,
            "all_dimensions_power_of_two": power2, "radix2_complex_reference_scalar_ops": reference,
            "reference_is_native_FLOP_count_or_hardware_bound": False,
            "normalization_and_real_transform_optimizations_not_identified": True,
            "reason": "Native FFT algorithm unknown" if power2 else "Non-power-of-two transform: radix2 expression is inapplicable; native FFT algorithm unknown"}


def _fused_recurrence(name, args, kwargs):
    tensor = args[0]; weights = args[2]
    if tensor.ndim != 3 or not isinstance(weights, (list, tuple)):
        raise NotImplementedError("Packed/projected/unknown recurrent structure is not modeled")
    biases = bool(_argument(args, kwargs, 3, "has_biases")); layers = int(_argument(args, kwargs, 4, "num_layers"))
    dropout = float(_argument(args, kwargs, 5, "dropout", 0)); training = bool(_argument(args, kwargs, 6, "train", False))
    directions = 2 if _argument(args, kwargs, 7, "bidirectional", False) else 1
    batch_first = bool(_argument(args, kwargs, 8, "batch_first", False))
    if training and dropout:
        raise NotImplementedError("Training recurrent dropout is outside eval arithmetic")
    count = 4 if biases else 2
    if len(weights) != layers*directions*count:
        raise NotImplementedError("Projected recurrent matrices require another audited reference")
    time_steps = int(tensor.shape[1 if batch_first else 0]); batch = int(tensor.shape[0 if batch_first else 1])
    gates, other = (4, 13) if name == "lstm" else (3, 11) if name == "gru" else (1, 2)
    result = 0
    for index in range(layers*directions):
        w_ih, w_hh = weights[index*count:index*count+2]
        hidden = int(w_hh.shape[1]); input_size = int(w_ih.shape[1])
        if w_ih.shape[0] != gates*hidden or w_hh.shape[0] != gates*hidden:
            raise NotImplementedError("Inconsistent/projected recurrent gate sizes")
        result += time_steps*batch*(2*gates*hidden*(input_size+hidden)+other*hidden+(2*gates*hidden if biases else 0))
    return result


def _cost(name, args, kwargs, output):
    out = _first_tensor(output); n = out.numel() if out is not None else 0
    plain = name[:-1] if name.endswith("_") and not name.startswith("_") else name
    floating_inputs = any(_floating(t) for t in _tensors((args, kwargs)))
    if name in ("dropout", "native_dropout", "feature_dropout", "feature_alpha_dropout", "alpha_dropout"):
        probability = float(_argument(args, kwargs, 1, "p", .5)); training = bool(_argument(args, kwargs, 2, "train", True))
        if not training or probability in (0, 1): return 0
        if "alpha_dropout" in name: raise NotImplementedError("Training alpha-dropout reference not defined")
        return 2*n  # Multiply mask and scale survivors; RNG is explicitly excluded.
    if name in _ZERO or plain in _ZERO:
        return 0
    if name in ("_conj_physical", "conj_physical"):
        return n if args[0].dtype.is_complex else 0
    if name == "resolve_conj":
        return n if args[0].is_conj() else 0
    if not floating_inputs and not any(_floating(t) for t in _tensors(output)):
        return 0
    if plain in _UNARY:
        if _complex_factor(args, 1, 2) == 2:
            if plain == "abs": return n*4
            if plain == "exp": return n*5
            if plain == "neg": return n*2
            raise NotImplementedError("Complex nonlinear convention unavailable")
        return n
    if plain in _COMPARE:
        return n if floating_inputs else 0
    if plain in ("add", "sub"):
        alpha = _argument(args, kwargs, 2, "alpha", 1)
        return n*(_complex_factor(args, 1, 2)+(0 if alpha == 1 else _complex_factor((args, alpha), 1, 6)))
    if plain in _BINARY:
        factor = {"mul": 6, "div": 11, "true_divide": 11}.get(plain)
        return n*_complex_factor(args, 1, factor)
    if name in ("mm", "bmm", "matmul", "mv", "dot"):
        return n*int(args[0].shape[-1])*_complex_factor(args[:2], 2, 8)
    if name in ("addmm", "addmv", "baddbmm"):
        alpha = kwargs.get("alpha", 1); beta = kwargs.get("beta", 1)
        multiply = n*int(args[1].shape[-1])*_complex_factor(args[1:3], 2, 8) if alpha != 0 else 0
        result = multiply
        if alpha not in (0, 1): result += n*_complex_factor((args[1:3], alpha), 1, 6)
        if beta != 0:
            if multiply: result += n*_complex_factor(args, 1, 2)
            if beta != 1: result += n*_complex_factor((args[0], beta), 1, 6)
        return result
    if name == "linear":
        bias = _argument(args, kwargs, 2, "bias")
        return n*int(args[0].shape[-1])*_complex_factor(args[:2], 2, 8)+(n*_complex_factor(bias, 1, 2) if bias is not None else 0)
    if name == "einsum":
        return _einsum_cost(args, kwargs, out)
    if name in ("lstm", "gru", "rnn_tanh", "rnn_relu"):
        _complex_factor(args, 1)
        return _fused_recurrence(name, args, kwargs)
    if name in ("convolution", "_convolution", "conv1d", "conv2d", "conv3d", "mkldnn_convolution"):
        if name in ("convolution", "_convolution") and _argument(args, kwargs, 6, "transposed", False):
            raise NotImplementedError("Transposed convolution reference not defined")
        weight = args[1]; bias = _argument(args, kwargs, 2, "bias")
        _complex_factor(args, 1)  # Complex convolution is not silently treated as real.
        return 2*n*int(weight.shape[1])*math.prod(map(int, weight.shape[2:]))+(n if bias is not None else 0)
    if name in ("sum", "mean", "amax", "amin", "argmax", "argmin", "max", "min"):
        # Tensor-vs-tensor maximum/minimum are pointwise; a dim/int is reduction.
        if name in ("max", "min") and len(args) > 1 and isinstance(args[1], torch.Tensor):
            return n
        size = args[0].numel()
        factor = _complex_factor(args, 1, 2) if name in ("sum", "mean") else _complex_factor(args, 1)
        return max(0, size-n)*factor+(n*factor if name == "mean" else 0)
    if name in ("var", "std", "var_mean", "std_mean"):
        _complex_factor(args, 1)
        return 4*args[0].numel()+(n if name.startswith("std") else 0)
    if name in ("cumsum", "cumprod"):
        axis = int(_argument(args, kwargs, 1, "dim")) % args[0].ndim
        groups = args[0].numel() // int(args[0].shape[axis])
        return (args[0].numel()-groups)*_complex_factor(args, 1, 2 if name == "cumsum" else 6)
    if name in ("_softmax", "softmax", "_log_softmax", "log_softmax"):
        _complex_factor(args, 1)
        axis = int(_argument(args, kwargs, 1, "dim")) % args[0].ndim
        groups = n // int(args[0].shape[axis])
        return 5*n-(groups if "log_softmax" in name else 2*groups)
    if name in ("native_layer_norm", "layer_norm", "rms_norm"):
        _complex_factor(args, 1)
        size = args[0].numel(); group_size = math.prod(map(int, args[1])); groups = size//group_size
        if name == "rms_norm": return 3*size+2*groups+(size if _argument(args, kwargs, 2, "weight") is not None else 0)
        return 5*size+2*groups+(size if _argument(args, kwargs, 2, "weight") is not None else 0)+(size if _argument(args, kwargs, 3, "bias") is not None else 0)
    if name in ("native_group_norm", "group_norm"):
        _complex_factor(args, 1)
        size = args[0].numel()
        if name == "group_norm":
            groups = int(args[0].shape[0])*int(args[1]); weight = _argument(args, kwargs, 2, "weight"); bias = _argument(args, kwargs, 3, "bias")
        else:
            groups = int(args[3])*int(args[6]); weight = args[1]; bias = args[2]
        return 5*size+2*groups+(size if weight is not None else 0)+(size if bias is not None else 0)
    if name in ("native_batch_norm", "batch_norm", "_native_batch_norm_legit_no_training", "_native_batch_norm_legit"):
        if name != "_native_batch_norm_legit_no_training" and bool(_argument(args, kwargs, 5, "training", False)):
            raise NotImplementedError("Training batch normalization is outside eval-forward profiling")
        _complex_factor(args, 1)
        size = args[0].numel(); channels = int(args[0].shape[1])
        return 2*size+2*channels+(size if args[1] is not None else 0)+(size if args[2] is not None else 0)
    if name in ("avg_pool1d", "avg_pool2d", "avg_pool3d"):
        kernel = _argument(args, kwargs, 1, "kernel_size")
        volume = int(kernel) if isinstance(kernel, int) else math.prod(map(int, kernel))
        return n*volume  # k-1 additions and one division, including declared padding.
    if name in ("max_pool1d", "max_pool2d", "max_pool3d", "max_pool1d_with_indices", "max_pool2d_with_indices", "max_pool3d_with_indices"):
        kernel = _argument(args, kwargs, 1, "kernel_size")
        volume = int(kernel) if isinstance(kernel, int) else math.prod(map(int, kernel))
        return n*max(0, volume-1)
    if name == "_adaptive_avg_pool2d":
        # Only divisible, equal-sized bins receive the simple k-1+division count.
        inp = args[0]
        if any(int(inp.shape[i]) % int(out.shape[i]) for i in (-2, -1)):
            raise NotImplementedError("Nonuniform adaptive pooling bins are not modeled")
        return inp.numel()
    if name in ("_scaled_dot_product_flash_attention_for_cpu", "scaled_dot_product_attention"):
        q, k, v = args[:3]
        dropout_position = 3 if name == "_scaled_dot_product_flash_attention_for_cpu" else 4
        if float(_argument(args, kwargs, dropout_position, "dropout_p", 0)) != 0:
            raise NotImplementedError("Attention dropout arithmetic/RNG not modeled")
        _complex_factor(args[:3], 1)
        rows = math.prod(map(int, q.shape[:-1])); length = int(k.shape[-2]); scores = rows*length
        cost = 2*scores*(int(q.shape[-1])+int(v.shape[-1]))+scores+(5*scores-2*rows)
        mask = kwargs.get("attn_mask") if name == "_scaled_dot_product_flash_attention_for_cpu" else _argument(args, kwargs, 3, "attn_mask")
        if isinstance(mask, torch.Tensor) and mask.dtype.is_floating_point: cost += scores
        if name == "_scaled_dot_product_flash_attention_for_cpu": cost += 2*rows  # Returned logsumexp: log(sum)+max.
        return cost
    if name == "addcmul":
        return n*(2+(0 if kwargs.get("value", 1) == 1 else 1))
    if name == "addcdiv":
        return n*(2+(0 if kwargs.get("value", 1) == 1 else 1))
    if name == "lerp":
        return 3*n
    if name in ("_weight_norm_interface", "_weight_norm"):
        _complex_factor(args, 1)
        vector, gain = args[:2]
        # norm over each vector: square+sum+sqrt=2N; divide gain and scale=N+G.
        return 3*vector.numel()+gain.numel()
    if name == "linalg_vector_norm":
        if _argument(args, kwargs, 1, "ord", 2) != 2:
            raise NotImplementedError("Only the real L2 vector norm reference is defined")
        _complex_factor(args, 1)
        return 2*args[0].numel()
    raise NotImplementedError("Floating operator has no audited arithmetic rule")


class _ArithmeticCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operators = {}; self.unsupported = {}; self.fft = []; self.scalar_transfers = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}; output = func(*args, **kwargs)
        qualified = str(func); parts = qualified.split(".")
        name = parts[1] if parts[0] == "aten" and len(parts) >= 2 else qualified
        entry = self.operators.setdefault(qualified, {"calls": 0, "counted_scalar_ops": 0})
        entry["calls"] += 1
        if name == "_local_scalar_dense": self.scalar_transfers += 1
        try:
            if name in ("_fft_r2c", "_fft_c2r", "_fft_c2c") or name.startswith("fft_"):
                self.fft.append(_fft_reference(name, args, kwargs, _first_tensor(output)))
                raise NotImplementedError("Native FFT algorithm is not identified")
            cost = int(_cost(name, args, kwargs, output))
            if cost < 0: raise ValueError("Negative arithmetic count")
            entry["counted_scalar_ops"] += cost
        except (NotImplementedError, IndexError, TypeError, KeyError, ZeroDivisionError) as error:
            if any(_floating(t) for t in _tensors((args, kwargs, output))):
                record = self.unsupported.setdefault(qualified, {"operator": qualified, "calls": 0,
                    "reason": str(error), "input_shapes": [list(t.shape) for t in _tensors(args)][:6],
                    "output_shapes": [list(t.shape) for t in _tensors(output)][:4]})
                record["calls"] += 1
        return output


@contextmanager
def _eval_preserved(model):
    states = [(module, module.training) for module in model.modules()] if isinstance(model, nn.Module) else []
    if states: model.eval()
    try:
        with torch.inference_mode(): yield
    finally:
        for module, training in states: module.training = training


@contextmanager
def _cpu_threads(count):
    if type(count) is not int or count != 1:
        raise ValueError("The common CPU measurement policy requires exactly one torch thread")
    previous = torch.get_num_threads()
    if previous != count: torch.set_num_threads(count)
    try: yield
    finally:
        if previous != count: torch.set_num_threads(previous)


def profile_model(model, x):
    """Profile one CPU eval forward with no optimizer, backward or real-data IO.

    The first input tensor's leading dimension is the number of examples. Views,
    RNG and integer shape/index costs are explicitly outside this convention.
    Fused unknown kernels produce partial coverage, never a fabricated zero.
    """
    if not isinstance(model, nn.Module): raise TypeError("A neural nn.Module is required")
    args = x if isinstance(x, tuple) else (x,)
    tensors = _tensors(args)
    if not tensors or tensors[0].ndim == 0 or tensors[0].shape[0] < 1:
        raise ValueError("A positive leading batch dimension is required")
    if any(t.device.type != "cpu" for t in tensors+list(model.parameters())+list(model.buffers())):
        raise ValueError("This profiler is for the common CPU protocol")
    batch = int(tensors[0].shape[0]); counter = _ArithmeticCounter(); started = time.perf_counter()
    with _eval_preserved(model), counter:
        model(*args)
    subtotal = sum(r["counted_scalar_ops"] for r in counter.operators.values())
    complete = not counter.unsupported
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameter_count_scope": "All registered parameters, including registered inactive modules; not FLOPs",
            "coverage": "complete_under_convention" if complete else "partial",
            "estimated_flops_total": subtotal if complete else None,
            "estimated_flops_per_sample": subtotal/batch if complete else None,
            "counted_flops_total": subtotal, "counted_flops_per_sample": subtotal/batch,
            "partial_count_is_hardware_lower_bound": False,
            "unsupported_floating_ops": list(counter.unsupported.values()),
            "operation_counts": counter.operators, "FFT_references": counter.fft,
            "Python_scalar_transfers_observed": counter.scalar_transfers,
            "input_shapes": [list(t.shape) for t in tensors], "batch_size": batch,
            "scope": "Observed eval-forward path only; data-dependent routing can change the path or count",
            "convention": CONVENTION, "torch_version": torch.__version__,
            "profiler_elapsed_seconds_not_model_latency": time.perf_counter()-started,
            "peak_memory_bytes": None, "memory_scope": "Not measured here; runner records process RSS separately",
            "models_fitted": 0, "market_files_opened": 0}


def _batch_size(x):
    if isinstance(x, torch.Tensor) and x.device.type != "cpu":
        raise ValueError("CPU throughput protocol does not measure asynchronous CUDA calls")
    try: size = len(x)
    except TypeError as error: raise ValueError("Batch input must have a leading sample dimension") from error
    if size < 1: raise ValueError("A positive batch size is required")
    return size


def _check_prediction_batch(output, size):
    first = _first_tensor(output)
    if first is not None and first.device.type != "cpu":
        raise ValueError("Throughput outputs must be CPU tensors, not asynchronous GPU results")
    target = first if first is not None else output
    try: observed = len(target)
    except TypeError as error: raise ValueError("Predictor must return one output per batch sample") from error
    if observed != size: raise ValueError("Output batch size differs; cannot claim its requested samples/sec")


def summarize_throughput_blocks(block_seconds, *, batch_size=32, calls_per_block=32):
    seconds = list(map(float, block_seconds))
    if not seconds or any(not math.isfinite(t) or t <= 0 for t in seconds):
        raise ValueError("Positive finite measured block durations are required")
    if type(batch_size) is not int or batch_size < 1 or type(calls_per_block) is not int or calls_per_block < 1:
        raise ValueError("Positive integer batch and call counts required")
    samples = batch_size*calls_per_block; rates = [samples/t for t in seconds]
    pooled = len(seconds)*samples/sum(seconds)
    return {"samples_per_second": pooled, "throughput_samples_sec": pooled,
            "batches_per_second": len(seconds)*calls_per_block/sum(seconds),
            "block_rate_mean_samples_sec": statistics.mean(rates),
            "block_rate_SD_samples_sec": statistics.stdev(rates) if len(rates) > 1 else None,
            "block_rate_median_samples_sec": statistics.median(rates),
            "batch_size": batch_size, "blocks": len(seconds), "calls_per_block": calls_per_block,
            "timed_batches": len(seconds)*calls_per_block, "timed_samples": len(seconds)*samples,
            "block_seconds": seconds, "total_timed_seconds": sum(seconds),
            "scope": "Measured frozen-predictor CPU throughput; pooled samples/total elapsed, not inverse batch1 latency",
            "cross_model_interleaving": "Caller responsibility; use measure_throughput_block for a shared interleaved schedule"}


def measure_throughput_block(predictor, x_batch, *, calls_per_block=32, threads=1):
    """One warmed timed block, suitable for caller-controlled model interleaving."""
    if type(calls_per_block) is not int or calls_per_block < 1:
        raise ValueError("Positive integer calls_per_block required")
    size = _batch_size(x_batch)
    if isinstance(predictor, nn.Module) and any(t.device.type != "cpu" for t in list(predictor.parameters())+list(predictor.buffers())):
        raise ValueError("The common throughput protocol requires a CPU predictor")
    with _cpu_threads(threads), _eval_preserved(predictor):
        started = time.perf_counter_ns()
        for _ in range(calls_per_block): output = predictor(x_batch)
        elapsed = (time.perf_counter_ns()-started)/1e9
        _check_prediction_batch(output, size)
    return {"block_seconds": elapsed, "batch_size": size,
            "calls_per_block": calls_per_block, "samples": size*calls_per_block, "torch_threads": threads}


def measure_throughput(predictor, x_batch, *, warmup=32, blocks=31, calls_per_block=32, threads=1):
    """Measure batch throughput on CPU; the caller supplies frozen predictor/data.

    Weights, preprocessing and data acquisition are outside this function. Warmup
    and timed forwards occur under inference_mode; original module flags and
    torch thread count are restored. Calls are synchronous CPU calls. Serialize
    measurements because torch's intra-op thread setting is process-global.
    """
    if type(warmup) is not int or warmup < 0 or type(blocks) is not int or blocks < 1:
        raise ValueError("Nonnegative warmup and positive integer blocks required")
    if type(calls_per_block) is not int or calls_per_block < 1:
        raise ValueError("Positive integer calls_per_block required")
    size = _batch_size(x_batch); durations = []
    with _cpu_threads(threads), _eval_preserved(predictor):
        for _ in range(warmup): output = predictor(x_batch)
        if warmup: _check_prediction_batch(output, size)
        for _ in range(blocks):
            row = measure_throughput_block(predictor, x_batch, calls_per_block=calls_per_block, threads=threads)
            durations.append(row["block_seconds"])
    report = summarize_throughput_blocks(durations, batch_size=size, calls_per_block=calls_per_block)
    return {**report, "warmup_calls": warmup, "torch_threads": threads,
            "helper_performs_weight_or_input_updates": False,
            "frozen_predictor_is_caller_precondition": True,
            "new_model_fits": 0, "clock": "time.perf_counter_ns, synchronousCPU"}
