"""Synthetic CPU audit of the *evaluated additive* latent head, not market fitting.

The bound is a real-arithmetic theorem on bounded latent states with fixed A,
scale and parameters. FP32/64 measurements are finite numerical checks only.
Run from repository root with the existing PyTorch CPU environment:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
    experiments/e3_diseno_riguroso_20261117/verify_theory.py
No checkpoint, market array, optimizer or GPU is loaded or used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from src.e3_innovation_model import InnovationModel, MIN_NORMALIZED_SCALE


SEED = 2026111707
N, D = 7, 12
SCALE, FIXED_A = 0.02, 0.0001  # declared synthetic constants, not fitted statistics
Q, W, TAU_MIN = 1.0, 4.0, 0.25
EXPECTED_SOURCES = {
    "src/original_model.py": "dce48ff772cd284c63a672c599febc66dfc6eb14f286f523703fe6b01d086c2f",
    "src/e3_development.py": "33dd121f0dd36ef1933c5f4338045fa5a17e85ad6986dfd2c1a43c2662158ebe",
    "src/e3_innovation_model.py": "7f362d38ca83ed302dc9c822b4e4e19a0d446e4c01477f42b0e9a8da823ae7b1",
}


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mixed_head(model, states):
    """Exact frozen class calls from the latent states onward; no encoder call."""
    mechanism = model.base.mechanism
    context = mechanism.attention_components(states)["context"]
    score = context.to(torch.float64) @ mechanism.effective_parameters()["w"]
    score = score + model.bias
    floor = torch.full_like(score, FIXED_A)
    return model._components_from_score(score, floor)["prediction"]


def real_reference(model, states):
    """Same real formula in FP64 using the effective parameter values of class."""
    pars = model.base.mechanism.effective_parameters()
    states = states.to(torch.float64)
    q = pars["q"].detach().to(torch.float64)
    tau = pars["temperature"].detach().to(torch.float64)
    w = pars["w"].detach().to(torch.float64)
    bias = model.bias.detach().to(torch.float64)
    alpha = torch.softmax((states @ q) / tau, dim=-1)
    context = (alpha.unsqueeze(-1) * states).sum(dim=1)
    score = context @ w + bias
    return math.sqrt(FIXED_A) + SCALE * (MIN_NORMALIZED_SCALE + F.softplus(score))


def build(profile):
    model = InnovationModel("additive_positive_head", u_scale=0.01,
                            y_scale=SCALE, seed=SEED).cpu().eval()
    m = model.base.mechanism
    assert not m.use_penalty and not m.use_saturation
    assert sum(p.numel() for p in model.parameters()) == 1210
    with torch.no_grad():
        if profile != "seeded_initialization":
            direction = torch.linspace(-1.0, 1.0, D)
            direction = direction / torch.linalg.vector_norm(direction)
            m.query_raw.copy_(direction * (0.0 if profile == "uniform_attention" else 200.0))
            m.readout_raw.copy_(direction.double() * 200.0)
            m.temperature_raw.fill_(-20.0)
            m.bias.fill_(4.0)
    return model


def run():
    torch.set_num_threads(1)
    source_records = {}
    for path, expected in EXPECTED_SOURCES.items():
        actual = checksum(ROOT / path)
        if actual != expected:
            raise RuntimeError("Frozen source mismatch: " + path)
        source_records[path] = {"sha256": actual, "matches_frozen_source": True}
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    normalized_bound = W * (1.0 + math.sqrt(N * D) * Q / (2.0 * TAU_MIN))
    bound = SCALE * normalized_bound
    records = []
    for profile in ("seeded_initialization", "near_parameter_bounds", "uniform_attention"):
        model = build(profile)
        pars = model.base.mechanism.effective_parameters()
        measured = {"q_norm": float(torch.linalg.vector_norm(pars["q"]).detach()),
                    "w_norm": float(torch.linalg.vector_norm(pars["w"]).detach()),
                    "temperature": float(pars["temperature"].detach())}
        # Float32 representations near the norm boundary allow a small margin.
        assert measured["q_norm"] <= Q + 2e-7
        assert measured["w_norm"] <= W + 2e-12
        assert measured["temperature"] >= TAU_MIN
        quotients, normalized_quotients, discrepancies = [], [], []
        actual_steps = []
        with torch.no_grad():
            for amplitude in (0.001, 0.1, 1.0):
                for perturbation in (0.001, 0.03, 0.3):
                    states = (torch.rand(64, N, D, generator=generator) * 2 - 1) * amplitude
                    change = torch.randn(64, N, D, generator=generator) * perturbation
                    altered = (states + change).clamp(-1.0, 1.0)
                    distance = torch.linalg.vector_norm((altered - states).double(), dim=(1, 2))
                    assert bool((distance > 0).all())
                    assert bool((states.abs() <= 1).all() and (altered.abs() <= 1).all())
                    first, second = mixed_head(model, states), mixed_head(model, altered)
                    delta = (second - first).abs()
                    tolerance = 1e-7 * SCALE + 1e-12
                    assert bool((delta <= bound * distance + tolerance).all())
                    quotients.extend((delta / distance).tolist())
                    normalized_quotients.extend((delta / distance / SCALE).tolist())
                    actual_steps.extend(distance.tolist())
                    discrepancies.extend((first - real_reference(model, states)).abs().tolist())
                    assert bool((first >= math.sqrt(FIXED_A)).all())
        # FP64 reference is differentiable; its Jacobian norm is compared to K.
        states64 = (torch.rand(32, N, D, generator=generator, dtype=torch.float64) * 2 - 1)
        states64.requires_grad_(True)
        reference = real_reference(model, states64)
        gradient, = torch.autograd.grad(reference.sum(), states64)
        gradient_norm = torch.linalg.vector_norm(gradient, dim=(1, 2))
        assert bool(torch.isfinite(gradient_norm).all())
        assert bool((gradient_norm <= bound + 1e-12).all())
        records.append({
            "profile": profile,
            "effective_parameters": measured,
            "pairs": len(quotients),
            "max_mixed_precision_difference_quotient": max(quotients),
            "max_quotient_divided_by_y_scale": max(normalized_quotients),
            "min_pair_Frobenius_distance": min(actual_steps),
            "max_pair_Frobenius_distance": max(actual_steps),
            "max_mixed_vs_FP64_reference_absolute_difference": max(discrepancies),
            "FP64_reference_gradient_cases": 32,
            "max_FP64_reference_gradient_Frobenius_norm": float(gradient_norm.max()),
            "violations": 0,
        })
    # A toy input confirms only the LSTM state bound, not a global encoder bound.
    model = build("seeded_initialization")
    with torch.no_grad():
        inputs = torch.randn(32, N, 5, generator=generator)
        encoded = model.base.encode(inputs)
    assert encoded.shape == (32, N, D)
    assert bool(torch.isfinite(encoded).all() and (encoded.abs() <= 1).all())
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "15f6f664217dc9a88dc12c6ce22ad1516ff9e384",
        "frozen_model_source_commit": "cf891213f1acb6967aef5e0857fa0ac63768f836",
        "seed": SEED,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "threads": 1},
        "model": {"variant": "additive_positive_head", "parameters": 1210,
                  "n": N, "d": D, "row_norm_bound": math.sqrt(D)},
        "synthetic_constants": {"y_scale": SCALE, "fixed_A": FIXED_A,
                                "scale_is_not_fitted_to_market": True},
        "theorem": {"Q": Q, "W": W, "tau_min": TAU_MIN,
                    "normalized_latent_bound": normalized_bound,
                    "latent_bound_in_output_units": bound,
                    "arithmetic": "real", "A_and_scale_fixed": True},
        "numerical_checks": {"total_pairs": sum(r["pairs"] for r in records),
                             "total_reference_gradient_cases": 96,
                             "amplitudes": [0.001, 0.1, 1.0],
                             "perturbation_standard_deviations": [0.001, 0.03, 0.3],
                             "latent_coordinates_clipped_to": [-1.0, 1.0],
                             "pair_bound_absolute_slack": 1e-7 * SCALE + 1e-12,
                             "profiles": records,
                             "violations": 0},
        "encoder_domain_sanity": {"synthetic_windows": 32,
                                  "max_absolute_hidden_coordinate": float(encoded.abs().max()),
                                  "global_encoder_bound_proved": False},
        "sources": source_records,
        "verification_script_sha256": checksum(__file__),
        "scope": {"uses_trained_market_weights": False, "market_arrays_loaded": False,
                  "market_training": False, "market_inference": False,
                  "R1_R2_R3_certified": False, "novel_theorem_claimed": False,
                  "empirical_predictive_benefit_established": False,
                  "FP32_universal_Lipschitz_guarantee_claimed": False},
        "limitations": [
            "Fixed inference parameters, fixed A and scale; bounded latent domain only.",
            "The real-arithmetic bound is not a proof for every finite-precision difference quotient.",
            "The finite checks use seeded initial and deliberately stressed parameters, not trained weights.",
            "No bound for raw OHLCV, preprocessor or encoder is certified.",
            "A varying floor adds |sqrt(A)-sqrt(A')|, not a uniform Lipschitz constant in A near zero.",
            "No ablation, forecasting improvement, novelty or academic defence follows from these checks.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("theory_verification.json"))
    args = parser.parse_args()
    record = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pairs": record["numerical_checks"]["total_pairs"],
                      "violations": 0, "normalized_bound": record["theorem"]["normalized_latent_bound"],
                      "output": str(args.output)}))
