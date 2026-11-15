"""Synthetic software/property checks, not Bitcoin experiments or evidence of R3."""
from dataclasses import replace
import io
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.original_model import (ABLATIONS, BoundedDiscordanceAttention,
                                CandidateConfig, OriginalModel, validate_input)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(batch=5, *, seed=731, dtype=torch.float32):
    return torch.randn(batch, 7, 5, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def test_seed_reproducibility_and_global_rng_isolation():
    before = torch.random.get_rng_state().clone()
    first = OriginalModel(seed=41)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    second = OriginalModel(seed=41)
    third = OriginalModel(seed=42)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    x = sample()
    torch.testing.assert_close(first(x), second(x), rtol=0, atol=0)
    assert not torch.allclose(first(x), third(x))


@pytest.mark.parametrize("bad_config", [0, False, {}, "not-a-config"])
def test_model_rejects_invalid_configuration_objects(bad_config):
    with pytest.raises(TypeError, match="CandidateConfig"):
        OriginalModel(bad_config)


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_shapes_positive_scaled_output_and_sample_independence(batch):
    model = OriginalModel(seed=3).eval()
    x = sample(batch)
    y = model(x)
    assert y.shape == (batch,)
    assert torch.isfinite(y).all() and (y > 0).all()
    alone = torch.cat([model(x[i:i + 1]) for i in range(batch)])
    torch.testing.assert_close(y, alone, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("bad", [torch.zeros(2, 6, 5), torch.zeros(2, 7, 4),
                                  torch.zeros(0, 7, 5), torch.zeros(7, 5),
                                  torch.zeros(2, 7, 5, dtype=torch.int64),
                                  torch.full((2, 7, 5), float("nan")),
                                  torch.full((2, 7, 5), float("inf")),
                                  [[0.0] * 5] * 7])
def test_invalid_input_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        OriginalModel()(bad)


@pytest.mark.parametrize("change", [{"radius": 0}, {"temperature_min": 0},
                                    {"initial_penalty": 2}, {"epsilon": float("nan")},
                                    {"input_features": 6}, {"hidden_size": True},
                                    {"ablation": "unknown"}])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError):
        CandidateConfig(**change)


def test_encoder_is_causal_and_reference_matrix_is_fixed():
    model = OriginalModel(seed=15).double().eval()
    x = sample(dtype=torch.float64)
    future_changed = x.clone()
    future_changed[:, 4:] += 7.0
    original, changed = model.encode(x), model.encode(future_changed)
    torch.testing.assert_close(original[:, :4], changed[:, :4], rtol=0, atol=0)
    assert not torch.allclose(original[:, 4:], changed[:, 4:])
    p = model.mechanism.P
    assert not p.requires_grad
    assert torch.count_nonzero(torch.triu(p, diagonal=1)) == 0
    torch.testing.assert_close(p.sum(dim=1), torch.ones(7, dtype=p.dtype), rtol=0, atol=4e-8)


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_all_trainable_parameters_receive_finite_nonzero_gradient(ablation):
    model = OriginalModel(replace(CandidateConfig(), ablation=ablation), seed=11)
    x = sample(batch=6)
    target = torch.linspace(0.3, 1.4, 6)
    loss = F.mse_loss(model(x), target)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0, name


def test_smooth_constraints_survive_large_raw_parameters():
    head = OriginalModel(seed=4).mechanism
    with torch.no_grad():
        head.query_raw.fill_(1e4)
        head.readout_raw.fill_(-1e4)
        head.penalty_raw.fill_(100)
        head.temperature_raw.fill_(-100)
    pars = head.effective_parameters()
    assert torch.linalg.vector_norm(pars["q"]) <= head.config.query_bound * (1 + 1e-6)
    assert torch.linalg.vector_norm(pars["w"]) <= head.config.readout_bound * (1 + 1e-6)
    assert 0 <= pars["penalty"] <= head.config.penalty_bound
    assert pars["temperature"] >= head.config.temperature_min


def test_latent_module_gradcheck_double():
    config = replace(CandidateConfig(), hidden_size=4)
    head = OriginalModel(config, seed=20).mechanism.double()
    states = torch.randn(2, 7, 4, generator=torch.Generator().manual_seed(12),
                         dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(head, (states,), eps=1e-6, atol=2e-5, rtol=1e-3)


def test_radial_values_are_bounded_and_numerically_one_lipschitz():
    head = OriginalModel(replace(CandidateConfig(), hidden_size=4), seed=14).mechanism.double()
    generator = torch.Generator().manual_seed(21)
    for scale in (0.0, 0.1, 1.0, 100.0):
        h = torch.randn(3, 7, 4, generator=generator, dtype=torch.float64) * scale
        other = h + torch.randn(3, 7, 4, generator=generator, dtype=torch.float64) * 0.03
        values = head.attention_components(h)["values"]
        other_values = head.attention_components(other)["values"]
        assert torch.linalg.vector_norm(values, dim=-1).max() <= head.config.radius + 1e-12
        assert torch.linalg.vector_norm(values - other_values) <= torch.linalg.vector_norm(h - other) + 1e-12


def test_latent_lipschitz_bound_on_fixed_pairs_and_jacobian():
    head = OriginalModel(replace(CandidateConfig(), hidden_size=4), seed=15).mechanism.double()
    bound = head.latent_lipschitz_bound()
    assert math.isfinite(bound) and bound > 0
    generator = torch.Generator().manual_seed(22)
    for scale in (0.0, 0.1, 1.0, 100.0):
        h = torch.randn(4, 7, 4, generator=generator, dtype=torch.float64) * scale
        other = h + torch.randn(4, 7, 4, generator=generator, dtype=torch.float64) * 1e-4
        differences = (head(h) - head(other)).abs()
        distances = torch.linalg.matrix_norm(h - other, ord="fro")
        assert torch.all(differences <= bound * distances + 1e-12)
    point = torch.randn(1, 7, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    gradient, = torch.autograd.grad(head(point).sum(), point)
    assert torch.linalg.vector_norm(gradient) <= bound + 1e-12


def test_three_ablation_components_are_isolated_and_distinct():
    full = OriginalModel(seed=7)
    states = torch.randn(3, 7, 12, generator=torch.Generator().manual_seed(36)) * 2
    baseline = full.mechanism.attention_components(states)
    outputs = [full.mechanism(states)]
    for ablation in ABLATIONS[1:]:
        variant = OriginalModel(replace(full.config, ablation=ablation), seed=7)
        # Bind shared parameters exactly. Ablations omit removed parameters.
        keep = {k: v for k, v in full.state_dict().items() if k in variant.state_dict()}
        variant.load_state_dict(keep, strict=True)
        parts = variant.mechanism.attention_components(states)
        if ablation == "without_penalty":
            assert variant.mechanism.penalty_raw is None
            torch.testing.assert_close(parts["values"], baseline["values"])
        elif ablation == "without_saturation":
            torch.testing.assert_close(parts["weights"], baseline["weights"])
            torch.testing.assert_close(parts["values"], states)
            with pytest.raises(ValueError, match="requires radial"):
                variant.mechanism.latent_lipschitz_bound()
        else:
            assert variant.mechanism.query_raw is None
            assert torch.count_nonzero(parts["content"]) == 0
            torch.testing.assert_close(parts["values"], baseline["values"])
        outputs.append(variant.mechanism(states))
    for i, first in enumerate(outputs):
        for second in outputs[i + 1:]:
            assert not torch.allclose(first, second)


def test_synthetic_tiny_batch_can_be_overfit():
    config = replace(CandidateConfig(), cnn_channels=6, hidden_size=8)
    model = OriginalModel(config, seed=18)
    x = sample(batch=8, seed=45)
    target = F.softplus(0.6 * x[:, -1, 0] - 0.3 * x[:, -2, 4]) + 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = F.mse_loss(model(x), target).item()
    for _ in range(350):
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    final = F.mse_loss(model(x), target).item()
    assert final < 0.005
    assert final < initial * 0.02


def test_save_load_and_torchscript_roundtrip(tmp_path):
    model = OriginalModel(seed=37).eval()
    x = sample(batch=4)
    memory = model(x).detach()
    buffer = io.BytesIO()
    torch.save({"config": model.configuration(), "state_dict": model.state_dict()}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = OriginalModel(CandidateConfig(**saved["config"]["config"]), seed=999).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(x), memory, rtol=0, atol=0)
    with torch.no_grad():
        traced = torch.jit.trace(model, x[:1], check_inputs=[(x[:3],)], strict=True)
        destination = tmp_path / "candidate_synthetic_smoke.torchscript.pt"
        traced.save(str(destination))
        exported = torch.jit.load(str(destination)).eval()
        validate_input(x, model.config.history_days, model.config.input_features)
        torch.testing.assert_close(exported(x), memory, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        validate_input(torch.zeros(1, 6, 5))
