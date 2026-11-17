"""Synthetic model, loss and fit-only initialization contracts; no market data."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import e3_moment_model as design
from src.e3_development import DevelopmentModel


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _features(n=8):
    return torch.randn(n, 7, 5, generator=torch.Generator().manual_seed(782))


def _model(name="moment_joint", **kwargs):
    options = dict(u_scale=.03, y_scale=.02, mean_v=.0018,
                   initial_mean_excess=.003, seed=711)
    options.update(kwargs)
    return design.MomentModel(name, **options)


def _bar(day):
    k = (day - date(2024, 1, 1)).days
    close = math.exp(5.0 + .0005 * k + .03 * math.sin(k / 3.0))
    opening = close * (1 + .001 * math.cos(k))
    return [opening, max(opening, close) * 1.01, min(opening, close) * .99,
            close, 300 + 2 * k + k % 5]


def _arrays():
    dates = np.array(["2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30",
                      "2024-07-08", "2024-07-09"])
    raw, targets = [], []
    for text in dates:
        origin = date.fromisoformat(text)
        eight = np.asarray([_bar(origin - timedelta(days=lag)) for lag in range(7, -1, -1)])
        raw.append(eight[:7])
        targets.append(np.std(np.diff(np.log(eight[:, 3])), ddof=0))
    return {"training_raw": np.asarray(raw), "training_y_original": np.asarray(targets),
            "training_dates": dates,
            "calibration_raw": np.array([{"forbidden": True}], dtype=object),
            "evaluation_y_original": np.array([{"forbidden": True}], dtype=object)}


def _save(tmp_path, arrays, name):
    path = tmp_path / name
    np.savez(path, **arrays)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", design.NEURAL_NAMES)
def test_exact_active_parameter_count_precisions_and_absence_of_old_readout(name):
    model = _model(name)
    assert sum(p.numel() for p in model.parameters()) == 1236
    assert model.configuration()["backbone_parameters"] == 1197
    assert model.readout_weight.shape == (3, 12) and model.readout_bias.shape == (3,)
    assert model.readout_weight.dtype == model.readout_bias.dtype == torch.float64
    assert model.query_raw.dtype == model.temperature_raw.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in model.encoder.parameters())
    assert all("readout_raw" not in n and "penalty_raw" not in n for n, _ in model.named_parameters())
    assert model.configuration()["prediction_interval_claimed"] is False
    assert model.configuration()["R1_R3_verified"] is False


def test_constructor_preserves_global_rng_and_shares_exact_frozen_M00_backbone():
    torch.manual_seed(81)
    state = torch.get_rng_state().clone()
    models = [_model(n) for n in design.NEURAL_NAMES]
    assert torch.equal(state, torch.get_rng_state())
    baseline = DevelopmentModel("M00", seed=711).model
    for model in models:
        for key, value in baseline.encoder.state_dict().items():
            torch.testing.assert_close(value, model.encoder.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(model.query_raw, baseline.mechanism.query_raw, rtol=0, atol=0)
        torch.testing.assert_close(model.temperature_raw, baseline.mechanism.temperature_raw, rtol=0, atol=0)
        values = _features()
        expected = baseline.mechanism.attention_components(baseline.encode(values))["context"]
        torch.testing.assert_close(model.context(values), expected, rtol=0, atol=0)


def test_moment_variants_start_with_identical_predictions_and_every_parameter():
    first, second = _model(), _model("moment_mean_only")
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    A = torch.linspace(0, .001, 8, dtype=torch.float64)
    torch.testing.assert_close(first(_features(), A), second(_features(), A), rtol=0, atol=0)
    out = first.forward_components(_features(), A)
    torch.testing.assert_close(out["m"], torch.full_like(A, .0018), rtol=2e-15, atol=0)
    torch.testing.assert_close(out["q"], 9 * out["m"].square(), rtol=2e-15, atol=0)
    torch.testing.assert_close(out["rho"], torch.full_like(A, .5), rtol=0, atol=0)


def test_torch_operator_matches_independent_formula_and_actual_derivative_gradcheck():
    A = torch.tensor([0., .02, .3], dtype=torch.float64)
    a = torch.tensor([-.7, .4, 2.], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1.1, -1.7, .8], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([.7, 1., -1.5], dtype=torch.float64, requires_grad=True)
    out = design.moment_head(A, a, b, d, u_scale=.4)
    m = .4 ** 2 * np.logaddexp(0, a.detach().numpy())
    q = m ** 2 * (1 + b.detach().numpy() ** 2)
    rho = d.detach().numpy() ** 2 / (1 + d.detach().numpy() ** 2)
    floor = np.sqrt(A.numpy())
    lower = floor + design.C * m / (np.sqrt(A.numpy() + design.C * q / m) + floor)
    upper = np.sqrt(A.numpy() + design.C * m)
    np.testing.assert_allclose(out["prediction"].detach(), lower + rho * (upper - lower), rtol=3e-15)
    assert torch.autograd.gradcheck(lambda x, y, z: design.moment_head(A, x, y, z, u_scale=.4)["prediction"],
                                  (a, b, d), eps=1e-6, atol=2e-8, rtol=2e-6)


def test_mean_envelope_contains_mean_but_not_individual_future_observations():
    lower, upper = design.moment_bounds(0., 1., 4.)
    root = math.sqrt(design.C)
    assert lower.item() == pytest.approx(.5 * root, rel=2e-15)
    assert upper.item() == pytest.approx(root, rel=2e-15)
    # For V=0 with probability .75 and V=4 with probability .25, each
    # observation can lie outside the region that constrains its conditional mean.
    assert 0 < lower < upper < 2 * root
    assert (.75 * 0 + .25 * 2 * root) == pytest.approx(lower.item())


def test_degenerate_second_moment_has_exact_equal_endpoint_branch_and_flat_d():
    a = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    d = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    out = design.moment_head(.1, a, b, d, u_scale=.3)
    assert out["L"] == out["H"] == out["prediction"]
    gradients = torch.autograd.grad(out["prediction"], (b, d))
    assert gradients[0] == gradients[1] == 0


def test_lower_endpoint_is_reachable_and_its_d_zero_flat_point_is_explicit():
    a = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    d = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    out = design.moment_head(.03, a, b, d, u_scale=.2)
    assert out["prediction"] == out["L"] < out["H"]
    assert torch.autograd.grad(out["prediction"], d)[0] == 0


@pytest.mark.parametrize("coordinates", [(-1000., 1., 1.), (1e200, 1., 1.),
                                          (1., 1e200, 1.), (1., 1., 1e200)])
def test_unrepresentable_reports_abort_without_clipping_or_positive_floor(coordinates):
    with pytest.raises(ValueError):
        design.moment_head(0., *coordinates, u_scale=1.)
    with pytest.raises(ValueError):
        design.moment_bounds(0., 1e-200, 1e-200)


def test_float_rounding_at_upper_endpoint_is_exposed_without_changing_report():
    out = design.moment_head(.2, 1., 1., 1e12, u_scale=.2)
    assert out["rho_rounded_to_one"]
    assert out["prediction"] == out["H"]
    assert out["upper_endpoint_reached_by_roundoff"]


def test_loss_uses_exact_normalized_mean_scores_and_is_unit_invariant():
    A = torch.tensor([.01, .02, .04], dtype=torch.float64)
    out = design.moment_head(A, torch.tensor([.7, 1., -.2], dtype=torch.float64), 1.2, .8, u_scale=.3)
    Y, U = torch.tensor([.2, .3, .4]), torch.tensor([0., .15, .7])
    terms = design.loss_components(out, Y, U, y_scale=.2, u_scale=.3, lambda_m=.1, lambda_q=.1)
    pred, m, q = (out[k].detach().numpy() for k in ("prediction", "m", "q"))
    v = (U.numpy().astype(np.float64) / .3) ** 2
    mn, qn = m / .3 ** 2, q / .3 ** 4
    expected_m, expected_q = np.mean(np.log(mn) + v / mn), np.mean(np.log(qn) + v ** 2 / qn)
    expected_mse = np.mean(((pred - Y.numpy()) / .2) ** 2)
    assert terms["m_score"].item() == pytest.approx(expected_m, rel=2e-14)
    assert terms["q_score"].item() == pytest.approx(expected_q, rel=2e-14)
    assert terms["loss"].item() == pytest.approx(expected_mse + .1 * (expected_m + expected_q), rel=2e-14)
    k = 7.
    scaled = dict(out, prediction=out["prediction"] * k, m=out["m"] * k ** 2, q=out["q"] * k ** 4)
    physical = design.loss_components(scaled, Y.double() * k, U.double() * k,
                                      y_scale=.2 * k, u_scale=.3 * k, lambda_m=.1, lambda_q=.1)
    for key in terms:
        torch.testing.assert_close(terms[key], physical[key], rtol=2e-14, atol=2e-14)


def test_same_head_mse_only_and_joint_have_different_auxiliary_gradients():
    mse_only, joint = _model("moment_mean_only"), _model()
    X, A = _features(), torch.linspace(0, .001, 8, dtype=torch.float64)
    Y, U = torch.linspace(.015, .055, 8), torch.linspace(.01, .1, 8)
    first = mse_only.loss_components(mse_only.forward_components(X, A), Y, U)
    second = joint.loss_components(joint.forward_components(X, A), Y, U)
    assert first["loss"] == first["mse_scaled"]
    torch.testing.assert_close(first["mse_scaled"], second["mse_scaled"], rtol=0, atol=0)
    first["loss"].backward()
    second["loss"].backward()
    assert not torch.equal(mse_only.readout_bias.grad, joint.readout_bias.grad)
    assert mse_only.configuration()["moment_identification_claimed"] is False


@pytest.mark.parametrize("name", design.NEURAL_NAMES)
def test_every_registered_parameter_affects_loss_after_nonzero_readout_weights(name):
    model = _model(name)
    with torch.no_grad():
        model.readout_weight.copy_(torch.linspace(-.2, .3, 36, dtype=torch.float64).reshape(3, 12))
    X, A = _features(), torch.linspace(0, .001, 8, dtype=torch.float64)
    out = model.forward_components(X, A)
    model.loss_components(out, torch.linspace(.015, .055, 8), torch.linspace(.01, .1, 8))["loss"].backward()
    for key, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
        assert parameter.grad.abs().sum() > 0, key


def test_zero_initial_readouts_give_zero_initial_encoder_gradient_but_active_readouts():
    model = _model()
    X, A = _features(), torch.linspace(0, .001, 8, dtype=torch.float64)
    model.loss_components(model.forward_components(X, A), torch.full((8,), .04), torch.full((8,), .03))["loss"].backward()
    assert model.readout_weight.grad.abs().sum() > 0
    for parameter in (*model.encoder.parameters(), model.query_raw, model.temperature_raw):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0


def test_observed_A_is_detached_at_zero_and_future_labels_are_not_forward_inputs():
    model = _model()
    X, A = _features(), torch.zeros(8, dtype=torch.float64, requires_grad=True)
    output = model(X, A)
    assert output.dtype == torch.float64 and torch.isfinite(output).all()
    output.sum().backward()
    assert A.grad is None
    with pytest.raises(TypeError):
        model.forward_components(X, A, U=torch.ones(8))
    with pytest.raises(ValueError):
        model(X.double(), A)
    with pytest.raises(ValueError):
        model(X, A[:2])


def test_fit_only_statistics_and_initial_mean_matching_ignore_validation_changes(tmp_path):
    arrays = _arrays()
    first_path, first_sha = _save(tmp_path, arrays, "first.npz")
    changed = deepcopy(arrays)
    changed["training_raw"][4:, :, 4] *= 10000
    changed["training_y_original"][4:] *= 4
    second_path, second_sha = _save(tmp_path, changed, "second.npz")
    first = design.load_moment_data(first_path, first_sha)
    second = design.load_moment_data(second_path, second_sha)
    for key in ("u_scale", "y_scale", "mean_v", "initial_mean_excess"):
        assert first["scaler"][key] == second["scaler"][key]
    assert first["scaler"]["mean_v"] == pytest.approx(np.mean(first["train_U"] ** 2), rel=2e-15)
    assert first["admission"]["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
    assert first["admission"]["forbidden_arrays_loaded"] is False
    assert first["admission"]["moment_initialization_uses_validation"] is False
    arguments = {key: first["scaler"][key] for key in ("u_scale", "y_scale", "mean_v", "initial_mean_excess")}
    models = [design.MomentModel(name, **arguments) for name in design.NEURAL_NAMES]
    X, A = torch.from_numpy(first["train_X"]), torch.from_numpy(first["train_A"])
    means = [(model(X, A) - A.sqrt()).mean() for model in models]
    for actual in means:
        assert actual.item() == pytest.approx(arguments["initial_mean_excess"], rel=2e-13, abs=1e-16)
    # Matching the fit mean is not falsely claimed as pointwise identity.
    assert not torch.equal(models[0](X, A), models[2](X, A))


def test_state_roundtrip_preserves_all_three_outputs():
    first, second = _model(), _model(seed=9)
    second.load_state_dict(first.state_dict())
    X, A = _features(), torch.linspace(0, .001, 8)
    for key in ("prediction", "m", "q", "rho"):
        torch.testing.assert_close(first.forward_components(X, A)[key],
                                   second.forward_components(X, A)[key], rtol=0, atol=0)


def test_invalid_reports_scales_and_targets_fail_explicitly():
    with pytest.raises(ValueError):
        _model("direct_moment_aux", initial_mean_excess=None)
    with pytest.raises(ValueError):
        _model(u_scale=0.)
    with pytest.raises(ValueError):
        _model(u_scale=1e200)
    with pytest.raises(ValueError):
        design.moment_bounds(0., 1., .9)
    out = _model().forward_components(_features(), torch.zeros(8))
    with pytest.raises(ValueError):
        design.loss_components(out, torch.ones(8), -torch.ones(8), y_scale=.2, u_scale=.3)
    with pytest.raises(ValueError):
        design.loss_components(out, torch.ones(8), torch.ones(8), y_scale=.2, u_scale=.3, lambda_m=True)
