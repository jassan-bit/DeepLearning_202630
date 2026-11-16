"""Synthetic scientific contracts for V2; no market arrays or market fits."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.e3_innovation_model as design


NAMES = ("additive_positive_head", "innovation_mean_only", "innovation_density_mean")


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _bar(day):
    k = (day - date(2024, 1, 1)).days
    close = math.exp(5.0 + 0.0005 * k + 0.03 * math.sin(k / 3.0))
    opening = close * (1 + 0.001 * math.cos(k))
    return [opening, max(opening, close) * 1.01,
            min(opening, close) * 0.99, close, 300 + 2 * k + k % 5]


def _arrays():
    dates = np.array(["2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30",
                      "2024-07-08", "2024-07-09"])
    raw, target = [], []
    for text in dates:
        origin = date.fromisoformat(text)
        bars = [_bar(origin - timedelta(days=lag)) for lag in range(7, -1, -1)]
        raw.append(bars[:7])
        target.append(np.std(np.diff(np.log(np.array(bars)[:, 3])), ddof=0))
    return {"training_raw": np.asarray(raw, dtype=np.float64),
            "training_y_original": np.asarray(target, dtype=np.float64),
            "training_dates": dates}


def _save(tmp_path, arrays=None, name="synthetic.npz"):
    path = tmp_path / name
    np.savez(path, **(_arrays() if arrays is None else arrays))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _model(name="innovation_density_mean", seed=17):
    return design.InnovationModel(name, seed=seed, u_scale=.03, y_scale=.02)


def _features():
    generator = torch.Generator().manual_seed(218)
    return torch.randn(6, 7, 5, generator=generator)


def _bias(model):
    matches = [value for name, value in model.named_parameters() if name.endswith("bias")
               and value.numel() == 1]
    assert len(matches) == 1
    return matches[0]


def _assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_fit_only_statistics_and_magnitudes_match_eight_close_population_target(tmp_path):
    arrays = _arrays()
    path, sha = _save(tmp_path, arrays)
    data = design.load_innovation_data(path, sha)
    raw, y = arrays["training_raw"], arrays["training_y_original"]
    r = np.diff(np.log(raw[:, :, 3]), axis=1)
    A = ((r - r.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / 7
    U = np.sqrt(np.maximum((y ** 2 - A) / (6 / 49), 0))
    future_returns = np.array([math.log(_bar(date.fromisoformat(text))[3])
                               - math.log(raw[i, -1, 3])
                               for i, text in enumerate(arrays["training_dates"])])
    np.testing.assert_allclose(U, np.abs(future_returns-r.mean(axis=1)), rtol=1e-11, atol=1e-14)
    transformed = raw.copy()
    transformed[:, :, :4] = np.log(raw[:, :, :4]) - np.log(raw[:, -1, 3])[:, None, None]
    transformed[:, :, 4] = np.log1p(raw[:, :, 4])
    mu, sd = transformed[:4].mean(axis=(0, 1)), transformed[:4].std(axis=(0, 1))
    np.testing.assert_allclose(data["train_X"], ((transformed[:4] - mu) / sd).astype(np.float32),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data["train_A"], A[:4], rtol=1e-13, atol=1e-18)
    np.testing.assert_allclose(data["train_U"], U[:4], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(data["val_A"], A[4:], rtol=1e-13, atol=1e-18)
    assert data["scaler"]["u_scale"] == pytest.approx(U[:4].mean(), rel=1e-12)
    assert data["scaler"]["y_scale"] == pytest.approx(y[:4].mean(), rel=1e-14)
    assert data["scaler"]["d_fit"] == pytest.approx(np.mean(y[:4] - np.sqrt(A[:4])), rel=1e-12)
    assert data["train_X"].dtype == np.float32 and data["val_X"].dtype == np.float32
    for key in ("train_A", "train_U", "train_y_original", "val_A", "val_U", "val_y_original"):
        assert data[key].dtype == np.float64
    assert len(data["train_dates"]) == 4 and len(data["val_dates"]) == 2


def test_validation_changes_cannot_change_fit_scalers_or_control_offset(tmp_path):
    a = _arrays()
    p1, h1 = _save(tmp_path, a, "first.npz")
    b = deepcopy(a)
    b["training_raw"][4:, :, 4] *= 10000
    b["training_y_original"][4:] *= 5
    p2, h2 = _save(tmp_path, b, "second.npz")
    first, second = design.load_innovation_data(p1, h1), design.load_innovation_data(p2, h2)
    for key in ("train_X", "train_A", "train_U", "train_y_original"):
        np.testing.assert_array_equal(first[key], second[key])
    for key in ("mean", "std", "u_scale", "y_scale", "d_fit"):
        np.testing.assert_array_equal(first["scaler"][key], second["scaler"][key])
    assert not np.array_equal(first["val_U"], second["val_U"])


def test_npz_admission_reads_only_three_raw_training_members(tmp_path, monkeypatch):
    arrays = _arrays()
    arrays["training_X"] = np.array([{"not_admitted": True}], dtype=object)
    arrays["calibration_raw"] = np.array([{"not_admitted": True}], dtype=object)
    arrays["evaluation_y_original"] = np.array([{"not_admitted": True}], dtype=object)
    path, sha = _save(tmp_path, arrays)
    original_load, accessed = design.np.load, []
    allowed = {"training_raw", "training_y_original", "training_dates"}

    class Guard:
        def __init__(self, source):
            self.source, self.files = source, source.files

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.source.close()

        def __contains__(self, name):
            return name in self.files

        def __getitem__(self, name):
            accessed.append(name)
            assert name in allowed, name
            return self.source[name]

    monkeypatch.setattr(design.np, "load", lambda *a, **kw: Guard(original_load(*a, **kw)))
    data = design.load_innovation_data(path, sha)
    assert set(accessed) == allowed
    assert len(data["train_X"]) == 4


def test_wrong_hash_is_rejected_before_npz_parsing(tmp_path, monkeypatch):
    path, _ = _save(tmp_path)
    monkeypatch.setattr(design.np, "load", lambda *_a, **_k: pytest.fail("Unadmitted bytes parsed"))
    with pytest.raises(ValueError):
        design.load_innovation_data(path, "0" * 64)


def test_impossible_variance_in_a_purged_row_aborts_without_dropping_it(tmp_path):
    a = _arrays()
    a["training_dates"] = np.insert(a["training_dates"], 4, "2024-07-04")
    a["training_raw"] = np.insert(a["training_raw"], 4, a["training_raw"][3], axis=0)
    a["training_y_original"] = np.insert(a["training_y_original"], 4, 0.)
    path, sha = _save(tmp_path, a)
    with pytest.raises(ValueError):
        design.load_innovation_data(path, sha)


def test_only_roundoff_negative_margin_is_clamped_without_removing_rows(tmp_path):
    a = _arrays()
    r = np.diff(np.log(a["training_raw"][0, :, 3]))
    A = np.sum((r - r.mean()) ** 2) / 7
    a["training_y_original"][0] = np.sqrt(A - 5e-13)
    path, sha = _save(tmp_path, a)
    data = design.load_innovation_data(path, sha)
    assert data["train_U"][0] == 0
    assert len(data["train_X"]) == 4 and len(data["val_X"]) == 2
    assert data["admission"]["roundoff_margins_clamped"] == 1
    assert data["admission"]["rows_dropped"] == 0


@pytest.mark.parametrize("problem", ["bad_shape", "nan_y", "invalid_bar", "duplicate_date",
                                    "unordered_date", "overlap", "zero_U"])
def test_invalid_data_or_degenerate_fit_is_rejected(tmp_path, problem):
    a, kwargs = _arrays(), {}
    if problem == "bad_shape":
        a["training_raw"] = a["training_raw"][:, :6]
    elif problem == "nan_y":
        a["training_y_original"][0] = np.nan
    elif problem == "invalid_bar":
        a["training_raw"][0, 0, 1] = a["training_raw"][0, 0, 2] - 1
    elif problem == "duplicate_date":
        a["training_dates"][1] = a["training_dates"][0]
    elif problem == "unordered_date":
        a["training_dates"][[0, 1]] = a["training_dates"][[1, 0]]
    elif problem == "overlap":
        kwargs["validation_start"] = "2024-07-07"
    else:
        r = np.diff(np.log(a["training_raw"][:, :, 3]), axis=1)
        a["training_y_original"] = np.sqrt(((r-r.mean(axis=1, keepdims=True))**2).sum(axis=1)/7)
    path, sha = _save(tmp_path, a)
    with pytest.raises((ValueError, RuntimeError)):
        design.load_innovation_data(path, sha, **kwargs)


def test_three_variants_share_seeded_parameters_with_only_final_readout_in_double():
    assert tuple(design.NEURAL_NAMES) == NAMES
    before = torch.random.get_rng_state().clone()
    models = {name: _model(name) for name in NAMES}
    torch.testing.assert_close(before, torch.random.get_rng_state(), atol=0, rtol=0)
    reference = dict(models[NAMES[0]].named_parameters())
    assert sum(x.numel() for x in reference.values()) == 1210
    for model in models.values():
        parameters = dict(model.named_parameters())
        assert parameters.keys() == reference.keys()
        for name, parameter in parameters.items():
            torch.testing.assert_close(parameter, reference[name], rtol=0, atol=0)
            is_readout = name.endswith("readout_raw") or (name.endswith("bias") and parameter.numel() == 1)
            assert parameter.dtype == (torch.float64 if is_readout else torch.float32), name
        assert not any("penalty_raw" in name for name in parameters)


def test_readout_is_linear_in_double_and_each_head_applies_softplus_once():
    X, A = _features(), torch.tensor([0, 1e-5, 2e-5, 1e-4, .0004, .001], dtype=torch.float64)
    for name in NAMES:
        model = _model(name)
        components = model.forward_components(X, A)
        score = model.raw_score(X)
        assert score.dtype == torch.float64 and components["prediction"].dtype == torch.float64
        states = model.base.encode(X)
        context = model.base.mechanism.attention_components(states)["context"]
        readout_raw = model.base.mechanism.readout_raw
        readout = 4.0*readout_raw/torch.sqrt(1+readout_raw.square().sum())
        expected_score = context.double() @ readout + model.base.mechanism.bias
        torch.testing.assert_close(score, expected_score, rtol=0, atol=0)
        if name == "additive_positive_head":
            expected = torch.sqrt(A) + .02 * (1e-4 + F.softplus(score))
            assert components["scale"] is None
        else:
            scale = .03 * (1e-4 + F.softplus(score))
            torch.testing.assert_close(model.scale(X), scale, rtol=1e-14, atol=1e-16)
            rule = json.loads((ROOT / "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json").read_text())
            a = torch.tensor(rule["a_i"], dtype=torch.float64)
            z = torch.tensor(rule["z_i"], dtype=torch.float64)
            expected = (a * torch.sqrt(A[:, None] + (6/49) * scale[:, None] ** 2 * z ** 2)).sum(dim=1)
        torch.testing.assert_close(components["prediction"], expected, rtol=2e-13, atol=2e-15)
        torch.testing.assert_close(model(X, A), components["prediction"], rtol=0, atol=0)
        assert bool((model(X, A) >= torch.sqrt(A)).all())


def test_auxiliary_nll_matches_independent_half_student_density_and_has_finite_score():
    u = torch.tensor([0, 1e-8, .001, .1, 10, 1e4], dtype=torch.float64)
    logs = torch.full_like(u, math.log(.03), requires_grad=True)
    s = torch.exp(logs)
    observed = design.half_student_nll(u, s)
    expected = -torch.distributions.StudentT(5., loc=0., scale=s).log_prob(u) - math.log(2)
    torch.testing.assert_close(observed, expected, rtol=2e-14, atol=2e-14)
    gradient, = torch.autograd.grad(observed.sum(), logs)
    assert torch.isfinite(gradient).all() and gradient.min() >= -5 and gradient.max() <= 1
    extreme = design.half_student_nll(torch.tensor([0., 1e250], dtype=torch.float64),
                                      torch.tensor([1e-200, 1e-200], dtype=torch.float64))
    assert torch.isfinite(extreme).all()


def test_module_quadrature_preserves_numeric_properties_and_scale_gradcheck():
    rule = design.load_quadrature()
    a, z = torch.tensor(rule["a"]), torch.tensor(rule["z"])
    generator = torch.Generator().manual_seed(43)
    A = torch.exp(torch.rand(32, generator=generator, dtype=torch.float64)*10-5)
    s = torch.exp(torch.rand(32, generator=generator, dtype=torch.float64)*10-5)
    other = torch.exp(torch.rand(32, generator=generator, dtype=torch.float64)*10-5)
    first = design.quadrature_mean(A, s, a, z)
    second = design.quadrature_mean(A, other, a, z)
    assert torch.all(first >= A.sqrt())
    assert torch.all((first-second).abs() <= rule["L_s"]*(s-other).abs()+1e-12)
    for k in (1e-4, 100.):
        torch.testing.assert_close(design.quadrature_mean(k*k*A, k*s, a, z), k*first,
                                   rtol=3e-15, atol=2e-14)
    scales = torch.tensor([.1, .2, .3], dtype=torch.float64, requires_grad=True)
    fixed_A = torch.tensor([0., .01, .2], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda value: design.quadrature_mean(fixed_A, value, a, z),
                                    (scales,), atol=1e-7, rtol=1e-5)


def test_analytic_controls_are_exactly_the_floor_and_fit_offset():
    A = np.array([0., 1e-6, .0004])
    result = design.analytic_predictions(A, .003)
    np.testing.assert_array_equal(result["floor_only"], np.sqrt(A))
    np.testing.assert_allclose(result["floor_plus_fit_mean"], np.sqrt(A)+.003, rtol=0, atol=0)


@pytest.mark.parametrize("name", NAMES)
def test_normalized_loss_is_exactly_specified_and_all_retained_parameters_receive_gradients(name):
    model, X = _model(name), _features()
    A = torch.tensor([0., 1e-6, 1e-5, .0002, .0004, .001], dtype=torch.float64)
    U = torch.tensor([.01, .02, .03, .04, .06, .08], dtype=torch.float64)
    Y = torch.sqrt(A + (6/49)*U**2)
    outputs = model.forward_components(X, A)
    lam = .1 if name == "innovation_density_mean" else 0.
    parts = design.loss_components(outputs, Y, U, y_scale=.02, u_scale=.03, lambda_nll=lam)
    expected_mse = torch.mean(((outputs["prediction"]-Y)/.02)**2)
    expected_nll = (design.half_student_nll(U/.03, outputs["scale"]/.03).mean()
                    if outputs["scale"] is not None else expected_mse.new_zeros(()))
    torch.testing.assert_close(parts["loss"], expected_mse+lam*expected_nll, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(design.innovation_loss(model, X, A, Y, U), parts["loss"], rtol=0, atol=0)
    parts["loss"].backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, parameter_name
        assert torch.isfinite(parameter.grad).all(), parameter_name
        assert parameter.grad.abs().sum() > 0, parameter_name


def test_nominal_quantiles_match_independent_student_and_are_not_claimed_calibrated():
    student = pytest.importorskip("scipy.stats").t
    A, s = torch.tensor([0., 1e-5, .001], dtype=torch.float64), torch.tensor([.01, .02, .03], dtype=torch.float64)
    for p in (.05, .5, .95):
        expected = torch.sqrt(A+(6/49)*s**2*student.ppf((1+p)/2, 5)**2)
        torch.testing.assert_close(design.predictive_quantile(A, s, p), expected, rtol=2e-13, atol=2e-15)
    low, high = design.nominal_interval(A, s, coverage=.9)
    assert torch.all(low >= torch.sqrt(A)) and torch.all(high > low)
    torch.testing.assert_close(low, design.predictive_quantile(A, s, .05), rtol=2e-13, atol=2e-15)
    torch.testing.assert_close(high, design.predictive_quantile(A, s, .95), rtol=2e-13, atol=2e-15)


def test_modified_quadrature_is_rejected_even_with_a_recomputed_sidecar(tmp_path):
    source = ROOT / "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json"
    rule = json.loads(source.read_text())
    rule["a_i"][0] *= 1.001
    path = tmp_path / source.name
    path.write_text(json.dumps(rule))
    path.with_suffix(path.suffix+".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+"\n")
    with pytest.raises((ValueError, RuntimeError)):
        design.InnovationModel("innovation_mean_only", seed=17, u_scale=.03, y_scale=.02, rule_path=path)


def test_bias_matching_uses_fit_mean_and_keeps_probability_pair_identical():
    X = _features()
    A = torch.tensor([0., 1e-5, 2e-5, 1e-4, .0004, .001], dtype=torch.float64)
    models = {name: _model(name) for name in NAMES}
    reports = design.match_initial_biases(models, X, A, .005, .02)
    assert reports
    for model in models.values():
        excess = float((model(X, A)-torch.sqrt(A)).mean().detach())
        assert abs(excess-.005) <= 1e-8*.02
    _assert_state_equal(models["innovation_mean_only"].state_dict(), models["innovation_density_mean"].state_dict())
    first = dict(models["additive_positive_head"].named_parameters())
    second = dict(models["innovation_mean_only"].named_parameters())
    for key in first:
        if not (key.endswith("bias") and first[key].numel() == 1):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_bias_matching_budget_failure_is_transactional_and_invalid_target_is_rejected():
    model, X = _model(), _features()
    A = torch.full((len(X),), .0001, dtype=torch.float64)
    state = deepcopy(model.state_dict())
    calls = 0

    def exhaust():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("Synthetic exhausted budget")

    with pytest.raises(RuntimeError, match="Synthetic exhausted budget"):
        model.calibrate_bias(X, A, .005, budget_callback=exhaust, batch_size=2)
    _assert_state_equal(model.state_dict(), state)
    for target in (0., 1e20):
        with pytest.raises((ValueError, RuntimeError)):
            model.calibrate_bias(X, A, target)
        _assert_state_equal(model.state_dict(), state)


def test_mixed_precision_state_roundtrip_preserves_outputs():
    first, second = _model(seed=17), _model(seed=99)
    X, A = _features(), torch.arange(6, dtype=torch.float64)*1e-5
    second.load_state_dict(deepcopy(first.state_dict()), strict=True)
    torch.testing.assert_close(first(X, A), second(X, A), rtol=0, atol=0)
