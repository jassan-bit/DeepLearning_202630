"""Synthetic admission, controlled-ablation and recovery checks; no market fits."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.e3_development as dev


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _arrays():
    """Six synthetic outcomes with physically valid, causally indexed windows."""
    dates = np.array(["2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30",
                      "2024-07-08", "2024-07-09"])

    def bar(day):
        k = (day - date(2024, 1, 1)).days
        close = 100.0 + 0.11 * k + 0.7 * np.sin(k / 3.0)
        opening = close + 0.2 * np.cos(k / 2.0)
        return [opening, max(opening, close) + 1.0,
                min(opening, close) - 1.0, close, 300.0 + k * 2.0 + k % 5]

    raw = np.array([[bar(date.fromisoformat(day) - timedelta(days=lag))
                     for lag in range(7, 0, -1)] for day in dates], dtype=np.float64)
    return {"training_raw": raw,
            "training_y_original": np.array([0.01, 0.02, 0.03, 0.04, 0.025, 0.035]),
            "training_dates": dates}


def _save_npz(tmp_path, arrays=None, name="synthetic.npz"):
    path = tmp_path / name
    np.savez(path, **(_arrays() if arrays is None else arrays))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path, sha, **kwargs):
    return dev.load_development_data(path, sha, **kwargs)


def _plan(**updates):
    plan = {"seed": 117,
            "training": {"batch_size": 2, "max_epochs": 2, "patience": 8}}
    plan.update(updates)
    return plan


def _create(tmp_path, *, name="run", plan=None):
    path, sha = _save_npz(tmp_path)
    destination = tmp_path / name
    dev.create_run(path, sha, destination, device="cpu", plan=_plan() if plan is None else plan)
    return destination, path, sha


def test_scaler_and_output_scale_are_fit_only_and_match_raw_contract(tmp_path):
    arrays = _arrays()
    path, sha = _save_npz(tmp_path, arrays)
    loaded = _load(path, sha)
    fit = arrays["training_raw"][:4]
    relative_prices = np.log(fit[:, :, :4]) - np.log(fit[:, -1, 3])[:, None, None]
    transformed = np.concatenate([relative_prices, np.log1p(fit[:, :, 4:5])], axis=2)
    means = transformed.mean(axis=(0, 1))
    stds = transformed.std(axis=(0, 1))
    expected_X = ((transformed - means) / stds).astype(np.float32)
    np.testing.assert_allclose(loaded["scaler"]["mean"], means, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(loaded["scaler"]["std"], stds, atol=1e-13, rtol=1e-12)
    assert loaded["scaler"]["target_scale"] == pytest.approx(0.025, abs=1e-15)
    np.testing.assert_allclose(loaded["train_X"], expected_X, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(loaded["train_y_scaled"], [0.4, 0.8, 1.2, 1.6], rtol=1e-6)
    assert loaded["train_X"].dtype == np.float32
    assert list(loaded["train_dates"]) == list(arrays["training_dates"][:4])
    assert list(loaded["validation_dates"]) == list(arrays["training_dates"][4:])


def test_changing_validation_cannot_change_fit_data_or_fitted_statistics(tmp_path):
    first, first_sha = _save_npz(tmp_path, name="first.npz")
    changed = _arrays()
    changed["training_raw"][4:, :, :4] *= np.linspace(1.0, 1.8, 7)[None, :, None]
    changed["training_raw"][4:, :, 4] *= 1000.0
    changed["training_y_original"][4:] *= 50.0
    second, second_sha = _save_npz(tmp_path, changed, name="changed_validation.npz")
    a, b = _load(first, first_sha), _load(second, second_sha)
    for key in ("train_X", "train_y_scaled", "train_y_original", "train_dates"):
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("mean", "std", "target_scale"):
        np.testing.assert_array_equal(a["scaler"][key], b["scaler"][key])
    assert not np.array_equal(a["validation_X"], b["validation_X"])
    assert not np.array_equal(a["validation_y_scaled"], b["validation_y_scaled"])


def test_loader_never_dereferences_calibration_or_evaluation_arrays(tmp_path, monkeypatch):
    arrays = _arrays()
    # Access to these arrays would require pickle; they are intentionally unusable.
    arrays["calibration_raw"] = np.array([{"must_not_be_read": True}], dtype=object)
    arrays["evaluation_y_original"] = np.array([{"must_not_be_read": True}], dtype=object)
    path, sha = _save_npz(tmp_path, arrays)
    real_load = np.load
    accessed = []
    allowed = {"training_raw", "training_y_original", "training_dates"}

    class AdmissionOnlyArchive:
        def __init__(self, archive):
            self.archive = archive
            self.files = archive.files

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.archive.close()

        def __contains__(self, key):
            return key in self.files

        def __iter__(self):
            return iter(self.files)

        def __len__(self):
            return len(self.files)

        def keys(self):
            return self.files

        def __getitem__(self, key):
            accessed.append(key)
            assert key in allowed, f"Forbidden held-out array was accessed: {key}"
            return self.archive[key]

        def close(self):
            self.archive.close()

    monkeypatch.setattr(dev.np, "load", lambda *a, **kw: AdmissionOnlyArchive(real_load(*a, **kw)))
    loaded = _load(path, sha)
    assert set(accessed) == allowed
    assert len(loaded["train_X"]) == 4 and len(loaded["validation_X"]) == 2


def test_bad_data_hash_is_rejected_before_numpy_loading(tmp_path, monkeypatch):
    path, _ = _save_npz(tmp_path)

    def forbidden_load(*_a, **_kw):
        pytest.fail("Unadmitted bytes must not be parsed")

    monkeypatch.setattr(dev.np, "load", forbidden_load)
    with pytest.raises((ValueError, RuntimeError)):
        _load(path, "0" * 64)


@pytest.mark.parametrize("problem", ["missing_key", "bad_shape", "length_mismatch", "nan_raw",
                                     "infinite_target", "negative_target", "nonpositive_price",
                                     "negative_volume", "invalid_bar", "duplicate_date",
                                     "unordered_date", "malformed_date", "empty_fit", "empty_validation"])
def test_invalid_admitted_training_schema_is_rejected(tmp_path, problem):
    arrays = _arrays()
    if problem == "missing_key":
        del arrays["training_raw"]
    elif problem == "bad_shape":
        arrays["training_raw"] = arrays["training_raw"][:, :6, :]
    elif problem == "length_mismatch":
        arrays["training_y_original"] = arrays["training_y_original"][:-1]
    elif problem == "nan_raw":
        arrays["training_raw"][0, 0, 0] = np.nan
    elif problem == "infinite_target":
        arrays["training_y_original"][0] = np.inf
    elif problem == "negative_target":
        arrays["training_y_original"][0] = -0.01
    elif problem == "nonpositive_price":
        arrays["training_raw"][0, 0, 3] = 0
    elif problem == "negative_volume":
        arrays["training_raw"][0, 0, 4] = -1
    elif problem == "invalid_bar":
        arrays["training_raw"][0, 0, 1] = arrays["training_raw"][0, 0, 2] - 1
    elif problem == "duplicate_date":
        arrays["training_dates"][1] = arrays["training_dates"][0]
    elif problem == "unordered_date":
        arrays["training_dates"][[0, 1]] = arrays["training_dates"][[1, 0]]
    elif problem == "malformed_date":
        arrays["training_dates"][0] = "not-a-date"
    elif problem == "empty_fit":
        arrays["training_dates"] = np.array(["2024-07-08", "2024-07-09", "2024-07-10",
                                              "2024-07-11", "2024-07-12", "2024-07-13"])
    else:
        arrays["training_dates"] = np.array(["2024-06-25", "2024-06-26", "2024-06-27",
                                              "2024-06-28", "2024-06-29", "2024-06-30"])
    path, sha = _save_npz(tmp_path, arrays)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _load(path, sha)


def test_outcomes_in_seven_day_gap_are_excluded_from_both_splits(tmp_path):
    arrays = _arrays()
    for key in ("training_raw", "training_y_original"):
        arrays[key] = np.concatenate([arrays[key][:4], arrays[key][3:4], arrays[key][4:]])
    arrays["training_dates"] = np.insert(arrays["training_dates"], 4, "2024-07-04")
    # It is not a fit observation, even if it would dominate fitted statistics.
    arrays["training_raw"][4, :, 4] *= 1e6
    arrays["training_y_original"][4] = 99.0
    path, sha = _save_npz(tmp_path, arrays)
    loaded = _load(path, sha)
    assert len(loaded["train_X"]) == 4 and len(loaded["validation_X"]) == 2
    assert "2024-07-04" not in set(loaded["train_dates"]) | set(loaded["validation_dates"])
    assert loaded["scaler"]["target_scale"] == pytest.approx(0.025)


def test_overlapping_target_support_cannot_be_configured_as_validation(tmp_path):
    path, sha = _save_npz(tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        _load(path, sha, fit_end="2024-06-30", validation_start="2024-07-03")


def test_m00_removes_both_mechanisms_and_keeps_exact_shared_initialization():
    full = dev.DevelopmentModel(model_name="full", seed=23)
    control = dev.DevelopmentModel(model_name="M00", seed=23)
    full_state, control_state = full.state_dict(), control.state_dict()
    removed = set(full_state) - set(control_state)
    assert removed and all(name.endswith("penalty_raw") for name in removed)
    assert not (set(control_state) - set(full_state))
    for name, value in control_state.items():
        torch.testing.assert_close(value, full_state[name], rtol=0, atol=0)

    states = torch.randn(4, 7, 12, generator=torch.Generator().manual_seed(81)) * 4.0
    head = control.model.mechanism
    components = head.attention_components(states)
    pars = head.effective_parameters()
    expected_weights = torch.softmax((states @ pars["q"]) / pars["temperature"], dim=1)
    expected_context = (expected_weights[..., None] * states).sum(dim=1)
    expected_output = F.softplus(expected_context @ pars["w"] + head.bias)
    torch.testing.assert_close(components["values"], states, rtol=0, atol=0)
    torch.testing.assert_close(components["weights"], expected_weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(head(states), expected_output, rtol=1e-6, atol=1e-7)
    assert head.penalty_raw is None
    assert not torch.allclose(full.model.mechanism(states), head(states))
    with pytest.raises(ValueError):
        head.latent_lipschitz_bound()


@pytest.mark.parametrize("model_name", ["full", "M00"])
def test_every_retained_control_parameter_has_an_active_finite_gradient(model_name):
    model = dev.DevelopmentModel(model_name=model_name, seed=23)
    features = torch.randn(4, 7, 5, generator=torch.Generator().manual_seed(91))
    loss = F.mse_loss(model(features), torch.tensor([0.3, 0.7, 1.1, 1.5]))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum().item() > 0, name


def test_preflight_preserves_initial_weights_and_empty_optimizer_history(tmp_path, monkeypatch):
    def forbidden_step(*_a, **_kw):
        pytest.fail("Preflight must not update model parameters")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    run_dir, _, _ = _create(tmp_path)
    assert (run_dir / "manifest.json").is_file()
    for model_name in ("full", "M00"):
        checkpoint = torch.load(run_dir / model_name / "last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 0
        assert not checkpoint["optimizer_state"]["state"]
        assert not checkpoint["history"]


def _assert_equal_state(first, second):
    """Compare persisted scientific state, excluding wall clocks and paths."""
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_equal_state(a, b)
    else:
        assert first == second


def _checkpoint(run_dir, model_name):
    # Only locally generated synthetic checkpoints are deserialized by these tests.
    return torch.load(run_dir / model_name / "last.pt", map_location="cpu", weights_only=False)


def test_resumed_training_matches_uninterrupted_optimizer_rng_and_predictions(tmp_path):
    path, sha = _save_npz(tmp_path)
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    for run_dir in (uninterrupted, resumed):
        dev.create_run(path, sha, run_dir, device="cpu", plan=_plan())
    dev.run_development(uninterrupted, path, sha, device="cpu")
    dev.run_development(resumed, path, sha, device="cpu", stop_after_epochs=2)
    for name in ("full", "M00"):
        paused = _checkpoint(resumed, name)
        assert paused["epoch"] == 1
        assert not paused["complete"]
    dev.run_development(resumed, path, sha, resume=True, device="cpu")
    for name in ("full", "M00"):
        expected, recovered = _checkpoint(uninterrupted, name), _checkpoint(resumed, name)
        assert expected["complete"] and recovered["complete"]
        assert expected["epoch"] == recovered["epoch"] == 2
        for key in ("model_state", "optimizer_state", "rng", "dataloader_generator_state",
                    "best_validation_rmse", "best_epoch", "no_improvement", "history"):
            _assert_equal_state(expected[key], recovered[key])
        with (uninterrupted / name / "validation_predictions.csv").open(newline="") as stream:
            a = list(csv.DictReader(stream))
        with (resumed / name / "validation_predictions.csv").open(newline="") as stream:
            b = list(csv.DictReader(stream))
        assert len(a) == len(b) == 2
        assert a == b


@pytest.mark.parametrize("boundary", ["best_publication", "last_publication_before_receipt"])
def test_recovery_after_atomic_publication_failure_preserves_best_and_exact_training(tmp_path, monkeypatch, boundary):
    """A published epoch or batch cannot be lost with a stale derived best/receipt."""
    path, sha = _save_npz(tmp_path)
    uninterrupted, interrupted = tmp_path / "reference", tmp_path / "publication_failure"
    plan = _plan()
    plan["training"]["max_epochs"] = 1
    for run_dir in (uninterrupted, interrupted):
        dev.create_run(path, sha, run_dir, device="cpu", plan=plan)
    dev.run_development(uninterrupted, path, sha, device="cpu")
    original_replace = dev.os.replace
    triggered = False

    def interrupted_publication(source, destination):
        nonlocal triggered
        destination = Path(destination)
        name = "best.pt" if boundary == "best_publication" else "last.pt"
        if not triggered and destination == interrupted / "full" / name:
            triggered = True
            if boundary == "last_publication_before_receipt":
                # Bytes are durable; the surrounding receipt commit is interrupted.
                original_replace(source, destination)
            raise OSError("Synthetic checkpoint publication interruption")
        return original_replace(source, destination)

    with monkeypatch.context() as fault:
        fault.setattr(dev.os, "replace", interrupted_publication)
        with pytest.raises(OSError, match="Synthetic checkpoint publication interruption"):
            dev.run_development(interrupted, path, sha, device="cpu")
    assert triggered
    charged_before = json.loads((interrupted / "ledger.json").read_text())["charged_seconds"]
    assert charged_before > 0
    dev.run_development(interrupted, path, sha, resume=True, device="cpu")
    assert json.loads((interrupted / "ledger.json").read_text())["charged_seconds"] >= charged_before
    for name in ("full", "M00"):
        expected, recovered = _checkpoint(uninterrupted, name), _checkpoint(interrupted, name)
        assert recovered["epoch"] == 1 and recovered["complete"]
        for key in ("model_state", "optimizer_state", "history", "best_epoch", "best_validation_rmse"):
            _assert_equal_state(expected[key], recovered[key])
        expected_best = torch.load(uninterrupted / name / "best.pt", map_location="cpu", weights_only=False)
        recovered_best = torch.load(interrupted / name / "best.pt", map_location="cpu", weights_only=False)
        _assert_equal_state(expected_best["model_state"], recovered_best["model_state"])
        with (uninterrupted / name / "validation_predictions.csv").open(newline="") as stream:
            expected_rows = list(csv.DictReader(stream))
        with (interrupted / name / "validation_predictions.csv").open(newline="") as stream:
            recovered_rows = list(csv.DictReader(stream))
        assert expected_rows == recovered_rows and len(recovered_rows) == 2


def test_os_lock_prevents_a_second_writer_before_optimizer_step(tmp_path, monkeypatch):
    import fcntl

    run_dir, path, sha = _create(tmp_path)

    def forbidden_step(*_a, **_kw):
        pytest.fail("A second writer must not train while another process owns the lock")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with (run_dir / ".run.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises((BlockingIOError, RuntimeError, ValueError)):
                dev.run_development(run_dir, path, sha, device="cpu")
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_corrupted_checkpoint_is_rejected_before_deserialization(tmp_path, monkeypatch):
    run_dir, path, sha = _create(tmp_path)
    checkpoint_path = run_dir / "full" / "last.pt"
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"synthetic corruption")

    def forbidden_deserialization(*_a, **_kw):
        pytest.fail("Corrupted checkpoint bytes must not reach torch.load")

    monkeypatch.setattr(dev.torch, "load", forbidden_deserialization)
    with pytest.raises((RuntimeError, ValueError)):
        dev.run_development(run_dir, path, sha, device="cpu")


@pytest.mark.parametrize("identity", ["data", "plan", "source"])
def test_resume_rejects_changed_data_plan_or_source_identity(tmp_path, identity, monkeypatch):
    run_dir, path, sha = _create(tmp_path)
    dev.run_development(run_dir, path, sha, device="cpu", stop_after_epochs=2)
    before = {name: _checkpoint(run_dir, name)["model_state"] for name in ("full", "M00")}
    if identity == "data":
        changed = _arrays()
        changed["training_raw"][0, :, 4] *= 1.02
        path, sha = _save_npz(tmp_path, changed)
    elif identity == "plan":
        plan_path = run_dir / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["training"]["learning_rate"] *= 2.0
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
    else:
        changed_sources = deepcopy(dev.source_fingerprints())
        first_source = next(iter(changed_sources))
        changed_sources[first_source] = "0" * 64
        monkeypatch.setattr(dev, "source_fingerprints", lambda: changed_sources)

    def forbidden_step(*_a, **_kw):
        pytest.fail("Identity mismatch must be rejected before modifying parameters")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with pytest.raises((ValueError, RuntimeError)):
        dev.run_development(run_dir, path, sha, resume=True, device="cpu")
    for name in ("full", "M00"):
        _assert_equal_state(before[name], _checkpoint(run_dir, name)["model_state"])


class _StepClock:
    def __init__(self, step=1.0):
        self.value = 0.0
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def test_failed_training_attempt_is_charged_to_the_same_persistent_ledger(tmp_path, monkeypatch):
    run_dir, path, sha = _create(tmp_path)

    def interrupt_before_update(*_a, **_kw):
        raise RuntimeError("Synthetic optimizer interruption")

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt_before_update)
    with pytest.raises(RuntimeError, match="Synthetic optimizer interruption"):
        dev.run_development(run_dir, path, sha, device="cpu", clock=_StepClock(step=0.01))
    ledger = json.loads((run_dir / "ledger.json").read_text())
    assert ledger["charged_seconds"] > 0
    assert not ledger["active_session"]
    for name in ("full", "M00"):
        assert _checkpoint(run_dir, name)["epoch"] == 0


def test_budget_charge_persists_and_cannot_be_reset_by_resuming_or_recreating(tmp_path, monkeypatch):
    run_dir, path, sha = _create(tmp_path, plan=_plan(budget={"total_seconds": 0.1}))

    def forbidden_step(*_a, **_kw):
        pytest.fail("Synthetic clock already consumed the budget before any training step")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    dev.run_development(run_dir, path, sha, device="cpu", clock=_StepClock())
    ledger_path = run_dir / "ledger.json"
    charged = json.loads(ledger_path.read_text())["charged_seconds"]
    assert charged >= 0.1
    try:
        dev.run_development(run_dir, path, sha, resume=True, device="cpu", clock=_StepClock())
    except (RuntimeError, ValueError) as error:
        assert any(word in str(error).lower() for word in ("budget", "presupuesto", "agot"))
    assert json.loads(ledger_path.read_text())["charged_seconds"] >= charged
    with pytest.raises((FileExistsError, RuntimeError, ValueError)):
        dev.create_run(path, sha, run_dir, device="cpu", plan=_plan(budget={"total_seconds": 0.1}))
    assert json.loads(ledger_path.read_text())["charged_seconds"] >= charged
    for name in ("full", "M00"):
        assert _checkpoint(run_dir, name)["epoch"] == 0


def test_wall_budget_cannot_be_raised_above_development_ceiling(tmp_path):
    path, sha = _save_npz(tmp_path)
    with pytest.raises(ValueError):
        dev.create_run(path, sha, tmp_path / "over_budget", device="cpu",
                       plan=_plan(budget={"total_seconds": 1200.1}))
