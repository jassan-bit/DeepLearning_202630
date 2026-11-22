"""Only synthetic optimizer fixtures; no panel collection or reserved data."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from engine import training


@pytest.fixture
def software(monkeypatch):
    calls = {"loss": 0, "transitions": []}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(35, 1)

        def forward(self, x):
            return self.linear(torch.as_tensor(x, dtype=torch.float32).flatten(1)).flatten()

    def build(name, seed, state):
        torch.manual_seed(seed)
        return Tiny()

    def loss(model, name, phase, raw, y, state):
        calls["loss"] += 1
        return ((model(raw) - torch.as_tensor(y, dtype=torch.float32))**2).mean()

    def predict(model, name, raw, state):
        value = model(raw).detach().numpy().astype(float)
        floor = np.full(len(value), .01)
        return value, np.maximum(value, floor), floor

    patches = {"fit_preprocessor": lambda raw, y: {"y_scale": 1.},
               "build_model": build, "model_config": lambda name: {"name": name},
               "training_phases": lambda name: ("ae1", "ae2", "supervised") if name == "sae" else ("supervised",),
               "phase_parameters": lambda model, name, phase: model.parameters(),
               "phase_model": lambda model, name, phase: model,
               "phase_loss": loss, "predict": predict,
               "transition_phase": lambda model, name, phase: calls["transitions"].append(phase)}
    for name, value in patches.items():
        monkeypatch.setattr(training.models, name, value)
    rng = np.random.default_rng(761)
    raw = rng.uniform(.1, 1., size=(12, 7, 5)).astype(np.float32)
    data = {"raw": raw, "y": np.linspace(.05, .3, len(raw)),
            "dates": np.arange(len(raw)).astype(str)}
    job = {"model": "tiny", "seed": 61, "protocol_sha256": "a" * 64,
           "identity": {"purpose": "software_fixture", "data_sha256": "b" * 64},
           "training": {"max_epochs": 3, "patience": 3, "batch_size": 4,
                        "total_budget_seconds": 60., "epoch_reservation_seconds": 10.}}
    return job, data, calls


def test_real_optimizer_snapshot_selected_and_skip_verified(tmp_path, software, monkeypatch):
    job, data, calls = software
    events = []
    receipt = training.fit_job(job, data, data, tmp_path / "job",
                               on_checkpoint=lambda directory, event: events.append(event["kind"]))
    assert receipt["optimizer_steps_committed"] == 9
    assert receipt["status"] == "completed_validated_fit"
    assert events[0] == "before_fit" and events[-1] == "job_completed"
    assert receipt["remaining_nominal_budget_seconds"] < 60
    state, pointer = training._load(tmp_path / "job", receipt["request_sha256"])
    assert state["optimizer"] is None  # Completed phase: optimizer retained in old immutable epoch snapshot.
    epoch_snapshots = []
    for path in (tmp_path / "job").glob("snapshot_*.pt"):
        payload = torch.load(path, weights_only=False)
        if payload["optimizer"] is not None:
            epoch_snapshots.append(payload)
    assert epoch_snapshots and epoch_snapshots[-1]["optimizer"]["state"]
    assert all(key in state["rng"] for key in ("python", "numpy", "torch"))
    assert state["phase_optimizers"]["supervised"]["state"]
    assert training.verify_job(job, tmp_path / "job") == receipt
    count = calls["loss"]
    assert training.fit_job(job, data, data, tmp_path / "job") == receipt
    assert calls["loss"] == count
    pred = training.predict_job(job, {"raw": data["raw"]}, tmp_path / "job", batch_size=2)
    assert pred["raw_sigma7"].shape == (12,)
    assert pred["selected_state_sha256"] == receipt["selected_state_sha256"]


def test_epoch_boundary_resume_matches_uninterrupted(tmp_path, software):
    job, data, calls = software
    complete = training.fit_job(job, data, data, tmp_path / "continuous")

    def stop(directory, event):
        if event["kind"] == "epoch_checkpoint" and event["epoch"] == 1:
            raise OSError("fixture remote unavailable")

    with pytest.raises(training.DurabilityError):
        training.fit_job(job, data, data, tmp_path / "resume", on_checkpoint=stop)
    before = calls["loss"]
    resumed = training.fit_job(job, data, data, tmp_path / "resume")
    assert calls["loss"] - before == 6
    assert complete["selected_state_sha256"] == resumed["selected_state_sha256"]
    assert resumed["interrupted_epochs"] == 0
    assert resumed["compute_uncertain"] is False


def test_initial_failed_save_blocks_all_steps(tmp_path, software):
    job, data, calls = software

    def failed(directory, event):
        raise PermissionError("credential content should not be recorded")

    with pytest.raises(training.DurabilityError):
        training.fit_job(job, data, data, tmp_path / "job", on_checkpoint=failed)
    assert calls["loss"] == 0
    assert "credential content" not in (tmp_path / "job" / "failures.json").read_text()
    assert not (tmp_path / "job" / "done.json").exists()


def test_partial_epoch_reservation_charged_once_and_marked_unknown(tmp_path, software, monkeypatch):
    job, data, calls = software
    original = training.models.phase_loss

    def crash(model, name, phase, raw, y, state):
        if calls["loss"] == 1:
            raise RuntimeError("fixture interruption after one optimizer update")
        return original(model, name, phase, raw, y, state)

    monkeypatch.setattr(training.models, "phase_loss", crash)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        training.fit_job(job, data, data, tmp_path / "job")
    assert (tmp_path / "job" / "epoch_pending.json").exists()
    monkeypatch.setattr(training.models, "phase_loss", original)
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    assert receipt["interrupted_epochs"] == 1
    assert receipt["compute_uncertain"] is True
    assert receipt["uncommitted_optimizer_steps"] is None
    assert receipt["active_seconds_charged"] >= 10
    assert receipt["optimizer_steps_committed"] == 9
    assert training.fit_job(job, data, data, tmp_path / "job") == receipt
    uninterrupted = training.fit_job(job, data, data, tmp_path / "continuous")
    assert receipt["selected_state_sha256"] == uninterrupted["selected_state_sha256"]


def test_three_sae_phases_share_one_total_budget(tmp_path, software):
    job, data, calls = software
    job["model"] = "sae"
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    assert calls["transitions"] == ["ae1", "ae2", "supervised"]
    assert [phase["name"] for phase in receipt["phase_receipts"]] == calls["transitions"]
    assert receipt["optimizer_steps_committed"] == 27
    assert receipt["phase_budget_seconds"] == 20
    assert all(phase["best_epoch"] >= 1 for phase in receipt["phase_receipts"])


def test_identity_protocol_data_and_payload_corruption_rejected(tmp_path, software):
    job, data, calls = software
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    other = copy.deepcopy(job)
    other["protocol_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="identity differs"):
        training.fit_job(other, data, data, tmp_path / "job")
    changed = dict(data, y=data["y"] + .1)
    with pytest.raises(ValueError, match="identity differs"):
        training.fit_job(job, changed, data, tmp_path / "job")
    payload = tmp_path / "job" / receipt["checkpoint_file"]
    with payload.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash/request mismatch"):
        training.predict_job(job, data, tmp_path / "job")


def test_symlink_and_unvalidated_done_refused(tmp_path, software):
    job, data, calls = software
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        training.fit_job(job, data, data, link / "job")
    training.fit_job(job, data, data, actual / "job")
    done = actual / "job" / "done.json"
    value = json.loads(done.read_text())
    value["selected_state_sha256"] = "d" * 64
    done.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt"):
        training.predict_job(job, data, actual / "job")


def test_prediction_batching_and_rng_are_idempotent(tmp_path, software):
    job, data, calls = software
    training.fit_job(job, data, data, tmp_path / "job")
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    state = training._rng()
    one = training.predict_job(job, data, tmp_path / "job", batch_size=1)
    observed = (random.random(), np.random.rand(), torch.rand(1))
    training._restore_rng(state)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    assert observed[0] == expected[0] and observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])
    many = training.predict_job(job, data, tmp_path / "job", batch_size=12)
    np.testing.assert_allclose(one["raw_sigma7"], many["raw_sigma7"], rtol=1e-6, atol=1e-7)


def test_no_budget_reset_after_crash(tmp_path, software, monkeypatch):
    job, data, calls = software
    ticks = iter(range(10000))
    monkeypatch.setattr(training.time, "perf_counter", lambda: next(ticks) * .01)
    job["training"].update(total_budget_seconds=.001, epoch_reservation_seconds=.001)
    with pytest.raises(training.TrainingIncomplete):
        training.fit_job(job, data, data, tmp_path / "job")
    with pytest.raises(training.TrainingIncomplete, match="budget exhausted"):
        training.fit_job(job, data, data, tmp_path / "job")
    assert not (tmp_path / "job" / "done.json").exists()


def test_nominal_budget_selects_validated_best_and_records_partial_work(tmp_path, software, monkeypatch):
    job, data, calls = software
    ticks = iter(range(10000))
    monkeypatch.setattr(training.time, "perf_counter", lambda: next(ticks) * .0001)
    job["training"].update(max_epochs=100, patience=100,
                            total_budget_seconds=.001, epoch_reservation_seconds=.0003)
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    assert receipt["phase_receipts"][0]["stop_reason"] == "nominal_budget"
    assert receipt["phase_receipts"][0]["best_epoch"] >= 1
    assert receipt["optimizer_steps_committed"] >= 3
    assert receipt["compute_uncertain"]
    assert receipt["cooperative_nominal_budget_overrun_seconds"] >= 0
    assert training.verify_job(job, tmp_path / "job") == receipt


def test_selection_uses_best_trained_epoch_instead_of_latest(tmp_path, software, monkeypatch):
    job, data, calls = software
    scores = iter([1., 3., 2.])
    monkeypatch.setattr(training, "_score", lambda *args: next(scores))
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    state, _ = training._load(tmp_path / "job", receipt["request_sha256"])
    assert receipt["phase_receipts"][0]["best_epoch"] == 1
    selected = training._weights_hash(state["selected"])
    latest = []
    for path in (tmp_path / "job").glob("snapshot_*.pt"):
        snap = torch.load(path, weights_only=False)
        if snap["phase_state"]["epoch"] == 3:
            latest.append(training._weights_hash(snap["current"]))
    assert latest and all(value != selected for value in latest)


def test_crash_after_payload_publication_does_not_double_charge(tmp_path, software, monkeypatch):
    job, data, calls = software
    from pathlib import Path
    original = Path.unlink
    crashed = [False]

    def interrupted_unlink(path, *args, **kwargs):
        if path.name == "epoch_pending.json" and not crashed[0]:
            crashed[0] = True
            raise OSError("fixture interrupted marker removal after durable epoch")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_unlink)
    with pytest.raises(OSError, match="marker removal"):
        training.fit_job(job, data, data, tmp_path / "job")
    monkeypatch.setattr(Path, "unlink", original)
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    assert receipt["interrupted_epochs"] == 0
    assert receipt["compute_uncertain"] is False
    assert receipt["active_seconds_charged"] < 10
    assert receipt["optimizer_steps_committed"] == 9


def test_failed_pointer_publication_restores_previous_snapshot(tmp_path, software, monkeypatch):
    job, data, calls = software
    original = training._write
    publications = [0]

    def failed_pointer(path, value, **kwargs):
        if path.name == "checkpoint.json":
            publications[0] += 1
            if publications[0] == 2:
                raise OSError("fixture interrupted atomic pointer publication")
        return original(path, value, **kwargs)

    monkeypatch.setattr(training, "_write", failed_pointer)
    with pytest.raises(OSError, match="pointer publication"):
        training.fit_job(job, data, data, tmp_path / "job")
    monkeypatch.setattr(training, "_write", original)
    receipt = training.fit_job(job, data, data, tmp_path / "job")
    assert receipt["interrupted_epochs"] == 1
    assert receipt["active_seconds_charged"] >= 10
    uninterrupted = training.fit_job(job, data, data, tmp_path / "continuous")
    assert receipt["selected_state_sha256"] == uninterrupted["selected_state_sha256"]


def _physical_fixture():
    rng = np.random.default_rng(20261121)
    close = np.exp(4.5 + np.cumsum(rng.normal(0, .001, 180)))
    bars = np.stack((close * .999, close * 1.01, close * .99,
                     close, np.arange(180) + 100.), axis=1)
    raw = np.stack([bars[i - 7:i] for i in range(14, 174)])
    A = training.models.candidate_data.known_geometry(raw)[0]
    y = np.sqrt(A + (6 / 49) * rng.uniform(.003, .03, len(A))**2)
    return raw, y


@pytest.mark.parametrize("name", training.models.NEURAL)
def test_real_family_one_epoch_optimizer_selected_prediction(name, tmp_path):
    raw, y = _physical_fixture()
    fit = {"raw": raw[:4], "y": y[:4], "dates": np.arange(4).astype(str)}
    selection = {"raw": raw[4:7], "y": y[4:7], "dates": np.arange(4, 7).astype(str)}
    job = {"model": name, "seed": 2870326772251854041, "protocol_sha256": "e" * 64,
           "identity": {"purpose": "software_fixture", "no_market_observations": True},
           "training": {"max_epochs": 1, "patience": 1, "batch_size": 4,
                        "total_budget_seconds": 60., "epoch_reservation_seconds": 20.}}
    receipt = training.fit_job(job, fit, selection, tmp_path / "job")
    phases = training.models.training_phases(name)
    assert receipt["optimizer_steps_committed"] == len(phases)
    assert [p["name"] for p in receipt["phase_receipts"]] == list(phases)
    point = training.predict_job(job, {"raw": selection["raw"]}, tmp_path / "job", batch_size=2)
    assert np.isfinite(point["raw_sigma7"]).all()
    assert np.all(point["projected_sigma7"] >= point["floor"])
    assert training.verify_job(job, tmp_path / "job") == receipt


@pytest.mark.parametrize("name", training.models.DETERMINISTIC)
def test_real_deterministic_fit_roundtrip_is_one_system(name, tmp_path):
    raw, _ = _physical_fixture()
    rng = np.random.default_rng(271828)
    sigma = np.empty(170)
    innovation = rng.normal(0, .0008, len(sigma))
    sigma[:2] = .03
    for i in range(2, len(sigma)):
        sigma[i] = (.03 + .45 * (sigma[i - 1] - .03) - .2 * (sigma[i - 2] - .03)
                    + innovation[i] + .25 * innovation[i - 1])
    fit = {"raw": raw[:100], "y": sigma[:100], "history_end_positions": np.arange(7, 107),
           "dates": np.arange(100).astype(str)}
    selection = {"raw": raw[100:104], "y": sigma[100:104],
                 "known_sigma_history": np.stack([sigma[i - 7:i] for i in range(100, 104)]),
                 "observed_sigma_since_fit": np.r_[np.full(7, .029), sigma[:103]],
                 "history_end_positions": np.arange(107, 111)}
    job = {"model": name, "seed": 2870326772251854041, "protocol_sha256": "f" * 64,
           "identity": {"purpose": "software_fixture", "no_market_observations": True}}
    receipt = training.fit_job(job, fit, selection, tmp_path / "job")
    assert receipt["optimizer_steps_committed"] == 0
    assert receipt["independent_fit"] is False
    assert receipt["deterministic_seed_repetition_is_not_independent"] is True
    one = training.predict_job(job, selection, tmp_path / "job", batch_size=1)
    many = training.predict_job(job, selection, tmp_path / "job", batch_size=4)
    np.testing.assert_array_equal(one["raw_sigma7"], many["raw_sigma7"])
    assert training.fit_job(job, fit, selection, tmp_path / "job") == receipt


def test_interrupted_arima_cannot_restart_with_fresh_budget(tmp_path, monkeypatch):
    raw, y = _physical_fixture()
    data = {"raw": raw[:100], "y": y[:100]}
    job = {"model": "ARIMA", "seed": 2870326772251854041,
           "protocol_sha256": "f" * 64, "identity": {"purpose": "software_fixture"},
           "budget": {"total_budget_seconds": 1.}}
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("fixture interrupted non-resumable classical calibration")

    monkeypatch.setattr(training.models, "fit_deterministic", interrupted)
    with pytest.raises(RuntimeError, match="non-resumable"):
        training.fit_job(job, data, data, tmp_path / "job")
    with pytest.raises(training.TrainingIncomplete, match="never reset"):
        training.fit_job(job, data, data, tmp_path / "job")
    assert len(calls) == 1
    resources = training.export_job_resources(tmp_path / "job")
    assert resources["compute_uncertain"]
    assert resources["remaining_nominal_budget_seconds"] == 0
    assert not (tmp_path / "job" / "done.json").exists()


@pytest.mark.parametrize("field", ["python_major_minor", "torch", "numpy", "scipy", "statsmodels"])
def test_completed_job_cannot_cross_critical_software_cohort(tmp_path, software, monkeypatch, field):
    job, data, calls = software
    directory = tmp_path / "job"
    receipt = training.fit_job(job, data, data, directory)
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    runtime = training.critical_runtime_identity()
    monkeypatch.setattr(training, "critical_runtime_identity", lambda: {**runtime, field: "different-version"})
    count = calls["loss"]
    with pytest.raises(ValueError, match="Critical software runtime differs"):
        training.verify_job(job, directory)
    with pytest.raises(ValueError, match="Critical software runtime differs"):
        training.predict_job(job, data, directory)
    with pytest.raises(ValueError, match="Critical software runtime differs"):
        training.fit_job(job, data, data, directory)
    assert calls["loss"] == count
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before
    assert receipt["immutable_runtime"] == runtime


def test_incomplete_epoch_checkpoint_cannot_resume_under_other_python(tmp_path, software, monkeypatch):
    job, data, calls = software
    directory = tmp_path / "resume"

    def stop(directory, event):
        if event["kind"] == "epoch_checkpoint":
            raise OSError("software fixture stop before resume")

    with pytest.raises(training.DurabilityError):
        training.fit_job(job, data, data, directory, on_checkpoint=stop)
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    count = calls["loss"]
    runtime = training.critical_runtime_identity()
    monkeypatch.setattr(training, "critical_runtime_identity", lambda: {**runtime, "python_major_minor": "99.0"})
    with pytest.raises(ValueError, match="Critical software runtime differs"):
        training.fit_job(job, data, data, directory)
    assert calls["loss"] == count
    assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


def test_host_and_python_patch_observations_are_informational_not_resume_gates(tmp_path, software, monkeypatch):
    job, data, calls = software
    directory = tmp_path / "job"
    receipt = training.fit_job(job, data, data, directory)
    request = json.loads((directory / "request.json").read_text())
    assert request["immutable_runtime"]["python_major_minor"].count(".") == 1
    assert all(request["immutable_runtime"][name] for name in ("torch", "numpy", "scipy", "statsmodels"))
    observed = receipt["runtime_observed_at_creation"]
    assert observed["python_full_version"] and observed["platform"] and observed["machine"]
    monkeypatch.setattr(training.platform, "machine", lambda: "other-informational-machine")
    monkeypatch.setattr(training.platform, "python_version", lambda: "3.12.999")
    # The owned worker restores its prescribed two-thread setting, rather than
    # binding a job to the arbitrary setting of a caller's preceding command.
    torch.set_num_threads(1)
    assert training.verify_job(job, directory) == receipt
    assert torch.get_num_threads() == 2
    assert training.fit_job(job, data, data, directory) == receipt
    assert training.export_job_resources(directory)["runtime_observed_at_creation"] == observed
