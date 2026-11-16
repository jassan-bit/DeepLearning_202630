"""Synthetic-only integration tests for bounded V2 training and exact recovery."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.e3_innovation as trial
from src.e3_development import (DevelopmentModel, load_development_data,
    source_fingerprints as old_sources, _rng_state)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _write_json(path, value):
    Path(path).write_bytes(trial.canonical(value))


def _bar(day):
    k = (day - date(2024, 1, 1)).days
    close = np.exp(5 + .0005 * k + .03 * np.sin(k / 3))
    opening = close * (1 + .001 * np.cos(k))
    return [opening, max(opening, close) * 1.01, min(opening, close) * .99,
            close, 300 + 2 * k + k % 5]


@pytest.fixture
def fixture(tmp_path):
    ids = np.array(["2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30",
                    "2024-07-08", "2024-07-09"])
    raw, y = [], []
    for text in ids:
        origin = date.fromisoformat(text)
        bars = np.array([_bar(origin - timedelta(days=lag)) for lag in range(7, -1, -1)])
        raw.append(bars[:7])
        y.append(np.std(np.diff(np.log(bars[:, 3])), ddof=0))
    npz = tmp_path / "synthetic.npz"
    np.savez(npz, training_raw=np.array(raw), training_y_original=np.array(y), training_dates=ids,
             calibration_raw=np.array([{"forbidden": True}], dtype=object))
    sha = trial.sha256_file(npz)
    base = tmp_path / "frozen"
    (base / "M00").mkdir(parents=True)
    data = load_development_data(npz, sha)
    baseline_plan = {"seed": 11}
    plan_hash = hashlib.sha256(trial.canonical(baseline_plan)).hexdigest()
    data_hash = hashlib.sha256(trial.canonical(data["admission"])).hexdigest()
    sources = old_sources()
    cp = {"model_state": DevelopmentModel("M00", 11).state_dict(),
          "source_hashes": sources, "plan_sha256": plan_hash,
          "data_fingerprint": data_hash, "epoch": 1, "rng": _rng_state()}
    torch.save(cp, base / "M00/best.pt")
    _write_json(base / "data_admission.json", data["admission"])
    _write_json(base / "manifest.json", {"plan_sha256": plan_hash, "data_fingerprint": data_hash})
    _write_json(base / "plan.json", baseline_plan)
    members = ["M00/best.pt", "data_admission.json", "manifest.json", "plan.json"]
    snapshot = {"active_session": None, "state_status": "completed",
        "charged_seconds": trial.PRIOR_CHARGED, "source_hashes": sources,
        "data_fingerprint": data_hash, "plan_sha256": plan_hash,
        "files": {n: {"sha256": trial.sha256_file(base / n), "bytes": (base / n).stat().st_size} for n in members}}
    _write_json(base / "snapshot_manifest.json", snapshot)
    plan = json.loads((ROOT / "experiments/e3_innovacion_v2_20261116/plan.json").read_text())
    plan["data"].update(sha256=sha, fit_count=4, validation_count=2, purged_count=0)
    plan["training"].update(seed=117, batch_size=2, max_epochs_per_neural_variant=2)
    plan["frozen_control"].update(snapshot_manifest_sha256=trial.sha256_file(base / "snapshot_manifest.json"),
                                   best_checkpoint_sha256=trial.sha256_file(base / "M00/best.pt"))
    claim = {"version": trial.CLAIM_VERSION, "campaign_id": trial.CAMPAIGN,
        "run_id": "synthetic_run_001", "claim_id": "synthetic_claim_001",
        "v1_commit": trial.PRIOR_COMMIT, "prior_charged_seconds": trial.PRIOR_CHARGED,
        "additional_limit_seconds": 600, "global_limit_seconds": 1200.,
        "plan_sha256": hashlib.sha256(trial.canonical(plan)).hexdigest(),
        "npz_sha256": sha, "registry_commit": "a" * 40}
    return {"npz": npz, "sha": sha, "base": base, "plan": plan, "claim": claim, "tmp": tmp_path}


def _ledger(claim):
    return {"version": trial.VERSION, "limit_seconds": claim["additional_limit_seconds"],
        "charged_seconds": 0., "events": [], "active_session": {
            "session_id": "confirmed_reservation", "last_wall_utc": datetime.now(timezone.utc).isoformat()}}


def _create(f, name="run", **kwargs):
    directory = f["tmp"] / name
    trial.create_innovation_run(f["npz"], f["sha"], directory, plan=f["plan"],
        global_claim=f["claim"], initial_ledger=_ledger(f["claim"]), device="cpu",
        frozen_v1_dir=f["base"], snapshot_manifest_path=f["base"] / "snapshot_manifest.json", **kwargs)
    return directory


def _run(f, directory, **kwargs):
    # Simulate the coordinator's acknowledged reservation, retaining all prior
    # charges and any older ACTIVE timestamp that still needs recovery charging.
    ledger = trial._read_json(directory / "ledger.json")
    if ledger.get("active_session") is None:
        ledger["active_session"] = {"session_id": "confirmed_next_reservation",
            "last_wall_utc": datetime.now(timezone.utc).isoformat()}
        _write_json(directory / "ledger.json", ledger)
    return trial.run_innovation(directory, f["npz"], f["sha"], global_claim=f["claim"], device="cpu", **kwargs)


def _assert_same(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_same(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _assert_same(a, b)
    else:
        assert first == second


def test_claim_required_before_any_data_fit(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(trial, "load_innovation_data", lambda *a, **k: pytest.fail("Unclaimed fitting"))
    with pytest.raises(RuntimeError, match="claim"):
        trial.create_innovation_run(f["npz"], f["sha"], f["tmp"] / "no_claim", plan=f["plan"],
            frozen_v1_dir=f["base"], snapshot_manifest_path=f["base"] / "snapshot_manifest.json")
    assert not (f["tmp"] / "no_claim").exists()


def test_closed_preflight_receipt_cannot_fit_or_train_without_current_active_reservation(fixture, monkeypatch):
    f, directory = fixture, _create(fixture)
    ledger_before = trial._read_json(directory / "ledger.json")
    assert ledger_before["active_session"] is None
    monkeypatch.setattr(trial, "load_innovation_data", lambda *a, **k: pytest.fail("Fitting without ACTIVE"))
    monkeypatch.setattr(trial.torch.optim, "AdamW", lambda *a, **k: pytest.fail("Optimizer without ACTIVE"))
    with pytest.raises(RuntimeError, match="ACTIVE"):
        trial.run_innovation(directory, f["npz"], f["sha"], global_claim=f["claim"], device="cpu")
    assert trial._read_json(directory / "ledger.json") == ledger_before


@pytest.mark.parametrize("field,value", [("prior_charged_seconds", 0),
    ("global_limit_seconds", 1400), ("additional_limit_seconds", 601), ("registry_commit", "missing")])
def test_claim_cannot_reset_or_expand_global_budget(fixture, field, value):
    f = fixture
    claim = {**f["claim"], field: value}
    with pytest.raises(RuntimeError):
        trial.validate_global_claim(claim, f["plan"], f["sha"])


def test_prepare_is_billed_bias_fit_with_six_registered_recipes(fixture):
    f, directory = fixture, _create(fixture)
    manifest = trial._read_json(directory / "manifest.json")
    assert manifest["initialized"]
    assert manifest["run_id"] == f["claim"]["run_id"]
    assert trial._read_json(directory / "ledger.json")["charged_seconds"] > 0
    assert trial._read_json(directory / "ledger.json")["active_session"] is None
    assert trial._read_json(directory / "state.json")["status"] == "initialized"
    summary = trial._read_json(directory / "summary.json")
    assert len(summary["models"]) == 6 and not summary["comparison_complete"]
    for name in trial.NEURAL_NAMES:
        cp = trial._load_checkpoint(directory / name / "last.pt", manifest)
        assert cp["epoch"] == 0 and not cp["optimizer_state"]["state"]
        # The new checkpoints need no NumPy/pickle allowlist.
        safe = torch.load(directory / name / "last.pt", weights_only=True)
        assert safe["model_name"] == name
        assert sum(p.numel() for p in trial.InnovationModel(name, seed=117,
            u_scale=manifest["models"][name]["u_scale"], y_scale=manifest["models"][name]["y_scale"]).parameters()) == 1210
    pair = [trial._load_checkpoint(directory / n / "last.pt", manifest)["model_state"] for n in trial.NEURAL_NAMES[1:]]
    _assert_same(*pair)
    assert summary["models"]["floor_only"]["parameters"] == 0
    assert summary["models"]["floor_plus_fit_mean"]["parameters"] == 1
    assert summary["models"]["frozen_M00_projected"]["parameters"] == 1210


def test_interleaved_pause_resume_equals_continuous_including_rng_optimizer_and_csv(fixture):
    f = fixture
    uninterrupted = _create(f, "continuous")
    result = _run(f, uninterrupted)
    assert result["status"] == "completed" and result["comparison_complete"]
    paused = _create(f, "paused")
    partial = _run(f, paused, stop_after_epochs=4)
    assert partial["status"] == "paused" and not partial["decision"]["continuation_gate_passed"]
    assert [partial["models"][n]["completed_epochs"] for n in trial.NEURAL_NAMES] == [2, 1, 1]
    final = _run(f, paused, resume=True)
    assert final["comparison_complete"] and len(final["metrics"]) == 6
    for name in trial.NEURAL_NAMES:
        for checkpoint in ("last.pt", "best.pt"):
            left = torch.load(uninterrupted / name / checkpoint, weights_only=True)
            right = torch.load(paused / name / checkpoint, weights_only=True)
            _assert_same(left, right)
        assert (uninterrupted / name / "validation_predictions.csv").read_bytes() == (paused / name / "validation_predictions.csv").read_bytes()
    assert final["models"]["innovation_density_mean"]["coverage90"] is not None
    assert final["models"]["additive_positive_head"]["coverage90"] is None


class FakeClock:
    def __init__(self, step=100.):
        self.value, self.step = 0., step

    def __call__(self):
        self.value += self.step
        return self.value


def test_initialization_timeout_charges_every_attempt_and_never_declares_winner(fixture):
    f = fixture
    directory = _create(f, clock=FakeClock())
    summary = trial._read_json(directory / "summary.json")
    assert summary["status"] == "budget_exhausted"
    assert summary["pilot_incomplete"] and not summary["winner_declared"]
    assert not summary["decision"]["continuation_gate_passed"]
    assert not trial._read_json(directory / "manifest.json")["initialized"]
    previous = summary["charged_seconds"]
    result = _run(f, directory, resume=True, clock=FakeClock())
    assert result["status"] == "budget_exhausted"
    assert result["charged_seconds"] >= previous >= 600
    assert result["global_charged_seconds"] == pytest.approx(trial.PRIOR_CHARGED + result["charged_seconds"])


def test_training_timeout_leaves_honest_partial_checkpoints(fixture):
    f, directory = fixture, _create(fixture)
    result = _run(f, directory, clock=FakeClock(step=80))
    assert result["status"] == "budget_exhausted"
    assert not result["comparison_complete"] and not result["decision"]["continuation_gate_passed"]
    assert sum(result["models"][n]["completed_epochs"] for n in trial.NEURAL_NAMES) < 6
    assert trial._read_json(directory / "ledger.json")["active_session"] is None


def test_active_reservation_gap_is_charged_before_fit(fixture, monkeypatch):
    f = fixture
    ledger = _ledger(f["claim"])
    ledger["active_session"]["last_wall_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=700)).isoformat()
    monkeypatch.setattr(trial, "load_innovation_data", lambda *a, **k: pytest.fail("Expired reservation reached a fit"))
    directory = f["tmp"] / "expired"
    trial.create_innovation_run(f["npz"], f["sha"], directory, plan=f["plan"], device="cpu",
        global_claim=f["claim"], initial_ledger=ledger, frozen_v1_dir=f["base"],
        snapshot_manifest_path=f["base"] / "snapshot_manifest.json")
    stored = trial._read_json(directory / "ledger.json")
    assert stored["charged_seconds"] >= 600
    assert stored["events"][0]["previous_last_wall_utc"] == ledger["active_session"]["last_wall_utc"]


def test_frozen_hash_is_checked_before_safe_torch_load(fixture, monkeypatch):
    f = fixture
    with (f["base"] / "M00/best.pt").open("ab") as stream:
        stream.write(b"corrupted")
    monkeypatch.setattr(trial, "_safe_frozen_load", lambda *_: pytest.fail("Unverified checkpoint deserialized"))
    with pytest.raises(RuntimeError, match="hash/size"):
        _create(f)
    assert trial._read_json(f["tmp"] / "run/ledger.json")["charged_seconds"] > 0


def test_corrupt_resume_checkpoint_rejected_before_deserialization(fixture, monkeypatch):
    f, directory = fixture, _create(fixture)
    with (directory / trial.NEURAL_NAMES[0] / "last.pt").open("ab") as stream:
        stream.write(b"untrusted")
    manifest = trial._read_json(directory / "manifest.json")
    monkeypatch.setattr(trial.torch, "load", lambda *a, **k: pytest.fail("Invalid bytes deserialized"))
    with pytest.raises(RuntimeError, match="hash"):
        trial._load_checkpoint(directory / trial.NEURAL_NAMES[0] / "last.pt", manifest)


def test_resume_fails_if_plan_or_source_changed(fixture, monkeypatch):
    f, directory = fixture, _create(fixture)
    source = trial.source_fingerprints()
    monkeypatch.setattr(trial, "source_fingerprints", lambda: {**source, "src/e3_innovation.py": "f" * 64})
    with pytest.raises(RuntimeError, match="Source fingerprint"):
        _run(f, directory)


def test_failed_best_publication_is_recovered_from_authoritative_last(fixture, monkeypatch):
    f = fixture
    continuous = _create(f, "continuous")
    _run(f, continuous)
    interrupted = _create(f, "interrupted")
    original_save = trial._save_checkpoint
    fault = {"armed": True}

    def failed_best(path, value):
        if Path(path).name == "best.pt" and fault["armed"]:
            fault["armed"] = False
            raise OSError("simulated closed artifact publication failure")
        return original_save(path, value)

    monkeypatch.setattr(trial, "_save_checkpoint", failed_best)
    with pytest.raises(OSError):
        _run(f, interrupted)
    before = trial._read_json(interrupted / "ledger.json")["charged_seconds"]
    final = _run(f, interrupted, resume=True)
    assert final["comparison_complete"] and final["charged_seconds"] >= before
    for name in trial.NEURAL_NAMES:
        _assert_same(torch.load(continuous / name / "last.pt", weights_only=True),
                     torch.load(interrupted / name / "last.pt", weights_only=True))
        assert (continuous / name / "validation_predictions.csv").read_bytes() == (interrupted / name / "validation_predictions.csv").read_bytes()


def test_interruption_after_one_saved_minibatch_resumes_exactly(fixture, monkeypatch):
    f = fixture
    continuous = _create(f, "continuous")
    _run(f, continuous)
    interrupted = _create(f, "interrupted")
    original_save = trial._save_checkpoint
    armed = {"value": True}

    def stop_after_batch(path, value):
        original_save(path, value)
        if Path(path).name == "last.pt" and value["batch_cursor"] == 2 and armed["value"]:
            armed["value"] = False
            raise KeyboardInterrupt("synthetic disconnect after durable minibatch")

    monkeypatch.setattr(trial, "_save_checkpoint", stop_after_batch)
    with pytest.raises(KeyboardInterrupt):
        _run(f, interrupted)
    manifest = trial._read_json(interrupted / "manifest.json")
    cp = trial._load_checkpoint(interrupted / trial.NEURAL_NAMES[0] / "last.pt", manifest)
    assert cp["epoch"] == 0 and cp["batch_cursor"] == 2 and cp["active_permutation"] is not None
    assert _run(f, interrupted, resume=True)["comparison_complete"]
    for name in trial.NEURAL_NAMES:
        _assert_same(torch.load(continuous / name / "last.pt", weights_only=True),
                     torch.load(interrupted / name / "last.pt", weights_only=True))


def test_checkpoint_published_before_receipt_compaction_remains_recoverable(fixture, monkeypatch):
    f = fixture
    continuous = _create(f, "continuous")
    _run(f, continuous)
    interrupted = _create(f, "interrupted")
    original_replace = trial.os.replace
    armed = {"value": True}

    def lost_ack(source, destination):
        original_replace(source, destination)
        if (Path(destination) == interrupted / trial.NEURAL_NAMES[0] / "last.pt"
                and armed["value"]):
            armed["value"] = False
            raise OSError("synthetic disconnect after atomic checkpoint publication")

    monkeypatch.setattr(trial.os, "replace", lost_ack)
    with pytest.raises(OSError):
        _run(f, interrupted)
    cp_path = interrupted / trial.NEURAL_NAMES[0] / "last.pt"
    assert len(cp_path.with_suffix(".pt.sha256").read_text().splitlines()) == 2
    assert _run(f, interrupted, resume=True)["comparison_complete"]
    for name in trial.NEURAL_NAMES:
        _assert_same(torch.load(continuous / name / "last.pt", weights_only=True),
                     torch.load(interrupted / name / "last.pt", weights_only=True))


def test_interrupted_initialization_restarts_billed_matching_with_same_claim_on_new_path(fixture, monkeypatch):
    f = fixture
    continuous = _create(f, "continuous")
    _run(f, continuous)
    original_save = trial._save_checkpoint
    armed = {"value": True}

    def stop_during_initial_commit(path, value):
        original_save(path, value)
        if value["model_name"] == trial.NEURAL_NAMES[1] and value["epoch"] == 0 and armed["value"]:
            armed["value"] = False
            raise KeyboardInterrupt("synthetic disconnect before third initial checkpoint")

    monkeypatch.setattr(trial, "_save_checkpoint", stop_during_initial_commit)
    with pytest.raises(KeyboardInterrupt):
        _create(f, "interrupted")
    interrupted = f["tmp"] / "interrupted"
    assert not trial._read_json(interrupted / "manifest.json")["initialized"]
    assert not (interrupted / trial.NEURAL_NAMES[2] / "last.pt").exists()
    before = trial._read_json(interrupted / "ledger.json")["charged_seconds"]
    relocated = f["tmp"] / "new_vm_baseline_path"
    f["base"].rename(relocated)
    final = _run(f, interrupted, resume=True, frozen_v1_dir=relocated,
                 snapshot_manifest_path=relocated / "snapshot_manifest.json")
    assert final["comparison_complete"] and final["charged_seconds"] > before > 0
    assert final["run_id"] == f["claim"]["run_id"] and final["claim_id"] == f["claim"]["claim_id"]
    for name in trial.NEURAL_NAMES:
        _assert_same(torch.load(continuous / name / "last.pt", weights_only=True),
                     torch.load(interrupted / name / "last.pt", weights_only=True))


def test_all_three_completed_networks_without_controls_are_still_incomplete(fixture):
    f, directory = fixture, _create(fixture)
    assert _run(f, directory)["comparison_complete"]
    (directory / "controls.json").unlink()
    manifest = trial._read_json(directory / "manifest.json")
    summary = trial._summary(directory, manifest, "completed")
    assert summary["pilot_incomplete"] and not summary["comparison_complete"]
    assert not summary["decision"]["complete"]
    assert not summary["decision"]["continuation_gate_passed"]
    with pytest.raises(RuntimeError, match="control"):
        _run(f, directory, resume=True)


@pytest.mark.parametrize("failure", ["budget_exhausted", "interrupted"])
def test_setup_failure_preserves_progress_for_recipes_not_yet_loaded(fixture, monkeypatch, failure):
    f, directory = fixture, _create(fixture)
    partial = _run(f, directory, stop_after_epochs=3)
    assert [partial["models"][n]["completed_epochs"] for n in trial.NEURAL_NAMES] == [1, 1, 1]
    previous = trial._read_json(directory / "state.json")
    assert set(previous["models"]) == set(trial.NEURAL_NAMES)
    charged = trial._read_json(directory / "ledger.json")["charged_seconds"]
    if failure == "budget_exhausted":
        # The fourth tick exhausts the remaining counter while setting up the
        # second NN, after the first checkpoint was loaded and before any fit.
        result = _run(f, directory, resume=True, clock=FakeClock(step=160.))
        assert result["status"] == "budget_exhausted"
    else:
        original_tick = trial._Budget.tick
        counter = {"value": 0}

        def interrupt_setup(self):
            result = original_tick(self)
            counter["value"] += 1
            if counter["value"] == 4:
                raise KeyboardInterrupt("synthetic disconnect during second recipe setup")
            return result

        monkeypatch.setattr(trial._Budget, "tick", interrupt_setup)
        with pytest.raises(KeyboardInterrupt):
            _run(f, directory, resume=True)
    state = trial._read_json(directory / "state.json")
    assert state["status"] == failure
    assert state["models"] == previous["models"]
    assert trial._read_json(directory / "ledger.json")["charged_seconds"] >= charged
    summary = trial._read_json(directory / "summary.json")
    assert [summary["models"][n]["completed_epochs"] for n in trial.NEURAL_NAMES] == [1, 1, 1]
