"""Synthetic-only bounded runner integration; no market NPZ or prior weights."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import e3_moment_training as trial
from src.e3_innovation_model import load_innovation_data


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_json(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(trial.canonical(record))


def bar(day):
    k = (day - date(2024, 1, 1)).days
    close = np.exp(5 + .0005 * k + .03 * np.sin(k / 3))
    opening = close * (1 + .001 * np.cos(k))
    return [opening, max(opening, close) * 1.01, min(opening, close) * .99,
            close, 300 + 2 * k + k % 5]


def seal_reference(f):
    root = f["reference"]
    snapshot = {"registry": f["prior_registry"], "progress": {"state_status": "completed",
        "models": {name: {"epoch": 40, "complete": True} for name in
                   ("additive_positive_head", "innovation_mean_only", "innovation_density_mean")}},
        "scientific_identity": {"data_fingerprint": f["prior_manifest"]["data_fingerprint"]},
        "files": {name: {"bytes": (root / name).stat().st_size,
                         "sha256": trial.sha256_file(root / name)} for name in trial.REFERENCE_FILES}}
    write_json(root / "snapshot_manifest.json", snapshot)
    f["plan"]["frozen_control"].update(
        snapshot_manifest_sha256=trial.sha256_file(root / "snapshot_manifest.json"),
        predictions_sha256=trial.sha256_file(root / "additive_positive_head/validation_predictions.csv"))
    f["claim"] = {"version": trial.CLAIM_VERSION, "campaign_id": trial.CAMPAIGN,
        "prior_commit": trial.PRIOR_COMMIT,
        "prior_snapshot_sha256": f["plan"]["frozen_control"]["snapshot_manifest_sha256"],
        "prior_charged_seconds": trial.PRIOR_CHARGED, "additional_limit_seconds": 600.,
        "global_limit_seconds": 1200., "npz_sha256": f["sha"],
        "plan_sha256": hashlib.sha256(trial.canonical(f["plan"])).hexdigest(),
        "run_id": "synthetic_moment_run", "claim_id": "synthetic_moment_claim", "registry_commit": "a" * 40}


@pytest.fixture
def fixture(tmp_path):
    ids = np.asarray(["2024-06-27", "2024-06-28", "2024-06-29", "2024-06-30", "2024-07-08", "2024-07-09"])
    raw, targets = [], []
    for text in ids:
        origin = date.fromisoformat(str(text))
        bars = np.asarray([bar(origin - timedelta(days=lag)) for lag in range(7, -1, -1)])
        raw.append(bars[:7])
        targets.append(np.std(np.diff(np.log(bars[:, 3])), ddof=0))
    npz = tmp_path / "artificial.npz"
    np.savez(npz, training_raw=np.asarray(raw), training_y_original=np.asarray(targets), training_dates=ids,
             calibration_raw=np.asarray([{"forbidden": True}], dtype=object),
             evaluation_raw=np.asarray([{"forbidden": True}], dtype=object))
    sha = trial.sha256_file(npz)
    old_data = load_innovation_data(npz, sha)
    plan = json.loads((ROOT / "experiments/e3_piloto_momentos_20261116/plan.json").read_text())
    plan["data"].update(sha256=sha, fit_count=4, validation_count=2, purged_count=0)
    plan["training"].update(seed=117, batch_size=2, max_epochs_per_neural_variant=2)
    reference = tmp_path / "reference"
    reference.mkdir()
    prior_plan = {"seed": 11, "fixture": "artificial sealed text reference"}
    prior_manifest = {"plan_sha256": hashlib.sha256(trial.canonical(prior_plan)).hexdigest(),
        "data_fingerprint": hashlib.sha256(trial.canonical(old_data["admission"])).hexdigest()}
    registry = {"active_session": None, "prior_charged_seconds": 369.4779596389991,
        "charged_seconds": 186.73469972300006, "plan_sha256": prior_manifest["plan_sha256"], "npz_sha256": sha}
    for name, record in (("plan.json", prior_plan), ("manifest.json", prior_manifest),
        ("data_admission.json", old_data["admission"]), ("global_registry.json", registry),
        ("state.json", {"status": "completed"}),
        ("ledger.json", {"active_session": None, "charged_seconds": registry["charged_seconds"]}),
        ("summary.json", {"status": "completed", "global_charged_seconds": trial.PRIOR_CHARGED})):
        write_json(reference / name, record)
    rows = [{"date": str(day), "target_sigma7": float(y), "prediction_sigma7": float(y + .001),
             "known_floor": float(np.sqrt(a))} for day, y, a in zip(old_data["validation_dates"],
                old_data["validation_y_original"], old_data["validation_A"])]
    (reference / "additive_positive_head").mkdir(exist_ok=True)
    trial._csv(reference / "additive_positive_head/validation_predictions.csv", rows)
    f = {"npz": npz, "sha": sha, "tmp": tmp_path, "reference": reference,
         "plan": plan, "prior_manifest": prior_manifest, "prior_registry": registry}
    seal_reference(f)
    return f


def ledger(f):
    return {"limit_seconds": f["claim"]["additional_limit_seconds"], "charged_seconds": 0., "events": [],
        "active_session": {"session_id": "acknowledged_artificial_reservation",
                           "last_wall_utc": datetime.now(timezone.utc).isoformat()}}


def create(f, name="run", **kwargs):
    directory = f["tmp"] / name
    trial.create_moment_run(f["npz"], f["sha"], directory, plan=f["plan"],
        global_claim=f["claim"], initial_ledger=ledger(f), device="cpu",
        frozen_v2_dir=f["reference"], snapshot_manifest_path=f["reference"] / "snapshot_manifest.json", **kwargs)
    return directory


def run(f, directory, **kwargs):
    current = trial._read_json(directory / "ledger.json")
    if current["active_session"] is None:
        current["active_session"] = {"session_id": "acknowledged_next_artificial_reservation",
                                    "last_wall_utc": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "ledger.json", current)
    return trial.run_moment(directory, f["npz"], f["sha"], global_claim=f["claim"], device="cpu", **kwargs)


def assert_same(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same(a, b)
    else:
        assert first == second


def test_claim_and_active_ack_required_before_loading_data_or_creating_run(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(trial, "load_moment_data", lambda *a, **k: pytest.fail("Unclaimed fit"))
    with pytest.raises(RuntimeError, match="claim"):
        trial.create_moment_run(f["npz"], f["sha"], f["tmp"] / "no_claim", plan=f["plan"])
    assert not (f["tmp"] / "no_claim").exists()
    closed = ledger(f)
    closed["active_session"] = None
    with pytest.raises(RuntimeError, match="ACTIVE"):
        trial.create_moment_run(f["npz"], f["sha"], f["tmp"] / "no_active", plan=f["plan"],
            global_claim=f["claim"], initial_ledger=closed, device="cpu", frozen_v2_dir=f["reference"],
            snapshot_manifest_path=f["reference"] / "snapshot_manifest.json")
    assert not (f["tmp"] / "no_active").exists()


@pytest.mark.parametrize("key,value", [("prior_charged_seconds", 0.), ("global_limit_seconds", 1400.),
    ("additional_limit_seconds", 601.), ("prior_commit", "b" * 40), ("registry_commit", "missing")])
def test_global_claim_cannot_reset_expand_or_switch_predecessor(fixture, key, value):
    f = fixture
    with pytest.raises(RuntimeError):
        trial.validate_global_claim({**f["claim"], key: value}, f["plan"], f["sha"])


def test_preflight_loads_only_training_members_and_never_prior_checkpoint(fixture, monkeypatch):
    f = fixture
    observed = []
    original = torch.load
    def tracked(path, *args, **kwargs):
        observed.append(str(path))
        assert not str(path).startswith(str(f["reference"]))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", tracked)
    directory = create(f)
    admission = trial._read_json(directory / "data_admission.json")
    assert admission["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
    assert admission["forbidden_arrays_loaded"] is False
    manifest = trial._read_json(directory / "manifest.json")
    assert manifest["initialized"] and len(manifest["models"]) == 3
    assert all(x["trainable_parameters"] == 1236 for x in manifest["models"].values())
    budget = trial._read_json(directory / "ledger.json")
    assert budget["active_session"] is None and budget["charged_seconds"] > 0
    assert len(observed) == 3  # Our newly initialized last.pt files only.
    assert trial._read_json(directory / "controls.json")["frozen_control"]["prior_checkpoint_loaded"] is False


def test_closed_preflight_cannot_train_without_a_fresh_active_reservation(fixture, monkeypatch):
    f, directory = fixture, create(fixture)
    before = (directory / "ledger.json").read_bytes()
    monkeypatch.setattr(trial, "load_moment_data", lambda *a, **k: pytest.fail("Unreserved fit"))
    with pytest.raises(RuntimeError, match="ACTIVE"):
        trial.run_moment(directory, f["npz"], f["sha"], global_claim=f["claim"], device="cpu")
    assert (directory / "ledger.json").read_bytes() == before


def test_resume_matches_uninterrupted_checkpoints_optimizer_and_curves_exactly(fixture):
    f = fixture
    direct, resumed = create(f, "direct"), create(f, "resumed")
    full = run(f, direct)
    partial = run(f, resumed, stop_after_epochs=1)
    assert partial["status"] == "paused" and partial["comparison_complete"] is False
    assert partial["exploratory_continuation_gate_passed"] is False
    assert sum(m["completed_epochs"] for n, m in partial["models"].items() if n in trial.NEURAL_NAMES) == 1
    final = run(f, resumed, resume=True)
    assert full["status"] == final["status"] == "completed"
    assert full["metrics"] == final["metrics"]
    for name in trial.NEURAL_NAMES:
        manifest = trial._read_json(direct / "manifest.json")
        assert_same(trial._load_checkpoint(direct / name / "last.pt", manifest),
                    trial._load_checkpoint(resumed / name / "last.pt", trial._read_json(resumed / "manifest.json")))
        assert (direct / name / "curves.csv").read_bytes() == (resumed / name / "curves.csv").read_bytes()
        assert (resumed / name / "validation_predictions.csv").is_file()
    assert final["prior_charged_seconds"] == trial.PRIOR_CHARGED
    assert final["global_charged_seconds"] == trial.PRIOR_CHARGED + final["charged_seconds"]
    assert final["calibration_scored"] is False and final["evaluation_scored"] is False
    assert final["winner_declared"] is False


def test_midbatch_interrupt_resumes_cursor_and_preserves_exact_training(fixture, monkeypatch):
    f = fixture
    direct, interrupted = create(f, "direct"), create(f, "interrupted")
    run(f, direct)
    original = trial._save_checkpoint
    triggered = False
    def once(path, value):
        nonlocal triggered
        original(path, value)
        if not triggered and value.get("epoch") == 0 and value.get("batch_cursor", 0) > 0:
            triggered = True
            raise KeyboardInterrupt("artificial interruption after committed batch")
    monkeypatch.setattr(trial, "_save_checkpoint", once)
    with pytest.raises(KeyboardInterrupt):
        run(f, interrupted)
    assert trial._read_json(interrupted / "state.json")["status"] == "interrupted"
    cp = trial._load_checkpoint(interrupted / "moment_joint/last.pt", trial._read_json(interrupted / "manifest.json"))
    assert cp["batch_cursor"] == 2 and cp["active_permutation"] is not None
    monkeypatch.setattr(trial, "_save_checkpoint", original)
    run(f, interrupted, resume=True)
    for name in trial.NEURAL_NAMES:
        assert_same(trial._load_checkpoint(direct / name / "last.pt", trial._read_json(direct / "manifest.json")),
                    trial._load_checkpoint(interrupted / name / "last.pt", trial._read_json(interrupted / "manifest.json")))


def test_resealed_reference_with_wrong_targets_still_fails_alignment(fixture):
    f = fixture
    path = f["reference"] / "additive_positive_head/validation_predictions.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["target_sigma7"] = float(rows[0]["target_sigma7"]) + .001
    trial._csv(path, rows)
    seal_reference(f)
    with pytest.raises(RuntimeError, match="targets differ"):
        create(f)
    assert trial._read_json(f["tmp"] / "run/state.json")["status"] == "failed"
    assert trial._read_json(f["tmp"] / "run/ledger.json")["active_session"] is None


def test_reference_bytes_rejected_before_any_prior_model_load(fixture):
    f = fixture
    with (f["reference"] / "summary.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(RuntimeError, match="byte hash/size"):
        create(f)


def test_crash_recovery_conservatively_charges_old_active_time(fixture):
    f = fixture
    active = ledger(f)
    active["charged_seconds"] = 7.
    active["active_session"]["last_wall_utc"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    directory = f["tmp"] / "recovered"
    trial.create_moment_run(f["npz"], f["sha"], directory, plan=f["plan"], global_claim=f["claim"],
        initial_ledger=active, device="cpu", frozen_v2_dir=f["reference"],
        snapshot_manifest_path=f["reference"] / "snapshot_manifest.json")
    result = trial._read_json(directory / "ledger.json")
    assert result["charged_seconds"] >= 12.
    assert any(e["kind"] == "unclosed_session_recovery" and e["seconds"] >= 5 for e in result["events"])


def test_budget_exhaustion_is_terminal_and_never_claims_complete_gate(fixture, monkeypatch):
    f = fixture
    step = iter([float(i * 400) for i in range(100)])
    directory = create(f, clock=lambda: next(step))
    report = trial._read_json(directory / "summary.json")
    assert report["status"] == "budget_exhausted" and report["comparison_complete"] is False
    assert report["exploratory_continuation_gate_passed"] is False
    assert report["winner_declared"] is False
    monkeypatch.setattr(trial, "load_moment_data", lambda *a, **k: pytest.fail("Terminal budget reopened"))
    before = trial._read_json(directory / "ledger.json")["charged_seconds"]
    result = trial.run_moment(directory, f["npz"], f["sha"], global_claim=f["claim"], resume=True, device="cpu")
    assert result["status"] == "budget_exhausted" and result["charged_seconds"] == before


def test_numerical_failure_is_terminal_and_keeps_checkpoint_evidence(fixture, monkeypatch):
    f, directory = fixture, create(fixture)
    def fail(*args, **kwargs):
        raise RuntimeError("nonfinite artificial loss")
    monkeypatch.setattr(trial, "loss_components", fail)
    with pytest.raises(RuntimeError, match="nonfinite"):
        run(f, directory)
    assert trial._read_json(directory / "state.json")["status"] == "failed"
    assert all((directory / name / "last.pt.sha256").is_file() for name in trial.NEURAL_NAMES)
    monkeypatch.setattr(trial, "load_moment_data", lambda *a, **k: pytest.fail("Failed trial reopened"))
    before = trial._read_json(directory / "ledger.json")["charged_seconds"]
    result = trial.run_moment(directory, f["npz"], f["sha"], global_claim=f["claim"], resume=True, device="cpu")
    assert result["status"] == "failed" and result["charged_seconds"] == before


def test_checkpoint_tampering_rejects_resume(fixture):
    f, directory = fixture, create(fixture)
    path = directory / "moment_joint/last.pt"
    with path.open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(RuntimeError, match="byte hash"):
        run(f, directory)
    assert trial._read_json(directory / "state.json")["status"] == "failed"


def test_runtime_change_rejected_before_data_refit(fixture, monkeypatch):
    f, directory = fixture, create(fixture)
    original = trial.runtime_info
    monkeypatch.setattr(trial, "runtime_info", lambda *a, **k: {**original(*a, **k), "torch": "changed"})
    monkeypatch.setattr(trial, "load_moment_data", lambda *a, **k: pytest.fail("Changed-runtime refit"))
    with pytest.raises(RuntimeError, match="runtime/device"):
        run(f, directory)


def test_equal_shaped_checkpoint_cannot_switch_recipe_with_valid_digest(fixture):
    f, directory = fixture, create(fixture)
    source = directory / "moment_joint/last.pt"
    target = directory / "moment_mean_only/last.pt"
    target.write_bytes(source.read_bytes())
    target.with_suffix(".pt.sha256").write_bytes(source.with_suffix(".pt.sha256").read_bytes())
    with pytest.raises(RuntimeError, match="variant/version"):
        run(f, directory)
    state = trial._read_json(directory / "state.json")
    assert state["status"] == "failed"
    assert all(row["epoch"] == 0 for row in state["models"].values())


def test_checkpoint_version_is_required_even_with_valid_digest(fixture):
    f, directory = fixture, create(fixture)
    manifest = trial._read_json(directory / "manifest.json")
    path = directory / "moment_joint/last.pt"
    checkpoint = trial._load_checkpoint(path, manifest)
    checkpoint["version"] = "some_other_trial"
    trial._save_checkpoint(path, checkpoint)
    with pytest.raises(RuntimeError, match="variant/version"):
        run(f, directory)


def test_optimizer_overflow_preserves_last_committed_checkpoint(fixture, monkeypatch):
    f, directory = fixture, create(fixture)
    before = {name: (directory / name / "last.pt").read_bytes() for name in trial.NEURAL_NAMES}
    original = torch.optim.AdamW.step
    def corrupt(self, *args, **kwargs):
        value = original(self, *args, **kwargs)
        with torch.no_grad():
            self.param_groups[0]["params"][0].flatten()[0] = float("nan")
        return value
    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt)
    with pytest.raises(RuntimeError, match="Nonfinite model/optimizer state"):
        run(f, directory)
    assert trial._read_json(directory / "state.json")["status"] == "failed"
    for name in trial.NEURAL_NAMES:
        assert (directory / name / "last.pt").read_bytes() == before[name]


def test_acknowledged_progress_cannot_repeat_an_older_valid_checkpoint(fixture, monkeypatch):
    f, directory = fixture, create(fixture)
    path = directory / "moment_joint/last.pt"
    initial = path.read_bytes()
    initial_receipt = path.with_suffix(".pt.sha256").read_bytes()
    run(f, directory, stop_after_epochs=1)
    assert trial._read_json(directory / "state.json")["models"]["moment_joint"]["epoch"] == 1
    path.write_bytes(initial)
    path.with_suffix(".pt.sha256").write_bytes(initial_receipt)
    monkeypatch.setattr(trial, "loss_components", lambda *a, **k: pytest.fail("Rolled-back training"))
    with pytest.raises(RuntimeError, match="progress regresses"):
        run(f, directory, resume=True)
