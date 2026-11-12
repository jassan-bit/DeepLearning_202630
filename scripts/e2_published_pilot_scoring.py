"""Score the fifteen frozen pilot fits only after a durable common freeze.

No fit, retry, selection or checkpoint resume is performed here. The complete
checkpoint domain is verified and loaded before the scoring-label loader is
called. A dated replay then predicts before revealing each daily observation.
CPU opt-in is exclusively a synthetic fixture and cannot close scientific R.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import torch

import e2_amended_pilot_power as power
from e2_amended_metrics import select_local_hpo
from e2_amended_pilot_data import load_pilot_panel
from e2_published_pilot_training import predict_pilot
from e2_published_registry import build_train_model, registered_trials
from e2_real_executor import (DatedReplay, FrozenSystem, canonical_hash,
                              run_frozen_block)
from e2_staged_training import state_sha256
from e2_temporal_training import prepare_b

VERSION = "e2-published-frozen-pilot-scoring-v1"
ROOT = Path(__file__).resolve().parents[1]
ORIGIN, END, CUTOFF = "2024-11-08", "2024-12-30", "2024-10-31"


def _runner_sha(value):
    """The worker's JSON identity predates compact power-payload digests."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_path(root, relative):
    root = Path(root).absolute()
    if not isinstance(relative, str) or not relative:
        raise ValueError("An explicit relative artifact path is required")
    child = Path(relative)
    if child.is_absolute() or ".." in child.parts:
        raise ValueError("Artifact paths must stay within their attempt")
    path = root / child
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink artifact paths are not admitted")
    if not path.is_file():
        raise ValueError("Missing artifact: " + relative)
    return path


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP):
                raise
    finally:
        os.close(fd)


def _write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


class _ReplayJournal:
    """Executor-compatible journal; file fsync remains mandatory on FUSE."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        _fsync_directory(self.directory.parent)
        self.file = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self.sequence, self.previous, self.failed = 0, "0" * 64, False
        _fsync_directory(self.directory)

    def append(self, kind, **payload):
        if self.failed:
            raise RuntimeError("Replay journal stopped after write failure")
        event = {"sequence": self.sequence, "previous_sha256": self.previous,
                 "kind": kind, **payload}
        token = canonical_hash(event)
        event["event_sha256"] = token
        try:
            self.file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            self.file.flush()
            os.fsync(self.file.fileno())
        except Exception:
            self.failed = True
            raise
        self.sequence, self.previous = self.sequence + 1, token
        return token

    def close(self):
        self.file.close()


def _winners(receipt, output):
    saved = _safe_path(output, "october_winners.json")
    if json.loads(saved.read_text(encoding="utf-8")) != receipt:
        raise ValueError("The supplied October freeze differs from its saved receipt")
    if (receipt.get("power_pilot_labels_used") is not False
            or set(receipt.get("winners", {})) != set(power.MODELS)):
        raise ValueError("Freeze exactly the three October recipes before power labels")
    for name in power.MODELS:
        winner = receipt["winners"][name]
        trials = registered_trials(name)
        if (winner.get("trial") not in trials or type(winner.get("fixed_epochs")) is not int
                or not 1 <= winner["fixed_epochs"] <= 200):
            raise ValueError("An exact registered October trial and fixed epoch count are required")
        selection = winner.get("selection", {})
        choice = select_local_hpo(selection.get("ledger", []), anchor="2024-10-01",
                                 registered_trial_ids=[row["trial_id"] for row in trials])
        if choice != selection or choice["winner_trial_id"] != winner["trial"]["trial_id"]:
            raise ValueError("The October winner differs from the complete HPO comparison")


def _runtime(runtime, device, fixture):
    if any(runtime.get(key) != value for key, value in
           (("dtype", "float32"), ("amp", False), ("tf32", False), ("deterministic", True))):
        raise ValueError("The frozen FP32 deterministic execution policy differs")
    selected = torch.device(device)
    if selected.type == "cpu":
        if not fixture:
            raise ValueError("CPU scoring is restricted to explicitly synthetic fixtures")
    elif selected.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("A usable CUDA device is required for registered pilot scoring")
    elif (runtime.get("device") != str(selected)
          or runtime.get("torch") != torch.__version__
          or runtime.get("cuda") != torch.version.cuda
          or runtime.get("device_name") != torch.cuda.get_device_name(selected)
          or runtime.get("total_memory_bytes") != torch.cuda.get_device_properties(selected).total_memory):
        raise ValueError("Scoring hardware or PyTorch differs from the fitted policy")
    return selected


def _verify_result(result, *, output, source_identity, runtime_identity, winner_receipt,
                   train, data_receipt, source_roots, fixture):
    job = result.get("job", {})
    name, seed = job.get("model"), job.get("seed")
    if (name not in power.MODELS or type(seed) is not int or seed not in power.SEEDS
            or job.get("stage") != "power" or job.get("origin") != ORIGIN or job.get("end") != END
            or result.get("status") != "completed"):
        raise ValueError("Each result must be a completed registered power model/seed job")
    winner = winner_receipt["winners"][name]
    if (job.get("trial") != winner["trial"] or job.get("fixed_epochs") != winner["fixed_epochs"]
            or job.get("winner_receipt_sha256") != _runner_sha(winner_receipt)):
        raise ValueError("Power fit does not use its frozen October recipe")
    if not fixture and (result.get("synthetic") is not False
                        or result.get("test_only_cpu_execution") is not False):
        raise ValueError("Synthetic or CPU fits cannot become registered market pilots")
    attempt = result.get("attempt_id")
    if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{64}", attempt):
        raise ValueError("A full SHA256 attempt identity is required")
    if type(result.get("optimizer_steps")) is not int or result["optimizer_steps"] < 1:
        raise ValueError("A completed fit needs known positive optimizer steps")
    expected = {"prepared_signature": train.signature,
                "data_sha256": _runner_sha(data_receipt["source_sha256"]),
                "model_source_sha256": _runner_sha(source_identity),
                "execution_policy_sha256": _runner_sha(runtime_identity),
                "recipe_id": power.RECIPES[name]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("A fitted source, data, preparation or execution context differs")
    attempt_root = Path(output) / "attempts" / attempt
    artifacts = result.get("artifacts_sha256", {})
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("The fit must retain its verified artifact manifest")
    for relative, digest in artifacts.items():
        if _file_sha(_safe_path(attempt_root, relative)) != digest:
            raise ValueError("Fit artifact SHA256 mismatch: " + relative)
    selected = result.get("selected_checkpoint", {})
    relative = selected.get("path")
    checkpoint_path = _safe_path(attempt_root, relative)
    if (not Path(relative).is_relative_to(Path("artifacts") / attempt / "checkpoints")
            or artifacts.get(relative) != selected.get("sha256")
            or _file_sha(checkpoint_path) != selected.get("sha256")):
        raise ValueError("Selected checkpoint SHA256 or attempt binding differs")
    identity_path = _safe_path(attempt_root, f"artifacts/{attempt}/identity.json")
    bound = json.loads(identity_path.read_text(encoding="utf-8"))
    identity = bound.get("identity", {})
    if (bound.get("attempt_id") != attempt or bound.get("identity_sha256") != power.canonical_sha256(identity)
            or result.get("identity_sha256") != _runner_sha(identity)
            or identity.get("job") != job or identity.get("sources") != source_identity
            or identity.get("runtime") != runtime_identity or identity.get("data") != data_receipt
            or identity.get("prepared_signature") != train.signature
            or identity.get("scaler") != train.scaler.record()):
        raise ValueError("Durable artifact identity differs from the verified fitted context")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if (checkpoint.get("version") != "e2-published-pilot-training-v1"
            or checkpoint.get("fixed_epochs") != winner["fixed_epochs"]
            or checkpoint.get("epoch") != winner["fixed_epochs"]
            or checkpoint.get("best_epoch") != winner["fixed_epochs"]
            or checkpoint.get("optimizer_steps") != result["optimizer_steps"]
            or checkpoint.get("target_scale") != train.scaler.target_scale
            or config.get("seed") != seed
            or any(config.get(key) != job["trial"].get(key, 0) for key in
                   ("learning_rate", "weight_decay", "batch_size", "warmup_fraction"))):
        raise ValueError("Selected checkpoint does not describe the completed fixed-epoch fit")
    model = build_train_model(name, job["trial"], seed=seed, source_roots=source_roots)
    if model.architecture_metadata.get("training_recipe_id") != power.RECIPES[name]:
        raise ValueError("The constructed published body has a different recipe identity")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if state_sha256(model) != checkpoint.get("model_state_sha256"):
        raise ValueError("Selected weights differ from their recorded tensor digest")
    return model, {"model_id": name, "seed": seed, "status": "completed",
        "fresh_initialization": True, "optimizer_steps": result["optimizer_steps"],
        **expected, "hpo_sha256": power.canonical_sha256(job["trial"]),
        "artifact_sha256": selected["sha256"], "attempt_id": attempt,
        "train_cutoff": CUTOFF, "recipe_selection_origin": "2024-10-01"}


def _read_vectors(path, model_name):
    previous, pending, dates, truth, forecasts, persistence, mask = "0" * 64, None, [], [], [], [], []
    completed = False
    with path.open(encoding="utf-8") as handle:
        for sequence, line in enumerate(handle):
            event = json.loads(line)
            digest = event.pop("event_sha256")
            if (event.get("sequence") != sequence or event.get("previous_sha256") != previous
                    or canonical_hash(event) != digest):
                raise ValueError("Replay journal hash chain differs")
            previous = digest
            if event["kind"] == "predictions":
                if pending is not None:
                    raise ValueError("Replay prediction is missing its next observation")
                pending = (event, digest)
            elif event["kind"] == "observation":
                if (pending is None or pending[0]["date"] != event["date"]
                        or event.get("prediction_event_sha256") != pending[1]):
                    raise ValueError("Replay observation lacks its prior durable prediction")
                prediction = pending[0]
                eligible = prediction["eligible"]
                dates.append(event["date"][:10]); mask.append(eligible)
                truth.append(event["value"] if eligible else None)
                forecasts.append(prediction["systems"][model_name]["value"] if eligible else None)
                persistence.append(prediction["systems"]["Persistence"]["value"] if eligible else None)
                pending = None
            elif event["kind"] == "block_completed":
                completed = True
    if not completed or pending is not None:
        raise ValueError("A complete dated scoring journal is required")
    return dates, mask, truth, persistence, forecasts


def score_frozen_pilots(power_results, *, output, data_root, source_roots,
                        source_identity, runtime_identity, winner_receipt,
                        device="cuda:0", allow_cpu_for_tests=False):
    """Return the prediction payload and bound context for conditional power.

    Checkpoint paths are relative to ``output/attempts/<attempt_id>``. Existing
    scoring output is never overwritten or automatically retried. The caller
    must reconcile an interrupted scoring worker with the shared budget ledger.
    """
    if type(allow_cpu_for_tests) is not bool:
        raise TypeError("The fixture opt-in must be an explicit boolean")
    selected = _runtime(runtime_identity, device, allow_cpu_for_tests)
    fixture = selected.type == "cpu" or allow_cpu_for_tests
    actual_sources = {str(path.relative_to(ROOT)): _file_sha(path)
                      for path in sorted((ROOT / "scripts").glob("*.py"))}
    if source_identity != actual_sources:
        raise ValueError("Scoring source bytes differ from the fitted source identity")
    if isinstance(source_roots, (str, Path)):
        source_roots = json.loads(Path(source_roots).read_text(encoding="utf-8"))
    output = Path(output).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)) or not output.is_dir():
        raise ValueError("Use an existing nonsymlink campaign output directory")
    _winners(winner_receipt, output)
    results = list(power_results)
    domain = [(row.get("job", {}).get("model"), row.get("job", {}).get("seed")) for row in results]
    if (len(results) != 15 or set(domain) != {(name, seed) for name in power.MODELS for seed in power.SEEDS}
            or len({row.get("attempt_id") for row in results}) != 15):
        raise ValueError("Freeze all fifteen distinct model/seed fits before scoring labels")
    fit_panel, fit_receipt = load_pilot_panel(data_root, "power_fit")
    train = prepare_b(fit_panel, ORIGIN, END).refit_train
    loaded = []
    for result in sorted(results, key=lambda row: (row["job"]["model"], row["job"]["seed"])):
        loaded.append(_verify_result(result, output=output, source_identity=source_identity,
            runtime_identity=runtime_identity, winner_receipt=winner_receipt, train=train,
            data_receipt=fit_receipt, source_roots=source_roots, fixture=fixture))
    context = {"model_recipes": dict(power.RECIPES),
        "data_sha256": _runner_sha(fit_receipt["source_sha256"]),
        "recipe_freeze_sha256": _runner_sha(winner_receipt),
        "execution_policy_sha256": _runner_sha(runtime_identity),
        "model_source_sha256": _runner_sha(source_identity)}
    scoring = output / "power_scoring"
    scoring.mkdir(exist_ok=False)
    _fsync_directory(output)
    freeze = {"version": VERSION, "status": "all_fifteen_checkpoints_frozen_before_scoring_labels",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "context": context,
        "records": [record for _, record in loaded], "power_scoring_labels_opened": False,
        "synthetic_only": fixture, "model_fits": 0}
    _write_new(scoring / "frozen_pilots.json", freeze)
    # This is the first invocation that may decode any registered pilot label.
    panel, score_receipt = load_pilot_panel(data_root, "power_score")
    prepared = prepare_b(panel, ORIGIN, END)
    if (prepared.refit_train.signature != train.signature
            or score_receipt["source_sha256"] != fit_receipt["source_sha256"]):
        raise ValueError("Scoring admission changed frozen fit data or its train-only scaler")
    (scoring / "replays").mkdir()
    _fsync_directory(scoring)
    predictions, records, common = {name: {} for name in power.MODELS}, [], None
    for model, record in loaded:
        name, seed = record["model_id"], record["seed"]
        def forecast(_target, window, model=model):
            tensor = torch.as_tensor(train.scaler.transform(window[None, :, :]), dtype=torch.float32)
            values = predict_pilot(model, tensor, batch_size=1, device=str(selected),
                                   allow_cpu_for_tests=fixture, deterministic=True)
            return float(train.scaler.inverse_targets(values.numpy())[0])
        system = FrozenSystem(name, "window", predict=forecast,
            fingerprint=lambda model=model: {"state": state_sha256(model), "scaler": train.scaler.record()})
        replay = DatedReplay(panel.calendar, panel.features, panel.targets, quality=panel.quality)
        journal = _ReplayJournal(scoring / "replays" / record["attempt_id"])
        try:
            run_frozen_block(prepared.outer_plan, replay,
                             (system, FrozenSystem("Persistence", "naive", lag=1)), journal)
        finally:
            journal.close()
            model.cpu()
        vectors = _read_vectors(journal.directory / "events.jsonl", name)
        if common is None:
            common = vectors[:4]
        elif common != vectors[:4]:
            raise ValueError("Pilot models differ in calendar, eligibility, truth or deterministic baseline")
        predictions[name][str(seed)] = vectors[4]
        record.update(dates_sha256=power.canonical_sha256(vectors[0]),
            eligible_mask_sha256=power.canonical_sha256(vectors[1]),
            predictions_sha256=power.canonical_sha256(vectors[4]))
        records.append(record)
    dates, mask, truth, persistence = common
    payload = {"version": VERSION, "dates": dates, "eligible_mask": mask,
        "truth": truth, "persistence": persistence, "predictions": predictions,
        "provenance": {"synthetic_only": fixture, "market_training_executed": not fixture,
            "artifact_hashes_verified": True, "context_sha256": power.canonical_sha256(context),
            "truth_sha256": power.canonical_sha256(truth),
            "persistence_sha256": power.canonical_sha256(persistence),
            "dates_sha256": power.canonical_sha256(dates),
            "eligible_mask_sha256": power.canonical_sha256(mask), "records": records,
            "all_fits_freeze_sha256": _file_sha(scoring / "frozen_pilots.json"),
            "prediction_order": "durable_prediction_before_daily_observation", "prediction_batch_size": 1}}
    power._validate(payload, context, "synthetic_fixture" if fixture else "registered_market_pilot")
    _write_new(scoring / "payload.json", payload)
    _write_new(scoring / "context.json", context)
    return payload, context
