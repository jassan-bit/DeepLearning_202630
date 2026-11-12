"""Artificial CSVs and tiny CPU modules exercise freeze-before-label ordering."""
from contextlib import contextmanager
from copy import deepcopy
import errno
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_published_pilot_scoring as scoring
from e2_amended_metrics import select_local_hpo
from e2_amended_pilot_data import load_pilot_panel
from e2_temporal_training import prepare_b
from test_e2_development_data import registered_cache


class TinyPublishedFixture(nn.Module):
    def __init__(self, name, seed):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(float(seed % 13) / 100))
        self.architecture_metadata = {"training_recipe_id": scoring.power.RECIPES[name]}

    def forward(self, x):
        return torch.nn.functional.softplus(x.mean(dim=(1, 2)) * self.weight)


def builder(name, trial, *, seed, source_roots):
    return TinyPublishedFixture(name, seed)


@contextmanager
def fixture():
    with registered_cache("2024-12-30") as (data_root, _), tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        panel, receipt = load_pilot_panel(data_root, "power_fit")
        train = prepare_b(panel, scoring.ORIGIN, scoring.END).refit_train
        sources = {str(path.relative_to(scoring.ROOT)): scoring._file_sha(path)
                   for path in sorted((scoring.ROOT / "scripts").glob("*.py"))}
        runtime = {"device": "cpu", "dtype": "float32", "amp": False,
                   "tf32": False, "deterministic": True}
        winners = {}
        for name in scoring.power.MODELS:
            trials = scoring.registered_trials(name)
            rows = [{"trial_id": trial["trial_id"], "status": "completed",
                     "seed_rmse": {"271201": 1. + i, "271202": 1. + i}}
                    for i, trial in enumerate(trials)]
            selection = select_local_hpo(rows, anchor="2024-10-01",
                                        registered_trial_ids=[row["trial_id"] for row in trials])
            winners[name] = {"trial": trials[0], "fixed_epochs": 1, "selection": selection}
        winner_receipt = {"winners": winners, "power_pilot_labels_used": False}
        (output / "october_winners.json").write_text(json.dumps(winner_receipt))
        results = []
        for name in scoring.power.MODELS:
            for seed in scoring.power.SEEDS:
                job = {"stage": "power", "model": name, "seed": seed,
                    "origin": scoring.ORIGIN, "end": scoring.END,
                    "trial": winners[name]["trial"], "fixed_epochs": 1,
                    "winner_receipt_sha256": scoring._runner_sha(winner_receipt)}
                attempt = scoring._runner_sha(job)
                attempt_root = output / "attempts" / attempt
                artifacts = attempt_root / "artifacts" / attempt
                (artifacts / "checkpoints").mkdir(parents=True)
                identity = {"job": job, "sources": sources, "runtime": runtime,
                            "data": receipt, "prepared_signature": train.signature,
                            "scaler": train.scaler.record()}
                bound = {"attempt_id": attempt, "identity": identity,
                         "identity_sha256": scoring.power.canonical_sha256(identity)}
                (artifacts / "identity.json").write_text(json.dumps(bound))
                model = builder(name, job["trial"], seed=seed, source_roots={})
                config = {key: job["trial"].get(key, 0) for key in
                          ("learning_rate", "weight_decay", "batch_size", "warmup_fraction")}
                config["seed"] = seed
                checkpoint = {"version": "e2-published-pilot-training-v1",
                    "state_dict": model.state_dict(), "model_state_sha256": scoring.state_sha256(model),
                    "fixed_epochs": 1, "epoch": 1, "best_epoch": 1, "optimizer_steps": 2,
                    "target_scale": train.scaler.target_scale, "config": config}
                checkpoint_path = artifacts / "checkpoints" / "best.pt"
                torch.save(checkpoint, checkpoint_path)
                hashes = {str(path.relative_to(attempt_root)): scoring._file_sha(path)
                          for path in sorted(artifacts.rglob("*")) if path.is_file()}
                results.append({"job": job, "status": "completed", "attempt_id": attempt,
                    "optimizer_steps": 2, "synthetic": True, "test_only_cpu_execution": True,
                    "prepared_signature": train.signature, "recipe_id": scoring.power.RECIPES[name],
                    "data_sha256": scoring._runner_sha(receipt["source_sha256"]),
                    "model_source_sha256": scoring._runner_sha(sources),
                    "execution_policy_sha256": scoring._runner_sha(runtime),
                    "identity_sha256": scoring._runner_sha(identity), "artifacts_sha256": hashes,
                    "selected_checkpoint": {"path": str(checkpoint_path.relative_to(attempt_root)),
                                            "sha256": scoring._file_sha(checkpoint_path)}})
        options = {"output": output, "data_root": data_root, "source_roots": {},
                   "source_identity": sources, "runtime_identity": runtime,
                   "winner_receipt": winner_receipt, "device": "cpu", "allow_cpu_for_tests": True}
        with patch.object(scoring, "build_train_model", side_effect=builder):
            yield results, options


class PublishedPilotScoringTests(unittest.TestCase):
    def test_all_fifteen_closed_before_labels_and_dated_predictions_bind_to_artifacts(self):
        with fixture() as (results, options):
            calls = []
            def admitted(root, stage):
                calls.append(stage)
                if stage == "power_score":
                    freeze_path = options["output"] / "power_scoring" / "frozen_pilots.json"
                    frozen = json.loads(freeze_path.read_text())
                    self.assertEqual(len(frozen["records"]), 15)
                    self.assertFalse(frozen["power_scoring_labels_opened"])
                    self.assertEqual(built.call_count, 15)
                return load_pilot_panel(root, stage)
            with patch.object(scoring, "load_pilot_panel", side_effect=admitted), \
                 patch.object(scoring, "build_train_model", side_effect=builder) as built, \
                 patch.object(scoring, "predict_pilot", wraps=scoring.predict_pilot) as predict:
                payload, context = scoring.score_frozen_pilots(results, **options)
            self.assertEqual(calls, ["power_fit", "power_score"])
            self.assertEqual(predict.call_count, 15 * 53)
            self.assertTrue(all(call.kwargs["batch_size"] == 1 for call in predict.call_args_list))
            self.assertEqual(payload["dates"][0], "2024-11-08")
            self.assertEqual(payload["dates"][-1], "2024-12-30")
            self.assertEqual(len(payload["provenance"]["records"]), 15)
            scoring.power._validate(payload, context, "synthetic_fixture")
            with self.assertRaisesRegex(ValueError, "synthetic/unverified"):
                scoring.power._validate(payload, context, "registered_market_pilot")
            for path in (options["output"] / "power_scoring" / "replays").glob("*/events.jsonl"):
                dates, mask, truth, baseline, _ = scoring._read_vectors(path, json.loads(path.read_text().splitlines()[0])["systems"][0])
                self.assertEqual((dates, mask, truth, baseline),
                                 (payload["dates"], payload["eligible_mask"], payload["truth"], payload["persistence"]))
            with self.assertRaises(FileExistsError):
                scoring.score_frozen_pilots(results, **options)

    def test_missing_duplicate_changed_source_or_last_checkpoint_never_opens_score_labels(self):
        with fixture() as (results, options):
            variants = [results[:-1], results[:-1] + [results[0]]]
            changed = deepcopy(results); changed[-1]["model_source_sha256"] = "a" * 64
            variants.append(changed)
            tampered = deepcopy(results); tampered[-1]["selected_checkpoint"]["sha256"] = "f" * 64
            variants.append(tampered)
            for rows in variants:
                with self.subTest(length=len(rows)), \
                     patch.object(scoring, "load_pilot_panel", wraps=load_pilot_panel) as admission:
                    with self.assertRaises(ValueError):
                        scoring.score_frozen_pilots(rows, **options)
                    self.assertFalse(any(call.args[1] == "power_score" for call in admission.call_args_list))
            self.assertFalse((options["output"] / "power_scoring").exists())

    def test_hash_matching_checkpoint_with_wrong_fixed_epoch_is_refused_before_labels(self):
        with fixture() as (results, options):
            row = results[-1]
            relative = row["selected_checkpoint"]["path"]
            path = options["output"] / "attempts" / row["attempt_id"] / relative
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["fixed_epochs"] = 2
            torch.save(checkpoint, path)
            digest = scoring._file_sha(path)
            row["selected_checkpoint"]["sha256"] = row["artifacts_sha256"][relative] = digest
            with patch.object(scoring, "load_pilot_panel", wraps=load_pilot_panel) as admission:
                with self.assertRaisesRegex(ValueError, "fixed-epoch"):
                    scoring.score_frozen_pilots(results, **options)
                self.assertEqual([call.args[1] for call in admission.call_args_list], ["power_fit"])

    def test_fixture_cpu_requires_opt_in_and_source_snapshot_must_be_current(self):
        with fixture() as (results, options):
            for override in ({"allow_cpu_for_tests": False}, {"source_identity": {}}):
                with self.subTest(override=override), patch.object(scoring, "load_pilot_panel") as admission:
                    with self.assertRaises(ValueError):
                        scoring.score_frozen_pilots(results, **{**options, **override})
                    admission.assert_not_called()

    def test_directory_fsync_only_tolerates_unsupported_filesystem(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(scoring.os, "fsync", side_effect=OSError(errno.EINVAL, "FUSE")):
                scoring._fsync_directory(directory)
            with patch.object(scoring.os, "fsync", side_effect=OSError(errno.EIO, "disk")):
                with self.assertRaises(OSError):
                    scoring._fsync_directory(directory)


if __name__ == "__main__":
    unittest.main()
