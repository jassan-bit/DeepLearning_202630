"""Check published development-pilot evidence without fitting or loading models.

Only the standard library is imported. Checkpoint tensors remain unpublished;
their byte/state hashes and the report's weights-only verification receipts are
bound to each completed attempt here.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "book_e2/_static/development_pilot_20261109"
SNAPSHOT = ROOT / "experiments/e2_development_pilot_20261109"
PROTOCOL_SHA = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
REGISTRATION_SHA = "cc10ddba82f0c59cfcd8caf70e713481ec0701e71860862ed13ead0df0bf4797"
REGISTRATION_COMMIT = "cf68db00f4d7c3626bdb517bf922265a3b229ea4"
AMENDMENT_SHA = "025fe220405050a5cb8246bf22638e8bebc301093fcc571e878ebbdffa17ebc1"
SOURCE_DATA = {
    "results/eda_development/daily_market_development.csv":
        "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv":
        "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}
SEEDS = set(range(271901, 271906))
HYBRID = "CNN-1D + LSTM + atención temporal"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_evidence():
    result = {}
    for name in ("summary", "checkpoints", "resources_table", "decisions"):
        result[name] = json.loads((DATA / (name + ".json")).read_bytes())
    for name in ("resources_full", "variance_full", "ledgers"):
        result[name] = json.loads(gzip.decompress((DATA / (name + ".json.gz")).read_bytes()))
    return result


def snapshot_file(name):
    canonical = SNAPSHOT / name
    return canonical if canonical.is_file() else ROOT / name


class PilotEvidenceTests(unittest.TestCase):
    def assert_chain(self, events, *, resource=False):
        previous = "0" * 64
        for index, event in enumerate(events):
            field = "sha256" if resource else "event_sha256"
            plain = {key: value for key, value in event.items() if key != field}
            options = {"sort_keys": True, "allow_nan": False}
            if resource:
                options.update(separators=(",", ":"), ensure_ascii=False)
            self.assertEqual(event["sequence"], index)
            self.assertEqual(event["previous_sha256"], previous)
            self.assertEqual(sha(json.dumps(plain, **options).encode()), event[field])
            previous = event[field]
        return previous

    def test_artifacts_registration_data_and_loaded_sources_match_snapshot(self):
        evidence = load_evidence()
        summary = evidence["summary"]
        names = {"resources_full.json.gz", "variance_full.json.gz", "ledgers.json.gz",
                 "decisions.json", "resources_table.json", "checkpoints.json"}
        self.assertEqual(set(summary["artifacts"]), names)
        self.assertEqual({path.name for path in DATA.iterdir()
                          if path.is_file() and path.suffix in (".json", ".gz")},
                         names | {"summary.json"})
        for name, receipt in summary["artifacts"].items():
            with self.subTest(artifact=name):
                raw = (DATA / name).read_bytes()
                self.assertEqual(len(raw), receipt["bytes"])
                self.assertEqual(sha(raw), receipt["sha256"])
                if name.endswith(".gz"):
                    decoded = gzip.decompress(raw)
                    self.assertEqual(len(decoded), receipt["uncompressed_bytes"])
                    self.assertEqual(sha(decoded), receipt["uncompressed_sha256"])
                    json.loads(decoded)

        protocol = ROOT / "book_e2/_static/search_protocol_20261109/protocol.json"
        self.assertEqual(summary["protocol_sha256"], PROTOCOL_SHA)
        self.assertEqual(sha(protocol.read_bytes()), PROTOCOL_SHA)
        self.assertEqual((SNAPSHOT / "source_protocol.json").read_bytes(), protocol.read_bytes())
        registered = snapshot_file("registration.json").read_bytes()
        self.assertEqual(sha(registered), REGISTRATION_SHA)
        registration = json.loads(registered)
        self.assertEqual(registration["source_protocol_sha256"], PROTOCOL_SHA)
        self.assertEqual(registration["source_files"], SOURCE_DATA)
        self.assertEqual(summary["registration_sha256"], REGISTRATION_SHA)
        self.assertEqual(summary["registration_commit"], REGISTRATION_COMMIT)
        self.assertEqual(sha(snapshot_file("analysis_amendment.json").read_bytes()), AMENDMENT_SHA)
        self.assertEqual(summary["analysis_amendment_sha256"], AMENDMENT_SHA)

        resources, variance = evidence["resources_full"], evidence["variance_full"]
        observations = 0

        def check_sources(sources):
            for name, expected in sources.items():
                with self.subTest(source=name):
                    path = (ROOT / name).resolve()
                    self.assertTrue(path.is_relative_to((ROOT / "scripts").resolve()))
                    self.assertEqual(path.suffix, ".py")
                    self.assertEqual(sha(path.read_bytes()), expected)

        def check_data(receipt, admitted_end):
            self.assertEqual(receipt["source_sha256"], SOURCE_DATA)
            self.assertEqual(receipt["source_protocol_sha256"],
                             "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec")
            self.assertEqual(receipt["admitted_end"], admitted_end)
            self.assertTrue(receipt["population_target_reconstruction_matches"])
            self.assertTrue(receipt["source_values_parsed_only_through_admission"])
            self.assertFalse(receipt["target_is_intraday_realized_variance"])
            self.assertEqual(receipt["external_2025_2026_files_opened"], 0)
            self.assertEqual(receipt["raw_archives_opened"], 0)
            for parsing in receipt["parsing"].values():
                self.assertFalse(parsing["numerical_values_after_admission_decoded"])

        self.assertEqual(resources["registration_sha256"], REGISTRATION_SHA)
        self.assertEqual(resources["source_script_sha256"],
                         sha((ROOT / "scripts/run_e2_development_pilot.py").read_bytes()))
        for row in resources["results"]:
            worker = row.get("worker_profile") or {}
            sources = worker.get("source_sha256", worker.get("source_sha256_before", {}))
            check_sources(sources)
            observations += len(sources)
            if worker:
                self.assertEqual(worker["source_sha256_before"], worker["source_sha256_after"])
                self.assertTrue(worker["sources_unchanged_during_fit"])
                self.assertEqual(worker["registration_sha256"], REGISTRATION_SHA)
                self.assertEqual(worker["fit_receipt"]["protocol_file_sha256"], PROTOCOL_SHA)
                check_data(worker["data_receipt"], "2023-09-23")
            ledger = evidence["ledgers"]["resources/worker-%02d" % row["index"]]
            self.assertEqual(ledger, row["durable_events"])
            self.assertEqual(self.assert_chain(ledger["events"], resource=True),
                             ledger["last_event_sha256"])
            self.assertFalse(ledger["incomplete_tail"])
            for event in ledger["events"]:
                check_sources(event.get("source_sha256", {}))
                if "data_receipt" in event:
                    check_data(event["data_receipt"], "2023-09-23")
        self.assertEqual(variance["registration_file_sha256"], REGISTRATION_SHA)
        for model in variance["models"].values():
            for rows in model.values():
                for row in rows:
                    worker = row["worker_profile"]
                    check_sources(worker["source_sha256"])
                    observations += len(worker["source_sha256"])
                    check_data(worker["data_receipt"], "2024-12-30")
                    self.assertEqual(worker["registration_file_sha256"], REGISTRATION_SHA)
                    self.assertEqual(worker["registration_commit"], REGISTRATION_COMMIT)
        self.assertEqual(observations, 750)
        self.assertEqual(observations, summary["source_hash_observations_checked"])

    def test_failures_cost_checkpoints_and_paired_replays_are_preserved(self):
        evidence = load_evidence()
        summary = evidence["summary"]
        resources, variance = evidence["resources_full"], evidence["variance_full"]
        resource_rows = resources["results"]
        self.assertEqual(len(resource_rows), 15)
        self.assertEqual({row["index"] for row in resource_rows}, set(range(15)))
        self.assertEqual(Counter(row["status"] for row in resource_rows),
                         {"success": 14, "timeout": 1})
        censored = [row for row in resource_rows if row["optimizer_steps_censored"]]
        self.assertEqual(len(censored), 1)
        self.assertEqual((censored[0]["model"], censored[0]["case"], censored[0]["error_code"]),
                         ("TimesNet", "largest_timesnet", "external_wall_timeout"))
        self.assertIsNone(censored[0]["optimizer_steps"])
        self.assertIsNone(censored[0]["worker_profile"])
        self.assertEqual(resources["unknown_optimizer_step_workers"], 1)
        self.assertEqual(sum(row["optimizer_steps"] for row in resource_rows
                             if row["optimizer_steps"] is not None), 7074)
        self.assertEqual(summary["resource_known_optimizer_steps"], 7074)

        self.assertEqual(set(variance["models"]), {"TimeMixer", HYBRID, "GRU32"})
        variance_rows = []
        completed = []
        for name, model in variance["models"].items():
            self.assertEqual({row["seed"] for row in model["selection"]}, SEEDS)
            self.assertEqual(len(model["selection"]), 5)
            if name == "TimeMixer":
                self.assertEqual(model["refit"], [])
                self.assertTrue(all(row["status"] == "failed" for row in model["selection"]))
                self.assertTrue(all(row["worker_profile"]["failure"]["stage_receipt"]["error_code"]
                                    == "resource_limit" for row in model["selection"]))
            else:
                self.assertEqual({row["seed"] for row in model["refit"]}, SEEDS)
                self.assertEqual(len(model["refit"]), 5)
                median = math.ceil(statistics.median(row["best_epoch"] for row in model["selection"]))
                self.assertEqual(median, variance["fixed_epoch_counts"][name])
            for phase, rows in model.items():
                for row in rows:
                    variance_rows.append(row)
                    worker = row["worker_profile"]
                    self.assertFalse(row["replacement_attempt"])
                    self.assertFalse(worker["replacement_attempt"])
                    self.assertFalse(worker["weights_reused"])
                    if row["status"] == "completed":
                        completed.append((name, phase, row))
                        self.assertTrue(worker["fresh_optimizer"])
                        self.assertEqual(worker["training_split"]["dates"][-1],
                                         "2024-06-25" if phase == "selection" else "2024-10-31")
                        if phase == "refit":
                            self.assertIsNone(worker["validation_split"])
                            self.assertFalse(worker["epoch_selection_inputs_passed_to_refit"])
                            self.assertEqual(worker["fixed_epoch_count"], variance["fixed_epoch_counts"][name])
        self.assertEqual(len(variance_rows), 25)
        self.assertEqual(Counter(row["status"] for row in variance_rows),
                         {"completed": 20, "failed": 5})
        self.assertEqual(len(resource_rows) + len(variance_rows), 40)
        self.assertEqual(14 + len(completed), 34)
        self.assertEqual(set(variance["blocked_refits"]), {"TimeMixer"})
        self.assertEqual(variance["replacement_attempts"], 0)
        self.assertEqual(summary["reference_pilot"]["blocked_refits_not_attempted"], 5)
        completed_steps = sum(row["optimizer_steps"] for row in variance_rows if row["status"] == "completed")
        failed_steps = sum(row["optimizer_steps"] for row in variance_rows if row["status"] == "failed")
        self.assertEqual((completed_steps, failed_steps), (39290, 3645))
        self.assertEqual(completed_steps + failed_steps, 42935)
        self.assertEqual(summary["reference_pilot"]["known_optimizer_steps"], 42935)
        self.assertEqual(7074 + completed_steps + failed_steps, 50009)
        self.assertEqual(summary["resource_unknown_optimizer_step_workers"], 1)
        self.assertEqual(summary["reference_pilot"]["unknown_optimizer_step_workers"], 0)
        self.assertFalse(summary["reference_paired_3x5_complete"])
        self.assertIsNone(summary["final_R"])
        self.assertIsNone(summary["achieved_final_power"])

        checkpoints = evidence["checkpoints"]
        expected = {(name, phase, row["seed"]): row["worker_profile"]["checkpoint"]
                    for name, phase, row in completed}
        self.assertEqual(len(checkpoints), 20)
        self.assertEqual(len({row["file"] for row in checkpoints}), 20)
        self.assertEqual(summary["checkpoints_safely_reloaded_for_report"], 20)
        for receipt in checkpoints:
            identity = (receipt["model"], receipt["phase"], receipt["seed"])
            self.assertEqual({key: value for key, value in receipt.items()
                              if key not in ("model", "phase", "seed")}, expected.pop(identity))
            self.assertTrue(receipt["safe_load_weights_only"])
            self.assertFalse(receipt["optimizer_state_saved"])
            self.assertGreater(receipt["bytes"], 0)
            for key in ("sha256", "state_sha256"):
                self.assertRegex(receipt[key], r"^[0-9a-f]{64}$")
        self.assertEqual(expected, {})
        self.assertFalse(summary["weights_published"])

        dates = [(datetime(2024, 11, 8, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
                 for i in range(53)]
        common_truth = None
        refits = [(name, row) for name, phase, row in completed if phase == "refit"]
        self.assertEqual(len(refits), 10)
        self.assertEqual(set(evidence["ledgers"]),
                         {"resources/worker-%02d" % i for i in range(15)}
                         | {name + "/seed" + str(row["seed"]) for name, row in refits})
        for name, row in refits:
            worker = row["worker_profile"]
            system = name + "/seed" + str(row["seed"])
            events = evidence["ledgers"][system]
            self.assert_chain(events)
            raw = "".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                          for event in events).encode()
            self.assertEqual(sha(raw), worker["replay_ledger_sha256"])
            self.assertEqual(Counter(event["kind"] for event in events),
                             {"block_plan": 1, "bridge_observation": 7,
                              "predictions": 53, "observation": 53, "block_completed": 1})
            self.assertEqual(events[0]["plan"]["evaluation_dates"], [date[:10] for date in dates])
            self.assertEqual(events[0]["plan"]["eligible"], [True] * 53)
            predictions, observations, pending = {}, {}, {}
            for event in events:
                if event["kind"] == "predictions":
                    self.assertEqual(pending, {})
                    self.assertEqual(set(event["systems"]), {system})
                    self.assertTrue(event["eligible"])
                    self.assertNotIn(event["date"], predictions)
                    value = event["systems"][system]["value"]
                    self.assertTrue(math.isfinite(value))
                    predictions[event["date"]] = value
                    pending[event["date"]] = event["event_sha256"]
                elif event["kind"] == "observation":
                    self.assertEqual(event["prediction_event_sha256"], pending.pop(event["date"]))
                    self.assertEqual(event["previous_sha256"], event["prediction_event_sha256"])
                    self.assertNotIn(event["date"], observations)
                    self.assertTrue(math.isfinite(event["value"]))
                    observations[event["date"]] = event["value"]
            self.assertEqual(pending, {})
            self.assertEqual(list(predictions), dates)
            self.assertEqual(list(observations), dates)
            truths = list(observations.values())
            if common_truth is None:
                common_truth = truths
            self.assertEqual(truths, common_truth)
            self.assertEqual(worker["evaluation"], {"dates": dates,
                "predictions": list(predictions.values()), "truths": truths})
            self.assertTrue(worker["observation_after_durable_prediction"])
            self.assertTrue(worker["frozen_fingerprint_invariant"])
            self.assertEqual(worker["replay"]["scored_days"], 53)
            csv_buffer = io.StringIO(newline="")
            writer = csv.writer(csv_buffer)
            writer.writerow(("date_utc", "prediction_sigma7", "observed_sigma7"))
            writer.writerows(zip(dates, predictions.values(), truths))
            self.assertEqual(sha(csv_buffer.getvalue().encode()), worker["predictions_file"]["sha256"])


if __name__ == "__main__":
    unittest.main()
