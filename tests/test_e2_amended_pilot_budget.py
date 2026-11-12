"""Shared accounting, exact journal replay and interruptions (no market fits)."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_amended_pilot_budget as budget


def hashed(value):
    result = deepcopy(value)
    result["sha256"] = budget.digest(result)
    return result


def rehash(value):
    result = deepcopy(value)
    result.pop("sha256", None)
    return hashed(result)


def snapshots():
    legacy = hashed({"branch": budget.LEGACY_BRANCH, "head_sha": "1" * 40,
                     "plan_sha256": budget.LEGACY_PLAN, "journal_verified": True,
                     "snapshot_manifest_sha256": "a" * 64,
                     "unresolved_job_ids": [], "unclosed_session_ids": [],
                     "completed_jobs": 1802, "charged_wall_seconds": 156553.03253392776,
                     "campaign_budget_seconds": 172800})
    optimized = hashed({"branch": budget.OPTIMIZED_BRANCH, "head_sha": "2" * 40,
                        "plan_sha256": budget.OPTIMIZED_PLAN, "journal_verified": True,
                        "snapshot_manifest_sha256": "b" * 64,
                        "unresolved_job_ids": [], "unclosed_session_ids": [],
                        "charged_wall_seconds": 66374.95698550006,
                        "campaign_budget_seconds": 73846.96746607224,
                        "legacy_cache": {"commit": legacy["head_sha"],
                                         "plan_sha256": legacy["plan_sha256"],
                                         "completed_jobs": legacy["completed_jobs"],
                                         "spent_seconds": legacy["charged_wall_seconds"],
                                         "limit_seconds": legacy["campaign_budget_seconds"]}})
    heads = {budget.LEGACY_BRANCH: legacy["head_sha"], budget.OPTIMIZED_BRANCH: optimized["head_sha"]}
    return legacy, optimized, heads


def reconciliation():
    legacy, optimized, heads = snapshots()
    return budget.reconcile_snapshots(legacy, optimized, observed_heads=heads)


class CampaignFixture:
    def __init__(self, root, *, closed=True, result=True):
        self.root = Path(root)
        (self.root / "events").mkdir()
        (self.root / "results").mkdir()
        self.plan = {"remote_storage": {"branch": budget.LEGACY_BRANCH}, "fixture": True}
        self.plan_sha = budget.digest(self.plan)
        self.events = []
        self.write("campaign.json", hashed({"version": "e2-search-session-v1", "plan": self.plan,
                                           "plan_sha256": self.plan_sha,
                                           "controls": {"campaign_budget_seconds": 86400}}))
        self.event("session_started", session_id="s1", session_budget_seconds=100,
                   campaign_budget_seconds=86400)
        start = self.event("job_started", job_id="job", session_id="s1",
                           job_payload_sha256="c" * 64, reservation_seconds=30)
        if result:
            outcome = {"status": "completed", "synthetic_fixture": True}
            record = hashed({"version": "e2-search-session-v1", "plan_sha256": self.plan_sha,
                             "job_id": "job", "session_id": "s1",
                             "job_payload_sha256": "c" * 64,
                             "started_event_sha256": start["sha256"],
                             "outcome": outcome, "outcome_sha256": budget.digest(outcome),
                             "charged_wall_seconds": 20})
            self.write("results/" + hashlib.sha256(b"job").hexdigest() + ".json", record)
            self.event("job_finished", job_id="job", session_id="s1", result_sha256=record["sha256"])
        if closed:
            self.event("session_closed", session_id="s1", charged_overhead_seconds=5)
        charge = (20 if result else 0) + (5 if closed else 100 - (20 if result else 0))
        self.write("status.json", {"plan_sha256": self.plan_sha, "status": "fixture_only",
                                   "budget": {"plan_sha256": self.plan_sha,
                                              "completed_jobs": int(result),
                                              "charged_wall_seconds": charge,
                                              "campaign_budget_seconds": 86400,
                                              "remaining_campaign_seconds": 86400 - charge,
                                              "conservative_unclosed_session_charges": {} if closed else {"s1": 100 - (20 if result else 0)}}})
        self.manifest()

    def write(self, name, value):
        (self.root / name).write_bytes(budget.canonical(value) + b"\n")

    def event(self, kind, **fields):
        record = hashed({"version": "e2-search-session-v1", "plan_sha256": self.plan_sha,
                         "sequence": len(self.events), "kind": kind,
                         "previous_sha256": self.events[-1]["sha256"] if self.events else budget.ZERO,
                         "utc": "2026-11-12T00:00:00Z", **fields})
        self.write("events/%09d.json" % len(self.events), record)
        self.events.append(record)
        return record

    def manifest(self):
        from e2_github_campaign import _Inventory, _descriptor
        inventory = _Inventory()
        for path in self.root.rglob("*.json"):
            if path.name == "snapshot.json":
                continue
            relative = path.relative_to(self.root).as_posix()
            inventory.set(relative, _descriptor(relative, path.read_bytes()))
        self.write("snapshot.json", hashed({"repository": budget.REPOSITORY,
                                           "branch": budget.LEGACY_BRANCH, "prefix": budget.LEGACY_BRANCH,
                                           "plan_sha256": self.plan_sha,
                                           "inventory_algorithm": "sha256-path-bucket-merkle-v1",
                                           "record_count": len(inventory.records),
                                           "record_bytes": inventory.bytes,
                                           "inventory_root_sha256": inventory.root}))

    def replay(self):
        return budget.replay_campaign(self.root, branch=budget.LEGACY_BRANCH, head_sha="1" * 40,
                                      expected_plan_sha256=self.plan_sha)


class ReplayTests(unittest.TestCase):
    def test_all_records_and_inventory_are_verified_not_status_only(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CampaignFixture(root)
            report = fixture.replay()
            self.assertEqual(report["charged_wall_seconds"], 25)
            self.assertEqual(report["completed_jobs"], 1)
            self.assertTrue(report["journal_verified"])
            self.assertEqual(report["unclosed_session_ids"], [])
            status = budget._read(Path(root) / "status.json")
            status["budget"]["charged_wall_seconds"] = 1
            fixture.write("status.json", status)
            with self.assertRaisesRegex(budget.BudgetError, "inventory"):
                fixture.replay()
            fixture.manifest()
            with self.assertRaisesRegex(budget.BudgetError, "disagrees"):
                fixture.replay()

    def test_unclosed_allocations_are_conservatively_charged(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CampaignFixture(root, closed=False)
            report = fixture.replay()
            self.assertEqual(report["charged_wall_seconds"], 100)
            self.assertEqual(report["conservative_unclosed_session_charges"], {"s1": 80})
            self.assertEqual(report["unclosed_session_ids"], ["s1"])

    def test_unknown_attempt_is_never_interpreted_as_zero_cost(self):
        with tempfile.TemporaryDirectory() as root:
            report = CampaignFixture(root, closed=False, result=False).replay()
            self.assertEqual(report["charged_wall_seconds"], 100)
            self.assertEqual(report["unresolved_job_ids"], ["job"])

    def test_missing_or_tampered_event_and_result_rejected(self):
        for mutation in ("missing", "tampered", "result"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                fixture = CampaignFixture(root)
                if mutation == "missing":
                    (Path(root) / "events/000000001.json").unlink()
                elif mutation == "tampered":
                    event = fixture.events[1]
                    event["reservation_seconds"] = 1
                    fixture.write("events/000000001.json", event)
                else:
                    path = next((Path(root) / "results").glob("*.json"))
                    row = budget._read(path)
                    row["outcome"]["status"] = "changed"
                    fixture.write(path.relative_to(root).as_posix(), rehash(row))
                with self.assertRaises(budget.BudgetError):
                    fixture.replay()

    def test_result_without_finish_keeps_its_charge(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = CampaignFixture(root, closed=False)
            (Path(root) / "events/000000002.json").unlink()
            fixture.manifest()
            report = fixture.replay()
            self.assertEqual(report["completed_jobs"], 1)
            self.assertEqual(report["charged_wall_seconds"], 100)


class SharedBudgetTests(unittest.TestCase):
    def test_observed_balance_does_not_reset_to_64_hours(self):
        report = reconciliation()
        self.assertAlmostEqual(report["remaining_seconds_before_pilots"], 7472.0104805721785, places=6)
        self.assertFalse(report["budget_reset"])
        self.assertGreater(report["historical_charged_seconds"], 61 * 3600)

    def test_changed_head_cache_or_budget_must_be_reconciled(self):
        for mutation in ("head", "cache", "cap", "unresolved", "unclosed"):
            with self.subTest(mutation=mutation):
                legacy, optimized, heads = snapshots()
                if mutation == "head":
                    heads[budget.OPTIMIZED_BRANCH] = "f" * 40
                elif mutation == "cache":
                    optimized["legacy_cache"]["spent_seconds"] = 0
                elif mutation == "cap":
                    optimized["campaign_budget_seconds"] += 100
                elif mutation == "unresolved":
                    optimized["unresolved_job_ids"] = ["lost"]
                else:
                    optimized["unclosed_session_ids"] = ["lost"]
                optimized = rehash(optimized)
                with self.assertRaises(budget.BudgetError):
                    budget.reconcile_snapshots(legacy, optimized, observed_heads=heads)

    def test_no_implicit_budget_increase(self):
        legacy, optimized, heads = snapshots()
        with self.assertRaises(budget.BudgetError):
            budget.reconcile_snapshots(legacy, optimized, observed_heads=heads, authorized_total_hours=72)

    def test_original_7500_second_reservation_does_not_fit(self):
        with tempfile.TemporaryDirectory() as root, budget.PilotBudgetLedger(root, reconciliation()) as ledger:
            with self.assertRaises(budget.BudgetPause):
                ledger.reserve("new", 7500)
            self.assertEqual(ledger.report()["reserved_seconds"], 0)

    def test_reserve_charge_overhead_and_reopen_keep_all_cost(self):
        with tempfile.TemporaryDirectory() as root:
            report = reconciliation()
            with budget.PilotBudgetLedger(root, report) as ledger:
                ledger.reserve("a", 100)
                ledger.record_progress("a", 30, 7)
                ledger.charge("a", 60, status="completed")
                ledger.charge_overhead(4)
                self.assertAlmostEqual(ledger.remaining_seconds, report["remaining_seconds_before_pilots"] - 64)
            with budget.PilotBudgetLedger(root, report) as ledger:
                self.assertEqual(ledger.report()["pilot_charged_seconds"], 64)
                self.assertEqual(ledger.report()["attempt_cost_lower_bounds"]["a"]["optimizer_steps_lower_bound"], 7)
                with self.assertRaises(budget.BudgetError):
                    ledger.reserve("a", 100)

    def test_dirty_death_blocks_reopen_and_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = budget.PilotBudgetLedger(root, reconciliation())
            ledger.reserve("lost", 100)
            ledger.record_progress("lost", 12, 3)
            ledger.close()
            with self.assertRaises(budget.BudgetReconciliationRequired):
                budget.PilotBudgetLedger(root, reconciliation())
            rows = [json.loads(x) for x in (Path(root) / "pilot_budget.jsonl").read_text().splitlines()]
            self.assertEqual([x["kind"] for x in rows], ["reserved", "progress"])

    def test_lock_prevents_two_controllers(self):
        with tempfile.TemporaryDirectory() as root, budget.PilotBudgetLedger(root, reconciliation()):
            with self.assertRaisesRegex(budget.BudgetError, "active"):
                budget.PilotBudgetLedger(root, reconciliation())

    def test_write_failure_does_not_release_known_reservation(self):
        with tempfile.TemporaryDirectory() as root, budget.PilotBudgetLedger(root, reconciliation()) as ledger:
            ledger.reserve("a", 100)
            with mock.patch("builtins.open", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    ledger.charge("a", 10, status="failed")
            self.assertEqual(ledger.report()["reserved_seconds"], 100)

    def test_overrun_is_charged_before_pause(self):
        with tempfile.TemporaryDirectory() as root, budget.PilotBudgetLedger(root, reconciliation()) as ledger:
            ledger.reserve("a", 100)
            with self.assertRaisesRegex(budget.BudgetPause, "overrun"):
                ledger.charge("a", 110, status="failed")
            self.assertEqual(ledger.report()["pilot_charged_seconds"], 110)
            self.assertEqual(ledger.report()["reserved_seconds"], 0)

    def test_truncated_journal_and_changed_anchor_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            report = reconciliation()
            with budget.PilotBudgetLedger(root, report) as ledger:
                ledger.charge_overhead(1)
            changed = deepcopy(report)
            changed["historical_heads"][budget.LEGACY_BRANCH] = "f" * 40
            changed = rehash(changed)
            with self.assertRaises(budget.BudgetError):
                budget.PilotBudgetLedger(root, changed)
            with open(Path(root) / "pilot_budget.jsonl", "ab") as stream:
                stream.write(b'{"broken":')
            with self.assertRaises((ValueError, budget.BudgetError)):
                budget.PilotBudgetLedger(root, report)


class SnapshotFetchTests(unittest.TestCase):
    def test_automatic_evidence_refuses_changed_heads_and_anchor(self):
        legacy, optimized, heads = snapshots()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(budget, "fetch_public_heads", return_value=heads), \
                    mock.patch.object(budget, "_restore_public_snapshot"), \
                    mock.patch.object(budget, "replay_campaign", side_effect=[legacy, optimized]):
                result = budget.fetch_reconciliation(root)
            self.assertEqual(budget._read(Path(root) / "budget_evidence.json"), result)
            changed = dict(heads)
            changed[budget.OPTIMIZED_BRANCH] = "f" * 40
            with mock.patch.object(budget, "fetch_public_heads", side_effect=[heads, changed]), \
                    mock.patch.object(budget, "_restore_public_snapshot"), \
                    mock.patch.object(budget, "replay_campaign", side_effect=[legacy, optimized]):
                with self.assertRaisesRegex(budget.BudgetError, "changed_during"):
                    budget.fetch_reconciliation(root)
            optimized["charged_wall_seconds"] += 1
            optimized = rehash(optimized)
            with mock.patch.object(budget, "fetch_public_heads", return_value=heads), \
                    mock.patch.object(budget, "_restore_public_snapshot"), \
                    mock.patch.object(budget, "replay_campaign", side_effect=[legacy, optimized]):
                with self.assertRaisesRegex(budget.BudgetReconciliationRequired, "anchor_changed"):
                    budget.fetch_reconciliation(root)

    def test_api_fallback_git_receives_no_token(self):
        _, _, heads = snapshots()
        process = mock.Mock(returncode=0, stdout="\n".join(sha + "\trefs/heads/" + branch
                                                          for branch, sha in heads.items()))
        with mock.patch.object(budget, "_public_get", side_effect=budget.BudgetError("budget_remote_read_unavailable")), \
                mock.patch.dict("os.environ", {"GITHUB_TOKEN": "private-read-credential", "GH_TOKEN": "another"}), \
                mock.patch.object(budget.subprocess, "run", return_value=process) as run:
            self.assertEqual(budget.fetch_public_heads(token="private-read-credential"), heads)
            environment = run.call_args.kwargs["env"]
            self.assertNotIn("GITHUB_TOKEN", environment)
            self.assertNotIn("GH_TOKEN", environment)
            self.assertNotIn("private-read-credential", " ".join(run.call_args.args[0]))

    def test_archives_are_pinned_and_extraction_confined(self):
        with tempfile.TemporaryDirectory() as root:
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr("repo/" + budget.LEGACY_BRANCH + "/campaign.json", "{}")
                archive.writestr("repo/" + budget.LEGACY_BRANCH + "/snapshot.json", "{}")
                archive.writestr("repo/" + budget.LEGACY_BRANCH + "/status.json", "{}")
            with mock.patch.object(budget, "_public_get", return_value=payload.getvalue()) as get:
                budget._restore_public_snapshot(Path(root) / "snapshot", budget.LEGACY_BRANCH, "1" * 40)
                self.assertIn("/zip/" + "1" * 40, get.call_args.args[0])
                self.assertNotIn("token", get.call_args.kwargs)
            self.assertTrue((Path(root) / "snapshot/snapshot.json").exists())

    def test_traversal_and_non_campaign_archives_are_rejected(self):
        for path in ("repo/../../bad.json", "repo/README.md"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as root:
                payload = io.BytesIO()
                with zipfile.ZipFile(payload, "w") as archive:
                    archive.writestr(path, "{}")
                with mock.patch.object(budget, "_public_get", return_value=payload.getvalue()):
                    with self.assertRaises(budget.BudgetError):
                        budget._restore_public_snapshot(Path(root) / "snapshot", budget.LEGACY_BRANCH, "1" * 40)


if __name__ == "__main__":
    unittest.main()
