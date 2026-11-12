"""Storage integrity tests only: no market data, torch import, or model fits."""
import errno
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_amended_pilot_artifacts import (  # noqa: E402
    ArtifactIdentityMismatch, ArtifactReconciliationRequired, ArtifactStateError,
    PilotArtifactStore, _fsync_directory,
)


IDENTITY = {"registration_sha256": "a" * 64, "model": "fixture", "seed": 271701}
METADATA = {"config": {"epochs": 2}, "source": {"sha256": "b" * 64},
            "data": {"sha256": "c" * 64, "scope": "synthetic_fixture"},
            "seed": 271701, "resources": {"device": "fixture_only"}}


class PilotArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.directory.cleanup()

    def store(self, identity=None):
        store = PilotArtifactStore(self.root, "attempt-1", identity or IDENTITY)
        self.stores.append(store)
        return store

    def started(self):
        store = self.store()
        store.prepare(METADATA)
        store.checkpoint(b"initial-state", label="initial")
        store.begin_training({"resume_exact_supported": False})
        return store

    def completed(self):
        store = self.started()
        store.event("epoch_completed", {"epoch": 1, "optimizer_steps": 2,
                                         "known_cost_seconds": 0.5,
                                         "train_loss": 1.0, "peak_memory_bytes": 1024})
        store.checkpoint(b"optimizer-and-rng", label="latest")
        store.checkpoint(b"best-state", label="best")
        store.write_json("predictions", {"fixture": [1, 2]})
        store.complete({"final_score": 0.1})
        return store

    def test_complete_roundtrip_and_hashed_artifacts(self):
        store = self.completed()
        manifest = store.manifest
        self.assertEqual(manifest["optimizer_steps_known_lower_bound"], 2)
        self.assertEqual(manifest["known_cost_seconds"], 0.5)
        self.assertEqual(manifest["state"], "completed")
        for receipt in manifest["checkpoint_history"]:
            raw = (store.path / receipt["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["sha256"])
        store.close()
        recovered = self.store()
        self.assertEqual(recovered.require_reconciled(), manifest)
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.prepare(METADATA)
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.begin_training({})
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.event("new_epoch", {})

    def test_prepare_has_required_pre_optimizer_metadata_and_is_idempotent(self):
        store = self.store()
        with self.assertRaises(ValueError):
            store.prepare({"seed": 1})
        store.prepare(METADATA)
        count = store.manifest["event_count"]
        store.prepare(METADATA)
        self.assertEqual(store.manifest["event_count"], count)
        with self.assertRaises(ArtifactIdentityMismatch):
            store.prepare({**METADATA, "seed": 2})
        with self.assertRaises(ValueError):
            store.event("bad_metadata", {"score": float("nan")})
        with self.assertRaises(TypeError):
            store.event("bad_metadata", {1: "non-string-key"})

    def test_dirty_prepared_or_running_attempt_cannot_be_retried(self):
        store = self.started()
        store.event("optimizer_step_started", {"epoch": 1})
        store.close()  # Simulates lost runtime; releasing a lock does not seal the attempt.
        recovered = self.store()
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.require_reconciled()
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.prepare(METADATA)
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.begin_training({})
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.checkpoint(b"unsafe retry", label="latest")
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.finish("failed", {})

    def test_identity_mismatch_and_concurrent_writer_rejected(self):
        store = self.started()
        with self.assertRaises(ArtifactStateError):
            self.store()
        store.close()
        with self.assertRaises(ArtifactIdentityMismatch):
            self.store({**IDENTITY, "seed": 271702})

    def test_clean_pause_requires_checkpoint_and_epoch_boundary(self):
        store = self.started()
        with self.assertRaises(ArtifactStateError):
            store.pause_clean({"epoch_boundary": True})
        latest = store.checkpoint(b"complete-optimizer-rng-state", label="latest")
        metadata = {"epoch_boundary": True, "checkpoint_sha256": latest["sha256"]}
        store.event("optimizer_step_started", {})
        with self.assertRaises(ArtifactStateError):
            store.pause_clean(metadata)
        store.event("optimizer_step_completed", {"optimizer_steps": 1})
        store.pause_clean(metadata)
        store.close()
        recovered = self.store()
        self.assertEqual(recovered.require_reconciled()["state"], "paused_clean")
        with self.assertRaises(ArtifactStateError):
            recovered.resume_clean(metadata)
        recovered.resume_clean({**metadata, "resume_state_restored": True})
        self.assertEqual(recovered.status, "training")

    def test_interrupted_and_failed_are_preserved_but_require_reconciliation(self):
        store = self.started()
        store.finish("interrupted", {"reason": "fixture interruption", "outcome": "unknown"})
        self.assertTrue(store.manifest["terminal_metadata"]["requires_reconciliation"])
        store.close()
        recovered = self.store()
        self.assertEqual(recovered.status, "interrupted")
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.require_reconciled()

    def test_truncated_journal_is_not_silently_repaired(self):
        store = self.completed()
        journal = store.path / "events.jsonl"
        store.close()
        original = journal.read_bytes()
        journal.write_bytes(original[:-3])
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()
        self.assertEqual(journal.read_bytes(), original[:-3])

    def test_changed_journal_or_manifest_is_rejected(self):
        store = self.completed()
        journal = store.path / "events.jsonl"
        store.close()
        original = journal.read_bytes()
        journal.write_bytes(original.replace(b'"train_loss":1.0', b'"train_loss":9.0'))
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()
        journal.write_bytes(original)
        manifest = store.path / "manifest.json"
        value = json.loads(manifest.read_bytes())
        value["known_cost_seconds"] = 0
        manifest.write_text(json.dumps(value))
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()

    def test_all_historical_checkpoints_are_verified(self):
        store = self.started()
        earlier = store.checkpoint(b"earlier-latest", label="latest")
        store.checkpoint(b"new-latest", label="latest")
        store.checkpoint(b"best", label="best")
        store.complete({})
        store.close()
        (store.path / earlier["path"]).write_bytes(b"changed")
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()

    def test_disk_failure_never_produces_clean_completion(self):
        store = self.started()
        with patch("e2_amended_pilot_artifacts.os.fsync", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaises(OSError):
                store.checkpoint(b"partial-write", label="latest")
        with self.assertRaises(ArtifactReconciliationRequired):
            store.complete({})
        store.close()
        recovered = self.store()
        with self.assertRaises(ArtifactReconciliationRequired):
            recovered.require_reconciled()

    def test_atomic_manifest_failure_is_detected_after_recovery(self):
        store = self.started()
        with patch.object(store, "_write_json", side_effect=OSError(errno.EIO, "publication failed")):
            with self.assertRaises(OSError):
                store.event("epoch_completed", {"epoch": 1, "known_cost_seconds": 0.2})
        store.close()
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()

    def test_monotonic_counters_and_immutable_results(self):
        store = self.started()
        store.event("epoch_completed", {"optimizer_steps": 10, "known_cost_seconds": 1})
        for payload in ({"optimizer_steps": 9}, {"known_cost_seconds": 0.9}, {"optimizer_steps": 10.5}):
            with self.assertRaises(ArtifactStateError):
                store.event("epoch_completed", payload)
        with self.assertRaises(ArtifactStateError):
            store.checkpoint(b"new-initial", label="initial")
        store.write_json("score", {"value": 1})
        with self.assertRaises(ArtifactStateError):
            store.write_json("score", {"value": 2})

    def test_names_symlinks_and_incomplete_identity_are_rejected(self):
        with self.assertRaises(ValueError):
            PilotArtifactStore(self.root, "../escape", IDENTITY)
        with self.assertRaises(ValueError):
            PilotArtifactStore(self.root, "empty", {})
        (self.root / "attempt-1").mkdir()
        (self.root / "attempt-1" / "orphan.bin").write_bytes(b"orphan")
        with self.assertRaises(ArtifactReconciliationRequired):
            self.store()

    def test_file_fsync_errors_propagate_directory_fuse_limit_is_supported(self):
        with patch("e2_amended_pilot_artifacts.os.fsync", side_effect=OSError(errno.EINVAL, "FUSE directory")):
            self.assertFalse(_fsync_directory(self.root))
        with patch("e2_amended_pilot_artifacts.os.fsync", side_effect=OSError(errno.EIO, "disk")):
            with self.assertRaises(OSError):
                _fsync_directory(self.root)


if __name__ == "__main__":
    unittest.main()
