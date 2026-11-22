"""Offline root integration harness; fake pins/network/journal, zero market data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fresh_entry_review", HERE / "fresh_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
RUNTIME = Path("/workspace/e3_hypothesis_adequacy_20261122/runtime")


class EntryTests(unittest.TestCase):
    def test_metadata_paths_export_recovery_not_target_admission_or_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            private, kit, repair = work / "private", work / "kit", work / "cm"
            panel = private / "panels/F00/development-v1"
            panel.mkdir(parents=True)
            (panel / "recovery_metadata.json").write_bytes(b'{"synthetic":"bundle"}')
            (panel / "admission.json").write_bytes(b'{"synthetic":"admission"}')
            (panel / "train.npz").write_bytes(b"not-real-data")
            kit.mkdir(); repair.mkdir()
            (kit / "driver.py").write_text("# synthetic code\n")
            paths = entry.metadata_paths(dict(WORK=work, PRIVATE=private, repair_dir=repair), kit)
            self.assertIn(private / "recovery-exports/F00.json", paths)
            self.assertNotIn(panel / "admission.json", paths)
            self.assertNotIn(panel / "train.npz", paths)
            self.assertEqual((private / "recovery-exports/F00.json").read_bytes(),
                             (panel / "recovery_metadata.json").read_bytes())

    def test_entry_calls_one_private_credential_read_and_backup_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); content = base / "content"; content.mkdir()
            events = []
            original_credentials = lambda: (events.append("credentials") or
                                           {"token":"synthetic-only-token", "backup_secret":"synthetic-secret"})
            ns = dict(FRESH_REBUILD_COMMIT="a"*40, FRESH_REBUILD_MANIFEST_SHA256="b"*64)
            def mapped_path(value):
                value = str(value)
                if value == "/content" or value.startswith("/content/"):
                    return content / value.removeprefix("/content").lstrip("/")
                return Path(value)
            def fake_fetch(commit, prefix, relative, expected, root):
                target = Path(root) / relative; target.parent.mkdir(parents=True, exist_ok=True)
                if relative == "RUNTIME_MANIFEST.json":
                    value = {"version":"fresh-cpu-development-rebuild-1", "files":{
                        "ops/fresh_worker.py":"0"*64,"assets/pinned_T2_notebook.json":"1"*64}}
                    target.write_text(json.dumps(value))
                elif relative == "ops/REPAIR_MANIFEST.json":
                    target.write_text('{"files":{}}')
                else:
                    target.write_text('{}')
                return target
            class Journal:
                def __init__(self, storage, **kwargs):
                    events.append("journal_init")
                    self.case = Path(kwargs["journal_dir"])
                    self.store = types.SimpleNamespace(status=lambda:{"exists":False}, _request=lambda *a,**k:None)
                def checkpoint(self, paths, *, event):
                    self_outer.assertTrue(paths)
                    events.append("backup_"+event)
                    return {"remote_verified":True,"commit":"c"*40}
                def close(self):
                    events.append("close")
            self_outer = self
            def fake_rebuild(current, notebook_bytes, checkpoint, *, before_metadata_callback):
                work = content / "e3-r3-crypto-engine-v2"
                current.update(WORK=work, PRIVATE=work/"private", SOURCE=RUNTIME,
                    PLANNING_BRANCH="e3-r3-crypto-engine-v2-planning-fixture",
                    PLANNING_PREFIX="runs/e3_r3_crypto_engine_v2/planning-fixture",
                    PLANNING_CASE=work/"cases/fixture", github_credentials=original_credentials)
                before_metadata_callback(current)
                inputs = current["PRIVATE"] / "inputs"; inputs.mkdir(parents=True, exist_ok=True)
                (inputs/"owner.json").write_text('{"synthetic":true}')
                checkpoint(current, "before_acquisition")
                events.append("acquisition_stub")
                return {"status":"synthetic_complete"}
            def fake_load(path, name):
                return (types.SimpleNamespace(fresh_rebuild=fake_rebuild) if name.endswith("driver") else
                        types.SimpleNamespace(MetadataJournal=Journal))
            with patch.object(entry, "Path", side_effect=mapped_path), patch.object(entry, "fetch", side_effect=fake_fetch), patch.object(entry, "load", side_effect=fake_load):
                result = entry.run_fresh_cpu(ns)
            self.assertEqual(result["status"], "synthetic_complete")
            self.assertEqual(events.count("credentials"), 1)
            self.assertLess(events.index("backup_before_acquisition"), events.index("acquisition_stub"))
            self.assertEqual(events[-1], "close")
            self.assertIs(ns["github_credentials"], original_credentials)


if __name__ == "__main__":
    unittest.main(verbosity=2)
