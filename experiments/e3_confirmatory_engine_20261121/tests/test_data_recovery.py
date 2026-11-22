"""Generated observations and mocked HTTP; no market acquisition or fitting."""
import base64
import datetime as dt
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import io
import zipfile
from unittest.mock import patch

import numpy as np

from engine import data


def prepared(root):
    notice = root/"license.txt"; notice.write_text("SOFTWARE FIXTURE LICENSE ONLY")
    actor = root/"declaration.json"; actor.write_text('{"actor":"SOFTWARE FIXTURE, NO REAL AUTHOR CLAIM"}')
    common = {"actor_name": "SOFTWARE FIXTURE", "actor_contact": "fixture@example.invalid",
              "evidence_path": str(actor), "evidence_sha256": data.sha256_bytes(actor.read_bytes())}
    rights = dict(common, status="approved", allow_derived_publication=True, allow_raw_redistribution=True,
                  license="FIXTURE", scope="software fixture only", source_url="https://example.invalid")
    exposure = dict(common, status="audited", final_reserve_claim=False,
        form={"reserve_mode": "development_only", "observation_cutoff_utc": "2020-01-01T00:00:00+00:00",
              "source_license_evidence_path": str(notice), "source_license_evidence_sha256": data.sha256_bytes(notice.read_bytes())})
    dates = [(dt.date(2010, 1, 1)+dt.timedelta(days=i)).isoformat() for i in range(140)]
    close = 100*np.exp(.001*np.arange(140)+.01*np.sin(np.arange(140)/3))
    columns = [{"name": k} for k in ("ticker", "date", "open", "high", "low", "close", "volume", "split_ratio")]
    payload = json.dumps({"datatable": {"columns": columns,
        "data": [["IBM", date, float(c*.999), float(c*1.01), float(c*.99), float(c), 1000+i, 1]
                 for i, (date, c) in enumerate(zip(dates, close))]}, "meta": {"next_cursor_id": None}}).encode()
    config = {"provider": "nasdaq_wiki", "instrument": "IBM", "development_start": dates[0], "development_end": dates[-1]}
    with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "ORIGINAL_PRIVATE_FIXTURE_KEY"}), \
         patch.object(data, "_network_get", return_value=payload):
        data.collect_development(config, root/"raw", rights_receipt=rights, exposure_receipt=exposure,
                                 enable_market_acquisition=True)
    split = {"raw_start": dates[0], "raw_end": dates[-1], "calendar": "24/7", "embargo_bars": 2,
             "raw_bounds": {"fit": [0, 34], "selection": [35, 69], "planning": [70, 104], "calibration": [105, 139]}}
    admission = data.prepare_development(root/"raw/raw_manifest.json", split, root/"panel", rights=rights, exposure=exposure)
    bundle = root/"saved-metadata.json"
    bundle.write_bytes((root/"panel/recovery_metadata.json").read_bytes())
    return payload, admission, bundle


class RecoveryTests(unittest.TestCase):
    def test_explicit_activation_precedes_bundle_or_network_access(self):
        with patch.object(data, "_read_recovery_bundle", side_effect=AssertionError("bundle accessed")), \
             patch.object(data, "_network_get", side_effect=AssertionError("network accessed")), \
             self.assertRaisesRegex(data.AdmissionError, "activation"):
            data.reacquire_development("unread", "unwritten", "unwritten", "unwritten")

    def test_metadata_only_replay_preserves_original_sources_partitions_and_actor_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); payload, admission, bundle = prepared(root)
            original_admission = (root/"panel/admission.json").read_bytes()
            original_manifest = (root/"raw/raw_manifest.json").read_bytes()
            original_partitions = {part: (root/"panel"/name).read_bytes() for part, name in admission["partition_paths"].items()}
            # The pinned NumPy writer fixes ZIP member timestamps. Exact bytes
            # therefore do not depend on two writes accidentally sharing a
            # wall-clock second (the common source of false replay tests).
            for archive_raw in original_partitions.values():
                with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
                    self.assertEqual({item.date_time for item in archive.infolist()}, {(1980, 1, 1, 0, 0, 0)})
            value = json.loads(bundle.read_text())
            self.assertFalse(value["contains_raw_arrays"])
            self.assertFalse(value["contains_credentials"])
            self.assertEqual(value["original_raw_dir"], str(root/"raw"))
            self.assertNotIn("ORIGINAL_PRIVATE_FIXTURE_KEY", bundle.read_text())
            self.assertNotIn(payload, base64.b64decode(value["raw_manifest"]["content_base64"]))
            shutil.rmtree(root/"raw"); shutil.rmtree(root/"panel")
            (root/"license.txt").unlink(); (root/"declaration.json").unlink()
            with patch.object(data, "_network_get", side_effect=AssertionError("metadata restore queried network")):
                result = data.restore_development_metadata(bundle, root, expected_bundle_sha256=data.sha256_bytes(bundle.read_bytes()))
            self.assertEqual(result["status"], "DEVELOPMENT_AUTHORIZATION_METADATA_RESTORED")
            self.assertTrue((root/"license.txt").is_file())
            self.assertFalse((root/"panel").exists())
            with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "REACQUIRED_PRIVATE_FIXTURE_KEY"}), \
                 patch.object(data, "utc_now", return_value="2030-03-04T05:06:07+00:00"), \
                 patch.object(data, "_network_get", return_value=payload) as network:
                result = data.reacquire_development(bundle, root/"panel", root/"raw", root,
                    expected_bundle_sha256=data.sha256_bytes(bundle.read_bytes()), enable_market_acquisition=True)
            self.assertEqual(result["status"], "DEVELOPMENT_RESTORED_ORIGINAL_IDENTITY")
            self.assertIn("REACQUIRED_PRIVATE_FIXTURE_KEY", network.call_args.args[0])
            self.assertEqual((root/"raw/raw_manifest.json").read_bytes(), original_manifest)
            self.assertEqual((root/"panel/admission.json").read_bytes(), original_admission)
            for part, expected in original_partitions.items():
                self.assertEqual((root/"panel"/admission["partition_paths"][part]).read_bytes(), expected)
                data.load_partition(root/"panel", part)
            self.assertEqual((root/"panel/recovery_metadata.json").read_bytes(), bundle.read_bytes())
            events = [json.loads(line) for line in (root/"raw/reacquisition_events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["finished_utc"], "2030-03-04T05:06:07+00:00")
            self.assertNotEqual(events[-1]["original_acquired_utc"], events[-1]["finished_utc"])
            self.assertFalse(events[-1]["new_acquisition_manifest_created"])
            self.assertFalse(events[-1]["new_owner_declaration"])
            self.assertNotIn("PRIVATE_FIXTURE_KEY", json.dumps(events))
            with patch.object(data, "_network_get", side_effect=AssertionError("complete replay repeated network")):
                reused = data.reacquire_development(bundle, root/"panel", root/"raw", root, enable_market_acquisition=True)
            self.assertEqual(reused["status"], "DEVELOPMENT_ALREADY_RESTORED")

    def test_changed_upstream_bytes_are_not_accepted_under_old_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, bundle = prepared(root)
            with patch.dict("os.environ", {"NASDAQ_DATA_LINK_API_KEY": "PRIVATE_FIXTURE_KEY"}), \
                 patch.object(data, "_network_get", return_value=b"UPSTREAM CHANGED"), \
                 self.assertRaisesRegex(data.AdmissionError, "raw bytes differ"):
                data.reacquire_development(bundle, root/"new-panel", root/"new-raw", root, enable_market_acquisition=True)
            self.assertFalse((root/"new-panel/admission.json").exists())
            event = json.loads((root/"new-raw/reacquisition_events.jsonl").read_text().splitlines()[-1])
            self.assertEqual(event["status"], "failed")

    def test_frozen_bundle_hash_and_workspace_boundaries_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, _, bundle = prepared(root)
            with patch.object(data, "_network_get", side_effect=AssertionError("network accessed")):
                with self.assertRaisesRegex(data.AdmissionError, "frozen campaign"):
                    data.reacquire_development(bundle, root/"new", root/"raw2", root,
                        expected_bundle_sha256="0"*64, enable_market_acquisition=True)
                with self.assertRaisesRegex(data.AdmissionError, "inside the supplied workspace"):
                    data.reacquire_development(bundle, root.parent/"forbidden", root/"raw2", root,
                        enable_market_acquisition=True)
            value = json.loads(bundle.read_text()); value.pop("metadata_sha256")
            value["evidence"][0]["path"] = str(root.parent/(root.name+"-missing-external-proof"))
            value["metadata_sha256"] = data.identity(value)
            forged = root/"modified.json"; forged.write_text(json.dumps(value))
            with self.assertRaisesRegex(data.AdmissionError, "inside the supplied workspace"):
                data.restore_development_metadata(forged, root)


if __name__ == "__main__":
    unittest.main()
