"""Synthetic serialization/guard checks; never instantiate or fit a model."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_e1_replay_forensics as audit


class ReplayForensicsChecks(unittest.TestCase):
    def curve(self, path, value="0.1", newline="\n"):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator=newline)
            writer.writerow(audit.HEADER)
            writer.writerow([1, "inner", value, "0.1", "0.1", 1, "0.003"])

    def test_canonical_csv_ignores_decimal_spelling_and_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            self.curve(a, "0.1")
            self.curve(b, "1e-1", "\r\n")
            self.assertEqual(audit.csv_value_hash(a), audit.csv_value_hash(b))
            self.assertNotEqual(audit.sha(a), audit.sha(b))

    def test_small_change_is_tolerance_pass_but_never_exact_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            self.curve(a, "0.1")
            self.curve(b, "0.100000001")
            self.assertNotEqual(audit.csv_value_hash(a), audit.csv_value_hash(b))
            self.assertTrue(audit.numerical_comparison(a, b, "history_inner.csv")["within_tolerance"])
            self.curve(b, "0.11")
            self.assertFalse(audit.numerical_comparison(a, b, "history_inner.csv")["within_tolerance"])

    def test_different_curve_length_fails_numerical_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            self.curve(a)
            self.curve(b)
            with b.open("a") as stream:
                stream.write("2,inner,0.1,0.1,0.1,1,0.003\n")
            result = audit.numerical_comparison(a, b, "history_inner.csv")
            self.assertFalse(result["structure_equal"])
            self.assertFalse(result["within_tolerance"])

    def test_tensor_hash_ignores_zip_name_and_storage_but_preserves_shape(self):
        import torch
        checkpoint = {"state_dict": {"synthetic": torch.tensor([0.1, 0.2])}, "family": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "model.pt", Path(directory) / "different.pt"
            torch.save(checkpoint, a)
            torch.save(checkpoint, b)
            self.assertNotEqual(audit.sha(a), audit.sha(b))
            self.assertEqual(audit.tensor_value_hash(a), audit.tensor_value_hash(b))
            checkpoint["state_dict"]["synthetic"] = checkpoint["state_dict"]["synthetic"].reshape(1, 2)
            torch.save(checkpoint, b)
            self.assertNotEqual(audit.tensor_value_hash(a), audit.tensor_value_hash(b))
            self.assertFalse(audit.numerical_comparison(a, b, "model.pt")["within_tolerance"])

    def test_panel_is_fixed_before_new_fits(self):
        self.assertEqual([(row["model"], row["seed"]) for row in audit.panel()],
            [("GRU", 8101), ("LSTM", 8101), ("TCN", 8101), ("TCN", 8110), ("TCN", 8416), ("TCN", 8552)])

    def test_unpublished_or_wrong_commit_cannot_register(self):
        with patch.object(audit, "protocol", return_value={}), patch.object(audit, "git", return_value="a" * 40):
            with self.assertRaisesRegex(ValueError, "already-published"):
                audit.register(Path(tempfile.gettempdir()) / "unused-e1-audit", "b" * 40)

    def test_source_and_other_checkouts_cannot_store_private_payloads(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            audit.checked_private(audit.ROOT / "private")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "another Git"):
                audit.checked_private(root / "private")

    def test_existing_replicate_is_rejected_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            with patch.object(audit, "registered", return_value=(root, {})), patch.dict(audit.os.environ, {"PYTHONHASHSEED": "0"}):
                with self.assertRaisesRegex(ValueError, "fresh directory"):
                    audit.fit(root, "a")


if __name__ == "__main__":
    unittest.main()
