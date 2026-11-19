"""Fail-closed contracts for the five-model serving verifier, without network."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("verify_serving_v2.py")
spec = importlib.util.spec_from_file_location("serving_verifier_v2", PATH)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ServingVerifierContracts(unittest.TestCase):
    def rows(self):
        return [{"id":model,"slug":slug} for model,slug in verifier.EXPECTED.items()]

    def test_exact_registry_accepts_permutation_but_rejects_missing_extra_duplicate_or_renamed(self):
        verifier.validate_registry(list(reversed(self.rows())))
        valid = self.rows()
        cases = [valid[:-1], valid+[dict(valid[0])], valid[:-1]+[dict(valid[0])],
                 valid[:-1]+[{"id":"unverified replacement","slug":"tsmixer"}],
                 valid[:-1]+[{"id":"TSMixer","slug":"unverified_weights"}]]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    verifier.validate_registry(rows)

    def test_missing_or_duplicate_runtime_model_fails_before_input_archive_is_opened(self):
        for rows in [self.rows()[:-1],self.rows()[:-1]+[self.rows()[0]]]:
            with patch.object(verifier.inference,"_registry",return_value=rows), \
                 patch.object(verifier,"pinned_cohort") as load:
                with self.assertRaises(ValueError):
                    verifier.run(Path("nonexistent.npz"))
                load.assert_not_called()

    def test_changed_archive_hash_fails_before_numpy_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"cohort.npz"
            path.write_bytes(b"different or truncated archive")
            with patch.object(verifier.np,"load") as load:
                with self.assertRaisesRegex(ValueError,"bytes changed"):
                    verifier.pinned_cohort(path)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
