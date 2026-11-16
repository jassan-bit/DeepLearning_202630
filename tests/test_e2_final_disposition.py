"""Guard against turning a terminal management decision into scientific approval."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "book_e2/_static/e2_resolucion_final_20261116"


def load_verifier():
    spec = importlib.util.spec_from_file_location("e2_final_disposition_verifier", PACKAGE / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalDispositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()
        cls.state = json.loads((PACKAGE / "decisions.json").read_text(encoding="utf-8"))

    def test_verified_final_package_preserves_nonacceptance(self):
        result = self.verifier.verify(PACKAGE)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["full_guide_compliance_certified"])
        self.assertFalse(result["scientific_acceptance_changed"])

    def test_closing_management_cannot_certify_audit(self):
        state = deepcopy(self.state)
        state["full_audit_accepted"] = True
        with self.assertRaisesRegex(ValueError, "Unsupported scientific acceptance"):
            self.verifier.validate_state(state)

    def test_closing_a_single_observation_cannot_approve_its_requirement(self):
        state = deepcopy(self.state)
        state["observations"][0]["full_guide_acceptance_established"] = True
        with self.assertRaisesRegex(ValueError, "cannot approve"):
            self.verifier.validate_state(state)

    def test_missing_terminal_disposition_is_rejected(self):
        state = deepcopy(self.state)
        state["observations"].pop()
        with self.assertRaisesRegex(ValueError, "four final dispositions"):
            self.verifier.validate_state(state)

    def test_changed_scientific_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            isolated = Path(folder) / "package"
            isolated.mkdir()
            shutil.copy(PACKAGE / "manifest.json", isolated / "manifest.json")
            manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
            for row in manifest["files"]:
                destination = isolated / row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(PACKAGE / row["path"], destination)
            with (isolated / "evidence/negative_controls.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "Changed package input"):
                self.verifier.verify(isolated)

    def test_five_partial_and_two_NAS_items_are_preserved(self):
        current = json.loads((PACKAGE / "appendix_B_final.json").read_text(encoding="utf-8"))
        partial = {r["id"] for r in current["items"] if r["status"] == "parcial"}
        self.assertEqual(partial, {"B01", "B03", "B06", "B09", "B21"})
        self.assertEqual({r["id"] for r in current["items"] if r["status"] == "no_aplica"}, {"B14", "B15"})


if __name__ == "__main__":
    unittest.main()
