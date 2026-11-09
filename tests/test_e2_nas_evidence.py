"""Audit committed conformance evidence against code and exhaustive case identities."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"book_e2/_static/nas_factories_20261109"


class EvidenceTests(unittest.TestCase):
    def test_evidence_is_bound_to_the_exact_factory_and_checker_sources(self):
        report=json.loads((DATA/"summary.json").read_text())
        self.assertFalse(report["market_data_read"])
        self.assertEqual(report["optimizer_steps"],0)
        for path,expected in report["source_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected)

    def test_ledgers_have_complete_distinct_spaces_and_matching_checksums(self):
        report=json.loads((DATA/"summary.json").read_text())
        models=defaultdict(set)
        for filename,artifact in report["artifacts"].items():
            compressed=(DATA/filename).read_bytes()
            self.assertEqual(hashlib.sha256(compressed).hexdigest(),artifact["gzip_sha256"])
            raw=gzip.decompress(compressed)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),artifact["uncompressed_sha256"])
            rows=[json.loads(line) for line in raw.splitlines()]
            self.assertEqual(len(rows),artifact["records"])
            if filename.startswith("predictors_"):
                for row in rows:
                    self.assertTrue(row["all_parameter_gradients_connected"])
                    self.assertTrue(row["numerical_finiteness_checked"])
                    models[row["model"]].add(row["architecture_id"])
        self.assertEqual(len(models),6)
        self.assertTrue(all(len(ids)==1000 for ids in models.values()))


if __name__=="__main__":
    unittest.main()

