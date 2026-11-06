"""Guard documentary provenance, original row identity, and missing-data semantics."""

import copy
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e2_fulltext", ROOT / "scripts/compile_e2_fulltext_review.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
PUBLIC = ROOT / "references/literature_e2/fulltext_20261105"
STATIC = ROOT / "book_e2/_static/literature_fulltext"


class FulltextReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = review.read_json(PUBLIC / "source_inputs.json")
        cls.manifest = review.read_json(PUBLIC / "fulltext_manifest.json")

    def model_input(self, inputs, name):
        return next(m for group in inputs["groups"]
                    for m in group["facts"].get("models", group["facts"].get("papers", []))
                    if m["model"] == name)

    def test_ten_distinct_papers_actual_pages_exact_versions_and_receipts(self):
        papers = self.manifest["papers"]
        self.assertEqual(len(papers), 10)
        self.assertEqual(len({p["model"] for p in papers}), 10)
        self.assertEqual(sum(p["pdf_pages"] for p in papers), 215)
        actual_pages = {"S-Mamba": 14, "iTransformer": 25, "Autoformer": 20, "Informer": 15,
                        "Crossformer": 21, "TimesNet": 23, "PatchTST": 24, "TimeMixer": 28,
                        "TiDE": 21, "TSMixer": 24}
        self.assertEqual({p["model"]: p["pdf_pages"] for p in papers}, actual_pages)
        for p in papers:
            self.assertEqual(p["receipt_kind"], "user_provided")
            self.assertEqual(Path(p["filename"]).name, p["filename"])
            self.assertRegex(p["sha256"], r"^[a-f0-9]{64}$")
            if p["model"] != "Crossformer":
                self.assertEqual(p["source_url"], "https://arxiv.org/abs/" + p["filename"][:-4])
        cross = next(p for p in papers if p["model"] == "Crossformer")
        self.assertEqual(cross["source_url"], "https://openreview.net/forum?id=vSVLM2j9eie")

    def test_fixed_rows_preserve_precision_and_nonuniform_horizons(self):
        compiled = review.compile_models(self.inputs, self.manifest)
        models = {m["model"]: m for m in compiled["models"]}
        self.assertEqual(len(models), 10)
        expected = {"TSMixer": ("0.361", "0.392", 96, 512), "TiDE": ("0.375", "0.398", 96, 720),
                    "TimeMixer": ("0.375", "0.400", 96, 96), "PatchTST": ("0.370", "0.400", 96, 512),
                    "Informer": ("0.577", "0.549", 24, None), "Autoformer": ("0.384", "0.425", 24, 96),
                    "TimesNet": ("0.384", "0.402", 96, 96), "S-Mamba": ("0.386", "0.405", 96, 96),
                    "iTransformer": ("0.386", "0.405", 96, 96), "Crossformer": ("0.305", "0.367", 24, None)}
        for name, values in expected.items():
            p = models[name]["canonical_metric_pair"]
            self.assertEqual(tuple(p[k] for k in ("MSE", "MAE", "horizon_steps", "context_steps")), values)
            self.assertTrue(p["same_row_pair"])
        self.assertEqual(models["S-Mamba"]["source_version"], "2403.11144v3")
        self.assertEqual(models["S-Mamba"]["publication_year"], 2024)

    def test_other_variants_horizons_and_mixed_columns_are_rejected(self):
        for name, mutate in (
            ("TimeMixer", lambda m: m["verified_metric_pairs"].reverse()),
            ("PatchTST", lambda m: m["verified_metric_pairs"].reverse()),
            ("Informer", lambda m: m["numeric_evidence"][0].update(horizon=96)),
            ("Autoformer", lambda m: m["numeric_evidence"][0].update(dataset="ETTm2")),
            ("TSMixer", lambda m: m["canonical_metric_pair"].update(MSE=0.359)),
            ("TSMixer", lambda m: m["canonical_metric_pair"].update(model_variant="TSMixer-Ext with future/static covariates")),
            ("Informer", lambda m: m["numeric_evidence"][0].update(model_variant="Informer† dense attention")),
            ("iTransformer", lambda m: m["exact_metric_pair"]["full_row"][0].update(MAE=0.395)),
        ):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.inputs)
                mutate(self.model_input(changed, name))
                with self.assertRaisesRegex(ValueError, "Canonical"):
                    review.compile_models(changed, self.manifest)

    def test_unknown_context_is_not_imputed_and_missing_metrics_are_not_zero(self):
        compiled = review.compile_models(self.inputs, self.manifest)
        for m in compiled["models"]:
            if m["model"] in {"Informer", "Crossformer"}:
                self.assertIsNone(m["canonical_metric_pair"]["context_steps"])
                self.assertEqual(m["canonical_metric_pair"]["context_status"], "NR")
        for value in (None, "NR", "NV", False):
            changed = copy.deepcopy(self.inputs)
            self.model_input(changed, "TSMixer")["canonical_metric_pair"]["MSE"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Missing original numeric MSE"):
                review.compile_models(changed, self.manifest)

    def test_selected_row_uncertainty_and_historical_status_do_not_transfer(self):
        compiled = review.compile_models(self.inputs, self.manifest)
        models = {m["model"]: m for m in compiled["models"]}
        for name in set(models) - {"TiDE"}:
            self.assertEqual(models[name]["quality"]["variance_selected_pair"]["status"], "NR")
            self.assertIsNone(models[name]["quality"]["variance_selected_pair"]["value"])
        self.assertEqual(models["TiDE"]["quality"]["variance_selected_pair"]["value"]["kind"], "standard_error")
        self.assertEqual(models["iTransformer"]["quality"]["repetitions"]["status"], "NV")
        self.assertEqual(models["PatchTST"]["quality"]["seed_ids"]["value"], [2021])
        counts = compiled["counts"]
        self.assertEqual(counts["fulltext_verified_current"], 10)
        self.assertEqual(counts["author_code_available_from_immutable_prior_receipts"], 10)
        self.assertEqual(counts["fulltext_retrieved_in_historical_code_only_stage"], 0)
        self.assertEqual(counts["models_with_benchmark_descriptions"], 10)
        self.assertEqual(counts["benchmark_downloads_validated_this_review"], 0)
        self.assertEqual(counts["new_model_fits_this_review"], 0)
        self.assertEqual(counts["original_papers_numerically_reproduced_this_review"], 0)

    def test_duplicate_missing_model_hash_mismatch_and_page_overflow_fail(self):
        duplicate = copy.deepcopy(self.inputs)
        duplicate["groups"][0]["facts"]["models"].append(copy.deepcopy(duplicate["groups"][0]["facts"]["models"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review.compile_models(duplicate, self.manifest)
        incomplete = copy.deepcopy(self.inputs)
        incomplete["groups"].pop()
        with self.assertRaisesRegex(ValueError, "Exactly ten"):
            review.compile_models(incomplete, self.manifest)
        wrong_hash = copy.deepcopy(self.manifest)
        wrong_hash["papers"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "hash differs"):
            review.compile_models(self.inputs, wrong_hash)
        for name, keys in (("TiDE", ("paper",)), ("TSMixer", ("verification", "source_file"))):
            wrong_nested_hash = copy.deepcopy(self.inputs)
            source = self.model_input(wrong_nested_hash, name)
            for key in keys:
                source = source[key]
            source["sha256"] = "0" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash differs"):
                review.compile_models(wrong_nested_hash, self.manifest)
        invalid_page = copy.deepcopy(self.inputs)
        self.model_input(invalid_page, "S-Mamba")["exact_metric_pairs"][0]["locator"]["pdf_page"] = 15
        with self.assertRaisesRegex(ValueError, "outside 1..14"):
            review.compile_models(invalid_page, self.manifest)

    def test_private_ingestion_checks_actual_binary_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "paper.pdf"
            binary.write_bytes(b"an intentionally wrong PDF receipt")
            source = copy.deepcopy(self.manifest["papers"])
            source[0]["path"] = str(binary)
            with self.assertRaisesRegex(ValueError, "hash or byte size mismatch"):
                review.make_manifest(source)

    def test_sanitization_excludes_paths_excerpts_file_ids_and_abstract(self):
        private = {"model": "A", "pdf_path_private": "/workspace/attachments/secret.pdf",
                   "first_page_excerpt": "long source prose", "AB": "source abstract",
                   "extra": "/workspace/private/hidden.txt", "nested": [{"user_file_id": "file-secret", "pdf_page": 3}],
                   "code_path": "models/TimesNet.py", "metrics": {"MAE": "0.400"}}
        clean = review.sanitize(private)
        review.validate_public(clean)
        self.assertEqual(clean, {"model": "A", "nested": [{"pdf_page": 3}], "code_path": "models/TimesNet.py", "metrics": {"MAE": "0.400"}})
        with self.assertRaisesRegex(ValueError, "private path"):
            review.validate_public({"unsafe": "/workspace/attachments/a.pdf"})
        with self.assertRaisesRegex(ValueError, "uploaded-file identifier"):
            review.validate_public({"unsafe": "file_12345678secret"})
        for key in ("abstract", "AB", "quote", "long_verbatim_quote", "first_page_excerpt"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "excluded source field"):
                review.validate_public({key: "Do not publish original source prose"})
        for p in self.manifest["papers"]:
            self.assertNotIn("path", p)
            self.assertNotIn("text_path", p)
        review.validate_public(self.inputs)
        review.validate_public(self.manifest)

    def test_frozen_regeneration_hashes_csv_precision_and_static_mirrors(self):
        outputs = review.build_outputs(self.inputs, self.manifest)
        for filename, content in outputs.items():
            self.assertEqual((PUBLIC / filename).read_bytes(), content, filename)
            self.assertEqual((STATIC / filename).read_bytes(), content, filename)
        provenance = json.loads(outputs["provenance.json"])
        for filename, sha in provenance["output_hashes"].items():
            self.assertEqual(hashlib.sha256(outputs[filename]).hexdigest(), sha)
        for line in outputs["SHA256SUMS"].decode().splitlines():
            sha, filename = line.split("  ")
            self.assertEqual(hashlib.sha256(outputs[filename]).hexdigest(), sha)
        metrics = {r["model"]: r for r in csv.DictReader(io.StringIO(outputs["metrics.csv"].decode()))}
        self.assertEqual(metrics["PatchTST"]["MSE"], "0.370")
        self.assertEqual(metrics["TimeMixer"]["MAE"], "0.400")
        self.assertEqual(metrics["Crossformer"]["context_steps"], "NR")
        self.assertEqual(metrics["Informer"]["horizon_steps"], "24")


if __name__ == "__main__":
    unittest.main()
