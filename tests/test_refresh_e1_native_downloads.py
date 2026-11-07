"""Public download refresh tests with synthetic archives, never market data."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import refresh_e1_native_downloads as refresh


def public_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))


class NativeDownloadRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.output = Path(self.temporary.name) / "delivery"
        self.addCleanup(patch.stopall)
        patch.object(refresh, "ROOT", self.root).start()
        patch.object(refresh.legacy, "ROOT", self.root).start()
        self.write(".gitignore", "book/_build/\ndata/raw/\n")
        self.write("BUILD.md", "Current public build instructions\n")
        self.write("README.md", "Current repository documentation\n")
        self.write("requirements-e1-supplement.lock.txt", "supplement-package==1.0\n")
        self.write("requirements-e1-curves.lock.txt", "curve-package==1.0\n")
        for name in refresh.REPRODUCTION_PROTOCOL_FILES:
            self.write(name, '{"new_development_reproduction": true}\n' if name.endswith(".json")
                       else "Separately registered development curve reproduction\n")
        self.write("scripts/current.py", "print('public current code')\n")
        self.write("tests/test_current.py", "# current public verification\n")
        self.write("book/references.bib", "@article{current, title={Current literature}}\n")
        self.write("book/_static/literature_e1/README.md", "Current literature stage index\n")
        self.stage_directories = {}
        for stage, names in refresh.STAGES.items():
            prefix = "book/_static/literature_e1/" + stage
            directory = self.root / prefix
            for name in names:
                if name == "summary.json":
                    value = json.dumps({"stage": stage, "reviewed_records": 2}) + "\n"
                elif name.endswith(".json"):
                    value = json.dumps({"public_metadata": True}) + "\n"
                elif name.endswith(".csv"):
                    value = "record_id,title,doi\nR1,Public title,10.1000/example\n"
                elif name.endswith(".bib"):
                    value = "@article{new, title={Primary reviewed paper}}\n"
                elif name.endswith(".svg"):
                    value = '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
                else:
                    value = "Public sanitized literature metadata\n"
                self.write(prefix + "/" + name, value)
            mapping = {name: refresh.legacy.digest((directory / name).read_bytes()) for name in names}
            self.write(prefix + "/manifest.json", json.dumps({"files_sha256": mapping}) + "\n")
            self.stage_directories[stage] = directory
        self.dynamic_stage_names = {}
        for stage in refresh.MANIFEST_STAGES:
            prefix = "book/_static/literature_e1/" + stage
            directory = self.root / prefix
            names = {"README.md", "summary.json", "part_a/reviews.json", "query_log.json"}
            if stage == "open_access_replacement_20261107":
                names |= {"source_manifest.json", "evidence_matrix.json", "metrics.csv",
                          "references.bib", "inputs/prior_review_registry.json",
                          "part_upload_psogru/review.json", "part_upload_psogru/README.md"}
            for name in names:
                self.write(prefix + "/" + name,
                    '{"public_metadata": true}\n' if name.endswith(".json")
                    else "Reviewed nested metadata only\n")
            mapping = {name: {"sha256": refresh.legacy.digest((directory / name).read_bytes()),
                              "bytes": (directory / name).stat().st_size} for name in names}
            self.write(prefix + "/manifest.json", json.dumps({
                "publication_schema": refresh.PUBLICATION_SCHEMA,
                "approved_public_files": sorted(names), "files": mapping}) + "\n")
            self.dynamic_stage_names[stage] = names
            self.stage_directories[stage] = directory
        self.supplement_directories = {}
        for label, spec in refresh.SUPPLEMENTS.items():
            prefix, names = spec["directory"], spec["files"]
            directory = self.root / prefix
            for name in names:
                if name == "summary.json":
                    value = json.dumps({"supplement": label, "post_hoc": True}) + "\n"
                elif name.endswith(".json"):
                    value = json.dumps({"public_metrics": True}) + "\n"
                elif name.endswith(".csv"):
                    value = "model,rmse\nGRU,0.005\n"
                elif name.endswith(".png"):
                    value = public_png()
                else:
                    value = "Public post-hoc evidence; no new fits\n"
                self.write(prefix + "/" + name, value)
            mapping = {name: {"sha256": refresh.legacy.digest((directory / name).read_bytes()),
                              "bytes": (directory / name).stat().st_size} for name in names}
            self.write(prefix + "/" + spec["manifest"], json.dumps({"files": mapping}) + "\n")
            self.supplement_directories[label] = directory
        self.base = self.root / "book/_static/e1_downloads"
        self.base.mkdir(parents=True)
        records = []
        self.base_payloads = {}
        for item, filename in refresh.ITEMS.items():
            payload = {"original_metadata.json": b'{"historical": true}\n'}
            if item == "8.2":
                payload.update({"scripts/obsolete.py": b"# stale code\n",
                                "tests/test_obsolete.py": b"# stale test\n",
                                "experiments/original/design.json": b'{"seed": 73}\n',
                                "requirements-training.lock.txt": b"historical==1.0\n",
                                "BUILD.md": b"Old build\n", "README.md": b"Old README\n"})
            if item == "8.4":
                payload.update({"book/references.bib": b"@article{old}\n",
                    "book/_static/literature_e1/history_20261102/studies.json": b'{"n": 7}\n'})
            if item == "8.6":
                payload.update({"experiments/original/design.json": b'{"original_config": true}\n',
                                "requirements-training.lock.txt": b"original-runtime==1.0\n"})
            self.base_payloads[item] = payload
            records.append(refresh.legacy.archive(self.base / filename, payload, item, "Product " + item,
                                                  "a" * 40, "Original phase I/II notes\n"))
        self.catalog = {"source_commit": "a" * 40, "models": ["GRU", "LSTM", "TCN"],
            "active_entrypoint": "scripts/run_gru_lstm_tcn.py",
            "active_entrypoints": {"phase_I": "scripts/run_gru_lstm_tcn.py",
                                    "phase_II": "scripts/run_gru_lstm_tcn_planned.py"},
            "baseline_selected_by_cv": {"model": "GRU", "seed_count": 10},
            "planned_D7_phase": {"definitive_seeds_per_family": 465, "temporal_power_claimed": False},
            "scope": "Two separate historical experimental phases",
            "declaration_manifest_sha256": "c" * 64,
            "presentation_refresh": {"source_commit": "a" * 40,
                                     "preserved_archive_sha256": {"8_6_configuraciones.zip": "b" * 64}},
            "packages": records, "complete_package": {"title": "Complete E1"},
            "scientific_limitations_preserved": True}
        self.write("book/_static/e1_downloads/catalogo.json", json.dumps(self.catalog) + "\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.commit()
        self.build()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            path.write_bytes(data)
        else:
            path.write_text(data, encoding="utf-8")
        return path

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.PIPE, text=True).strip()

    def commit(self):
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "--quiet", "-m", "Public source")

    def build(self):
        self.write("book/_build/html/_static/eda/results/summary.json", '{"private": "one"}')
        self.write("book/_build/html/_static/eda_temporal_reference/summary.json", '{"private": "two"}')
        self.write("book/_build/html/_static/e1_downloads/8_5_benchmark.zip", b"must be excluded")
        self.write("book/_build/html/intro.html", '<html><a href="_static/e1_downloads/8_5_benchmark.zip">Download</a></html>')

    def test_refresh_preserves_science_and_updates_only_reviewed_delivery(self):
        baseline_archives = {item: (self.base / refresh.ITEMS[item]).read_bytes()
                             for item in refresh.PRESERVED_ITEMS}
        with patch.object(refresh.legacy, "load_crypto_stage", side_effect=AssertionError("Legacy validator must not run")):
            catalog = refresh.package(self.output)
        for item, data in baseline_archives.items():
            self.assertEqual((self.output / refresh.ITEMS[item]).read_bytes(), data)
        for key in ("models", "active_entrypoint", "active_entrypoints", "baseline_selected_by_cv",
                    "planned_D7_phase", "scope", "declaration_manifest_sha256"):
            self.assertEqual(catalog[key], self.catalog[key])
        self.assertEqual(catalog["source_commit"], self.git("rev-parse", "HEAD"))
        for row in catalog["packages"] + [catalog["complete_package"]]:
            self.assertEqual(row["url"], refresh.legacy.PUBLIC_BOOK + "_static/e1_downloads/" + row["file"])
        self.assertFalse(catalog["literature_refresh"]["retrospective_preregistration"])
        self.assertNotIn("presentation_refresh", catalog)
        self.assertEqual(catalog["prior_presentation_refreshes"], [self.catalog["presentation_refresh"]])
        self.assertEqual(catalog["new_model_fits_by_packaging"], 0)
        with ZipFile(self.output / refresh.ITEMS["8.2"]) as saved:
            self.assertEqual(saved.read("scripts/current.py"), (self.root / "scripts/current.py").read_bytes())
            self.assertNotIn("scripts/obsolete.py", saved.namelist())
            self.assertNotIn("tests/test_obsolete.py", saved.namelist())
            self.assertEqual(saved.read("experiments/original/design.json"), b'{"seed": 73}\n')
            self.assertEqual(saved.read("requirements-training.lock.txt"), b"historical==1.0\n")
            self.assertEqual(saved.read("BUILD.md"), b"Current public build instructions\n")
            self.assertEqual(saved.read("requirements-e1-supplement.lock.txt"), b"supplement-package==1.0\n")
            self.assertEqual(saved.read("requirements-e1-curves.lock.txt"), b"curve-package==1.0\n")
        with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
            self.assertEqual(saved.read("book/references.bib"), (self.root / "book/references.bib").read_bytes())
            self.assertEqual(saved.read("book/_static/literature_e1/history_20261102/studies.json"), b'{"n": 7}\n')
            for stage, names in refresh.STAGES.items():
                for name in names | {"manifest.json"}:
                    self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            for stage, names in self.dynamic_stage_names.items():
                for name in names | {"manifest.json"}:
                    self.assertEqual(saved.read("book/_static/literature_e1/" + stage + "/" + name),
                                     (self.stage_directories[stage] / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
            for stage, names in refresh.STAGES.items():
                for name in ("summary.json", "manifest.json", "source_manifest.json"):
                    if name == "manifest.json" or name in names:
                        self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            for stage in refresh.MANIFEST_STAGES:
                for name in ("summary.json", "manifest.json"):
                    self.assertIn("book/_static/literature_e1/" + stage + "/" + name, saved.namelist())
            self.assertEqual(saved.read("original_metadata.json"), b'{"historical": true}\n')
            self.assertIn("prerregistro retrospectivo", saved.read("AMPLIACION_LITERARIA_20261106.md").decode())
            self.assertIn("SUPLEMENTO_EVIDENCIAS_20261106.md", saved.namelist())
            for label, spec in refresh.SUPPLEMENTS.items():
                for name in ("summary.json", spec["manifest"]):
                    self.assertIn(spec["directory"] + "/" + name, saved.namelist())
        for label, spec in refresh.SUPPLEMENTS.items():
            with ZipFile(self.output / refresh.ITEMS[spec["item"]]) as saved:
                for name in spec["files"] | {spec["manifest"]}:
                    self.assertEqual(saved.read(spec["directory"] + "/" + name),
                                     (self.supplement_directories[label] / name).read_bytes())
                # Supplemental analyses add separate namespaces; original science remains byte-identical.
                for name, data in self.base_payloads[spec["item"]].items():
                    self.assertEqual(saved.read(name), data)
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["statistics"]["product"], "8.5")
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["curves"]["product"], "8.5")
        self.assertEqual(catalog["evidence_refresh"]["supplements"]["academic_margin"]["product"], "8.7")
        with ZipFile(self.output / refresh.ITEMS["8.6"]) as saved:
            for name, data in self.base_payloads["8.6"].items():
                self.assertEqual(saved.read(name), data)
            for name in refresh.REPRODUCTION_PROTOCOL_FILES:
                self.assertEqual(saved.read(name), (self.root / name).read_bytes())
            self.assertEqual(saved.read("requirements-e1-curves.lock.txt"), b"curve-package==1.0\n")
            curve_prefix = refresh.SUPPLEMENTS["curves"]["directory"] + "/"
            for name in ("reproduction_registration.json", "runtime.json", "execution.json"):
                self.assertEqual(saved.read(curve_prefix + name),
                    (self.supplement_directories["curves"] / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
            for name in refresh.REPRODUCTION_PROTOCOL_FILES:
                self.assertEqual(saved.read(name), (self.root / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.1"]) as saved:
            self.assertNotIn("informe/_static/e1_downloads/8_5_benchmark.zip", saved.namelist())
            self.assertNotIn("informe/_static/eda/results/summary.json", saved.namelist())
            self.assertIn(refresh.legacy.PUBLIC_BOOK.encode(), saved.read("informe/intro.html"))
        with ZipFile(self.output / "entregable1_completo.zip") as saved:
            self.assertEqual(set(saved.namelist()), set(refresh.ITEMS.values()) |
                             {"CATALOGO_PRODUCTOS.json", "LEEME.md", "MANIFIESTO.json"})
            self.assertEqual(json.loads(saved.read("CATALOGO_PRODUCTOS.json")), catalog["packages"])
        for line in (self.output / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split("  ")
            self.assertEqual(refresh.legacy.digest((self.output / name).read_bytes()), expected)
        for filename in [*refresh.ITEMS.values(), "entregable1_completo.zip"]:
            self.assertEqual(refresh.verify_archive(self.output / filename)["member_hashes"], "PASS")
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_rejects_dirty_source_and_mismatched_commit(self):
        self.write("README.md", "Uncommitted edit\n")
        with self.assertRaisesRegex(ValueError, "Commit final source"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())
        self.git("checkout", "--", "README.md")
        with self.assertRaisesRegex(subprocess.CalledProcessError, "git"):
            refresh.source_revision("does-not-exist")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                 "commit", "--allow-empty", "--quiet", "-m", "Second commit")
        with self.assertRaisesRegex(ValueError, "current HEAD"):
            refresh.source_revision("HEAD~1")

    def test_open_access_replacement_routes_nested_metadata_and_current_builder(self):
        stage = "open_access_replacement_20261107"
        self.assertIn(stage, refresh.MANIFEST_STAGES)
        builder = "scripts/build_e1_open_access_replacement.py"
        self.write(builder, "# Build reviewed literature metadata without training.\n")
        self.git("add", builder)
        self.commit()
        self.build()
        catalog = refresh.package(self.output)
        prefix = "book/_static/literature_e1/" + stage + "/"
        directory = self.stage_directories[stage]
        approved = self.dynamic_stage_names[stage] | {"manifest.json"}
        with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
            included = {name[len(prefix):] for name in saved.namelist() if name.startswith(prefix)}
            self.assertEqual(included, approved)
            for name in approved:
                self.assertEqual(saved.read(prefix + name), (directory / name).read_bytes())
        with ZipFile(self.output / refresh.ITEMS["8.7"]) as saved:
            included = {name[len(prefix):] for name in saved.namelist() if name.startswith(prefix)}
            self.assertEqual(included, {"summary.json", "source_manifest.json", "manifest.json"})
        with ZipFile(self.output / refresh.ITEMS["8.2"]) as saved:
            self.assertEqual(saved.read(builder), (self.root / builder).read_bytes())
        self.assertEqual(catalog["literature_refresh"]["stages"][stage]["public_files"], len(approved))
        for key in ("new_model_fits_by_packaging", "new_checkpoint_forecasts_by_packaging",
                    "raw_market_archive_reads_by_packaging", "publications_by_packaging"):
            self.assertEqual(catalog[key], 0)

    def test_rejects_unbuilt_stale_report_and_unsafe_destination(self):
        intro = self.root / "book/_build/html/intro.html"
        intro.unlink()
        with self.assertRaisesRegex(ValueError, "regular public source"):
            refresh.package(self.output)
        self.build()
        self.write("book/references.bib", "@article{newer, title={Newer source}}\n")
        self.git("add", "book/references.bib")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Rebuild E1"):
            refresh.package(self.output)
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            refresh.package(self.root / "inside")
        self.output.mkdir()
        (self.output / "existing.txt").write_text("Do not overwrite")
        with self.assertRaisesRegex(ValueError, "empty output"):
            refresh.package(self.output)
        self.assertEqual((self.output / "existing.txt").read_text(), "Do not overwrite")

    def test_rejects_baseline_archive_hash_mismatch(self):
        (self.base / refresh.ITEMS["8.5"]).write_bytes(b"tampered archive")
        with self.assertRaisesRegex(ValueError, "does not match catalog"):
            refresh.baseline_products(self.base)

    def test_rejects_manifest_path_escape_and_unreviewed_member(self):
        directory = self.stage_directories["native_search_20261106"]
        manifest = directory / "manifest.json"
        record = json.loads(manifest.read_text())
        record["files_sha256"]["../private.csv"] = "0" * 64
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "approved public files"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])
        for name in ("../private.csv", "/tmp/file", "C:/file", "a\\b", "a//b", "./a", "a\nfile"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                refresh.safe_member(name)

    def test_rejects_changed_allowlisted_file_and_symlink(self):
        directory = self.stage_directories["native_search_20261106"]
        target = directory / "records.csv"
        original = target.read_bytes()
        target.write_bytes(original + b"unexpected change\n")
        with self.assertRaisesRegex(ValueError, "allowlist hash mismatch"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])
        target.unlink()
        outside = Path(self.temporary.name) / "outside.csv"
        outside.write_bytes(original)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes|Symlink"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])

    def test_accepts_sized_native_manifest_and_checks_byte_counts(self):
        directory = self.stage_directories["native_search_20261106"]
        path = directory / "manifest.json"
        mapping = json.loads(path.read_text())["files_sha256"]
        record = {"files": {name: {"sha256": sha, "bytes": (directory / name).stat().st_size}
                            for name, sha in mapping.items()}}
        path.write_text(json.dumps(record))
        files, info = refresh.public_stage(directory, refresh.STAGES[directory.name])
        self.assertIn("records.csv", files)
        self.assertEqual(info["public_files"], len(mapping) + 1)
        record["files"]["records.csv"]["bytes"] += 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            refresh.public_stage(directory, refresh.STAGES[directory.name])

    def test_unlisted_uploads_are_never_copied_or_opened(self):
        # Ignored files mimic private uploads kept alongside public working data.
        stage = self.stage_directories["native_search_20261106"]
        self.write(".gitignore", (self.root / ".gitignore").read_text() + "raw_upload.csv\noriginal.pdf\n")
        self.git("add", ".gitignore")
        self.commit()
        self.build()
        upload = stage / "raw_upload.csv"
        upload.write_text("Abstract,Author Email\nPrivate abstract,private@example.test\n")
        nested_uploads = set()
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            nested_upload = directory / "raw_upload.csv"
            nested_upload.write_text("Abstract,Author Email\nPrivate abstract,private@example.test\n")
            original_pdf = directory / "original.pdf"
            original_pdf.write_bytes(b"Private original supplied separately")
            nested_uploads.update({nested_upload, original_pdf})
        market = self.write("data/raw/market.csv", "Do not read market observations\n")
        original_read_bytes = Path.read_bytes
        def public_reads_only(path):
            if path in {upload, market} | nested_uploads:
                raise AssertionError("Private input was opened: " + str(path))
            return original_read_bytes(path)
        with patch.object(Path, "read_bytes", public_reads_only):
            refresh.package(self.output)
        with ZipFile(self.output / refresh.ITEMS["8.4"]) as saved:
            self.assertFalse(any("raw_upload" in name for name in saved.namelist()))
            self.assertFalse(any(name.endswith(".pdf") for name in saved.namelist()))

    def test_rejects_abstract_contact_fields_in_metadata(self):
        cases = {"records.csv": b"Title,Abstract\nPublic title,Private abstract\n",
                 "source_manifest.json": b'{"records": [{"author_email": "private@example.test"}]}',
                 "references.bib": b"@article{x,\n abstract = {Private abstract}\n}\n",
                 "reviews.json": b'{"raw_abstract": "Exported copyrighted text"}'}
        for name, data in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Private/abstract"):
                refresh.validate_public_bytes(name, data)

    def test_nested_stage_needs_explicit_closed_approval(self):
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            path = directory / "manifest.json"
            original = json.loads(path.read_text())
            files, _ = refresh.public_stage(directory, None)
            self.assertIn("part_a/reviews.json", files)
            for change in ("missing_schema", "unapproved_file", "duplicate_approval",
                           "unsorted_approval", "hash_mismatch", "byte_mismatch"):
                changed = json.loads(json.dumps(original))
                if change == "missing_schema":
                    changed.pop("publication_schema")
                elif change == "unapproved_file":
                    changed["files"]["other.json"] = {"bytes": 1, "sha256": "a" * 64}
                elif change == "duplicate_approval":
                    changed["approved_public_files"].append("summary.json")
                elif change == "unsorted_approval":
                    changed["approved_public_files"].reverse()
                elif change == "byte_mismatch":
                    changed["files"]["part_a/reviews.json"]["bytes"] += 1
                else:
                    changed["files"]["part_a/reviews.json"]["sha256"] = "a" * 64
                path.write_text(json.dumps(changed))
                with self.subTest(stage=stage, change=change), self.assertRaises(ValueError):
                    refresh.public_stage(directory, None)

    def test_nested_approval_cannot_ship_private_payload_or_pdf(self):
        for stage in refresh.MANIFEST_STAGES:
            directory = self.stage_directories[stage]
            path = directory / "manifest.json"
            original = json.loads(path.read_text())
            for name in ("predictions.csv", "part_a/paper.pdf", "../escape.json"):
                changed = json.loads(json.dumps(original))
                changed["files"][name] = {"bytes": 1, "sha256": "a" * 64}
                changed["approved_public_files"] = sorted(changed["files"])
                path.write_text(json.dumps(changed))
                # Failure must precede reading any payload named by the bad entry.
                with self.subTest(stage=stage, name=name), self.assertRaisesRegex(
                        ValueError, "Private experimental|Unsupported public|Unsafe"):
                    refresh.public_stage(directory, None)

    def test_reproduction_protocol_must_be_committed_before_any_output(self):
        name = sorted(refresh.REPRODUCTION_PROTOCOL_FILES)[0]
        self.git("rm", name)
        self.commit()
        self.build()
        with self.assertRaisesRegex(ValueError, "Reproduction protocol source is not committed"):
            refresh.package(self.output)
        self.assertFalse(self.output.exists())

    def test_statistics_manifest_is_closed_and_public_payload_is_validated(self):
        spec = refresh.SUPPLEMENTS["statistics"]
        directory = self.supplement_directories["statistics"]
        files, _ = refresh.public_stage(directory, spec["files"], spec["manifest"])
        self.assertIn("qq_differences.png", files)
        path = directory / spec["manifest"]
        original = json.loads(path.read_text())
        changed = json.loads(path.read_text())
        changed["files"]["predictions.csv"] = {"sha256": "a" * 64, "bytes": 1}
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "approved public files"):
            refresh.public_stage(directory, spec["files"], spec["manifest"])
        private = b"model,Author Email\nGRU,private@example.test\n"
        (directory / "pairwise_sensitivity.csv").write_bytes(private)
        original["files"]["pairwise_sensitivity.csv"] = {
            "sha256": refresh.legacy.digest(private), "bytes": len(private)}
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "Private/abstract"):
            refresh.public_stage(directory, spec["files"], spec["manifest"])

    def test_rejects_broken_plot_trailing_payload_and_unsupported_format(self):
        refresh.validate_public_bytes("qq.png", public_png())
        with self.assertRaisesRegex(ValueError, "Trailing data"):
            refresh.validate_public_bytes("qq.png", public_png() + b"hidden payload")
        corrupted = bytearray(public_png())
        corrupted[20] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            refresh.validate_public_bytes("qq.png", bytes(corrupted))
        with self.assertRaisesRegex(ValueError, "Unsupported public evidence extension"):
            refresh.validate_public_bytes("model.pt", b"checkpoint")

    def test_rejects_archive_manifest_mismatch_unsafe_path_and_symlink(self):
        path = Path(self.temporary.name) / "test.zip"
        refresh.legacy.archive(path, {"public.txt": b"public"}, "8.4", "Literature", "a" * 40, "Readme")
        with ZipFile(path) as saved:
            files = {name: saved.read(name) for name in saved.namelist()}
        files["public.txt"] = b"tampered"
        with ZipFile(path, "w") as saved:
            for name, data in files.items():
                saved.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "member hash mismatch"):
            refresh.verify_archive(path)
        refresh.legacy.archive(path, {"../escape.txt": b"public"}, "8.4", "Literature", "a" * 40, "Readme")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            refresh.verify_archive(path)
        with ZipFile(path, "w") as saved:
            link = ZipInfo("link.txt")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            saved.writestr(link, "../outside")
        with self.assertRaisesRegex(ValueError, "symlink"):
            refresh.verify_archive(path)


if __name__ == "__main__":
    unittest.main()
