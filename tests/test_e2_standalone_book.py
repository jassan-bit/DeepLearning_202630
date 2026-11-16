"""Offline book contract checks; no payload build, download or scientific run."""
import hashlib
import io
from contextlib import redirect_stderr
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_e2_standalone_book_payload as builder
import e2_standalone_book as reader


COMMIT = "a" * 40
PATHS = tuple(builder.OPTIONAL_ASSET_PINS)


def catalogue():
    return {name: {"repository_path": name, **builder.OPTIONAL_ASSET_PINS[name],
                   "download_commit": COMMIT,
                   "url": f"{builder.OPTIONAL_DOWNLOAD_PREFIX}/{COMMIT}/{name}",
                   "reason": "Redundant archive, available online."} for name in PATHS}


class OptionalSourceIdentityTests(unittest.TestCase):
    def setUp(self):
        forbid_git = patch.object(builder, "git", side_effect=AssertionError("Unexpected Git call outside a mock"))
        forbid_git.start()
        self.addCleanup(forbid_git.stop)
        self.raw = {name: f"scientific fixture {i}".encode() for i, name in enumerate(PATHS)}
        self.pins = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                     for name, raw in self.raw.items()}
        self.inventory = {name: {"git_blob": f"source{i}", "git_mode": "100644"}
                          for i, name in enumerate(PATHS)}
        self.blobs = {f"{kind}{i}": self.raw[name] for i, name in enumerate(PATHS)
                      for kind in ("source", "download")}

    def git(self, *arguments):
        if arguments[0] == "rev-parse":
            return (COMMIT + "\n").encode()
        if arguments[0] == "ls-tree":
            name = arguments[-1]
            return f"100644 blob download{PATHS.index(name)}\t{name}\0".encode()
        if arguments[:2] == ("cat-file", "blob"):
            return self.blobs[arguments[2]]
        self.fail(f"Unexpected Git operation: {arguments}")

    def test_explicit_commit_binds_both_snapshots_and_immutable_https(self):
        self.assertEqual(builder.optional_download_catalogue(None, {}), {})
        with patch.object(builder, "OPTIONAL_ASSET_PINS", self.pins), patch.object(builder, "git", self.git):
            result = builder.optional_download_catalogue(COMMIT, self.inventory)
        self.assertEqual(set(result), set(PATHS))
        for name, record in result.items():
            self.assertEqual(record["sha256"], self.pins[name]["sha256"])
            self.assertEqual(record["url"], f"{builder.OPTIONAL_DOWNLOAD_PREFIX}/{COMMIT}/{name}")

    def test_mutable_or_short_revision_and_changed_asset_are_rejected(self):
        for revision in ("main", "a" * 7, "g" * 40):
            with self.assertRaises(ValueError):
                builder.optional_download_catalogue(revision, self.inventory)
        with patch.object(builder, "OPTIONAL_ASSET_PINS", self.pins), patch.object(builder, "git", self.git):
            self.blobs["download0"] = b"changed optional commit bytes"
            with self.assertRaisesRegex(ValueError, "Bytes/SHA"):
                builder.optional_download_catalogue(COMMIT, self.inventory)
            self.blobs["download0"] = self.raw[PATHS[0]]
            self.blobs["source0"] = b"changed selected source bytes"
            with self.assertRaisesRegex(ValueError, "Bytes/SHA"):
                builder.optional_download_catalogue(COMMIT, self.inventory)

    def test_executed_builder_must_match_its_selected_git_blob(self):
        with tempfile.TemporaryDirectory() as temporary:
            executed = Path(temporary) / "builder.py"
            raw = b"print('snapshot builder')\n"
            executed.write_bytes(raw)
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            inventory = {builder.BUILDER_SOURCE_PATH: {"git_blob": blob, "git_mode": "100644"}}
            with patch.object(builder, "__file__", str(executed)):
                builder.verify_executed_builder(inventory)
                executed.write_bytes(raw + b"# uncommitted change\n")
                with self.assertRaisesRegex(ValueError, "constructor ejecutado no coinciden"):
                    builder.verify_executed_builder(inventory)


class OptionalPathAndBuildTests(unittest.TestCase):
    def test_only_exact_opted_in_paths_and_sphinx_copies_are_excluded(self):
        records = catalogue()
        for name in PATHS:
            site = Path(*Path(name).parts[1:])
            copies = (Path("source") / name, site, Path("site") / site,
                      Path("_downloads") / ("b" * 32) / site.name)
            for path in copies:
                self.assertIsNone(builder.exclusion(path))
                self.assertIn("optional_scientific", builder.exclusion(path, records))
            for path in (Path("unrelated") / site.name,
                         Path("_static/another_closure") / site.name,
                         Path("_downloads/not-a-sphinx-hash") / site.name):
                self.assertIsNone(builder.exclusion(path, records))
        self.assertIsNone(builder.exclusion(Path("_static/evidence/other.zip"), records))

    def test_site_copy_checks_optional_content_before_omitting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site, stage = root / "html", root / "stage"
            site.mkdir()
            (site / "intro.html").write_text("<html>fixture</html>")
            logical = Path(*Path(PATHS[0]).parts[1:])
            archived = site / logical
            archived.parent.mkdir(parents=True)
            raw = b"verified scientific archive fixture"
            archived.write_bytes(raw)
            download = site / "_downloads" / ("b" * 32) / logical.name
            download.parent.mkdir(parents=True)
            download.write_bytes(raw)
            unrelated = site / "unrelated.zip"
            unrelated.write_bytes(b"retain unrelated ZIP")
            record = {**catalogue()[PATHS[0]], "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()}
            excluded = []
            builder.copy_site(site, stage, excluded, {PATHS[0]: record})
            self.assertEqual(len(excluded), 2)
            self.assertFalse((stage / "site" / logical).exists())
            self.assertEqual((stage / "site/unrelated.zip").read_bytes(), unrelated.read_bytes())
            download.write_bytes(b"a different ZIP with the same filename")
            with self.assertRaisesRegex(ValueError, "copia opcional"):
                builder.copy_site(site, root / "bad-stage", [], {PATHS[0]: record})
            download.unlink()
            download.mkdir()
            with self.assertRaisesRegex(ValueError, "archivo regular"):
                builder.copy_site(site, root / "directory-stage", [], {PATHS[0]: record})

    def test_local_markdown_myst_and_notebook_dependencies_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            book = stage / "source/book_e2"
            section = book / "sections/intro.md"
            section.parent.mkdir(parents=True)
            name = "../" + "/".join(Path(PATHS[0]).parts[1:])
            cases = (f"[archive]({name})", f"{{download}}`archive <{name}>`",
                     f":download:`archive <{name}>`", f"[archive]: {name}",
                     f'<a href="{name}">archive</a>')
            for text in cases:
                section.write_text(text)
                with self.assertRaisesRegex(ValueError, "Referencia local"):
                    builder.check_optional_source_links(stage, catalogue())
            section.unlink()
            notebook = section.with_suffix(".ipynb")
            notebook.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": [cases[1]]}]}))
            with self.assertRaisesRegex(ValueError, "Referencia local"):
                builder.check_optional_source_links(stage, catalogue())

    def test_canonical_remote_source_link_passes_but_mutable_link_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            book = stage / "source/book_e2"
            book.mkdir(parents=True)
            source = book / "intro.md"
            url = catalogue()[PATHS[0]]["url"]
            source.write_text(f"[optional archive]({url})")
            before = source.read_bytes()
            report = builder.check_optional_source_links(stage, catalogue())
            self.assertEqual(report["canonical_optional_links_seen"], [PATHS[0]])
            self.assertEqual(source.read_bytes(), before)
            source.write_text(f"[optional archive]({url.replace(COMMIT, 'main')})")
            with self.assertRaisesRegex(ValueError, "HTTPS inmutable"):
                builder.check_optional_source_links(stage, catalogue())

    def test_compiled_sources_remain_exact_and_local_omitted_html_link_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            source, compiled = stage / "source/book_e2/intro.md", stage / "site/_sources/intro.md"
            source.parent.mkdir(parents=True)
            compiled.parent.mkdir(parents=True)
            source.write_text("exact Git source fixture")
            compiled.write_bytes(source.read_bytes())
            builder.check_compiled_sources(stage)
            compiled.write_text("edited after compilation")
            with self.assertRaisesRegex(ValueError, "ajeno"):
                builder.check_compiled_sources(stage)
            page = stage / "site/intro.html"
            page.write_text(f'<a href="{catalogue()[PATHS[0]]["url"]}">optional</a>')
            reader.validate_html(stage / "site")
            page.write_text(f'<a href="{Path(*Path(PATHS[0]).parts[1:])}">missing</a>')
            with self.assertRaisesRegex(ValueError, "falta"):
                reader.validate_html(stage / "site")

    def test_generated_readme_discloses_optional_downloads_without_changing_default(self):
        default = builder.readme(COMMIT)
        opted_in = builder.readme(COMMIT, catalogue())
        self.assertNotIn("Descargas científicas opcionales", default)
        self.assertIn("no están incluidos en esta carpeta", opted_in)
        self.assertIn("reconstrucción del libro no necesitan estos ZIP", opted_in)
        for record in catalogue().values():
            self.assertIn(record["url"], opted_in)
            self.assertIn(record["sha256"], opted_in)


class ReaderCompatibilityTests(unittest.TestCase):
    def test_v1_v2_verify_without_optional_catalogue_and_v3_discloses_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = root / "site/intro.html"
            page.parent.mkdir()
            page.write_text("<html>offline fixture</html>")
            manifest = {"source_commit": COMMIT,
                        "files": {"site/intro.html": {"bytes": page.stat().st_size,
                                                        "sha256": reader.digest(page)}}}
            for version in ("e2-standalone-book-payload-v1", "e2-standalone-book-payload-v2"):
                manifest["version"] = version
                (root / reader.MANIFEST_NAME).write_text(json.dumps(manifest))
                self.assertNotIn("optional_downloads", reader.verify_files(root, COMMIT))
            manifest.update(version="e2-standalone-book-payload-v3", optional_downloads=list(catalogue().values()))
            (root / reader.MANIFEST_NAME).write_text(json.dumps(manifest))
            report = reader.verify_files(root, COMMIT)
            self.assertEqual(len(report["optional_downloads"]), 2)
            report["html_pages"] = 1
            reader.write_entry(root, report)
            entry = (root / "index.html").read_text()
            self.assertIn("requiere Internet", entry)
            self.assertIn(catalogue()[PATHS[0]]["url"], entry)
            manifest["optional_downloads"][0]["url"] = "https://example.com/changed.zip"
            with self.assertRaisesRegex(ValueError, "HTTPS inmutable"):
                reader.optional_downloads(manifest)


class OptionalJournalTests(unittest.TestCase):
    def setUp(self):
        forbid_git = patch.object(builder, "git", side_effect=AssertionError("Unexpected Git call outside a mock"))
        forbid_git.start()
        self.addCleanup(forbid_git.stop)
        self.roots = {prefix: {**specification, "expected_count": 2}
                      for prefix, specification in builder.OPTIONAL_JOURNAL_ROOTS.items()}
        self.inventory, self.blobs, self.archives, self.journal_bytes = {}, {}, {}, {}
        self.downloads = catalogue()
        for number, (prefix, specification) in enumerate(self.roots.items()):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                for i in range(2):
                    relative = f"{number * 2 + i + 1:064x}/supervised.jsonl.gz"
                    raw = f"opaque inner gzip fixture {number}-{i}".encode()
                    path = prefix + relative
                    self.journal_bytes[path] = raw
                    archive.writestr(specification["member_prefix"] + relative, raw)
            raw = stream.getvalue()
            bundle_path = specification["bundle_repository_path"]
            self.archives[bundle_path] = raw
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            self.inventory[bundle_path] = {"git_blob": blob, "git_mode": "100644"}
            self.blobs[blob] = raw
            self.downloads[bundle_path].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def git(self, *arguments):
        if arguments[:2] != ("cat-file", "blob"):
            self.fail(f"Unexpected Git operation: {arguments}")
        return self.blobs[arguments[2]]

    def read_asset(self, commit, path):
        self.assertEqual(commit, COMMIT)
        return self.archives[path]

    def build_catalogue(self):
        with patch.object(builder, "OPTIONAL_JOURNAL_ROOTS", self.roots), \
                patch.object(builder, "read_git_asset", self.read_asset), \
                patch.object(builder, "git", self.git):
            return builder.optional_journal_catalogue(True, self.downloads, self.inventory)

    def test_journals_bind_zip_source_and_member_without_individual_git_blobs(self):
        self.assertEqual(builder.optional_journal_catalogue(False, {}, {}), {})
        with self.assertRaisesRegex(ValueError, "requiere --optional-download-commit"):
            builder.optional_journal_catalogue(True, {}, self.inventory)
        records = self.build_catalogue()
        self.assertEqual(len(records), 4)
        for path, record in records.items():
            source = self.journal_bytes[path]
            self.assertNotIn(path, self.inventory)
            self.assertIs(record["source_individual_git_present"], False)
            self.assertEqual(record["source_archive_git_blob"], self.inventory[record["bundle_repository_path"]]["git_blob"])
            self.assertEqual(record["sha256"], hashlib.sha256(source).hexdigest())
            self.assertEqual(record["bytes"], len(source))
            with zipfile.ZipFile(io.BytesIO(self.archives[record["bundle_repository_path"]])) as archive:
                self.assertEqual(archive.read(record["member_path"]), source)
            self.assertIsNone(builder.exclusion(Path("source") / path))
            self.assertIn("optional_scientific", builder.exclusion(Path("source") / path, records))

    def test_cli_rejects_journal_opt_in_without_download_commit_before_git(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            builder.main(["--source-commit", COMMIT, "--site-root", "unused-site",
                          "--output", "unused-payload.tar.xz", "--optional-scientific-journals"])
        self.assertEqual(caught.exception.code, 2)

    def test_changed_source_zip_or_false_individual_git_source_claim_fails(self):
        first = next(iter(self.blobs))
        before = self.blobs[first]
        self.blobs[first] += b"changed C1 source"
        with self.assertRaisesRegex(ValueError, "fuente C1 difiere"):
            self.build_catalogue()
        self.blobs[first] = before
        extra = next(iter(self.journal_bytes))
        self.inventory[extra] = {"git_blob": first, "git_mode": "100644"}
        with self.assertRaisesRegex(ValueError, "journals individuales"):
            self.build_catalogue()

    def test_same_named_sphinx_journals_match_content_and_unrelated_gzip_is_retained(self):
        records = self.build_catalogue()
        values = list(records.values())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "html"
            site.mkdir()
            (site / "intro.html").write_text("<html>fixture</html>")
            known = site / "_downloads" / ("b" * 32) / "supervised.jsonl.gz"
            known.parent.mkdir(parents=True)
            record = values[1]
            known.write_bytes(self.journal_bytes[record["repository_path"]])
            other = site / "_downloads" / ("c" * 32) / "supervised.jsonl.gz"
            other.parent.mkdir(parents=True)
            other.write_bytes(b"unrelated same-named gzip")
            excluded = []
            builder.copy_site(site, root / "stage", excluded, records, records)
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["optional_download"]["repository_path"], record["repository_path"])
            self.assertEqual((root / "stage/site" / other.relative_to(site)).read_bytes(), other.read_bytes())

    def test_local_journal_myst_dependency_blocks_rebuild_and_readme_discloses_omission(self):
        records = self.build_catalogue()
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            book = stage / "source/book_e2"
            book.mkdir(parents=True)
            path = next(iter(records))
            local = Path(*Path(path).parts[1:]).as_posix()
            (book / "intro.md").write_text(f"{{download}}`journal <{local}>`")
            with self.assertRaisesRegex(ValueError, "Referencia local"):
                builder.check_optional_source_links(stage, {**self.downloads, **records})
        readme = builder.readme(COMMIT, self.downloads, records)
        self.assertIn("catálogo describe 4 journals gzip", readme)
        self.assertIn("no contiene todos los insumos", readme)

    def test_clean_site_needs_no_journals_and_dirty_static_or_sphinx_copy_requires_opt_in(self):
        records = self.build_catalogue()
        record = next(iter(records.values()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "html"
            site.mkdir()
            (site / "intro.html").write_text("<html>clean source fixture</html>")
            result = builder.copy_site(site, root / "clean-stage", [], None, records)
            self.assertEqual(result["files"], 1)
            static = site / Path(*Path(record["repository_path"]).parts[1:])
            static.parent.mkdir(parents=True)
            static.write_bytes(self.journal_bytes[record["repository_path"]])
            with self.assertRaisesRegex(ValueError, "Journal residual"):
                builder.copy_site(site, root / "dirty-stage", [], None, records)
            static.unlink()
            sphinx = site / "_downloads" / ("b" * 32) / static.name
            sphinx.parent.mkdir(parents=True)
            sphinx.write_bytes(self.journal_bytes[record["repository_path"]])
            with self.assertRaisesRegex(ValueError, "Sphinx residual"):
                builder.copy_site(site, root / "sphinx-stage", [], None, records)
            excluded = []
            builder.copy_site(site, root / "opted-stage", excluded, records, records)
            self.assertEqual(len(excluded), 1)
            self.assertFalse((root / "opted-stage/site" / sphinx.relative_to(site)).exists())

    def test_reader_requires_exact_206_plus_90_member_bindings(self):
        downloads = catalogue()
        records = []
        for prefix, (bundle_path, member_prefix, count) in reader.OPTIONAL_JOURNAL_ROOTS.items():
            bundle = downloads[bundle_path]
            for i in range(count):
                relative = f"{i:064x}/supervised.jsonl.gz"
                path = prefix + relative
                records.append({"repository_path": path, "bytes": 1,
                                "sha256": hashlib.sha256(path.encode()).hexdigest(),
                                "download_commit": COMMIT, "url": bundle["url"],
                                "bundle_repository_path": bundle_path,
                                "bundle_bytes": bundle["bytes"], "bundle_sha256": bundle["sha256"],
                                "source_individual_git_present": False,
                                "source_archive_git_blob": "d" * 40,
                                "source_binding": "Git-tracked ZIP member; no individual journal Git source is claimed",
                                "member_path": member_prefix + relative, "reason": "fixture"})
        manifest = {"version": "e2-standalone-book-payload-v3", "optional_scientific_journals": records}
        self.assertEqual(len(reader.optional_scientific_journals(manifest, list(downloads.values()))), 296)
        with self.assertRaisesRegex(ValueError, "dos ZIP"):
            reader.optional_scientific_journals(manifest, [])
        manifest["optional_scientific_journals"] = records[:-1]
        with self.assertRaisesRegex(ValueError, "206\\+90"):
            reader.optional_scientific_journals(manifest, list(downloads.values()))
        manifest["optional_scientific_journals"] = records
        records[0]["member_path"] = "unrelated/journal.gz"
        with self.assertRaisesRegex(ValueError, "miembro ZIP"):
            reader.optional_scientific_journals(manifest, list(downloads.values()))


if __name__ == "__main__":
    unittest.main()
