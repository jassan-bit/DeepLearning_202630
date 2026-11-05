"""Protection and preservation checks for publication; no model/data execution."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_books_site import assemble
from check_book_links import check_html


class MultiBookPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for number, name in enumerate(("book", "book_e2", "book_e3"), 1):
            source = self.root / name
            html = source / "_build" / "html"
            html.mkdir(parents=True)
            (source / "_config.yml").write_text(f'title: "Entregable {number}"\n')
            (source / "_toc.yml").write_text("format: jb-book\nroot: intro\n")
            (html / "intro.html").write_text(
                f'<html><body><h1 id="title">E{number}</h1><a href="index.html">Inicio</a></body></html>')
            (html / "index.html").write_text('<html><a href="intro.html#title">Libro</a></html>')
            (html / "objects.inv").write_bytes(b"inventory")
            (html / "searchindex.js").write_text("Search.setIndex({});")

    def test_combined_site_preserves_e1_and_independent_subbooks(self):
        before = (self.root / "book/_build/html/intro.html").read_bytes()
        result = assemble(self.root, self.root / "_build/site", "a" * 40)
        output = Path(result["output_dir"])
        self.assertEqual((output / "intro.html").read_bytes(), before)
        self.assertEqual(result["e1_intro_sha256_preserved"], sha256(before).hexdigest())
        for name, expected in (("entregable2", "E2"), ("entregable3", "E3")):
            self.assertIn(expected, (output / name / "intro.html").read_text())
        manifest = json.loads((output / "books_manifest.json").read_text())
        self.assertEqual([b["entrypoint"] for b in manifest["books"]],
                         ["intro.html", "entregable2/intro.html", "entregable3/intro.html"])
        self.assertEqual(manifest["new_model_fits"], 0)
        self.assertEqual(check_html(output)["html_pages"], 7)

    def test_missing_book_does_not_create_partial_output(self):
        (self.root / "book_e3/_build/html/intro.html").unlink()
        output = self.root / "not-created" / "site"
        with self.assertRaisesRegex(ValueError, "Incomplete HTML"):
            assemble(self.root, output)
        self.assertFalse(output.parent.exists())

    def test_existing_content_is_not_deleted_or_replaced(self):
        output = self.root / "previous-site"
        output.mkdir()
        (output / "keep.txt").write_text("Previous publication")
        with self.assertRaisesRegex(ValueError, "existing contents"):
            assemble(self.root, output)
        self.assertEqual((output / "keep.txt").read_text(), "Previous publication")
        self.assertFalse((output / "intro.html").exists())

    def test_source_and_repository_cannot_be_output(self):
        for output in (self.root, self.root / "book", self.root / "book_e2/_build/replacement"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                assemble(self.root, output)
        self.assertTrue((self.root / "book/_build/html/intro.html").is_file())

    def test_book_local_link_cannot_depend_on_a_sibling_book(self):
        html = self.root / "book_e2/_build/html"
        outside = self.root / "e1-only.html"
        outside.write_text("External local file")
        (html / "intro.html").write_text('<a href="../../../../e1-only.html">Dependent</a>')
        with self.assertRaisesRegex(ValueError, "escapes this HTML tree"):
            check_html(html)

    def test_compiled_symlink_is_refused_without_copying_it(self):
        target = self.root / "private.txt"
        target.write_text("Not a publication asset")
        (self.root / "book_e2/_build/html/symlink.txt").symlink_to(target)
        output = self.root / "site"
        with self.assertRaisesRegex(ValueError, "symlinks"):
            assemble(self.root, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
