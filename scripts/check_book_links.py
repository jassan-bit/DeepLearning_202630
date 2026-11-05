"""Validate one standalone book or the combined site's local links and anchors."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("href", "src"):
            if attr in attrs:
                self.links.append(attrs[attr])

def check_html(html_dir):
    root = Path(html_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"No HTML directory found at {root}; build first.")
    pages = {}
    for path in root.rglob("*.html"):
        if "_static" in path.relative_to(root).parts:
            continue  # Theme assets include unrendered Jinja template fragments.
        parser = Page()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    if not pages:
        raise ValueError(f"No HTML found in {root}; build first.")
    errors, checked = [], 0
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root):
                errors.append(f"{path.relative_to(root)}: local link escapes this HTML tree: {link}")
            elif not target.exists():
                errors.append(f"{path.relative_to(root)}: missing {link}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                errors.append(f"{path.relative_to(root)}: missing anchor {link}")
            checked += 1
    if errors:
        raise ValueError("\n".join(errors))
    return {"html_pages": len(pages), "local_links_and_assets_checked": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--book-dir", type=Path,
                        help="Source book directory; defaults to this repository's book/.")
    inputs.add_argument("--html-dir", type=Path,
                        help="Built HTML tree, including an assembled multi-book site.")
    args = parser.parse_args()
    html_dir = args.html_dir or (args.book_dir or PROJECT_ROOT / "book") / "_build" / "html"
    try:
        result = check_html(html_dir)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"OK: {result['html_pages']} HTML pages; "
          f"{result['local_links_and_assets_checked']} local links/assets checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
