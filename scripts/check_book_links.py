"""Validate local HTML href/src targets and anchors after the book build."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

root = Path(__file__).resolve().parents[1] / "book" / "_build" / "html"

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

pages = {}
for path in root.rglob("*.html"):
    if "_static" in path.relative_to(root).parts:
        continue  # Theme distribution includes unrendered Jinja template fragments.
    parser = Page()
    parser.feed(path.read_text(encoding="utf-8"))
    pages[path.resolve()] = parser
assert pages, f"No HTML found in {root}; build first."
errors = []
checked = 0
for path, page in pages.items():
    for link in page.links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc:
            continue
        target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
        if target.is_dir():
            target /= "index.html"
        if not target.exists():
            errors.append(f"{path.name}: missing {link}")
        elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
            errors.append(f"{path.name}: missing anchor {link}")
        checked += 1
assert not errors, "\n".join(errors)
print(f"OK: {len(pages)} HTML pages; {checked} local links/assets checked.")
