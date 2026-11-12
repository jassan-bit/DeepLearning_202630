"""Acquire hash-pinned architecture evidence and author code, without market data."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

try:
    from . import download_e2_completion_sources as acquisition
except ImportError:
    import download_e2_completion_sources as acquisition

ROOT = Path(__file__).resolve().parents[1]
MODELS = frozenset(("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "iTransformer"))


def acquire(output):
    """Preserve every existing unequal source file and fail before importing it."""
    design = json.loads((ROOT / "experiments/e2_completion_20261106/design.json").read_text())
    design["source_locations"] = {n: row for n, row in design["source_locations"].items() if n in MODELS}
    design["author_sources"] = [row for row in design["author_sources"] if row["model"] in MODELS]
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError("The author destination cannot be a symbolic link")
    receipt = acquisition.acquire(design, output)
    extras = json.loads((ROOT / "experiments/e2_published_architectures_20261112/supporting_sources.json").read_text())
    checked = []
    for row in extras["files"]:
        target = acquisition.destination(output, row["local_path"])
        data = target.read_bytes() if target.exists() else acquisition.fetch(row["url"])
        if len(data) != row["bytes"] or sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Supporting author evidence changed: " + row["local_path"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        checked.append({k: row[k] for k in ("local_path", "sha256", "bytes", "url")})
    result = {"version": "e2-published-source-acquisition-v1",
              "author_files_verified": len(receipt["verified_sources"]),
              "supporting_files_verified": checked, "market_data_read": False,
              "model_imported": False, "weights_downloaded": False, "training_started": False,
              "roots_file": str(output / "source_roots.json")}
    (output / "published_source_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = acquire(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
