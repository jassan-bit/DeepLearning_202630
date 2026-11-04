"""Package existing development run artifacts without opening market data."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/dl_benchmark_20261104"


def main():
    files = sorted(p for p in (RESULTS / "runs").rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in files):
        raise ValueError("Expected existing regular run files; no fitting is performed")
    entries = {
        str(p.relative_to(RESULTS)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in files
    }
    archive = RESULTS / "run_artifacts.tar.gz"
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as bundle:
                for p in files:
                    info = bundle.gettarinfo(str(p), arcname=str(p.relative_to(RESULTS)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with p.open("rb") as source:
                        bundle.addfile(info, source)
    manifest = {
        "scope": "Existing development run files, not a new experiment or fit",
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "file_count": len(entries),
        "checkpoint_count": sum(p.name == "model.pt" for p in files),
        "checkpoint_policy": "Pilot and development benchmark only; search logs and predictions retained without checkpoints",
        "files": entries,
    }
    (RESULTS / "run_artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}))


if __name__ == "__main__":
    main()
