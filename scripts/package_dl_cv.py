"""Archive existing monthly CV artifacts; no fitting or market-data access."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/dl_r2_cv_20261104"


def main():
    files = sorted(p for p in (RESULTS / "runs").rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in files):
        raise ValueError("Existing regular CV run files required")
    entries = {str(p.relative_to(RESULTS)): {
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        for p in files}
    path = RESULTS / "run_artifacts.tar.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for p in files:
                    info = archive.gettarinfo(str(p), arcname=str(p.relative_to(RESULTS)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    with p.open("rb") as source:
                        archive.addfile(info, source)
    manifest = {"scope": "Complete existing CV artifacts, not a new experiment",
        "archive": path.name, "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_bytes": path.stat().st_size, "file_count": len(files),
        "checkpoint_count": sum(p.name == "model.pt" for p in files), "files": entries}
    (RESULTS / "run_artifacts_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}))


if __name__ == "__main__":
    main()
