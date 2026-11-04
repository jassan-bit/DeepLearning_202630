"""Retain every new CV/final/test run in verifiable, size-bounded archives."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest_path = OUT / "run_artifacts_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest["files_sha256"].items():
            if digest(OUT / name) != expected:
                raise ValueError("Retained run changed: " + name)
        for record in manifest["archives"]:
            if digest(OUT / record["archive"]) != record["sha256"]:
                raise ValueError("Retained archive changed")
        print(json.dumps({"status": "retained_artifacts_verified", "files": len(manifest["files_sha256"])}))
        return
    cv = json.loads((OUT / "execution_summary.json").read_text())
    test = json.loads((OUT / "test/execution_summary.json").read_text())
    if (not cv["all_registered_trajectories_complete"] or test["model_runs"] != 120
            or test["actual_checkpoint_forward_calls"] != 120):
        raise ValueError("Complete CV and the single-pass final test before packaging")
    jobs = []
    for history in (14, 21, 28):
        directories = sorted((OUT / "runs").glob(f"relative_window_*_D{history}_s*"))
        if len(directories) != 30:
            raise ValueError("Thirty complete trajectories required per new D")
        jobs.append((f"cv_D{history}_artifacts.tar.gz", directories))
    jobs += [("final_fit_artifacts.tar.gz", [OUT / "final_fit"]),
             ("test_predictions_by_run.tar.gz", [OUT / "test/runs"])]
    records, all_hashes = [], {}
    for archive_name, directories in jobs:
        archive_path = OUT / archive_name
        if archive_path.exists():
            raise FileExistsError("Archive exists without a complete retention manifest; audit before replacing it")
        files = sorted(path for directory in directories for path in directory.rglob("*") if path.is_file())
        with archive_path.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w") as tar:
                    for path in files:
                        name = path.relative_to(OUT).as_posix()
                        all_hashes[name] = digest(path)
                        metadata = tarfile.TarInfo(name=name)
                        metadata.size = path.stat().st_size
                        metadata.mode = 0o644
                        metadata.mtime = 0
                        with path.open("rb") as stream:
                            tar.addfile(metadata, stream)
        if archive_path.stat().st_size >= 90*1024*1024:
            raise ValueError("Split retention archives before publishing files >=90MiB")
        records.append({"archive": archive_name, "files": len(files), "bytes": archive_path.stat().st_size,
                        "sha256": digest(archive_path), "torch_checkpoints": sum(p.suffix == ".pt" for p in files)})
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "All new D14/D21/D28 CV and all four-D final-fit/test run artifacts; original D7 CV remains in its signed original archive.",
                "archives": records, "files_sha256": all_hashes,
                "cv_new_project_fits": cv["completed_project_fits"], "final_development_fits": 120,
                "test_checkpoint_forward_calls": 120, "test_period_model_refits": 0}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": "all_run_artifacts_retained", "files": len(all_hashes), "archives": records}))


if __name__ == "__main__":
    main()
