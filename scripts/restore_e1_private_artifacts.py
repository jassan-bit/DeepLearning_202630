"""Restore the completed E1 experiment in an isolated, private directory.

This utility never fits a model, executes checkpoint inference or reads Binance
market ZIPs. It copies a private evidence package, checks archive identities and
each regular-file member, and can run the independent saved-result auditor.
The signed original experiment and its first-test-access marker remain intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("results/dl_r2_cv_20261104")
NEW = Path("results/filipe_multihistory_20261104")
EXPERIMENT = Path("experiments/filipe_multihistory_20261104")
FREEZE_SHA256 = "7c789040819598a4339b60c7be56e104b82ef94dbba6b70ae904128387f47837"
BUNDLE_MANIFEST = "private_e1_bundle_manifest.json"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def relative_file(name):
    """Accept only a canonical POSIX path to a file inside the package root."""
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        raise ValueError("Invalid package member name")
    candidate = PurePosixPath(name)
    if (candidate.is_absolute() or not candidate.parts or any(part in {".", ".."} for part in candidate.parts)
            or candidate.as_posix() != name or ":" in candidate.parts[0]):
        raise ValueError("Package members must be canonical relative file names: " + name)
    return Path(*candidate.parts)


def check_destination(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a separate output directory outside the source checkout/package")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Output directory must be absent or empty; existing evidence is never overwritten")
    if any(part in {"_static", "_build", "gh-pages"} for part in destination.parts):
        raise ValueError("Private evidence cannot be restored into a publication/build directory")
    # Managed execution roots may contain the platform's own read-only Git
    # metadata; sibling task evidence belongs outside the actual project checkout.
    platform_metadata_roots = {Path("/workspace"), Path("/tmp")}
    if any((parent / ".git").exists() and parent not in platform_metadata_roots
           for parent in (destination, *destination.parents)):
        raise ValueError("Private evidence must remain outside every Git checkout")
    return source, destination


def archive_catalog(source):
    """Return the six signed run archives and their member checksums."""
    source = Path(source)
    old = read_json(source / OLD / "run_artifacts_manifest.json")
    entries = [{"path": (OLD / old["archive"]).as_posix(), "base": OLD.as_posix(),
                "sha256": old["archive_sha256"],
                "members": {name: value["sha256"] for name, value in old["files"].items()}}]
    current = read_json(source / NEW / "run_artifacts_manifest.json")
    all_members = current["files_sha256"]
    for record in current["archives"]:
        name = record["archive"]
        if name.startswith("cv_D"):
            history = name.removeprefix("cv_D").split("_", 1)[0]
            members = {path: checksum for path, checksum in all_members.items()
                       if path.startswith("runs/") and f"_D{history}_s" in path.split("/")[1]}
        elif name == "final_fit_artifacts.tar.gz":
            members = {path: checksum for path, checksum in all_members.items() if path.startswith("final_fit/")}
        elif name == "test_predictions_by_run.tar.gz":
            members = {path: checksum for path, checksum in all_members.items() if path.startswith("test/runs/")}
        else:
            raise ValueError("Unexpected retention archive: " + name)
        if len(members) != record["files"]:
            raise ValueError("Archive member grid differs from the retention manifest")
        entries.append({"path": (NEW / name).as_posix(), "base": NEW.as_posix(),
                        "sha256": record["sha256"], "members": members})
    if len(entries) != 6 or len({entry["path"] for entry in entries}) != 6:
        raise ValueError("The original D7 archive and all five extension archives are required")
    return entries


def verify_archive(path, expected_sha256, members, *, destination=None):
    """Verify bytes and members before optional safe extraction; never extractall."""
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch: " + str(path))
    for name in members:
        relative_file(name)
    seen = set()
    # A complete validation pass precedes every write.
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            relative_file(member.name)
            if not member.isfile() or member.name in seen or member.name not in members:
                raise ValueError("Unexpected, duplicate or non-regular archive member: " + member.name)
            seen.add(member.name)
            checksum = hashlib.sha256()
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing regular-file payload")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
            if checksum.hexdigest() != members[member.name]:
                raise ValueError("Archived member SHA-256 mismatch: " + member.name)
    if seen != set(members):
        raise ValueError("Missing files in the signed archive")
    if destination is not None:
        destination = Path(destination).resolve()
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                output = destination / relative_file(member.name)
                if not output.resolve().is_relative_to(destination) or output.exists():
                    raise ValueError("Restoration would escape or overwrite the destination")
                output.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing regular-file payload")
                with output.open("xb") as target:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
                if digest(output) != members[member.name]:
                    raise ValueError("Restored member differs from its signed bytes")
    return len(seen)


def source_catalog(source):
    source = Path(source)
    required = [EXPERIMENT / "protocol.json", EXPERIMENT / "freeze.json",
                OLD / "run_artifacts_manifest.json", NEW / "run_artifacts_manifest.json",
                NEW / "test/predictions.csv", NEW / "test/daily_market_test.csv",
                NEW / "test/daily_volatility_test.csv", NEW / "test/raw_access_started.json",
                NEW / "test/raw_test_audit.json", NEW / "test/cohort_manifest.json",
                NEW / "final_models_manifest.json", NEW / "final_test_events.jsonl",
                Path("results/eda_development/daily_market_development.csv"),
                Path("results/eda_development/daily_volatility_development.csv"),
                Path("scripts/verify_filipe_multihistory.py"), Path("scripts/restore_e1_private_artifacts.py"),
                Path("data/manifests/test_raw_reserve_20250101_20260831.json")]
    absent = [path.as_posix() for path in required if not (source / path).is_file()]
    if absent:
        raise FileNotFoundError("Incomplete private evidence package. Public summaries are insufficient; obtain the private package. Missing: " + ", ".join(absent))
    if digest(source / EXPERIMENT / "freeze.json") != FREEZE_SHA256:
        raise ValueError("The private package does not match the published configuration freeze")
    freeze = read_json(source / EXPERIMENT / "freeze.json")
    code = set(freeze["code_sha256"])
    code.update(read_json(source / OLD / "code_manifest_before_fitting.json")["code_sha256"])
    code.update({"scripts/verify_filipe_multihistory.py", "scripts/restore_e1_private_artifacts.py"})
    selected = set(required)
    selected.update(Path(name) for name in code)
    for base in (OLD, NEW):
        for path in (source / base).rglob("*"):
            relative = path.relative_to(source / base)
            if (not path.is_file() or path.is_symlink() or any(part in {"runs", "final_fit", "incomplete_preserved"} for part in relative.parts)
                    or path.suffix not in {".csv", ".json", ".jsonl", ".png", ".md", ".txt", ".gz"}):
                continue
            selected.add(path.relative_to(source))
    selected.update(path.relative_to(source) for path in source.glob("requirements*.txt") if path.is_file())
    selected.add(Path("BUILD.md"))
    # Required identities include additional manifests/sources named in the frozen records.
    for mapping in (freeze["code_sha256"], freeze["source_cv_sha256"], freeze["analysis_cv_sha256"]):
        for name, checksum in mapping.items():
            path = relative_file(name)
            if digest(source / path) != checksum:
                raise ValueError("Signed source changed: " + name)
            selected.add(path)
    plan = read_json(source / EXPERIMENT / "protocol.json")
    for name, checksum in plan["sources"].items():
        path = relative_file(name)
        if digest(source / path) != checksum:
            raise ValueError("Registered development input changed: " + name)
        selected.add(path)
    entries = archive_catalog(source)
    selected.update(Path(entry["path"]) for entry in entries)
    catalog = {}
    for relative in sorted(selected):
        path = source / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Missing or non-regular private input: " + str(relative))
        catalog[relative.as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    existing = source / BUNDLE_MANIFEST
    if existing.exists():
        bundle = read_json(existing)
        if bundle.get("freeze_sha256") != FREEZE_SHA256:
            raise ValueError("Bundle freeze differs")
        for name, expected in bundle["files"].items():
            relative_file(name)
            if name not in catalog or digest(source / name) != expected["sha256"]:
                raise ValueError("Private bundle member changed: " + name)
    return catalog, entries


def build_private_copy(source, output, *, mode, verify=False):
    if mode not in {"bundle", "restore"} or (verify and mode != "restore"):
        raise ValueError("Verification requires restore mode; choose bundle or restore before writing")
    source, output = check_destination(source, output)
    catalog, archives = source_catalog(source)
    counts = []
    for entry in archives:
        count = verify_archive(source / entry["path"], entry["sha256"], entry["members"])
        counts.append({"archive": Path(entry["path"]).name, "members": count, "sha256": entry["sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in catalog.items():
        target = output / relative_file(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if digest(target) != expected["sha256"]:
            raise ValueError("Private input changed while copying: " + name)
    bundle = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Private evidence only; never publish this directory or its full archive as book static assets.",
              "experiment": "filipe_multihistory_20261104", "freeze_sha256": FREEZE_SHA256,
              "files": catalog, "archives": counts, "raw_market_archives_included": 0,
              "new_model_fits": 0, "new_checkpoint_forward_calls": 0}
    write_json(output / BUNDLE_MANIFEST, bundle)
    if mode == "restore":
        for entry in archives:
            verify_archive(output / entry["path"], entry["sha256"], entry["members"], destination=output / entry["base"])
    elif mode != "bundle":
        raise ValueError("Unknown copy mode")
    audit_status, audit_report = None, None
    if verify:
        if mode != "restore":
            raise ValueError("--verify requires restore mode")
        audit_report = output / "verification/independent_saved_result_audit.json"
        subprocess.run([sys.executable, str(output / "scripts/verify_filipe_multihistory.py"),
                        "--phase", "test", "--report", str(audit_report)], cwd=output, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        audit_status = read_json(audit_report)["status"]
        if audit_status != "PASS":
            raise ValueError("Restored evidence failed the independent saved-result audit")
    # Recheck source files: restoration must never alter original evidence.
    if any(digest(source / name) != entry["sha256"] for name, entry in catalog.items()):
        raise ValueError("Original evidence changed during private restoration")
    receipt = {"status": "PASS", "mode": mode, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_sha256": FREEZE_SHA256, "bundle_manifest_sha256": digest(output / BUNDLE_MANIFEST),
               "source_bundle_manifest_sha256": digest(source / BUNDLE_MANIFEST) if (source / BUNDLE_MANIFEST).is_file() else None,
               "copied_input_files": len(catalog), "retention_archives_checked": len(archives),
               "restored_artifact_members": sum(item["members"] for item in counts) if mode == "restore" else 0,
               "independent_audit": audit_status, "independent_audit_sha256": digest(audit_report) if audit_report else None,
               "python_version": sys.version.split()[0], "source_evidence_unchanged": True,
               "new_model_fits": 0, "new_checkpoint_forward_calls": 0, "raw_market_archives_opened": 0,
               "privacy_scope": "The receipt contains counts and hashes only. The associated evidence directory remains private."}
    write_json(output / "replication_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Complete local checkout or private handoff package")
    parser.add_argument("--output-dir", type=Path, required=True, help="Absent/empty private directory outside the source")
    parser.add_argument("--mode", choices=("bundle", "restore"), default="restore")
    parser.add_argument("--verify", action="store_true", help="Run the independent saved-result audit inside the restored copy")
    args = parser.parse_args()
    if args.verify and args.mode != "restore":
        parser.error("--verify requires --mode restore")
    result = build_private_copy(args.source_root, args.output_dir, mode=args.mode, verify=args.verify)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
