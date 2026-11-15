"""Recover only hash-bound deployed E2 reproduction artifacts, without unpickling.

Downloads public, exact manifest URLs to a new local corrections directory.
Never reads credentials, changes source artifacts, imports Torch or fits models.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import time
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
import uuid
import zipfile

REPOSITORY = "jassan-bit/DeepLearning_202630"
SOURCE_COMMIT = "0bdbc42bd724c6ffbbb1dec67564327adedb9883"
SCIENTIFIC_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
PRODUCER_SNAPSHOT_COMMIT = "d604ede6bf0978e14a1b1d7bafaae6250fb8ed07"


def stamp(): return datetime.now(timezone.utc).isoformat()
def digest(path): return sha256(Path(path).read_bytes()).hexdigest()

def readj(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def save_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("Existing different recovery metadata retained: " + str(path))
    else:
        with path.open("xb") as stream: stream.write(raw)


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError("Unsafe manifest artifact path")
    return p


def local_json_index(source):
    index = {}
    for path in source.rglob("*.json"):
        raw = path.read_bytes()
        index.setdefault(sha256(raw).hexdigest(), (path, None))
    for path in source.rglob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.endswith((".json", ".jsonl")):
                    continue
                if member.file_size > 8 * 1024 * 1024:
                    continue
                raw = archive.read(member)
                index.setdefault(sha256(raw).hexdigest(), (path, member.filename))
    return index


def retrieve(row, destination, index):
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected, count = row["sha256"], row["bytes"]
    if destination.exists():
        if destination.stat().st_size == count and digest(destination) == expected:
            return {"status": "verified_existing", "path": str(destination), "sha256": expected, "bytes": count}
        raise ValueError("Existing different file retained: " + str(destination))
    source = index.get(expected)
    if source:
        path, member = source
        if member is None: raw = path.read_bytes()
        else:
            with zipfile.ZipFile(path) as archive: raw = archive.read(member)
        if len(raw) != count or sha256(raw).hexdigest() != expected:
            raise ValueError("Local artifact differs despite index identity")
        with destination.open("xb") as stream: stream.write(raw)
        return {"status": "verified_local", "path": str(destination), "sha256": expected, "bytes": count}
    if row["storage"] == "release_asset":
        url = row["url"]
    elif row["storage"] == "git_blob":
        # The immutable producer snapshot locates the manifest-bound blob.
        # SHA256/byte verification remains authoritative, not branch content.
        relative = safe_relative(row["remote_path"]).as_posix()
        url = "https://raw.githubusercontent.com/" + REPOSITORY + "/" + PRODUCER_SNAPSHOT_COMMIT + "/" + relative
    else:
        raise ValueError("Local archive member unavailable: " + row["path"])
    parsed = urlsplit(url)
    allowed = (parsed.hostname == "github.com" and parsed.path.startswith("/" + REPOSITORY + "/releases/download/")) or (parsed.hostname == "raw.githubusercontent.com" and parsed.path.startswith("/" + REPOSITORY + "/" + PRODUCER_SNAPSHOT_COMMIT + "/"))
    if parsed.scheme != "https" or not allowed or parsed.query:
        raise ValueError("Unexpected download URL outside exact registered release")
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    started = time.perf_counter()
    try:
        h, received = sha256(), 0
        request = Request(url, headers={"User-Agent": "E2-Hash-Bound-Corrections-Recovery/1.0"})
        with urlopen(request, timeout=45) as response, temporary.open("xb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                received += len(chunk)
                if received > count: raise ValueError("Response larger than registered bytes")
                h.update(chunk); stream.write(chunk)
        if received != count or h.hexdigest() != expected:
            raise ValueError("Downloaded artifact bytes/SHA256 differ")
        os.replace(temporary, destination)
        index.setdefault(expected, (destination, None))
    finally:
        if temporary.exists(): temporary.unlink()
    return {"status": "verified_download", "path": str(destination), "sha256": expected,
            "bytes": count, "download_seconds": time.perf_counter() - started, "url": url}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    project, source, output = args.project_root.resolve(), args.source_metadata.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing_index = output / "metadata/checkpoint_index.json"
    if existing_index.exists():
        previous = readj(existing_index)
        if previous.get("status") != "verified":
            raise ValueError("Retain the partial index under a new archival name before continuing recovery")
        for row in previous["rows"]:
            for label, expected in row["expectedsha"].items():
                path = Path(row[label + "_path"])
                receipt = row["artifact_status"][label]
                if not path.is_file() or path.stat().st_size != receipt["bytes"] or digest(path) != expected:
                    raise ValueError("Previously verified recovery changed: " + str(path))
        print(json.dumps({"status":"verified_existing_index","checkpoints_verified":len(previous["rows"]),
                          "index":str(existing_index)}),flush=True)
        return 0
    started_at, started = stamp(), time.perf_counter()
    frozen = readj(project / "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json")
    trials = {r["model"]: r["trial"] for r in frozen}
    jobs = []
    for path in sorted(source.glob("metadata/reproduction/jobs/*/result.json")):
        result = readj(path)
        if result.get("kind") != "reproduction": continue
        manifest_path = source / "runs" / result["attempt_id"] / "manifest.json"
        manifest = readj(manifest_path)
        artifacts = manifest["artifacts"]
        deployed = [r for r in artifacts if r["path"] in ("fresh/fit/selected_model.pt", "fresh/predictor.pkl", "fresh/predictor.json")]
        if len(deployed) != 1: raise ValueError("One deployable predictor required per reproduction")
        desired = {"checkpoint": deployed[0]}
        for label, relative in (("scaler", "fresh/scaler.json"), ("calibrator", "fresh/calibrator.json"),
                                ("calibration_predictions", "fresh/calibration_predictions.json"),
                                ("test_predictions", "test_predictions.json")):
            rows = [r for r in artifacts if r["path"] == relative]
            if label == "calibration_predictions" and not rows:
                rows = [r for r in artifacts if r["path"] == "fresh/predictions.json"]
            if len(rows) != 1: raise ValueError("One bound artifact required: " + relative)
            desired[label] = rows[0]
        jobs.append({"model": result["model"], "seed": result["seed"], "attempt_id": result["attempt_id"],
            "source_result": str(path), "source_result_sha256": digest(path),
            "source_manifest": str(manifest_path), "source_manifest_sha256": digest(manifest_path),
            "trial": trials.get(result["model"]), "artifact_rows": desired,
            "recorded_original_prediction_max_difference": result.get("original_read_only_prediction_maximum_difference"),
            "original_freeze_unchanged": result.get("original_freeze_unchanged")})
    if len(jobs) != 81: raise ValueError("Exactly 81 frozen reproduction predictors required")
    save_new(output / "metadata/recovery_plan.json", {"version": "e2-corrections-recovery-plan-v1", "source_commit": SOURCE_COMMIT,
        "scientific_commit": SCIENTIFIC_COMMIT, "predictors": len(jobs),
        "deployable_bytes": sum(j["artifact_rows"]["checkpoint"]["bytes"] for j in jobs), "jobs": jobs})
    local = local_json_index(source)
    archive_pointers = output / "metadata/release_archive_assets.json"
    if archive_pointers.exists():
        for archive in readj(archive_pointers)["archives"]:
            target = output / "metadata_archives" / (archive["sha256"] + ".zip")
            retrieve({**archive, "storage": "release_asset"}, target, local)
            with zipfile.ZipFile(target) as saved:
                for member in saved.infolist():
                    if not member.is_dir() and member.filename.endswith((".json", ".jsonl")):
                        raw = saved.read(member)
                        local.setdefault(sha256(raw).hexdigest(), (target, member.filename))
    outcomes = {}
    # Small bound JSON first, so data auditing can start before checkpoint downloads finish.
    def recover_small(job, label, artifact):
        target = output / "attempts" / job["attempt_id"] / safe_relative(artifact["path"])
        try: return retrieve(artifact, target, local)
        except Exception as exc: return {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(recover_small, job, label, artifact): (job["attempt_id"], label)
                   for job in jobs for label, artifact in job["artifact_rows"].items() if label != "checkpoint"}
        for future in as_completed(pending): outcomes[pending[future]] = future.result()
    def recover_checkpoint(job):
        artifact = job["artifact_rows"]["checkpoint"]
        target = output / "attempts" / job["attempt_id"] / safe_relative(artifact["path"])
        try: return retrieve(artifact, target, local)
        except Exception as exc: return {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(recover_checkpoint, j): j for j in jobs}
        for future in as_completed(pending):
            job = pending[future]; result = future.result(); outcomes[(job["attempt_id"], "checkpoint")] = result
            print(json.dumps({"model": job["model"], "seed": job["seed"], "status": result["status"]}), flush=True)
    rows = []
    for job in jobs:
        row = {k: v for k, v in job.items() if k != "artifact_rows"}
        row["expectedsha"] = {label: a["sha256"] for label, a in job["artifact_rows"].items()}
        row["artifact_status"] = {}
        for label, artifact in job["artifact_rows"].items():
            recovered = outcomes[(job["attempt_id"], label)]
            row[label + "_path"] = recovered.get("path")
            row[label + "_sha256"] = artifact["sha256"]
            row["artifact_status"][label] = recovered
        rows.append(row)
    verified = sum(r["artifact_status"]["checkpoint"]["status"].startswith("verified_") for r in rows)
    all_verified = all(o["status"].startswith("verified_") for o in outcomes.values())
    report = {"version": "e2-corrections-checkpoint-index-v1", "source_commit": SOURCE_COMMIT,
        "scientific_commit": SCIENTIFIC_COMMIT, "started_at_utc": started_at, "completed_at_utc": stamp(),
        "elapsed_wall_seconds": time.perf_counter() - started, "status": "verified" if all_verified and verified == 81 else "partial",
        "predictors_expected": 81, "checkpoints_verified": verified, "rows": rows,
        "scope": {"files_unpickled": False, "model_inference_performed": False, "model_fitting_performed": False,
                  "reproduction_checkpoint_hashes_are_original_freeze_hashes": False,
                  "historical_original_state_equivalence_is_bound_by_reproduction_receipts": True}}
    save_new(output / "metadata/checkpoint_index.json", report)
    print(json.dumps({"status": report["status"], "checkpoints_verified": verified,
        "index": str(output / "metadata/checkpoint_index.json"), "elapsed_wall_seconds": report["elapsed_wall_seconds"]}), flush=True)
    return int(report["status"] != "verified")


if __name__ == "__main__": raise SystemExit(main())
