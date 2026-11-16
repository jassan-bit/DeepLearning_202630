"""Audit the E2 handoff to E3 without fitting, inference, or unpickling weights.

The audit distinguishes release metadata, downloaded bytes, and scientific controls.
It cannot certify absence of leakage from dates, hashes, or reproduction receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from urllib.request import urlopen
import zipfile

SOURCE_COMMIT = "f65b14d38e38f8d5fd80c717d71f2abd9ea708e2"
PUBLIC = Path("book_e2/_static/e2_benchmark_final_20261113")
OUTPUT = Path("experiments/e3_bloques_ab_20261115")
RECEIPTS = OUTPUT / "source_receipts"
PARTITION = Path("book_e2/_static/e2_eda_current_v1/partition_audit.json")
GUARDS = Path("book_e2/_static/e2_data_training_uncertainty_current_v1/targeted_data_guard_checks.json")
ENVIRONMENT = Path("book_e2/_static/e2_repro_prereg_current_v1/environment_observed.json")
REPRO_DOC = Path("book_e2/sections/08_reproducibilidad.md")
REPRO_DOC_SNAPSHOT = RECEIPTS / "audit_reproducibilidad_f65b14d.md"
REPRO_DOC_SOURCE = RECEIPTS / "audit_reproducibilidad_source.json"
REPRO_DOC_SHA256 = "7997753f34e3ed5f96684e573702f41f4b4c49853c3bf112e2d654d62d59dcab"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def historical_reproduction_source(root: Path) -> Path:
    """Admit the exact documentary source used by the historical audit.

    REPRO_DOC remains its original identity in the saved summary, at
    SOURCE_COMMIT. The live chapter may evolve; it must never be a fallback
    for this snapshot or silently change the historical output's provenance.
    """
    metadata_path = root / REPRO_DOC_SOURCE
    require(metadata_path.is_file(), "historical reproduction source metadata is missing")
    metadata = load_json(metadata_path)
    expected = {"version": "e3-historical-document-source-v1",
                "original_path": str(REPRO_DOC), "source_commit": SOURCE_COMMIT,
                "snapshot_path": str(REPRO_DOC_SNAPSHOT),
                "sha256": REPRO_DOC_SHA256, "bytes": 29132}
    require(metadata == expected, "historical reproduction source metadata mismatch")
    snapshot = root / REPRO_DOC_SNAPSHOT
    require(snapshot.is_file(), "historical reproduction source snapshot is missing")
    require(snapshot.stat().st_size == metadata["bytes"], "historical reproduction source byte count mismatch")
    require(sha256(snapshot) == metadata["sha256"], "historical reproduction source hash mismatch")
    return snapshot


def validate_catalog(freeze: dict, a2: list[dict]) -> None:
    """Check system/seed identity; deterministic references have one trajectory."""
    models = freeze["models"]
    identities = [(m["model"], str(m["seed"])) for m in models]
    require(len(identities) == len(set(identities)), "duplicate predictor identity")
    require(len(a2) == len({r["model"] for r in a2}), "duplicate A2 system")
    require(set(freeze["systems"]) == {r["model"] for r in a2}, "A2/freeze system mismatch")
    require({m["model"] for m in models} == set(freeze["systems"]), "missing frozen system")
    seeds = set(freeze["final_seed_ids"])
    for row in a2:
        group = [m for m in models if m["model"] == row["model"]]
        require(int(row["R"]) == len(group), f"R mismatch: {row['model']}")
        require(int(row["n_common_dates"]) == 608, "unexpected evaluation count")
        require(finite_positive(float(row["RMSE"])), "invalid primary metric")
        if len(group) == 1:
            require(group[0]["seed"] == "deterministic", "R1 is not deterministic")
        else:
            require({m["seed"] for m in group} == seeds, "neural seed set mismatch")


def validate_release_record(artifact: dict, listings: dict) -> bool:
    if artifact.get("storage") != "release_asset":
        return False
    record = listings.get(artifact["asset_id"])
    return bool(record and record.get("state") == "uploaded"
                and not record.get("draft", False)
                and record["size"] == artifact["bytes"]
                and record["name"] == artifact["name"]
                and record["release_tag"] == artifact["release_tag"]
                and record["browser_download_url"] == artifact["url"])


def local_artifact(path: str, expected_hash: str, roots: list[Path]) -> Path | None:
    relative = PurePosixPath(path)
    require(not relative.is_absolute() and ".." not in relative.parts, "unsafe artifact path")
    found = None
    for root in roots:
        candidate = root.joinpath(*relative.parts)
        if candidate.is_file():
            require(sha256(candidate) == expected_hash, f"artifact hash mismatch: {path}")
            found = candidate
    return found


def validate_metadata(model: dict, scaler: dict, calibrator: dict, partition: dict) -> None:
    """Verify the actual downloaded metadata, not an assertion of pipeline fitting."""
    train = partition["final_partition"]["training"]["training_sample_IDs"]
    calibration = partition["final_partition"]["calibration"]["sample_IDs"]
    require(calibrator["predictor_sha256"] == model["checkpoint_sha256"], "calibrator/checkpoint binding mismatch")
    require(calibrator["dates"] == calibration, "calibration dates mismatch")
    require(calibrator["n"] == len(calibration), "calibration n mismatch")
    require(not isinstance(calibrator["q"], bool) and isinstance(calibrator["q"], (float, int)) and math.isfinite(calibrator["q"])
            and calibrator["q"] >= 0, "invalid calibration radius")
    require(finite_positive(scaler["target_scale"]), "invalid target scale")
    if model["seed"] != "deterministic":
        require(scaler["train_dates"] == train, "scaler training dates mismatch")
        require(len(scaler["mean"]) == len(scaler["std"]) == 5, "scaler shape mismatch")
        require(all(finite_positive(x) for x in scaler["std"]), "invalid scaler std")
        require(all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) for x in scaler["mean"]), "invalid scaler mean")
    else:
        require(scaler["fit_cutoff"][:10] == "2024-10-31", "classical fit cutoff mismatch")


def validate_partition(partition: dict) -> dict:
    final = partition["final_partition"]
    train = final["training"]["training_sample_IDs"]
    calibration = final["calibration"]["sample_IDs"]
    evaluation = final["evaluation"]["sample_IDs"]
    for name, dates, expected in [("training", train, 655), ("calibration", calibration, 53), ("evaluation", evaluation, 608)]:
        require(dates == sorted(set(dates)) and len(dates) == expected, f"invalid {name} IDs")
    require(not (set(train) & set(calibration) or set(train) & set(evaluation)
                 or set(calibration) & set(evaluation)), "partition objective IDs overlap")
    require(max(train) == "2024-10-31" and calibration[0] == "2024-11-08"
            and calibration[-1] == "2024-12-30" and evaluation[0] == "2025-01-01"
            and evaluation[-1] == "2026-08-31", "unexpected chronology")
    for fold in partition["hpo_fold_table"]:
        require(fold["train_validation_objective_ID_intersection"] == 0, "HPO IDs overlap")
        require(fold["inner_train"]["last"] < fold["validation"]["first"], "HPO chronology invalid")
    return {"training": len(train), "calibration": len(calibration), "evaluation": len(evaluation),
            "hpo_folds": len(partition["hpo_fold_table"]), "scope": "saved objective IDs and metadata; not tensor content"}


def extract_metadata(raw: bytes, backup: dict, destination: Path) -> None:
    """Validate archive/member identity and reject extra/unsafe paths before writing."""
    transport = backup["archive_transport"]
    require(len(raw) == transport["bytes"], "metadata archive byte count mismatch")
    require(hashlib.sha256(raw).hexdigest() == transport["sha256"], "metadata archive hash mismatch")
    expected = {a["archive_member"]: a for a in backup["artifacts"] if a["storage"] == "release_archive_member"}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), "unexpected ZIP members")
        for info in archive.infolist():
            rel = PurePosixPath(info.filename)
            require(not rel.is_absolute() and ".." not in rel.parts, "unsafe ZIP member")
            require(info.file_size == expected[info.filename]["bytes"], "metadata member size mismatch")
            data = archive.read(info)
            require(hashlib.sha256(data).hexdigest() == expected[info.filename]["sha256"], "metadata member hash mismatch")
            path = destination.joinpath(*rel.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def restore_metadata(backup: dict, destination: Path) -> None:
    """Download a verified small metadata ZIP; never download or load weights."""
    transport = backup["archive_transport"]
    with urlopen(transport["url"], timeout=60) as response:
        raw = response.read(transport["bytes"] + 1)
    extract_metadata(raw, backup, destination)


def audit(root: Path, artifact_roots: list[Path]) -> tuple[list[dict], dict, list[dict]]:
    root = root.resolve()
    public = root / PUBLIC
    freeze = load_json(public / "final_freeze.json")
    backup_path = root / RECEIPTS / "audit_backup_manifest.json"
    listing_path = root / RECEIPTS / "audit_release_listing.json"
    backup, listing = load_json(backup_path), load_json(listing_path)
    release_records = {a["id"]: a for a in listing["assets"]}
    artifacts = {a["path"]: a for a in backup["artifacts"]}
    require(len(artifacts) == len(backup["artifacts"]), "duplicate backup artifact")
    require(len(freeze["models"]) == 81 and len(freeze["systems"]) == 16, "unexpected catalog size")
    frozen = freeze["frozen_files_sha256"]
    require(all(p in artifacts and artifacts[p]["sha256"] == h for p, h in frozen.items()), "backup/freeze hash mismatch")
    require(artifacts["final_freeze.json"]["sha256"] == sha256(public / "final_freeze.json"), "backup/freeze identity mismatch")
    published_manifest, source = load_json(public / "manifest.json"), load_json(public / "source_evidence.json")
    require(sha256(backup_path) == source["source_backups"]["final"]["manifest_sha256"], "backup manifest source hash mismatch")
    transport = backup["archive_transport"]
    require(validate_release_record({**transport, "storage": "release_asset"}, release_records), "metadata ZIP release listing mismatch")
    checked_hashes = {}
    for name in ["final_freeze.json", "benchmark_A2.csv", "source_evidence.json"]:
        digest = sha256(public / name)
        require(digest == published_manifest["files"][name]["sha256"], f"public source hash mismatch: {name}")
        if name != "source_evidence.json":
            require(digest == source["exact_source_artifacts"][name]["sha256"], f"source lineage mismatch: {name}")
        checked_hashes[str(PUBLIC / name)] = digest
    with (public / "benchmark_A2.csv").open(newline="", encoding="utf-8") as stream:
        a2 = list(csv.DictReader(stream))
    validate_catalog(freeze, a2)
    partition = load_json(root / PARTITION)
    partition_result = validate_partition(partition)
    guards = load_json(root / GUARDS)
    require(guards["tests_run"] == 5 and guards["status"] == "passed", "unexpected guard receipt")
    require(guards["operations"]["real_model_inference_calls"] == 0, "guard scope changed")
    historical_repro = historical_reproduction_source(root)
    for rel in [PARTITION, GUARDS, ENVIRONMENT, REPRO_DOC, RECEIPTS / "audit_backup_manifest.json", RECEIPTS / "audit_release_listing.json"]:
        checked_hashes[str(rel)] = sha256(historical_repro if rel == REPRO_DOC else root / rel)

    release_ids = []
    for artifact in backup["artifacts"]:
        if artifact["storage"] == "release_asset":
            require(validate_release_record(artifact, release_records), f"release asset missing or inconsistent: {artifact['path']}")
            release_ids.append(artifact["asset_id"])
    metadata_verified = 0
    for artifact in backup["artifacts"]:
        if artifact["storage"] == "release_archive_member":
            p = local_artifact(artifact["path"], artifact["sha256"], artifact_roots)
            if p:
                require(p.stat().st_size == artifact["bytes"], "metadata byte count mismatch")
                metadata_verified += 1

    rows = []
    bindings = 0
    for model in sorted(freeze["models"], key=lambda m: (m["model"].lower(), str(m["seed"]))):
        row = {"model": model["model"], "seed": model["seed"]}
        found = {}
        for kind in ["checkpoint", "scaler", "calibrator"]:
            path, digest = model[kind], model[kind + "_sha256"]
            require(path in artifacts and artifacts[path]["sha256"] == digest, f"predictor/freeze mismatch: {path}")
            artifact = artifacts[path]
            found[kind] = local_artifact(path, digest, artifact_roots)
            row[kind + "_path"] = path
            row[kind + "_sha256"] = digest
            row[kind + "_bytes"] = artifact["bytes"]
            row[kind + "_storage"] = artifact["storage"]
            row[kind + "_release_url"] = artifact.get("url", backup["archive_transport"]["url"])
            row[kind + "_release_member"] = artifact.get("archive_member", "")
            row[kind + "_release_metadata_verified"] = validate_release_record(artifact, release_records) if artifact["storage"] == "release_asset" else True
            row[kind + "_downloaded_hash_verified"] = found[kind] is not None
        row["scaler_calibrator_binding_verified"] = False
        if found["scaler"] and found["calibrator"]:
            validate_metadata(model, load_json(found["scaler"]), load_json(found["calibrator"]), partition)
            row["scaler_calibrator_binding_verified"] = True
            bindings += 1
        row["export_ready"] = False  # Export and full round-trip require separate evidence.
        rows.append(row)

    environment = load_json(root / ENVIRONMENT)
    summary = {
        "version": "e3-e2-handoff-audit-v1", "source_commit": SOURCE_COMMIT,
        "scope": "read-only hashes, downloaded metadata, saved objective IDs and public release listings; not full leakage certification",
        "systems": len(a2), "predictors": len(rows), "neural_predictors": sum(r["seed"] != "deterministic" for r in rows),
        "deterministic_predictors": sum(r["seed"] == "deterministic" for r in rows),
        "frozen_file_hashes_reconciled": len(frozen), "backup_artifacts": len(artifacts),
        "release_weight_asset_metadata_verified": len(release_ids),
        "downloaded_metadata_hashes_verified": metadata_verified,
        "checkpoint_bytes_hash_verified": sum(r["checkpoint_downloaded_hash_verified"] for r in rows),
        "scaler_bytes_hash_verified": sum(r["scaler_downloaded_hash_verified"] for r in rows),
        "calibrator_bytes_hash_verified": sum(r["calibrator_downloaded_hash_verified"] for r in rows),
        "scaler_calibrator_predictor_bindings_verified": bindings,
        "partition_metadata": partition_result,
        "guard_receipt": {"tests_run": guards["tests_run"], "status": guards["status"], "reexecuted_here": False,
                          "scope": "existing synthetic/mocked common guard receipt, not randomized-label training"},
        "determinism": {"recorded_runtime": environment["runtime"], "global_numpy_seed_claimed": False,
                        "numpy_policy": "registered analysis generators; no global NumPy seed set by training core",
                        "source": str(REPRO_DOC), "new_runtime_reproduction": False},
        "A2_sha256": checked_hashes[str(PUBLIC / "benchmark_A2.csv")],
        "results_changed": False, "ranking_changed": False, "models_fit": 0, "model_inference_calls": 0,
        "block_A_closed": False, "E3_benchmark_freeze_issued": False,
        "sources_sha256": checked_hashes,
        "limits": ["Hashes and chronology do not certify tensor content, transform fitting or absence of leakage.",
                   "Release listing verifies existence and size metadata; only downloaded_hash_verified verifies file bytes.",
                   "E2 known evaluation calendar and R6 are conserved; no new confirmation or equivalence claim."]
    }
    checks = [
        {"id": "A-H01", "control": "A2, freeze and source lineage hashes", "status": "passed", "scope": "published bytes and declared lineage"},
        {"id": "A-H02", "control": "81 predictor identities and A2 counts", "status": "passed", "scope": "78 neural plus three deterministic; no synthetic replication"},
        {"id": "A-H03", "control": "245 frozen hashes reconciled with backup", "status": "passed", "scope": "manifest hash equality, distinct from byte verification"},
        {"id": "A-H04", "control": "79 weight release assets", "status": "passed", "scope": "GitHub release API listing, uploaded state, ID, name, URL and bytes"},
        {"id": "A-H05", "control": "Downloaded scaler/calibrator/predictor metadata", "status": "passed" if metadata_verified == 167 and bindings == 81 else "partial", "scope": f"{metadata_verified}/167 metadata files; {bindings}/81 bindings"},
        {"id": "A-H06", "control": "Chronology and objective ID separation", "status": "passed", "scope": "655/53/608 saved IDs, training scaler dates and three HPO fold metadata; not tensor content"},
        {"id": "A-H07", "control": "Determinism and seeds", "status": "documented", "scope": "historical runtime and explicit policies in E2 §8; not a new reproduction"},
        {"id": "A-P01", "control": "Exact/near-duplicate final tensors", "status": "pending", "scope": "needs effective final tensors and prespecified sensitivity analysis"},
        {"id": "A-P02", "control": "All real pipeline transformations and attributes", "status": "pending", "scope": "scaler dates alone cannot reconstruct or certify fitting logic for every author port"},
        {"id": "A-P03", "control": "Adversarial domain validation", "status": "pending", "scope": "requires real final cohorts and trained domain classifier; AUC difference is not automatically leakage"},
        {"id": "A-P04", "control": "Randomized-label negative control", "status": "pending", "scope": "requires trained current-pipeline control; synthetic guard receipt is insufficient"},
        {"id": "A-P05", "control": "Formal current train-validation gap with inherited delta_g", "status": "pending", "scope": "recover margin, comparable cohorts, estimand and dependent inference; no retrospective preregistration"},
        {"id": "A-P06", "control": "Block A closure and new benchmark freeze", "status": "pending", "scope": "do not issue closure until applicable controls and impact traceability are resolved"},
    ]
    return rows, summary, checks


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--download-metadata", type=Path, help="verified small metadata ZIP cache; no weights are downloaded")
    parser.add_argument("--metadata-zip", type=Path, help="local verified ZIP; defaults to the bundled audit_metadata.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="recompute without network and compare with the three saved audit outputs")
    args = parser.parse_args()
    try:
        require(not (args.check and args.download_metadata), "--check cannot download artifacts")
        backup = load_json(args.root / RECEIPTS / "audit_backup_manifest.json")
        with tempfile.TemporaryDirectory(prefix="e3-handoff-") as temporary:
            local_zip = args.metadata_zip or args.root / OUTPUT / "inputs/audit_metadata.zip"
            if local_zip.is_file():
                extract_metadata(local_zip.read_bytes(), backup, Path(temporary))
                args.artifact_root.append(Path(temporary))
            if args.download_metadata:
                restore_metadata(backup, args.download_metadata)
                args.artifact_root.append(args.download_metadata)
            rows, summary, checks = audit(args.root, args.artifact_root)
        output = args.output or args.root / OUTPUT
        outputs = {"audit_predictors.csv": csv_text(rows), "audit_checks.csv": csv_text(checks),
                   "audit_summary.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n"}
        if args.check:
            for name, text in outputs.items():
                require((output / name).read_text(encoding="utf-8") == text, f"saved audit output differs: {name}")
        else:
            output.mkdir(parents=True, exist_ok=True)
            for name, text in outputs.items():
                (output / name).write_text(text, encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ["systems", "predictors", "downloaded_metadata_hashes_verified", "scaler_calibrator_predictor_bindings_verified", "checkpoint_bytes_hash_verified", "block_A_closed"]}, ensure_ascii=False))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
