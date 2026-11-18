"""Atomic, retryable GitHub snapshots of preflight JSON/logs and their ZIP.

The caller supplies a token in memory. Only a new results branch can be
created; main and historical receipts are never updated. No models run here.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import urllib.error
import urllib.request
import zipfile

REPOSITORY = "jassan-bit/DeepLearning_202630"
MAX_TEXT_BYTES = 1024 * 1024
MAX_ZIP_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_FILES = 256
_PROFILES = {
    "preflight": {"receipt": "preflight_receipt.json", "kind": "synthetic_preflight_receipts",
                  "branch": "e3-colab-receipts-", "prefix": "runs/colab_preflight/",
                  "message": "Guardar recibos Colab preflight "},
    "dataset": {"receipt": "dataset_audit.json", "kind": "dataset_admission_audit",
                "branch": "e3-dataset-audit-", "prefix": "runs/dataset_admission/",
                "message": "Guardar auditoría técnica de admisión de datos "},
}


class StorageError(RuntimeError):
    """A fixed diagnostic; response bodies and credentials are not disclosed."""


class StorageConflict(StorageError):
    pass


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _token(value):
    if not isinstance(value, str) or not 8 <= len(value) <= 1024 or any(c.isspace() for c in value):
        raise StorageError("github_token_required_in_memory")
    return value


def _repo(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value) is None:
        raise StorageError("invalid_repository")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(method, path, token, payload=None, *, allow_missing=False):
    """GitHub-only requests; never interpolate a secret into URL or body."""
    request = urllib.request.Request(
        "https://api.github.com" + path,
        data=None if payload is None else _canonical(payload), method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "Content-Type": "application/json", "User-Agent": "E3-Colab-Receipts"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=90) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise StorageError("github_response_too_large")
        return json.loads(raw)
    except urllib.error.HTTPError as error:
        if error.code == 404 and allow_missing:
            return None
        if error.code == 429 or (error.code == 403 and error.headers.get("X-RateLimit-Remaining") == "0"):
            raise StorageError("github_rate_limit_retry_same_snapshot") from None
        if error.code in (401, 403):
            raise StorageError("github_auth_or_contents_write_permission_required") from None
        raise StorageError("github_http_" + str(error.code)) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise StorageError("github_network_error_retry_same_snapshot") from None
    except (ValueError, UnicodeError):
        raise StorageError("github_invalid_json_response") from None


def validate_credentials(token, *, repository=REPOSITORY):
    """Read-only authentication/role check; Contents-write is proven on upload."""
    token, repository = _token(token), _repo(repository)
    user = _request("GET", "/user", token)
    repo = _request("GET", "/repos/" + repository, token)
    if (not isinstance(user.get("login"), str)
            or repo.get("full_name", "").lower() != repository.lower()
            or repo.get("permissions", {}).get("push") is not True):
        raise StorageError("repository_push_access_required")
    return {"authenticated": True, "repository": repository, "login": user["login"],
            "repository_push_role": True, "token_contents_write_verified": False,
            "scope_note": "Read-only check; actual object creation confirms Contents-write."}


def _safe_relative(name):
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and str(p) == name
            and ".." not in p.parts and "source_package" not in p.parts
            and "\\" not in name and not any(ord(c) < 32 for c in name))


def _regular_path(path, *, directory=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise StorageError("snapshot_symlink_refused")
    if not (path.is_dir() if directory else path.is_file()):
        raise StorageError("snapshot_regular_path_required")
    return path


def _snapshot(output_dir, zip_path, token, source_commit, run_id, *, profile="preflight"):
    config = _PROFILES[profile]
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise StorageError("full_source_commit_required")
    if re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", run_id or "") is None:
        raise StorageError("preflight_timestamp_run_id_required")
    root, archive = _regular_path(output_dir, directory=True), _regular_path(zip_path)
    selected = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name != "source_package"]
        for name in dirs + names:
            if (Path(current) / name).is_symlink():
                raise StorageError("snapshot_symlink_refused")
        for name in names:
            path = Path(current) / name
            if path.suffix not in (".json", ".log"):
                continue
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            if not _safe_relative(relative) or size > MAX_TEXT_BYTES:
                raise StorageError("snapshot_path_or_text_size_refused")
            if len(selected) >= MAX_FILES or sum(map(len, selected.values())) + size > MAX_TOTAL_BYTES:
                raise StorageError("snapshot_total_size_refused")
            with path.open("rb") as stream:
                raw = stream.read(MAX_TEXT_BYTES + 1)
            if len(raw) > MAX_TEXT_BYTES:
                raise StorageError("snapshot_text_grew_beyond_limit")
            try:
                raw.decode("utf-8")
            except UnicodeError:
                raise StorageError("snapshot_text_must_be_utf8") from None
            if token.encode() in raw or token in relative:
                raise StorageError("credential_in_snapshot_refused")
            selected[relative] = raw
    if not selected or len(selected) > MAX_FILES or config["receipt"] not in selected:
        raise StorageError("complete_" + profile + "_receipt_required")
    other_receipt = _PROFILES["dataset" if profile == "preflight" else "preflight"]["receipt"]
    if other_receipt in selected:
        raise StorageError("mixed_snapshot_profiles_refused")
    try:
        receipt = json.loads(selected[config["receipt"]])
        if (receipt.get("repository_HEAD") != source_commit
                or any(type(receipt.get(k)) is not int or receipt[k] != 0 for k in
                       ("new_real_data_fits", "new_market_predictions", "new_hypothesis_tests"))):
            raise StorageError(profile + "_source_or_zero_market_contract_mismatch")
        if profile == "preflight" and not isinstance(receipt.get("status"), str):
            raise StorageError("invalid_preflight_status")
        if profile == "dataset":
            datasets, admitted = receipt.get("datasets"), receipt.get("datasets_admitted")
            if (receipt.get("kind") != "dataset_admission_audit" or not isinstance(datasets, list)
                    or not all(isinstance(item, dict) for item in datasets)
                    or type(admitted) is not int or not 0 <= admitted <= len(datasets)):
                raise StorageError("invalid_dataset_admission_audit_contract")
            if any(k in receipt and receipt[k] is not False for k in
                   ("R1_R3_verified", "R1_R3_certified", "R1_accredited", "R3_accredited")):
                raise StorageError("dataset_audit_cannot_certify_R1_R3")
            if all(type(item.get("admitted")) is bool for item in datasets):
                if sum(item["admitted"] for item in datasets) != admitted:
                    raise StorageError("dataset_admission_count_mismatch")
    except (ValueError, TypeError, AttributeError):
        raise StorageError("invalid_" + profile + "_receipt_json") from None
    if archive.stat().st_size > MAX_ZIP_BYTES or token in archive.name:
        raise StorageError("snapshot_zip_size_or_name_refused")
    zipped = archive.read_bytes()
    if token.encode() in zipped:
        raise StorageError("credential_in_snapshot_refused")
    try:
        with zipfile.ZipFile(io.BytesIO(zipped)) as package:
            entries = package.infolist()
            if (package.comment or len(entries) != len(selected) or len(entries) > MAX_FILES
                    or sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES):
                raise StorageError("zip_must_match_selected_receipts")
            seen = set()
            for entry in entries:
                name = entry.filename
                if (not _safe_relative(name) or name in seen or name not in selected
                        or stat.S_ISLNK(entry.external_attr >> 16) or entry.is_dir() or entry.comment or entry.extra
                        or entry.file_size != len(selected[name]) or entry.file_size > MAX_TEXT_BYTES):
                    raise StorageError("zip_unsafe_or_unexpected_member")
                seen.add(name)
                raw = package.read(entry)
                if token.encode() in raw:
                    raise StorageError("credential_in_snapshot_refused")
                if raw != selected[name]:
                    raise StorageError("zip_receipt_bytes_mismatch")
    except StorageError:
        raise
    except (zipfile.BadZipFile, RuntimeError, OSError):
        raise StorageError("invalid_or_encrypted_snapshot_zip") from None
    if sum(map(len, selected.values())) + len(zipped) > MAX_TOTAL_BYTES:
        raise StorageError("snapshot_total_size_refused")
    files = {"files/" + name: raw for name, raw in selected.items()}
    files["receipts.zip"] = zipped
    manifest = {"schema_version": "1.0", "kind": config["kind"],
                "source_commit": source_commit, "run_id": run_id,
                "scope": "Persistence only; no models or market experiments executed by uploader.",
                "files": {name: {"bytes": len(raw), "sha256": _sha(raw), "git_blob_sha1": _git_sha(raw)}
                          for name, raw in sorted(files.items())}}
    if profile == "preflight":
        manifest["preflight_status"] = receipt["status"]
    else:
        manifest.update(datasets_total=len(receipt["datasets"]),
                        datasets_admitted_reported=receipt["datasets_admitted"], R1_R3_verified=False)
    manifest_raw = _canonical(manifest)
    files["storage_manifest.json"] = manifest_raw
    return files, _sha(manifest_raw), _sha(zipped)


def _tree_map(value):
    if value.get("truncated") is not False or not isinstance(value.get("tree"), list):
        raise StorageError("complete_git_tree_required")
    return {row["path"]: row for row in value["tree"] if row.get("type") == "blob"}


def _verify_tree(tree, old, expected, prefix):
    actual = _tree_map(tree)
    if any(path not in actual or actual[path].get("sha") != row["sha"] for path, row in old.items()):
        raise StorageError("source_tree_preservation_failed")
    if {p for p in actual if p.startswith(prefix + "/")} != set(expected):
        raise StorageConflict("existing_run_has_different_files")
    if set(actual) != set(old) | set(expected):
        raise StorageConflict("unexpected_tree_paths")
    if any(actual[path].get("sha") != sha for path, sha in expected.items()):
        raise StorageConflict("run_blob_integrity_mismatch")


def save_run(output_dir, zip_path, token, source_commit, run_id, *, repository=REPOSITORY):
    """Save exact bytes once; identical retries return the existing verified branch.

    Keep any local upload receipt outside output_dir: the ZIP must contain
    exactly its JSON/log files. Returned URLs contain no credentials.
    """
    return _save_profile(output_dir, zip_path, token, source_commit, run_id,
                         repository=repository, profile="preflight")


def save_dataset_audit(output_dir, zip_path, token, source_commit, run_id, *, repository=REPOSITORY):
    """Persist JSON/log admission evidence, including a blocked zero-admission audit.

    Requires dataset_audit.json with kind, repository_HEAD, datasets,
    datasets_admitted and the three zero-market counters. It cannot certify
    R1/R3. Raw CSV, model checkpoints and other binary data are not selected.
    The ZIP must contain exactly the selected JSON/log evidence.
    """
    return _save_profile(output_dir, zip_path, token, source_commit, run_id,
                         repository=repository, profile="dataset")


def _save_profile(output_dir, zip_path, token, source_commit, run_id, *, repository, profile):
    token, repository = _token(token), _repo(repository)
    files, manifest_sha, zip_sha = _snapshot(output_dir, zip_path, token, source_commit, run_id, profile=profile)
    validate_credentials(token, repository=repository)
    api = "/repos/" + repository
    config = _PROFILES[profile]
    branch = config["branch"] + run_id
    prefix = config["prefix"] + run_id
    source = _request("GET", api + "/git/commits/" + source_commit, token)
    if source.get("sha") != source_commit or not re.fullmatch(r"[0-9a-f]{40}", source.get("tree", {}).get("sha", "")):
        raise StorageError("source_commit_identity_mismatch")
    base_tree = source["tree"]["sha"]
    old = _tree_map(_request("GET", api + "/git/trees/" + base_tree + "?recursive=1", token))
    if any(path.startswith(prefix + "/") for path in old):
        raise StorageConflict("source_commit_already_contains_this_run")
    expected = {prefix + "/" + name: _git_sha(raw) for name, raw in files.items()}

    def verify_existing(ref):
        if ref.get("ref") != "refs/heads/" + branch or ref.get("object", {}).get("type") != "commit":
            raise StorageConflict("unexpected_results_ref")
        sha = ref["object"]["sha"]
        commit = _request("GET", api + "/git/commits/" + sha, token)
        if commit.get("sha") != sha or [p.get("sha") for p in commit.get("parents", [])] != [source_commit]:
            raise StorageConflict("existing_run_has_different_source")
        _verify_tree(_request("GET", api + "/git/trees/" + commit["tree"]["sha"] + "?recursive=1", token), old, expected, prefix)
        return sha

    ref_path = api + "/git/ref/heads/" + branch
    existing = _request("GET", ref_path, token, allow_missing=True)
    already_saved = existing is not None
    if existing is not None:
        sha = verify_existing(existing)
    else:
        entries = []
        for name, raw in sorted(files.items()):
            blob = _request("POST", api + "/git/blobs", token,
                            {"encoding": "base64", "content": base64.b64encode(raw).decode()})
            if blob.get("sha") != _git_sha(raw):
                raise StorageError("uploaded_git_blob_integrity_mismatch")
            entries.append({"path": prefix + "/" + name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        tree = _request("POST", api + "/git/trees", token, {"base_tree": base_tree, "tree": entries})
        _verify_tree(_request("GET", api + "/git/trees/" + tree["sha"] + "?recursive=1", token), old, expected, prefix)
        commit = _request("POST", api + "/git/commits", token,
                          {"message": config["message"] + run_id,
                           "tree": tree["sha"], "parents": [source_commit]})
        checked = {"ref": "refs/heads/" + branch, "object": {"type": "commit", "sha": commit["sha"]}}
        verify_existing(checked)
        try:
            _request("POST", api + "/git/refs", token, {"ref": "refs/heads/" + branch, "sha": commit["sha"]})
        except StorageError:
            # A lost acknowledgement must not force another GPU run.
            recovered = _request("GET", ref_path, token, allow_missing=True)
            if recovered is None:
                raise
            verify_existing(recovered)
            already_saved = True
        final_ref = _request("GET", ref_path, token)
        sha = verify_existing(final_ref)
    return {"saved": True, "already_saved": already_saved, "repository": repository,
            "branch": branch, "commit_sha": sha,
            "branch_url": "https://github.com/" + repository + "/tree/" + branch,
            "run_url": "https://github.com/" + repository + "/tree/" + branch + "/" + prefix,
            "manifest_sha256": manifest_sha, "zip_sha256": zip_sha,
            "persisted_files": len(files), "all_files_verified": True, "main_updated": False}
