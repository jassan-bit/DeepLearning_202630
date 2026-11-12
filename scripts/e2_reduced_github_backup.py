"""Additive GitHub backup of completed E2 artifacts; never fit or retry a model.

Metadata/curves go to a dedicated branch, tensor checkpoints to prerelease
assets. A failed upload raises BackupPause while preserving the local fit.
Authentication is sent only to GitHub API/upload hosts, never Git command lines.
"""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

VERSION = "e2-reduced-github-backup-v1"
REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "results/e2-reduced-delivery-v1"
RELEASE_TAG = "e2-reduced-results-20261112-v1"
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".safetensors", ".pkl"}
ALLOWED_SUFFIXES = WEIGHT_SUFFIXES | {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".yaml", ".yml", ".png", ".svg"}
MAX_GIT_BYTES = 25 * 1024**2
MAX_ASSET_BYTES = 2 * 1024**3
METADATA_ARCHIVE_THRESHOLD = 32
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class BackupPause(RuntimeError):
    """Pause the controller; retry the upload, never the scientific fit."""
    retry_training = False
    fit_result_preserved = True

    def __init__(self, code, *, status=None):
        self.code, self.http_status = str(code), status
        super().__init__(self.code + (" (HTTP %s)" % status if status else ""))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".github-backup-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _public_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or not (host == "github.com" or host.endswith(".githubusercontent.com")
                    or host == "githubusercontent.com" or host.endswith(".s3.amazonaws.com"))):
        raise BackupPause("untrusted_asset_download_url")


class GitHubBackup:
    """One reusable uploader, with optimistic ref updates and immutable assets."""
    def __init__(self, token=None, *, timeout=45, max_conflicts=4):
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
        if not isinstance(self._token, str) or not self._token or any(c.isspace() for c in self._token):
            raise BackupPause("GITHUB_TOKEN_missing_or_invalid")
        self.timeout, self.max_conflicts = timeout, max_conflicts
        self._opener = urllib.request.build_opener(_NoRedirect())
        self._public_opener = urllib.request.build_opener(_PublicRedirect())
        self._release, self._assets, self._checked = None, {}, False

    def __repr__(self):
        return "GitHubBackup(repository=%r, branch=%r, token=<redacted>)" % (REPOSITORY, BRANCH)

    def _request(self, method, path, payload=None, *, upload=None, accept="application/vnd.github+json"):
        host = "uploads.github.com" if upload is not None else "api.github.com"
        url = "https://" + host + "/repos/" + REPOSITORY + path
        headers = {"Authorization": "Bearer " + self._token, "Accept": accept,
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": VERSION}
        body, stream = None, None
        if upload is not None:
            stream = Path(upload).open("rb")
            body = stream
            headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(Path(upload).stat().st_size)})
        elif payload is not None:
            body = canonical(payload)
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            # Never relay response bodies, credentials or Authorization headers.
            raise BackupPause("github_request_failed", status=error.code) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise BackupPause("github_transport_or_response_failed") from None
        finally:
            if stream is not None:
                stream.close()

    def _branch_head(self):
        return self._request("GET", "/git/ref/heads/" + BRANCH)["object"]["sha"]

    def _ensure_branch(self):
        try:
            return self._branch_head()
        except BackupPause as error:
            if error.http_status != 404:
                raise
        head = self._request("GET", "/git/ref/heads/main")["object"]["sha"]
        try:
            self._request("POST", "/git/refs", {"ref": "refs/heads/" + BRANCH, "sha": head})
            return head
        except BackupPause as error:
            if error.http_status != 422:
                raise
            return self._branch_head()  # Another uploader created the same branch.

    def _ensure_release(self):
        if self._release is not None:
            return self._release
        try:
            result = self._request("GET", "/releases/tags/" + RELEASE_TAG)
        except BackupPause as error:
            if error.http_status != 404:
                raise
            try:
                result = self._request("POST", "/releases", {"tag_name": RELEASE_TAG,
                    "target_commitish": BRANCH, "name": "E2 reduced results 2026-11-12 v1",
                    "body": "Completed E2 artifact backups. Preflight assets are transport checks, not scientific results.",
                    "draft": False, "prerelease": True})
            except BackupPause as race:
                if race.http_status != 422:
                    raise
                result = self._request("GET", "/releases/tags/" + RELEASE_TAG)
        if result.get("tag_name") != RELEASE_TAG or result.get("draft") is not False or result.get("prerelease") is not True:
            raise BackupPause("release_identity_or_prerelease_mismatch")
        self._release = result
        self._refresh_assets()
        return result

    def _refresh_assets(self):
        self._assets = {}
        for page in range(1, 1001):
            rows = self._request("GET", "/releases/%s/assets?per_page=100&page=%s" % (self._release["id"], page))
            if not isinstance(rows, list):
                raise BackupPause("invalid_release_asset_list")
            for row in rows:
                if row["name"] in self._assets:
                    raise BackupPause("duplicate_release_asset_name")
                self._assets[row["name"]] = row
            if len(rows) < 100:
                return
        raise BackupPause("release_asset_pagination_limit")

    def _verify_asset(self, asset, expected_sha, expected_size):
        if asset.get("size") != expected_size:
            raise BackupPause("existing_release_asset_size_mismatch")
        if asset.get("digest") == "sha256:" + expected_sha:
            return
        if asset.get("digest"):
            raise BackupPause("existing_release_asset_sha256_mismatch")
        # Public prerelease downloads do not carry the token through redirects.
        url = asset.get("browser_download_url", "")
        _public_url(url)
        value, count = hashlib.sha256(), 0
        try:
            with self._public_opener.open(url, timeout=self.timeout) as response:
                for block in iter(lambda: response.read(1024**2), b""):
                    value.update(block)
                    count += len(block)
        except (urllib.error.URLError, TimeoutError, OSError):
            raise BackupPause("asset_digest_download_verification_failed") from None
        if count != expected_size or value.hexdigest() != expected_sha:
            raise BackupPause("existing_release_asset_sha256_mismatch")

    def _asset(self, path, name, expected_sha):
        self._ensure_release()
        size = path.stat().st_size
        if size > MAX_ASSET_BYTES:
            raise BackupPause("checkpoint_exceeds_GitHub_asset_limit")
        asset = self._assets.get(name)
        if asset is None:
            try:
                asset = self._request("POST", "/releases/%s/assets?name=%s" %
                    (self._release["id"], urllib.parse.quote(name, safe="")), upload=path)
                self._assets[name] = asset
            except BackupPause as error:
                if error.http_status != 422:
                    raise
                self._refresh_assets()
                asset = self._assets.get(name)
                if asset is None:
                    raise BackupPause("asset_name_race_not_reconciled")
        self._verify_asset(asset, expected_sha, size)
        return {"asset_id": asset["id"], "name": name, "sha256": expected_sha,
                "bytes": size, "url": asset["browser_download_url"], "release_tag": RELEASE_TAG}

    def _content(self, path, head):
        try:
            return self._request("GET", "/contents/" + urllib.parse.quote(path, safe="/") + "?ref=" + head)
        except BackupPause as error:
            if error.http_status == 404:
                return None
            raise

    def _commit_files(self, files, manifest_path=None, expected_manifest=None):
        for _ in range(self.max_conflicts):
            head = self._branch_head()
            if manifest_path:
                previous = self._content(manifest_path, head)
                if previous is not None:
                    # The Contents API omits content for files above1MB.
                    encoded = previous.get("content")
                    if not encoded or previous.get("encoding") == "none":
                        encoded = self._request("GET", "/git/blobs/" + previous["sha"])["content"]
                    old = json.loads(base64.b64decode(encoded))
                    if old != expected_manifest:
                        raise BackupPause("attempt_identity_already_backed_up_with_different_content")
                    for path, data in files.items():
                        observed = self._content(path, head)
                        if observed is None or observed.get("sha") != git_blob_sha(data):
                            raise BackupPause("existing_metadata_artifact_changed")
                    return head, True
            tree = self._request("GET", "/git/commits/" + head)["tree"]["sha"]
            entries = []
            for path, data in sorted(files.items()):
                blob = self._request("POST", "/git/blobs", {"encoding": "base64", "content": base64.b64encode(data).decode()})
                if blob.get("sha") != git_blob_sha(data):
                    raise BackupPause("uploaded_git_blob_hash_mismatch")
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            new_tree = self._request("POST", "/git/trees", {"base_tree": tree, "tree": entries})["sha"]
            commit = self._request("POST", "/git/commits", {"message": "Preserve completed E2 artifacts", "tree": new_tree, "parents": [head]})["sha"]
            try:
                # The commit has exactly the observed head as parent. A normal
                # fast-forward PATCH rejects a concurrently advanced ref.
                self._request("PATCH", "/git/refs/heads/" + BRANCH, {"sha": commit, "force": False})
                return commit, False
            except BackupPause as error:
                if error.http_status not in (409, 422):
                    raise
                # Re-read and merge onto the new tree; never force or replace it.
        raise BackupPause("results_branch_conflict_limit")

    def preflight(self):
        """Prove branch and release-asset write access before any scientific fit."""
        self._ensure_branch()
        self._ensure_release()
        probe = b"E2 backup transport check v1; not a scientific result.\n"
        probe_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="e2-backup-probe-") as directory:
            path = Path(directory) / "probe.txt"
            path.write_bytes(probe)
            asset = self._asset(path, "backup-preflight-" + digest_bytes(probe)[:16] + "-" + probe_id + ".txt", digest_bytes(probe))
        audit = {"version": VERSION, "kind": "transport_preflight", "scientific_result": False,
                 "checked_at_utc": datetime.now(timezone.utc).isoformat(), "release_asset": asset}
        commit, _ = self._commit_files({"transport/preflight-" + probe_id + ".json": canonical(audit) + b"\n"})
        self._checked = True
        return {"version": VERSION, "status": "backup_write_access_verified", "repository": REPOSITORY,
                "branch": BRANCH, "release_tag": RELEASE_TAG, "remote_commit": commit,
                "scientific_training_started": False, "release_asset_sha256_verified": True}

    def _safe_artifacts(self, directory, receipt):
        directory = Path(directory).absolute()
        if directory.is_symlink() or not directory.is_dir():
            raise BackupPause("regular_fit_directory_required")
        claimed = receipt.get("artifacts_sha256")
        if not isinstance(claimed, dict) or not claimed:
            raise BackupPause("receipt_artifact_hashes_required")
        artifacts = []
        for relative, expected in sorted(claimed.items()):
            pure = PurePosixPath(relative)
            if (not isinstance(relative, str) or pure.is_absolute() or ".." in pure.parts
                    or any(part.lower() in {".git", ".env", "raw", "raw_data", "raw_market_data", "credentials", "secrets"} for part in pure.parts)
                    or not isinstance(expected, str) or not HEX64.fullmatch(expected)):
                raise BackupPause("unsafe_artifact_path_or_hash")
            path = directory.joinpath(*pure.parts)
            if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(directory))
                    or path.suffix.lower() not in ALLOWED_SUFFIXES):
                raise BackupPause("unapproved_or_missing_artifact")
            if digest_file(path) != expected:
                raise BackupPause("local_artifact_sha256_mismatch")
            if path.suffix.lower() not in WEIGHT_SUFFIXES:
                if path.stat().st_size > MAX_GIT_BYTES:
                    raise BackupPause("metadata_file_exceeds_GitHub_branch_limit")
                data = path.read_bytes()
                if self._token.encode() in data or re.search(rb"(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}", data):
                    raise BackupPause("credential_detected_in_artifact")
                if path.suffix.lower() in {".csv", ".tsv"}:
                    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter=delimiter), [])
                    if len({x.strip().lower() for x in header} & {"open", "high", "low", "close", "volume"}) >= 3:
                        raise BackupPause("raw_OHLCV_artifact_not_allowed")
            artifacts.append((relative, path, expected))
        return directory, artifacts

    def persist_result(self, directory, receipt):
        """Upload an already completed result; backup failure never repeats fit."""
        effective = receipt.get("result", receipt) if isinstance(receipt, dict) else receipt
        if (not isinstance(receipt, dict) or not isinstance(effective, dict)
                or effective.get("status") not in {"completed", "failed", "paused_clean", "interrupted"}):
            raise BackupPause("terminal_result_receipt_required")
        identity = effective.get("attempt_id") or effective.get("job_id") or receipt.get("attempt_id") or receipt.get("job_id")
        if not isinstance(identity, str) or not identity:
            raise BackupPause("attempt_id_or_job_id_required")
        key = identity if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", identity) else "fit-" + digest_bytes(identity.encode())
        try:
            root, artifacts = self._safe_artifacts(directory, effective)
            if self._token.encode() in canonical(receipt):
                raise BackupPause("credential_detected_in_receipt")
            if not self._checked:
                self.preflight()
            prefix, files, rows = "runs/" + key + "/", {}, []
            metadata = [(relative, path, expected) for relative, path, expected in artifacts
                        if path.suffix.lower() not in WEIGHT_SUFFIXES]
            archive_transport = None
            if len(metadata) > METADATA_ARCHIVE_THRESHOLD:
                # All inputs already passed path, credential and raw-OHLCV
                # checks. Archive transport preserves every original member.
                with tempfile.TemporaryDirectory(prefix="e2-metadata-backup-") as temporary:
                    archive_path = Path(temporary) / "metadata.zip"
                    members = []
                    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                                         compresslevel=9) as archive:
                        for relative, path, expected in metadata:
                            data = path.read_bytes()
                            if digest_bytes(data) != expected:
                                raise BackupPause("artifact_changed_during_backup")
                            info = zipfile.ZipInfo(relative, date_time=(2026, 11, 12, 0, 0, 0))
                            info.compress_type = zipfile.ZIP_DEFLATED
                            info.external_attr = 0o644 << 16
                            archive.writestr(info, data)
                            members.append({"path": relative, "storage": "release_archive_member",
                                            "archive_member": relative, "sha256": expected, "bytes": len(data)})
                    archive_sha = digest_file(archive_path)
                    name = key[:48] + "--metadata--" + archive_sha + ".zip"
                    asset = self._asset(archive_path, name, archive_sha)
                    archive_transport = {"kind": "deterministic_zip", "original_artifact_hashes_preserved": True,
                        "member_count": len(members), **asset}
                    for row in members:
                        row["archive_sha256"] = archive_sha
                    rows.extend(members)
            else:
                for relative, path, expected in metadata:
                    data = path.read_bytes()
                    if digest_bytes(data) != expected:
                        raise BackupPause("artifact_changed_during_backup")
                    target = prefix + "artifacts/" + relative
                    files[target] = data
                    rows.append({"path": relative, "storage": "git_blob", "remote_path": target,
                        "sha256": expected, "bytes": len(data), "git_blob_sha": git_blob_sha(data)})
            for relative, path, expected in artifacts:
                if path.suffix.lower() in WEIGHT_SUFFIXES:
                    basename = re.sub(r"[^A-Za-z0-9_.-]", "_", path.stem)[:32]
                    name = key[:48] + "--" + basename + "--" + expected + path.suffix.lower()
                    asset = self._asset(path, name, expected)
                    rows.append({"path": relative, "storage": "release_asset", **asset})
            manifest = {"version": VERSION, "attempt_id": identity, "fit_status": effective["status"],
                "receipt_sha256": digest_bytes(canonical(receipt)), "artifacts": rows,
                "retry_scientific_fit": False, "raw_market_data_included": False}
            if archive_transport is not None:
                manifest["archive_transport"] = archive_transport
            manifest_path = prefix + "manifest.json"
            files[manifest_path] = canonical(manifest) + b"\n"
            files[prefix + "receipt.json"] = canonical(receipt) + b"\n"
            commit, existing = self._commit_files(files, manifest_path, manifest)
            result = {"version": VERSION, "status": "backed_up", "attempt_id": identity,
                "repository": REPOSITORY, "branch": BRANCH, "release_tag": RELEASE_TAG,
                "remote_commit": commit, "manifest_path": manifest_path,
                "manifest_sha256": digest_bytes(files[manifest_path]), "artifact_count": len(rows),
                "existing_identical_result": existing, "retry_scientific_fit": False}
            _atomic_json(root / "github_backup_receipt.json", result)
            pending = root / "github_backup_pending.json"
            if pending.is_file():
                pending.unlink()
            return result
        except BackupPause as error:
            if Path(directory).is_dir():
                try:
                    _atomic_json(Path(directory) / "github_backup_pending.json", {"version": VERSION,
                        "status": "backup_required", "attempt_id": identity, "code": error.code,
                        "retry_scientific_fit": False, "local_result_preserved": True})
                except OSError:
                    pass
            raise
        except (OSError, ValueError, KeyError, TypeError):
            raise BackupPause("backup_failed_local_result_preserved") from None


def persist_result(directory, receipt, *, token=None, client=None):
    return (client or GitHubBackup(token)).persist_result(directory, receipt)
