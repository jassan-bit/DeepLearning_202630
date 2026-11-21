"""Encrypted, atomic GitHub persistence for the new SEOI development pilot.

Credentials and the independent backup passphrase live in memory only. Public
results use an explicit metadata/derived-results allowlist; NPZ, source code,
unused historical snapshots and plaintext weights are never uploaded. Prepared
with declared AI assistance. Importing this module performs no network request.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import uuid

from . import runner

VERSION = "e3-seoi-v5-encrypted-github-1"
MAGIC = b"E3SEOBK1"
MAX_PLAIN_ZIP = 20 * 1024 * 1024
MAX_CIPHER = 32 * 1024 * 1024
MAX_UNCOMPRESSED = 64 * 1024 * 1024
MAX_RESPONSE = 48 * 1024 * 1024
KDF = {"name": "scrypt", "n": 32768, "r": 8, "p": 1, "length": 32}
EXTRA_ROOT = ("github_session.json", "remote_restore.json", "recovery_history.json", "assessment_started.json")
ROOT_PRIVATE = tuple(dict.fromkeys(runner.ROOT_EXPORT + EXTRA_ROOT))
ROOT_PUBLIC = tuple(dict.fromkeys(runner.ROOT_EXPORT + EXTRA_ROOT))
JOB_PRIVATE = tuple(dict.fromkeys(runner.JOB_EXPORT + ("request.json", "checkpoint.json", "epoch_pending.json")))
RECIPE_PATTERN = "(?:" + "|".join(map(re.escape, runner.RECIPES)) + ")"
JOB_PATTERN = re.compile(r"jobs/seed[0-9]+/" + RECIPE_PATTERN + r"/(?:" + "|".join(map(re.escape, JOB_PRIVATE)) + r"|snapshot_[0-9a-f]{32}\.pt)\Z")
CACHE_PATTERN = re.compile(r"assessment_cache/" + RECIPE_PATTERN + r"_[0-9]+\.json\Z")


class StorageError(RuntimeError):
    """Sanitized persistence error; never contains credentials/response bodies."""


def _require(ok, message):
    if not ok:
        raise StorageError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _safe_name(name):
    return (isinstance(name, str) and bool(name) and "\\" not in name and not PurePosixPath(name).is_absolute()
            and PurePosixPath(name).as_posix() == name
            and all(x not in ("", ".", "..") for x in name.split("/")))


def _zip(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name, raw in sorted(files.items()):
            _require(_safe_name(name), "unsafe_archive_member")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            output.writestr(info, raw)
    result = stream.getvalue()
    _require(len(result) <= MAX_PLAIN_ZIP, "compressed_backup_exceeds_20MiB")
    return result


def classify_http(status, message, headers=None, *, now=None):
    """Classify GitHub limits without confusing secondary limits with auth."""
    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    now = time.time() if now is None else now
    lower = (message or "").lower()
    delay = None
    if headers.get("retry-after"):
        try:
            delay = max(0., float(headers["retry-after"]))
        except ValueError:
            try:
                delay = max(0., parsedate_to_datetime(headers["retry-after"]).timestamp() - now)
            except (ValueError, TypeError, OverflowError):
                pass
    if status == 429 or (status == 403 and ("secondary rate" in lower or "abuse" in lower)):
        return {"kind": "secondary_rate_limit" if status == 403 else "rate_limit", "retryable": True,
                "retry_after_seconds": 60. if delay is None else delay}
    if status == 403 and headers.get("x-ratelimit-remaining") == "0":
        try:
            reset = max(1., float(headers.get("x-ratelimit-reset", now + 60)) - now + 1.)
        except ValueError:
            reset = 60.
        return {"kind": "primary_rate_limit", "retryable": True,
                "retry_after_seconds": reset if delay is None else delay}
    if status in (401, 403):
        return {"kind": "authentication_or_contents_write_permission", "retryable": False}
    if status in (409, 412) or (status == 422 and ("fast forward" in lower or "reference" in lower or "already exists" in lower)):
        return {"kind": "ref_conflict", "retryable": False}
    if status in (408, 500, 502, 503, 504) or (status == 422 and "timed out" in lower):
        return {"kind": "transient_server_error", "retryable": True,
                "retry_after_seconds": 5. if delay is None else delay}
    return {"kind": "http_error", "retryable": False}


class GithubStore:
    def __init__(self, token, backup_secret, repository="jassan-bit/DeepLearning_202630",
                 branch="e3-seoi-v5-run-01", *, source_commit,
                 prefix="runs/e3_seoi_v5/run-01", sleep=time.sleep, clock=time.time, opener=None):
        _require(isinstance(token, str) and len(token) >= 20 and not any(c in token for c in "\r\n"), "github_token_required")
        _require(isinstance(backup_secret, str) and len(backup_secret) >= 32 and backup_secret != token,
                 "independent_backup_secret_at_least_32_characters_required")
        _require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)), "invalid_repository")
        _require(isinstance(branch, str) and _safe_name(branch) and not any(c in branch for c in " ~^:?*["), "invalid_branch")
        _require(_safe_name(prefix), "invalid_repository_prefix")
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", source_commit or "")), "immutable_source_commit_required")
        self.repository, self.branch, self.source_commit, self.prefix = repository, branch, source_commit, prefix
        self._token, self._secret = token, backup_secret
        self._sleep, self._clock = sleep, clock
        self._opener = opener or urllib.request.urlopen
        self._last_write = None
        self._authenticated_head = None
        self._root = "/repos/" + repository

    def __repr__(self):
        return "GithubStore(repository=" + repr(self.repository) + ",branch=" + repr(self.branch) + ",credentials=<hidden>)"

    def _wait(self, seconds):
        seconds = max(0., seconds)
        while seconds > 0:
            chunk = min(seconds, 30.)
            self._sleep(chunk); seconds -= chunk

    def _request(self, method, path, payload=None, *, allow_missing=False):
        body = None if payload is None else _canonical(payload)
        for attempt in range(5):
            if method != "GET":
                if self._last_write is not None:
                    self._wait(max(0., 1.25 - (self._clock() - self._last_write)))
                self._last_write = self._clock()
            request = urllib.request.Request("https://api.github.com" + path, data=body, method=method,
                                             headers={"Accept": "application/vnd.github+json", "Authorization": "Bearer " + self._token,
                                                      "Content-Type": "application/json", "User-Agent": VERSION,
                                                      "X-GitHub-Api-Version": "2022-11-28"})
            try:
                with self._opener(request, timeout=90) as response:
                    raw = response.read(MAX_RESPONSE + 1)
                _require(len(raw) <= MAX_RESPONSE, "github_response_too_large")
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as error:
                if error.code == 404 and allow_missing:
                    return None
                try:
                    diagnostic = json.loads(error.read(64 * 1024)).get("message", "")
                except (ValueError, UnicodeError):
                    diagnostic = ""
                category = classify_http(error.code, diagnostic, dict(error.headers), now=self._clock())
                if not category["retryable"] or attempt == 4:
                    raise StorageError("github_" + category["kind"] + " status=" + str(error.code)) from None
                self._wait(max(category.get("retry_after_seconds", 5.), 2. ** attempt))
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 4:
                    raise StorageError("github_transport_retry_same_pending_snapshot") from None
                self._wait(2. ** attempt)
            except (ValueError, UnicodeError):
                raise StorageError("github_invalid_json_response") from None
        raise StorageError("github_transport_retry_same_pending_snapshot")

    def _ref(self):
        return self._request("GET", self._root + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe=""), allow_missing=True)

    def _file_info(self, name, commit, *, allow_missing=False):
        path = self._root + "/contents/" + urllib.parse.quote(self.prefix + "/" + name, safe="/") + "?ref=" + commit
        result = self._request("GET", path, allow_missing=allow_missing)
        if result is None:
            return None
        _require(isinstance(result, dict) and result.get("type") == "file" and bool(re.fullmatch(r"[0-9a-f]{40}", result.get("sha", ""))),
                 "github_expected_regular_file")
        return result

    def _file_bytes(self, name, commit, limit):
        info = self._file_info(name, commit)
        _require(type(info.get("size")) is int and 0 <= info["size"] <= limit, "github_blob_exceeds_bound")
        blob = self._request("GET", self._root + "/git/blobs/" + info["sha"])
        try:
            raw = base64.b64decode(blob["content"], validate=False)
        except (ValueError, KeyError, TypeError):
            raise StorageError("github_invalid_blob_encoding") from None
        _require(len(raw) == info["size"] and _git_blob_sha(raw) == info["sha"], "github_blob_readback_mismatch")
        return raw

    def status(self):
        ref = self._ref()
        if ref is None:
            return {"exists": False, "repository": self.repository, "branch": self.branch, "source_commit": self.source_commit}
        head = ref.get("object", {}).get("sha")
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", head or "")), "github_invalid_ref")
        info = self._file_info("status.json", head, allow_missing=True)
        _require(info is not None, "existing_branch_has_no_managed_status_refuse_overwrite")
        try:
            value = json.loads(self._file_bytes("status.json", head, 128 * 1024))
        except (ValueError, UnicodeError):
            raise StorageError("github_invalid_status") from None
        _require(value.get("version") == VERSION and value.get("source_commit") == self.source_commit
                 and value.get("repository") == self.repository and value.get("branch") == self.branch
                 and value.get("prefix") == self.prefix, "github_remote_source_or_storage_identity_differs")
        return {**value, "exists": True, "head": head, "head_commit": head}

    def _context(self, plan):
        return {"version": VERSION, "repository": self.repository, "branch": self.branch,
                "prefix": self.prefix, "source_commit": self.source_commit,
                "case_id": plan["case_id"], "plan_sha256": runner.digest(plan)}

    def _encrypt(self, zipped, context):
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        _require(len(zipped) <= MAX_PLAIN_ZIP, "compressed_backup_exceeds_20MiB")
        salt, nonce = os.urandom(16), os.urandom(12)
        header = _canonical({"context": context, "kdf": KDF})
        key = Scrypt(salt=salt, length=32, n=32768, r=8, p=1).derive(self._secret.encode())
        aad = MAGIC + len(header).to_bytes(4, "big") + header + salt + nonce
        payload = aad + AESGCM(key).encrypt(nonce, zipped, aad)
        _require(len(payload) <= MAX_CIPHER, "encrypted_backup_exceeds_32MiB")
        return payload

    def _decrypt(self, payload, expected):
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        _require(len(payload) <= MAX_CIPHER and payload[:8] == MAGIC and len(payload) >= 56, "invalid_encrypted_backup_envelope")
        length = int.from_bytes(payload[8:12], "big")
        _require(0 < length <= 4096 and len(payload) > 12 + length + 28 + 16, "invalid_encrypted_backup_header")
        header = payload[12:12+length]
        try:
            envelope = json.loads(header)
        except (ValueError, UnicodeError):
            raise StorageError("invalid_encrypted_backup_context") from None
        _require(isinstance(envelope, dict) and envelope.get("context") == expected
                 and envelope.get("kdf") == KDF, "encrypted_backup_context_or_KDF_differs")
        salt, nonce = payload[12+length:28+length], payload[28+length:40+length]
        key = Scrypt(salt=salt, length=32, n=32768, r=8, p=1).derive(self._secret.encode())
        try:
            zipped = AESGCM(key).decrypt(nonce, payload[40+length:], payload[:40+length])
        except InvalidTag:
            raise StorageError("backup_decryption_failed_wrong_secret_or_tampering") from None
        _require(len(zipped) <= MAX_PLAIN_ZIP, "decrypted_backup_exceeds_20MiB")
        return zipped

    def _read_case_file(self, case, name):
        _require(_safe_name(name), "unsafe_case_member")
        path = runner.safe(case / name)
        _require(path.is_file() and path.stat().st_size <= MAX_UNCOMPRESSED, "case_member_too_large_or_nonregular")
        raw = path.read_bytes()
        _require(self._token.encode() not in raw and self._secret.encode() not in raw, "credential_material_detected_in_case_refuse_upload")
        return raw

    def _archives(self, case):
        case = runner.safe(case)
        plan = runner.read_json(case / "plan.json")
        _require(isinstance(plan.get("case_id"), str) and plan["case_id"].startswith("seoi-v5-dev-"), "owned_new_case_required")
        context = self._context(plan)
        private, public = {}, {}
        for name in ROOT_PRIVATE:
            if (case/name).exists():
                private[name] = self._read_case_file(case, name)
        for name in ROOT_PUBLIC:
            if (case/name).exists():
                public[name] = self._read_case_file(case, name)
        completed = 0
        for job in plan.get("jobs", []):
            job_id = job.get("job_id", "")
            _require(_safe_name(job_id) and job_id.startswith("seed"), "invalid_job_identity")
            directory = "jobs/" + job_id
            for name in JOB_PRIVATE:
                member = directory + "/" + name
                _require(bool(JOB_PATTERN.fullmatch(member)), "unexpected_job_member")
                if (case/member).exists():
                    private[member] = self._read_case_file(case, member)
            for name in runner.JOB_EXPORT:
                member = directory + "/" + name
                if (case/member).exists():
                    public[member] = self._read_case_file(case, member)
            pointer_path = case / directory / "checkpoint.json"
            if pointer_path.exists():
                pointer = runner.read_json(pointer_path)
                member = directory + "/" + pointer.get("file", "")
                _require(bool(JOB_PATTERN.fullmatch(member)) and member.endswith(".pt"), "invalid_latest_snapshot_pointer")
                raw = self._read_case_file(case, member)
                _require(len(raw) == pointer.get("bytes") and _sha(raw) == pointer.get("sha256"), "latest_snapshot_hash_differs")
                private[member] = raw
            job_path = case / directory / "job.json"
            if job_path.exists() and runner.read_json(job_path).get("status") == "completed_validated_fit":
                completed += 1
        cache = case / "assessment_cache"
        if cache.exists():
            for path in cache.iterdir():
                member = path.relative_to(case).as_posix()
                _require(bool(CACHE_PATTERN.fullmatch(member)), "unexpected_assessment_cache_member")
                private[member] = self._read_case_file(case, member)
        _require(sum(map(len, private.values())) <= MAX_UNCOMPRESSED and sum(map(len, public.values())) <= MAX_PLAIN_ZIP,
                 "backup_uncompressed_size_limit")
        manifest = {"context": context, "files": [{"path": name, "bytes": len(raw), "sha256": _sha(raw)} for name, raw in sorted(private.items())],
                    "input_NPZ_included": False, "latest_pointer_snapshots_only": True}
        private["SNAPSHOT_MANIFEST.json"] = _canonical(manifest)
        session = runner.read_json(case/"github_session.json") if (case/"github_session.json").exists() else {}
        return context, _zip(private), _zip(public), {"completed_neural_fits": completed,
                                                     "session_open": session.get("session_open") is True,
                                                     "private_files": len(private)-1}

    def _pending(self, case, stage):
        folder = runner.safe(case / ".github_pending")
        if folder.exists():
            record = runner.read_json(folder/"publication.json")
            _require(record.get("context") == self._context(runner.read_json(case/"plan.json")), "pending_snapshot_case_identity_differs")
            files = {name: self._read_case_file(folder, name) for name in ("checkpoint.aesgcm", "status.json", "public_results.zip")}
            _require(all(_sha(raw) == record["sha256"][name] for name, raw in files.items()), "pending_payload_hash_differs")
            return folder, record, files
        context, private_zip, public_zip, counts = self._archives(case)
        payload = self._encrypt(private_zip, context)
        event = stage.get("event", stage.get("stage", "checkpoint")) if isinstance(stage, dict) else stage
        event = str(event or "checkpoint")[:100]
        _require(self._token not in event and self._secret not in event, "credential_in_stage_refused")
        metadata = {**context, **counts, "backup_created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_sha256": _sha(payload), "checkpoint_bytes": len(payload),
                    "public_results_sha256": _sha(public_zip), "public_results_bytes": len(public_zip),
                    "stage": event, "input_NPZ_uploaded": False, "weights_uploaded_in_plaintext": False,
                    "backup_algorithm": "AES-256-GCM/Scrypt-n32768-r8-p1-independent-passphrase", "R3_accredited": False}
        files = {"checkpoint.aesgcm": payload, "status.json": _canonical(metadata), "public_results.zip": public_zip}
        record = {"context": context, "created_at_utc": metadata["backup_created_at_utc"],
                  "sha256": {name: _sha(raw) for name, raw in files.items()}, "base_head": None, "commit": None}
        # No network request can occur until this complete staging directory
        # is atomically installed. A crash leaves an unused sibling, not a
        # poisoned partially created pending transaction.
        staging = Path(tempfile.mkdtemp(prefix=".github-staging-", dir=case))
        try:
            for name, raw in files.items():
                runner.atomic_bytes(staging/name, raw, immutable=True)
            runner.write_json(staging/"publication.json", record, immutable=True)
            _require(not folder.exists(), "pending_transaction_appeared_concurrently")
            os.rename(staging, folder)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return folder, record, files

    def _verify_remote(self, commit, files):
        ref = self._ref()
        _require(ref is not None and ref.get("object", {}).get("sha") == commit, "github_ref_conflict_after_write")
        for name, raw in files.items():
            info = self._file_info(name, commit)
            _require(info.get("sha") == _git_blob_sha(raw) and info.get("size") == len(raw), "github_final_path_readback_mismatch")

    def backup(self, case_root, stage=None):
        case = runner.safe(case_root)
        remote = self.status()
        # Authenticate the existing remote key before creating any local pending
        # payload, so a mistyped passphrase cannot silently rotate the backup key.
        if remote["exists"] and self._authenticated_head != remote["head"]:
            current = self._file_bytes("checkpoint.aesgcm", remote["head"], MAX_CIPHER)
            _require(len(current) == remote.get("checkpoint_bytes")
                     and _sha(current) == remote.get("checkpoint_sha256"), "encrypted_backup_remote_hash_differs")
            expected_context = {k: remote[k] for k in
                                ("version", "repository", "branch", "prefix", "source_commit", "case_id", "plan_sha256")}
            self._decrypt(current, expected_context)
            self._authenticated_head = remote["head"]
        folder, pending, files = self._pending(case, stage)
        # Persisted pending bytes may have been created in another invocation.
        # Hash equality does not prove that the current user knows their key.
        self._decrypt(files["checkpoint.aesgcm"], pending["context"])
        if remote["exists"] and remote.get("checkpoint_sha256") == _sha(files["checkpoint.aesgcm"]):
            commit = remote["head"]
        else:
            marker = runner.read_json(case/".github_backup_receipt.json") if (case/".github_backup_receipt.json").exists() else None
            if remote["exists"]:
                _require(remote.get("case_id") == pending["context"]["case_id"] and remote.get("plan_sha256") == pending["context"]["plan_sha256"],
                         "github_remote_case_or_plan_differs")
                _require(marker is not None and marker.get("commit") == remote["head"], "stale_local_case_restore_remote_before_overwrite")
                expected = remote["head"]
            else:
                _require(marker is None, "previous_remote_branch_disappeared_refuse_recreate")
                expected = self.source_commit
            _require(pending.get("base_head") in (None, expected), "pending_snapshot_base_head_conflict")
            pending["base_head"] = expected; runner.write_json(folder/"publication.json", pending)
            base_commit = self._request("GET", self._root + "/git/commits/" + expected)
            tree_sha = base_commit.get("tree", {}).get("sha")
            _require(bool(re.fullmatch(r"[0-9a-f]{40}", tree_sha or "")), "github_base_tree_missing")
            entries = []
            for name, raw in files.items():
                blob = self._request("POST", self._root + "/git/blobs", {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
                _require(blob.get("sha") == _git_blob_sha(raw), "github_created_blob_identity_differs")
                entries.append({"path": self.prefix+"/"+name, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            tree = self._request("POST", self._root+"/git/trees", {"base_tree": tree_sha, "tree": entries})
            author = {"name": "E3 pilot automated checkpoint", "email": "e3-pilot@users.noreply.github.com", "date": pending["created_at_utc"]}
            candidate = self._request("POST", self._root+"/git/commits",
                                      {"message": "Preserve encrypted SEOI pilot checkpoint and public derived evidence", "tree": tree["sha"],
                                       "parents": [expected], "author": author, "committer": author})
            commit = candidate.get("sha")
            _require(bool(re.fullmatch(r"[0-9a-f]{40}", commit or "")), "github_candidate_commit_missing")
            pending["commit"] = commit; runner.write_json(folder/"publication.json", pending)
            try:
                if remote["exists"]:
                    self._request("PATCH", self._root+"/git/refs/heads/"+urllib.parse.quote(self.branch, safe=""), {"sha": commit, "force": False})
                else:
                    self._request("POST", self._root+"/git/refs", {"ref": "refs/heads/"+self.branch, "sha": commit})
            except StorageError:
                ref = self._ref()
                if ref is None or ref.get("object", {}).get("sha") != commit:
                    raise
        self._verify_remote(commit, files)
        metadata = json.loads(files["status.json"])
        receipt = {"status": "verified", "remote_verified": True, "commit": commit, "head_commit": commit,
                   "branch": self.branch, "source_commit": self.source_commit, "plan_sha256": metadata["plan_sha256"],
                   "case_id": metadata["case_id"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                   "public_results_sha256": metadata["public_results_sha256"],
                   "url": "https://github.com/"+self.repository+"/tree/"+commit+"/"+self.prefix}
        runner.write_json(case/".github_backup_receipt.json", receipt)
        self._authenticated_head = commit
        # First retire the complete transaction atomically. Cleanup failure can
        # leave only an ignored completed directory, never a broken pending one.
        retired = case / (".github-completed-cleanup-" + uuid.uuid4().hex)
        os.rename(folder, retired)
        try:
            shutil.rmtree(retired)
        except OSError:
            pass
        return receipt

    def _validated_members(self, zipped, context):
        try:
            archive = zipfile.ZipFile(io.BytesIO(zipped))
            infos = archive.infolist()
            _require(len(infos) <= 2000 and len({i.filename for i in infos}) == len(infos), "duplicate_or_excessive_archive_members")
            _require(sum(i.file_size for i in infos) <= MAX_UNCOMPRESSED, "archive_expansion_exceeds_64MiB")
            for info in infos:
                _require(_safe_name(info.filename) and not info.is_dir() and stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK,
                         "unsafe_or_symlink_archive_member")
            files = {i.filename: archive.read(i) for i in infos}
        except StorageError:
            raise
        except (zipfile.BadZipFile, RuntimeError, OSError):
            raise StorageError("invalid_backup_archive") from None
        try:
            manifest = json.loads(files.pop("SNAPSHOT_MANIFEST.json"))
        except (KeyError, ValueError, UnicodeError):
            raise StorageError("backup_snapshot_manifest_missing") from None
        _require(manifest.get("context") == context, "backup_snapshot_context_differs")
        records = manifest.get("files", [])
        _require({r.get("path") for r in records} == set(files) and len(records) == len(files), "backup_manifest_members_differ")
        for record in records:
            name = record["path"]; raw = files[name]
            _require(name in ROOT_PRIVATE or JOB_PATTERN.fullmatch(name) or CACHE_PATTERN.fullmatch(name), "backup_member_not_allowlisted")
            _require(len(raw) == record.get("bytes") and _sha(raw) == record.get("sha256"), "backup_member_hash_differs")
        referenced = set()
        for name, raw in files.items():
            if name.startswith("jobs/") and name.endswith("/checkpoint.json"):
                try:
                    pointer = json.loads(raw)
                    member = str(PurePosixPath(name).parent / pointer["file"])
                    referenced.add(member)
                    _require(member in files and member.endswith(".pt") and len(files[member]) == pointer["bytes"]
                             and _sha(files[member]) == pointer["sha256"], "restored_latest_snapshot_differs")
                except (KeyError, ValueError, TypeError):
                    raise StorageError("restored_checkpoint_pointer_invalid") from None
        _require({n for n in files if n.endswith(".pt")} == referenced, "unreferenced_checkpoint_payload_refused")
        return files

    def restore(self, destination):
        destination = runner.safe(destination)
        _require(not destination.exists(), "fresh_restore_destination_required")
        remote = self.status()
        _require(remote["exists"], "remote_case_not_found")
        payload = self._file_bytes("checkpoint.aesgcm", remote["head"], MAX_CIPHER)
        _require(len(payload) == remote.get("checkpoint_bytes") and _sha(payload) == remote.get("checkpoint_sha256"), "encrypted_backup_remote_hash_differs")
        context = {k: remote[k] for k in ("version", "repository", "branch", "prefix", "source_commit", "case_id", "plan_sha256")}
        files = self._validated_members(self._decrypt(payload, context), context)
        try:
            plan = json.loads(files["plan.json"])
        except (KeyError, ValueError, UnicodeError):
            raise StorageError("restored_plan_missing_or_invalid") from None
        _require(plan.get("case_id") == context["case_id"] and runner.digest(plan) == context["plan_sha256"], "restored_plan_identity_differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".seoi-restore-", dir=destination.parent))
        try:
            for name, raw in files.items():
                path = temporary/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
            session = json.loads(files.get("github_session.json", b"{}"))
            open_session = session.get("session_open") is True
            history = json.loads(files.get("recovery_history.json", b"[]"))
            _require(isinstance(history, list), "invalid_recovery_history")
            recovery = {"restored_at_utc": datetime.now(timezone.utc).isoformat(), "commit": remote["head"],
                        "source_commit": self.source_commit, "case_id": context["case_id"], "plan_sha256": context["plan_sha256"],
                        "unuploaded_progress_may_be_lost": True, "last_saved_session_open": open_session,
                        "unobserved_remote_compute_possible": open_session,
                        "budget_comparability_verified": not open_session and not any(r.get("unobserved_remote_compute_possible") for r in history),
                        "cost_and_optimizer_steps_are_lower_bounds_if_session_was_open": open_session,
                        "weights_curves_and_scientific_protocol_modified": False,
                        "input_NPZ_included": False, "input_NPZ_must_be_redownloaded_and_hash_verified": True,
                        "seed_and_case_identity_unchanged": True}
            if open_session:
                history.append(recovery)
                runner.write_json(temporary/"recovery_history.json", history)
            runner.write_json(temporary/"remote_restore.json", recovery)
            receipt = {"status": "restored", "remote_verified": True, "commit": remote["head"], "head_commit": remote["head"],
                       "branch": self.branch, "source_commit": self.source_commit, "case_id": context["case_id"],
                       "plan_sha256": context["plan_sha256"], "checkpoint_sha256": remote["checkpoint_sha256"],
                       "remote_restore": recovery, "input_npz_required": True}
            runner.write_json(temporary/".github_backup_receipt.json", receipt)
            _require(not destination.exists(), "restore_destination_appeared_during_validation")
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        return receipt
