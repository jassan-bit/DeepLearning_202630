"""Authenticated, crash-recoverable GitHub storage for the R3 engine.

The AES-GCM index and bounded encrypted shards are committed with the public
allowlist archive and status in one non-forced Git reference update. No request
is made at import. Credentials are memory-only and errors use fixed codes.
Adapted from the attributable e3_seoi_v5_pilot github_store/controller design;
this is a new storage identity and never accepts an SEOI-V5 case.
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
import uuid
import zipfile

VERSION = "e3-temporal-confirmation-github-1"
MAGIC = b"E3R3BK01"
MAX_BLOB = 8 * 1024 * 1024
CHUNK_BYTES = MAX_BLOB - 8192
MAX_RESPONSE = 16 * 1024 * 1024
MAX_MANIFEST = MAX_BLOB - 8192
MAX_FILE = 16 * 1024**3
MAX_TOTAL = 512 * 1024**3
MAX_FILES = 100000
KDF = {"name": "scrypt", "n": 32768, "r": 8, "p": 1, "length": 32}
ROOT_PRIVATE = frozenset((
    "plan.json", "status.json", "session.json", "selected_manifest.json", "training_seal.json", "analysis_receipt.json", "source_identity.json",
    "prepared.json", "preparation.json", "identity.json",
    "source_manifest.json", "protocol.json", "protocol_closed.json", "admission.json",
    "planning.json", "sizing.json", "budget.json", "selection.json", "fit_status.json",
    "failures.json", "attempts.json", "costs.json", "compute_ledger.json", "events.jsonl",
    "github_session.json", "remote_restore.json", "recovery_history.json",
    "evaluation_started.json", "evaluation_sealed.json", "evaluation_receipt.json",
    "evaluation.json", "final_evaluation.json", "results.json", "metrics.json",
))
JOB_META = frozenset(("job.json", "request.json", "status.json", "checkpoint.json", "latest.json",
                      "best.json", "curve.json", "curves.json", "metrics.json", "receipt.json",
                      "failure.json", "failures.json", "attempts.json", "epoch_pending.json", "selected.json",
                      "running.json", "done.json", "resources.json"))
POINTER_NAMES = ("checkpoint.json", "latest.json", "best.json", "selected.json")
DENY_COMPONENTS = frozenset(("inputs", "input", "raw", "datasets", "data", "sources", "source", "tokens", "secrets"))
PUBLIC_DIRS = frozenset(("predictions", "metrics", "status", "docs", "receipts"))
PUBLIC_SUFFIXES = frozenset((".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".npz", ".npy"))
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GITSHA = re.compile(r"[0-9a-f]{40}\Z")


class StorageError(RuntimeError):
    """Sanitized diagnostic; never contains HTTP response bodies or credentials."""


def _require(condition, code):
    if not condition:
        raise StorageError(code)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError):
        raise StorageError("noncanonical_storage_metadata") from None


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _safe_name(name):
    return (isinstance(name, str) and bool(name) and "\\" not in name and "\x00" not in name
            and not PurePosixPath(name).is_absolute() and PurePosixPath(name).as_posix() == name
            and all(p not in ("", ".", "..") for p in name.split("/")))


def _json(raw, code="invalid_storage_json"):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, code)
            value[key] = item
        return value
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, TypeError):
        raise StorageError(code) from None


def _atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def classify_http(status, message, headers=None, *, now=None):
    """Separate primary/secondary limits, permissions, ref races and transport."""
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
    if status == 404:
        return {"kind": "repository_or_object_missing_or_read_permission", "retryable": False}
    if status in (401, 403):
        return {"kind": "authentication_or_contents_write_permission", "retryable": False}
    if status in (409, 412) or (status == 422 and any(x in lower for x in ("fast forward", "reference", "already exists"))):
        return {"kind": "ref_conflict", "retryable": False}
    if status in (408, 500, 502, 503, 504) or (status == 422 and "timed out" in lower):
        return {"kind": "transient_server_error", "retryable": True, "retry_after_seconds": 5. if delay is None else delay}
    return {"kind": "http_error", "retryable": False}


class GithubStore:
    def __init__(self, token, backup_secret, repository, branch, *, source_commit, prefix,
                 sleep=time.sleep, clock=time.time, opener=None, chunk_bytes=CHUNK_BYTES):
        _require(isinstance(token, str) and 20 <= len(token) <= 4096 and token.isascii()
                 and all(33 <= ord(c) <= 126 for c in token), "github_token_required")
        _require(isinstance(backup_secret, str) and 32 <= len(backup_secret) <= 16384 and backup_secret != token,
                 "independent_backup_secret_at_least_32_characters_required")
        _require(isinstance(repository, str) and bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)), "invalid_repository")
        _require(_safe_name(branch) and not any(c in branch for c in " ~^:?*[") and not branch.endswith((".", ".lock"))
                 and ".." not in branch and "@{" not in branch, "invalid_branch")
        _require(_safe_name(prefix), "invalid_repository_prefix")
        _require(isinstance(source_commit, str) and bool(GITSHA.fullmatch(source_commit)), "immutable_source_commit_required")
        _require(type(chunk_bytes) is int and 1 <= chunk_bytes <= CHUNK_BYTES, "invalid_encrypted_chunk_size")
        _require(all(token not in value and backup_secret not in value for value in (repository, branch, prefix, source_commit)),
                 "credentials_in_storage_identity_refused")
        self.repository, self.branch, self.source_commit, self.prefix = repository, branch, source_commit, prefix
        self.chunk_bytes = chunk_bytes
        self._token, self._secret = token, backup_secret
        self._sleep, self._clock, self._opener = sleep, clock, opener or urllib.request.urlopen
        self._last_write = None
        self._root = "/repos/" + repository
        self._verified_chunks = set()
        self._verified_files = {}
        self._verified_objects = set()
        self._authenticated_head = None
        self._remote_manifest = None
        self._remote_status = None

    def __repr__(self):
        return "GithubStore(repository=" + repr(self.repository) + ",branch=" + repr(self.branch) + ",credentials=<hidden>)"

    def _wait(self, seconds):
        while seconds > 0:
            chunk = min(seconds, 30.)
            self._sleep(chunk)
            seconds -= chunk

    def _request(self, method, path, payload=None, *, allow_missing=False):
        body = None if payload is None else _canonical(payload)
        for attempt in range(5):
            if method != "GET":
                if self._last_write is not None:
                    self._wait(max(0., 1.25 - (self._clock() - self._last_write)))
                self._last_write = self._clock()
            request = urllib.request.Request("https://api.github.com" + path, data=body, method=method,
                headers={"Accept": "application/vnd.github+json", "Authorization": "Bearer " + self._token,
                         "Content-Type": "application/json", "User-Agent": VERSION, "X-GitHub-Api-Version": "2022-11-28"})
            try:
                with self._opener(request, timeout=90) as response:
                    raw = response.read(MAX_RESPONSE + 1)
                _require(len(raw) <= MAX_RESPONSE, "github_response_too_large")
                return _json(raw, "github_invalid_json_response") if raw else {}
            except urllib.error.HTTPError as error:
                if error.code == 404 and allow_missing:
                    return None
                try:
                    diagnostic = json.loads(error.read(65536)).get("message", "")
                except (ValueError, UnicodeError, TypeError, AttributeError):
                    diagnostic = ""
                category = classify_http(error.code, diagnostic, dict(error.headers or {}), now=self._clock())
                if not category["retryable"] or attempt == 4:
                    raise StorageError("github_" + category["kind"] + " status=" + str(error.code)) from None
                self._wait(max(category.get("retry_after_seconds", 5.), 2. ** attempt))
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 4:
                    raise StorageError("github_transport_retry_same_pending_snapshot") from None
                self._wait(2. ** attempt)
        raise StorageError("github_transport_retry_same_pending_snapshot")

    def _ref(self):
        return self._request("GET", self._root + "/git/ref/heads/" + urllib.parse.quote(self.branch, safe=""), allow_missing=True)

    def _file_info(self, name, commit, *, allow_missing=False):
        _require(_safe_name(name), "unsafe_remote_member")
        value = self._request("GET", self._root + "/contents/" + urllib.parse.quote(self.prefix + "/" + name, safe="/") + "?ref=" + commit,
                              allow_missing=allow_missing)
        if value is None:
            return None
        _require(isinstance(value, dict) and value.get("type") == "file" and bool(GITSHA.fullmatch(value.get("sha", "")))
                 and type(value.get("size")) is int and 0 <= value["size"] <= MAX_BLOB, "github_expected_bounded_regular_file")
        return value

    def _file_bytes(self, name, commit, limit=MAX_BLOB):
        info = self._file_info(name, commit)
        _require(info["size"] <= limit, "github_blob_exceeds_bound")
        blob = self._request("GET", self._root + "/git/blobs/" + info["sha"])
        try:
            _require(isinstance(blob, dict) and blob.get("encoding") == "base64", "github_invalid_blob_encoding")
            raw = base64.b64decode(blob["content"], validate=False)
        except (ValueError, KeyError, TypeError):
            raise StorageError("github_invalid_blob_encoding") from None
        _require(len(raw) == info["size"] and _git_blob_sha(raw) == info["sha"], "github_blob_readback_mismatch")
        self._verified_objects.add(info["sha"])
        return raw

    def _context(self, plan):
        _require(isinstance(plan, dict) and isinstance(plan.get("case_id"), str)
                 and plan["case_id"].startswith(("r3-engine-plan-", "r3-engine-final-")), "new_r3_engine_case_identity_required")
        _require(plan.get("source_commit") == self.source_commit, "plan_immutable_source_commit_differs")
        # Entire canonical plan binds every protocol/data/source/seed field; no
        # optional field can escape AAD by using another spelling in a caller.
        return {"version": VERSION, "repository": self.repository, "branch": self.branch,
                "prefix": self.prefix, "source_commit": self.source_commit,
                "case_id": plan["case_id"], "plan_sha256": _sha(_canonical(plan))}

    def _encrypt(self, plaintext, context, *, kind="manifest", content_sha256=None):
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        _require(len(plaintext) <= MAX_MANIFEST, "encrypted_object_plaintext_exceeds_bound")
        salt, nonce = os.urandom(16), os.urandom(12)
        header = _canonical({"context": context, "kind": kind, "content_sha256": content_sha256 or _sha(plaintext), "kdf": KDF})
        key = Scrypt(salt=salt, length=32, n=32768, r=8, p=1).derive(self._secret.encode())
        aad = MAGIC + len(header).to_bytes(4, "big") + header + salt + nonce
        raw = aad + AESGCM(key).encrypt(nonce, plaintext, aad)
        _require(len(raw) <= MAX_BLOB, "encrypted_git_blob_exceeds_8MiB")
        return raw

    def _decrypt(self, payload, context, *, kind="manifest", content_sha256=None):
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        _require(56 <= len(payload) <= MAX_BLOB and payload[:8] == MAGIC, "invalid_encrypted_backup_envelope")
        length = int.from_bytes(payload[8:12], "big")
        _require(0 < length <= 8192 and len(payload) >= 40 + length + 16, "invalid_encrypted_backup_header")
        header = _json(payload[12:12+length], "invalid_encrypted_backup_context")
        _require(isinstance(header, dict) and header.get("context") == context and header.get("kdf") == KDF and header.get("kind") == kind,
                 "encrypted_backup_context_or_KDF_differs")
        _require(bool(SHA256.fullmatch(header.get("content_sha256", ""))) and
                 (content_sha256 is None or header["content_sha256"] == content_sha256), "encrypted_backup_content_identity_differs")
        salt, nonce = payload[12+length:28+length], payload[28+length:40+length]
        key = Scrypt(salt=salt, length=32, n=32768, r=8, p=1).derive(self._secret.encode())
        try:
            raw = AESGCM(key).decrypt(nonce, payload[40+length:], payload[:40+length])
        except InvalidTag:
            raise StorageError("backup_decryption_failed_wrong_secret_or_tampering") from None
        _require(_sha(raw) == header["content_sha256"], "decrypted_backup_hash_differs")
        return raw

    def _case_path(self, case, name):
        _require(_safe_name(name), "unsafe_case_member")
        path = case / name
        _require(path.resolve().is_relative_to(case.resolve()), "case_member_escapes_root")
        current = path
        while current != case:
            _require(not current.is_symlink(), "case_symlink_refused")
            current = current.parent
        _require(path.is_file() and path.stat().st_size <= MAX_FILE, "case_member_nonregular_or_too_large")
        return path

    def _read_small(self, case, name):
        path = self._case_path(case, name)
        _require(path.stat().st_size <= MAX_MANIFEST, "metadata_file_exceeds_bound")
        raw = path.read_bytes()
        self._no_secrets(raw)
        return raw

    def _no_secrets(self, raw):
        _require(self._token.encode() not in raw and self._secret.encode() not in raw,
                 "credential_material_detected_refuse_upload")

    def _eligible(self, name):
        if not _safe_name(name) or any(p.lower() in DENY_COMPONENTS for p in name.split("/")):
            return False
        if name in ROOT_PRIVATE:
            return True
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == "jobs":
            return parts[-1] in JOB_META or (parts[-1].endswith(".pt") and bool(re.fullmatch(r"[A-Za-z0-9_.-]+\.pt", parts[-1])))
        return (parts[0] in {"receipts", "evaluation", "evaluations", "planning", "predictions", "metrics"}
                and Path(name).suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".npy", ".npz"})

    def _members(self, case, plan):
        names = {n for n in ROOT_PRIVATE if (case/n).exists()}
        _require("plan.json" in names, "case_plan_required")
        jobs = plan.get("jobs", [])
        _require(isinstance(jobs, list), "plan_jobs_list_required")
        for job in jobs:
            _require(isinstance(job, dict) and _safe_name(job.get("job_id", job.get("id"))), "invalid_job_identity")
            directory = "jobs/" + job.get("job_id", job.get("id"))
            for basename in JOB_META:
                name = directory + "/" + basename
                if (case/name).exists():
                    names.add(name)
            for basename in POINTER_NAMES:
                pointer_name = directory + "/" + basename
                if (case/pointer_name).exists():
                    pointer = _json(self._read_small(case, pointer_name), "invalid_checkpoint_pointer")
                    _require(isinstance(pointer, dict), "invalid_checkpoint_pointer")
                    # A pointer may identify latest/best together or be one record.
                    candidates = [pointer] + [pointer[k] for k in ("latest", "best", "selected") if isinstance(pointer.get(k), dict)]
                    for record in candidates:
                        filename = record.get("file", record.get("path"))
                        if filename is None:
                            continue
                        _require(_safe_name(filename), "unsafe_checkpoint_pointer")
                        name = filename if filename.startswith(directory + "/") else directory + "/" + filename
                        _require(self._eligible(name) and name.endswith(".pt"), "checkpoint_pointer_not_allowlisted")
                        path = self._case_path(case, name)
                        _require(type(record.get("bytes")) is int and path.stat().st_size == record["bytes"]
                                 and bool(SHA256.fullmatch(record.get("sha256", ""))), "checkpoint_pointer_size_or_hash_missing")
                        digest = hashlib.sha256()
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(self.chunk_bytes), b""):
                                digest.update(chunk)
                        _require(digest.hexdigest() == record["sha256"], "checkpoint_pointer_hash_differs")
                        names.add(name)
        for folder in ("receipts", "evaluation", "evaluations", "planning", "predictions", "metrics"):
            base = case/folder
            if base.exists():
                _require(not base.is_symlink(), "case_symlink_refused")
                for path in base.rglob("*"):
                    _require(not path.is_symlink(), "case_symlink_refused")
                    if path.is_file():
                        name = path.relative_to(case).as_posix()
                        _require(self._eligible(name), "private_derived_member_not_allowlisted")
                        names.add(name)
        _require(len(names) <= MAX_FILES and all(self._eligible(n) for n in names), "private_member_inventory_invalid")
        return sorted(names)

    def _sealed(self, case, plan):
        for name in ("evaluation_sealed.json", "evaluation_receipt.json"):
            if (case/name).exists():
                receipt = _json(self._read_small(case, name))
                if (receipt.get("sealed") is True or receipt.get("evaluation_sealed") is True) and receipt.get("case_id") == plan["case_id"]:
                    return True
        return False

    def _public_archive(self, case, plan, public_files):
        _require(public_files is None or isinstance(public_files, dict), "explicit_public_file_dictionary_required")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, value in sorted((public_files or {}).items()):
                _require(_safe_name(name) and not any(p.lower() in DENY_COMPONENTS for p in name.split("/")), "unsafe_public_member")
                parts = name.split("/")
                _require(parts[0] in PUBLIC_DIRS and Path(name).suffix.lower() in PUBLIC_SUFFIXES, "public_member_not_allowlisted")
                # Every prediction from a final case is considered final-test
                # material; names cannot evade the seal by omitting 'test'.
                if parts[0] == "predictions" and (plan["case_id"].startswith("r3-engine-final-") or "test" in name.lower() or "final" in name.lower()):
                    _require(self._sealed(case, plan), "final_predictions_require_sealed_evaluation_receipt")
                if isinstance(value, (str, Path)):
                    path = Path(value)
                    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BLOB,
                             "public_source_nonregular_or_too_large")
                    raw = path.read_bytes()
                else:
                    _require(isinstance(value, bytes), "public_file_bytes_or_path_required")
                    raw = value
                _require(len(raw) <= MAX_BLOB, "public_member_exceeds_bound")
                self._no_secrets(raw)
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(info, raw)
        result = stream.getvalue()
        _require(len(result) <= MAX_BLOB, "public_allowlist_archive_exceeds_8MiB_use_selected_summaries")
        return result

    def _validate_manifest(self, manifest, context):
        _require(isinstance(manifest, dict) and manifest.get("context") == context and manifest.get("format") == "chunked-private-files-1",
                 "backup_manifest_context_or_format_differs")
        records = manifest.get("files")
        _require(isinstance(records, list) and 0 < len(records) <= MAX_FILES, "backup_manifest_file_count_invalid")
        names = set()
        total = 0
        for record in records:
            _require(isinstance(record, dict), "backup_manifest_record_invalid")
            name, size, digest = record.get("path"), record.get("bytes"), record.get("sha256")
            _require(self._eligible(name) and name not in names and type(size) is int and 0 <= size <= MAX_FILE
                     and isinstance(digest, str) and bool(SHA256.fullmatch(digest)), "backup_manifest_member_invalid")
            names.add(name)
            total += size
            chunks = record.get("chunks")
            _require(isinstance(chunks, list) and len(chunks) <= (MAX_FILE // 1 + 1), "backup_manifest_chunks_invalid")
            chunk_size = 0
            for chunk in chunks:
                _require(isinstance(chunk, dict) and type(chunk.get("plain_bytes")) is int and 0 < chunk["plain_bytes"] <= CHUNK_BYTES
                         and type(chunk.get("cipher_bytes")) is int and 56 <= chunk["cipher_bytes"] <= MAX_BLOB
                         and isinstance(chunk.get("plain_sha256"), str) and bool(SHA256.fullmatch(chunk["plain_sha256"]))
                         and isinstance(chunk.get("cipher_sha256"), str) and bool(SHA256.fullmatch(chunk["cipher_sha256"]))
                         and isinstance(chunk.get("git_sha"), str) and bool(GITSHA.fullmatch(chunk["git_sha"]))
                         and chunk.get("path") == "chunks/" + chunk["cipher_sha256"][:2] + "/" + chunk["cipher_sha256"] + ".aesgcm",
                         "backup_manifest_chunk_invalid")
                chunk_size += chunk["plain_bytes"]
            _require(chunk_size == size, "backup_manifest_chunk_lengths_differ")
        _require("plan.json" in names and total <= MAX_TOTAL, "backup_manifest_total_or_plan_invalid")
        _require(type(manifest.get("public_results_bytes")) is int and 0 <= manifest["public_results_bytes"] <= MAX_BLOB
                 and isinstance(manifest.get("public_results_sha256"), str) and bool(SHA256.fullmatch(manifest["public_results_sha256"])),
                 "backup_public_identity_missing")
        return records

    def _chunk_plain(self, chunk, context, commit, *, local=None):
        if local is not None and (local/chunk["path"]).is_file():
            raw = (local/chunk["path"]).read_bytes()
        else:
            raw = self._file_bytes(chunk["path"], commit)
        _require(len(raw) == chunk["cipher_bytes"] and _sha(raw) == chunk["cipher_sha256"] and _git_blob_sha(raw) == chunk["git_sha"],
                 "encrypted_shard_remote_hash_differs")
        plain = self._decrypt(raw, context, kind="chunk", content_sha256=chunk["plain_sha256"])
        _require(len(plain) == chunk["plain_bytes"], "decrypted_shard_size_differs")
        return plain

    def _verify_files(self, manifest, context, commit, *, destination=None, local=None):
        records = self._validate_manifest(manifest, context)
        plan = None
        available = {r["path"]: r for r in records}
        referenced_checkpoints = set()
        def verify_pointer(record, metadata):
            # Cached pointer bytes are authenticated by this exact context and
            # full manifest-record identity. Their references must nevertheless
            # be checked against the CURRENT manifest on every verification.
            pointer = _json(metadata, "restored_checkpoint_pointer_invalid")
            _require(isinstance(pointer, dict), "restored_checkpoint_pointer_invalid")
            candidates = [pointer] + [pointer[k] for k in ("latest", "best", "selected") if isinstance(pointer.get(k), dict)]
            directory = str(PurePosixPath(record["path"]).parent)
            for item in candidates:
                filename = item.get("file", item.get("path"))
                if filename is not None:
                    _require(_safe_name(filename), "restored_checkpoint_pointer_invalid")
                    name = filename if filename.startswith(directory + "/") else directory + "/" + filename
                    referenced_checkpoints.add(name)
                    _require(name in available and available[name]["sha256"] == item.get("sha256") and available[name]["bytes"] == item.get("bytes"),
                             "restored_checkpoint_pointer_hash_differs")
        for record in records:
            record_key = (_sha(_canonical(context)), _sha(_canonical(record)))
            cached = self._verified_files.get(record_key)
            if destination is None and cached is not None:
                if local is None:
                    for chunk in record["chunks"]:
                        info = self._file_info(chunk["path"], commit)
                        _require(info["sha"] == chunk["git_sha"] and info["size"] == chunk["cipher_bytes"],
                                 "cached_shard_path_readback_mismatch")
                if record["path"] == "plan.json":
                    plan = _json(cached["metadata"], "restored_plan_invalid")
                elif record["path"].startswith("jobs/") and Path(record["path"]).name in POINTER_NAMES:
                    verify_pointer(record,cached["metadata"])
                continue
            target = None
            if destination is not None:
                target = destination/record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
            stream = target.open("wb") if target is not None else None
            digest = hashlib.sha256()
            captured = bytearray() if record["path"] == "plan.json" or (record["path"].startswith("jobs/") and Path(record["path"]).name in POINTER_NAMES) else None
            try:
                for chunk in record["chunks"]:
                    cache_key = (_sha(_canonical(context)), chunk["cipher_sha256"])
                    # Full file hashes and pointer verification are intentionally
                    # repeated on restoration. status may reuse authenticated,
                    # immutable shard identities in this store instance only.
                    plain = self._chunk_plain(chunk, context, commit, local=local)
                    self._verified_chunks.add(cache_key)
                    digest.update(plain)
                    if captured is not None:
                        _require(len(captured) + len(plain) <= MAX_MANIFEST, "restored_metadata_exceeds_bound")
                        captured.extend(plain)
                    if stream is not None:
                        stream.write(plain)
                _require(digest.hexdigest() == record["sha256"], "restored_file_hash_differs")
            finally:
                if stream is not None:
                    stream.close()
            self._verified_files[record_key] = {"metadata": bytes(captured) if captured is not None else b""}
            if record["path"] == "plan.json":
                plan = _json(captured, "restored_plan_invalid")
            elif captured is not None:
                verify_pointer(record,captured)
        _require({n for n in available if n.endswith(".pt")} == referenced_checkpoints, "unreferenced_checkpoint_payload_refused")
        _require(self._context(plan) == context, "restored_plan_identity_differs")
        return plan

    def status(self):
        ref = self._ref()
        if ref is None:
            return {"exists": False, "repository": self.repository, "branch": self.branch, "source_commit": self.source_commit}
        _require(isinstance(ref, dict) and isinstance(ref.get("object"), dict), "github_invalid_ref")
        head = ref["object"].get("sha")
        _require(isinstance(head, str) and bool(GITSHA.fullmatch(head)), "github_invalid_ref")
        if self._authenticated_head == head and self._remote_status is not None:
            return dict(self._remote_status)
        _require(self._file_info("status.json", head, allow_missing=True) is not None,
                 "existing_branch_has_no_managed_status_refuse_overwrite")
        value = _json(self._file_bytes("status.json", head, MAX_MANIFEST), "github_invalid_status")
        required = ("version", "repository", "branch", "prefix", "source_commit", "case_id", "plan_sha256")
        _require(isinstance(value, dict) and all(k in value for k in required), "github_status_identity_missing")
        context = {k: value[k] for k in required}
        _require(all(context[k] == getattr(self, k) for k in ("repository", "branch", "prefix", "source_commit"))
                 and context["version"] == VERSION, "github_remote_source_or_storage_identity_differs")
        encrypted = self._file_bytes("checkpoint.aesgcm", head)
        _require(len(encrypted) == value.get("checkpoint_bytes") and _sha(encrypted) == value.get("checkpoint_sha256"), "encrypted_backup_remote_hash_differs")
        manifest = _json(self._decrypt(encrypted, context), "invalid_decrypted_manifest")
        self._validate_manifest(manifest, context)
        _require(manifest["public_results_bytes"] == value.get("public_results_bytes")
                 and manifest["public_results_sha256"] == value.get("public_results_sha256"), "public_results_authenticated_identity_differs")
        public = self._file_bytes("public_results.zip", head)
        _require(len(public) == manifest["public_results_bytes"] and _sha(public) == manifest["public_results_sha256"], "public_results_remote_hash_differs")
        self._verify_files(manifest, context, head)
        expected = {"private_files": len(manifest["files"]), "private_plain_bytes": sum(r["bytes"] for r in manifest["files"]),
                    "raw_inputs_uploaded": False, "weights_uploaded_in_plaintext": False, "R3_accredited": False,
                    "backup_algorithm": "AES-256-GCM/Scrypt-n32768-r8-p1-independent-passphrase"}
        _require(all(value.get(k) == v for k, v in expected.items()), "github_status_inventory_or_safety_flags_differ")
        _require(isinstance(value.get("stage"), str) and len(value["stage"]) <= 100, "github_status_stage_invalid")
        self._no_secrets(_canonical(value))
        keep = set(context) | set(expected) | {"checkpoint_sha256", "checkpoint_bytes", "public_results_sha256", "public_results_bytes", "backup_created_at_utc", "stage"}
        value = {k: v for k, v in value.items() if k in keep}
        value = {**value, "exists": True, "head": head, "head_commit": head, "remote_verified": True}
        self._authenticated_head, self._remote_manifest, self._remote_status = head, manifest, value
        return dict(value)

    def _pending(self, case, public_files, stage, remote):
        folder = case/".github_pending"
        plan = _json(self._read_small(case, "plan.json"), "case_plan_invalid")
        context = self._context(plan)
        if folder.exists():
            _require(not folder.is_symlink(), "pending_symlink_refused")
            record = _json(self._read_small(folder, "publication.json"), "pending_record_invalid")
            _require(isinstance(record, dict) and record.get("context") == context and isinstance(record.get("sha256"), dict), "pending_snapshot_case_identity_differs")
            for name, digest in record["sha256"].items():
                path = self._case_path(folder, name)
                _require(path.stat().st_size <= MAX_BLOB and _sha(path.read_bytes()) == digest, "pending_payload_hash_differs")
            _require({"checkpoint.aesgcm", "status.json", "public_results.zip"}.issubset(record.get("sha256", {})), "pending_transaction_incomplete")
            manifest = _json(self._decrypt((folder/"checkpoint.aesgcm").read_bytes(), context), "pending_manifest_invalid")
            self._validate_manifest(manifest, context)
            allowed = {"checkpoint.aesgcm", "status.json", "public_results.zip"}
            allowed.update(c["path"] for f in manifest["files"] for c in f["chunks"])
            _require(set(record["sha256"]).issubset(allowed), "pending_member_not_allowlisted")
            metadata = _json((folder/"status.json").read_bytes(), "pending_status_invalid")
            _require(isinstance(metadata, dict) and all(metadata.get(k) == v for k, v in context.items())
                     and metadata.get("checkpoint_sha256") == record["sha256"]["checkpoint.aesgcm"]
                     and metadata.get("checkpoint_bytes") == (folder/"checkpoint.aesgcm").stat().st_size
                     and metadata.get("public_results_sha256") == manifest["public_results_sha256"]
                     and record["sha256"]["public_results.zip"] == manifest["public_results_sha256"]
                     and metadata.get("public_results_bytes") == manifest["public_results_bytes"]
                     and (folder/"public_results.zip").stat().st_size == manifest["public_results_bytes"], "pending_status_identity_differs")
            return folder, record, manifest
        public = self._public_archive(case, plan, public_files)
        staging = Path(tempfile.mkdtemp(prefix=".github-staging-", dir=case))
        old = {}
        if remote["exists"]:
            for f in self._remote_manifest["files"]:
                for chunk in f["chunks"]:
                    old[(chunk["plain_sha256"], chunk["plain_bytes"])] = chunk
        records, local_hashes = [], {}
        total = 0
        try:
            for name in self._members(case, plan):
                path = self._case_path(case, name)
                chunks, digest, size = [], hashlib.sha256(), 0
                overlap = b""
                with path.open("rb") as stream:
                    while True:
                        plain = stream.read(self.chunk_bytes)
                        if not plain:
                            break
                        self._no_secrets(overlap + plain)
                        overlap = (overlap + plain)[-max(len(self._token.encode()), len(self._secret.encode())):]
                        size += len(plain)
                        digest.update(plain)
                        chunk_key = (_sha(plain), len(plain))
                        if chunk_key in old:
                            chunk = old[chunk_key]
                        else:
                            cipher = self._encrypt(plain, context, kind="chunk", content_sha256=chunk_key[0])
                            sha = _sha(cipher)
                            shard = "chunks/" + sha[:2] + "/" + sha + ".aesgcm"
                            chunk = {"path": shard, "plain_sha256": chunk_key[0], "plain_bytes": len(plain),
                                     "cipher_sha256": sha, "cipher_bytes": len(cipher), "git_sha": _git_blob_sha(cipher)}
                            _atomic(staging/shard, cipher)
                            local_hashes[shard] = sha
                            old[chunk_key] = chunk
                        chunks.append(chunk)
                _require(size == path.stat().st_size, "case_file_changed_during_backup")
                total += size
                _require(total <= MAX_TOTAL, "private_snapshot_exceeds_512GiB")
                records.append({"path": name, "bytes": size, "sha256": digest.hexdigest(), "chunks": chunks})
            manifest = {"format": "chunked-private-files-1", "context": context, "files": records,
                        "public_results_sha256": _sha(public), "public_results_bytes": len(public),
                        "raw_inputs_included": False, "only_referenced_latest_best_checkpoints": True}
            self._validate_manifest(manifest, context)
            checkpoint = self._encrypt(_canonical(manifest), context)
            event = str(stage.get("event", stage.get("kind", "checkpoint")) if isinstance(stage, dict) else stage or "checkpoint")
            self._no_secrets(event.encode())
            event = event[:100]
            metadata = {**context, "checkpoint_sha256": _sha(checkpoint), "checkpoint_bytes": len(checkpoint),
                        "public_results_sha256": _sha(public), "public_results_bytes": len(public),
                        "private_files": len(records), "private_plain_bytes": total,
                        "backup_created_at_utc": datetime.now(timezone.utc).isoformat(), "stage": event,
                        "raw_inputs_uploaded": False, "weights_uploaded_in_plaintext": False,
                        "backup_algorithm": "AES-256-GCM/Scrypt-n32768-r8-p1-independent-passphrase", "R3_accredited": False}
            for name, raw in (("checkpoint.aesgcm", checkpoint), ("status.json", _canonical(metadata)), ("public_results.zip", public)):
                _atomic(staging/name, raw)
                local_hashes[name] = _sha(raw)
            record = {"context": context, "sha256": local_hashes, "base_head": None, "commit": None,
                      "created_at_utc": metadata["backup_created_at_utc"]}
            _atomic(staging/"publication.json", _canonical(record))
            _require(not folder.exists(), "pending_transaction_appeared_concurrently")
            os.rename(staging, folder)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return folder, record, manifest

    def _tree_update(self, base_sha, changes):
        """Update small immediate subtrees; never submit a huge flat source tree."""
        children = {}
        if base_sha is not None:
            existing = self._request("GET", self._root + "/git/trees/" + base_sha)
            _require(not existing.get("truncated", False), "github_immediate_tree_truncated")
            children = {item["path"]: item for item in existing.get("tree", [])}
        direct, nested = [], {}
        for name, sha in sorted(changes.items()):
            first, sep, rest = name.partition("/")
            if sep:
                nested.setdefault(first, {})[rest] = sha
            else:
                direct.append({"path": first, "mode": "100644", "type": "blob", "sha": sha})
        for name, subchanges in nested.items():
            previous = children.get(name)
            _require(previous is None or previous.get("type") == "tree", "github_prefix_collides_with_file")
            subtree = self._tree_update(None if previous is None else previous["sha"], subchanges)
            direct.append({"path": name, "mode": "040000", "type": "tree", "sha": subtree})
        current = base_sha
        # GitHub validates only <=128 immediate entries per mutation. The
        # existing source root is referenced by SHA, never expanded into JSON.
        for offset in range(0, len(direct), 128):
            payload = {"tree": direct[offset:offset+128]}
            if current is not None:
                payload["base_tree"] = current
            result = self._request("POST", self._root + "/git/trees", payload)
            current = result.get("sha")
            _require(isinstance(current, str) and bool(GITSHA.fullmatch(current)), "github_created_tree_identity_missing")
        _require(current is not None, "github_empty_tree_update")
        return current

    def backup(self, case_root, public_files=None, *, stage="checkpoint"):
        case = Path(case_root).absolute()
        _require(case.is_dir() and not case.is_symlink(), "existing_case_directory_required")
        # status authenticates every remote file before a pending snapshot or a
        # mutation exists, so a typo cannot silently rotate the encryption key.
        remote = self.status()
        plan = _json(self._read_small(case, "plan.json"), "case_plan_invalid")
        context = self._context(plan)
        if remote["exists"]:
            _require(all(remote.get(k) == v for k, v in context.items()), "github_remote_case_or_plan_differs")
        folder, pending, manifest = self._pending(case, public_files, stage, remote)
        metadata = _json((folder/"status.json").read_bytes())
        # Authenticate persisted pending key before any writes, including after
        # recovery from another process; hashes alone never establish key use.
        self._decrypt((folder/"checkpoint.aesgcm").read_bytes(), context)
        if remote["exists"] and remote["checkpoint_sha256"] == metadata["checkpoint_sha256"]:
            commit = remote["head"]
        else:
            marker_path = case/".github_backup_receipt.json"
            marker = _json(self._read_small(case, ".github_backup_receipt.json")) if marker_path.exists() else None
            if remote["exists"]:
                _require(marker is not None and marker.get("commit") == remote["head"], "stale_local_case_restore_remote_before_overwrite")
                expected = remote["head"]
            else:
                _require(marker is None, "previous_remote_branch_disappeared_refuse_recreate")
                expected = self.source_commit
            _require(pending.get("base_head") in (None, expected), "pending_snapshot_base_head_conflict")
            pending["base_head"] = expected
            _atomic(folder/"publication.json", _canonical(pending))
            # Verify all pending/reused chunks and plaintext file hashes before
            # publishing; incomplete local staging cannot produce a broken ref.
            self._verify_files(manifest, context, remote.get("head"), local=folder)
            changes = {}
            for name, digest in pending["sha256"].items():
                raw = (folder/name).read_bytes()
                _require(_sha(raw) == digest, "pending_payload_hash_differs")
                sha = _git_blob_sha(raw)
                if sha not in self._verified_objects:
                    blob = self._request("POST", self._root + "/git/blobs", {"content": base64.b64encode(raw).decode(), "encoding": "base64"})
                    _require(blob.get("sha") == sha, "github_created_blob_identity_differs")
                changes[self.prefix + "/" + name] = sha
            if remote["exists"]:
                old_chunks = {c["path"] for f in self._remote_manifest["files"] for c in f["chunks"]}
                new_chunks = {c["path"] for f in manifest["files"] for c in f["chunks"]}
                changes.update({self.prefix + "/" + n: None for n in old_chunks - new_chunks})
            base_commit = self._request("GET", self._root + "/git/commits/" + expected)
            tree_sha = base_commit.get("tree", {}).get("sha")
            _require(isinstance(tree_sha, str) and bool(GITSHA.fullmatch(tree_sha)), "github_base_tree_missing")
            tree = self._tree_update(tree_sha, changes)
            author = {"name": "E3 temporal confirmation checkpoint", "email": "e3-r3@users.noreply.github.com", "date": pending["created_at_utc"]}
            candidate = self._request("POST", self._root + "/git/commits", {
                "message": "Preserve encrypted R3 engine state and explicit public derived results", "tree": tree,
                "parents": [expected], "author": author, "committer": author})
            commit = candidate.get("sha")
            _require(isinstance(commit, str) and bool(GITSHA.fullmatch(commit)), "github_candidate_commit_missing")
            pending["commit"] = commit
            _atomic(folder/"publication.json", _canonical(pending))
            current = self._ref()
            _require((current is None and not remote["exists"]) or (current is not None and current.get("object", {}).get("sha") == expected),
                     "github_ref_conflict_before_write")
            try:
                if remote["exists"]:
                    self._request("PATCH", self._root + "/git/refs/heads/" + urllib.parse.quote(self.branch, safe=""), {"sha": commit, "force": False})
                else:
                    self._request("POST", self._root + "/git/refs", {"ref": "refs/heads/" + self.branch, "sha": commit})
            except StorageError:
                ref = self._ref()
                if ref is None or ref.get("object", {}).get("sha") != commit:
                    raise
        # Read actual bytes, not only size/SHA metadata, after the ref update.
        for name in pending["sha256"]:
            raw = self._file_bytes(name, commit)
            _require(_sha(raw) == pending["sha256"][name], "github_final_path_readback_mismatch")
        ref = self._ref()
        _require(ref is not None and ref.get("object", {}).get("sha") == commit, "github_ref_conflict_after_write")
        receipt = {"status": "verified", "remote_verified": True, "commit": commit, "head_commit": commit,
                   "branch": self.branch, "source_commit": self.source_commit, "case_id": context["case_id"],
                   "plan_sha256": context["plan_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                   "public_results_sha256": metadata["public_results_sha256"],
                   "url": "https://github.com/" + self.repository + "/tree/" + commit + "/" + self.prefix}
        _atomic(case/".github_backup_receipt.json", _canonical(receipt))
        self._authenticated_head, self._remote_manifest = commit, manifest
        self._remote_status = {**metadata, "exists": True, "head": commit, "head_commit": commit, "remote_verified": True}
        retired = case/(".github-completed-cleanup-" + uuid.uuid4().hex)
        os.rename(folder, retired)
        try:
            shutil.rmtree(retired)
        except OSError:
            pass
        return receipt

    def restore(self, case_root):
        destination = Path(case_root).absolute()
        _require(not destination.exists() and not destination.is_symlink(), "fresh_restore_destination_required")
        remote = self.status()
        if not remote["exists"]:
            return None
        context = self._remote_manifest["context"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".r3-restore-", dir=destination.parent))
        try:
            self._verify_files(self._remote_manifest, context, remote["head"], destination=temporary)
            session_path = temporary/"session.json" if (temporary/"session.json").exists() else temporary/"github_session.json"
            session = _json(session_path.read_bytes()) if session_path.exists() else {}
            recovery = {"restored_at_utc": datetime.now(timezone.utc).isoformat(), "commit": remote["head"],
                        "source_commit": self.source_commit, "case_id": context["case_id"],
                        "raw_inputs_restored": False, "raw_inputs_must_be_reacquired_and_hash_verified": True,
                        "unuploaded_progress_may_be_lost": True, "last_saved_session_open": session.get("session_open") is True,
                        "unobserved_remote_compute_possible": session.get("session_open") is True,
                        "cost_and_optimizer_steps_are_lower_bounds_if_session_was_open": session.get("session_open") is True}
            _atomic(temporary/"remote_restore.json", _canonical(recovery))
            history = _json((temporary/"recovery_history.json").read_bytes()) if (temporary/"recovery_history.json").exists() else []
            _require(isinstance(history, list), "invalid_recovery_history")
            history.append(recovery)
            _atomic(temporary/"recovery_history.json", _canonical(history))
            receipt = {"status": "restored", "remote_verified": True, "commit": remote["head"], "head_commit": remote["head"],
                       "branch": self.branch, "source_commit": self.source_commit, "case_id": context["case_id"],
                       "plan_sha256": context["plan_sha256"], "checkpoint_sha256": remote["checkpoint_sha256"],
                       "remote_restore": recovery, "input_data_required": True}
            _atomic(temporary/".github_backup_receipt.json", _canonical(receipt))
            _require(not destination.exists(), "restore_destination_appeared_during_validation")
            os.rename(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return receipt
