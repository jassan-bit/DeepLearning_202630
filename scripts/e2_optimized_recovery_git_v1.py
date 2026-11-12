"""Create-only Git receipts for one registered optimized E2 recovery.

This transport can write four JSON receipts on one dedicated recovery branch.
It does not write the scientific campaign or a source branch. Every new commit
has the fetched head as its only parent and is pushed without force. A racing
claim is returned to the caller for owner/hash validation; it is never replaced.
"""
from __future__ import annotations

import json
import math
import threading

from e2_github_git_transport import GitDataTransport
from e2_github_campaign import CampaignConflict, CampaignSyncError

REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "recoveries/e2-cuda-crop-11-v2/colab-loss-20261112-r1"
RECEIPT_FILES = frozenset({"authorization.json", "started.json", "observed.json", "completed.json"})
MAX_RECEIPT_BYTES = 4 * 1024 * 1024


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonfinite JSON number")


def _validate_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _validate_value(child)
    elif isinstance(value, list):
        for child in value:
            _validate_value(child)


def _validate_data(data):
    if type(data) is not bytes or not 0 < len(data) <= MAX_RECEIPT_BYTES:
        raise CampaignSyncError("recovery_receipt_bytes_invalid")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=_invalid_constant)
        if type(value) is not dict:
            raise ValueError("receipt must be a JSON object")
        _validate_value(value)
    except (UnicodeError, ValueError, RecursionError):
        raise CampaignSyncError("recovery_receipt_json_invalid") from None


class RecoveryGitTransport:
    """Durable, create-only receipts; exact owner semantics belong to the core.

    ``read`` and ``create`` accept only the four bare filenames above. ``create``
    returns existing bytes if another writer already created that path. Callers
    must validate the returned authorization, owner nonce, and receipt chain.
    """

    def __init__(self, repository=REPOSITORY, branch=BRANCH, *, token=None,
                 _test_remote=None, _allow_local_for_tests=False):
        if repository != REPOSITORY or branch != BRANCH:
            raise CampaignSyncError("fixed_recovery_repository_and_branch_required")
        # Reuse the frozen credential isolation and Git execution policy. Its
        # constructor only initializes a private bare cache; no push occurs.
        initial = "campaigns/e2-optimized-recovery-receipts"
        self._git = GitDataTransport(repository, initial, initial, token=token,
            _test_remote=_test_remote, _allow_local_for_tests=_allow_local_for_tests)
        self.repository, self.branch, self.prefix = repository, branch, branch
        self._git.branch = self._git.prefix = branch
        self._git._ref = "refs/heads/" + branch
        self._mutex = threading.RLock()
        self.head = None

    def _assert_binding(self):
        if (self.repository != REPOSITORY or self.branch != BRANCH or self.prefix != BRANCH
                or self._git.repository != REPOSITORY or self._git.branch != BRANCH
                or self._git.prefix != BRANCH or self._git._ref != "refs/heads/" + BRANCH):
            raise CampaignSyncError("recovery_transport_binding_changed")

    @staticmethod
    def _path(path):
        if type(path) is not str or path not in RECEIPT_FILES:
            raise CampaignSyncError("recovery_receipt_path_refused")
        return BRANCH + "/" + path

    def _inspect(self):
        self._assert_binding()
        code, _, body = self._git._fetch_ref()
        if code == 404:
            self.head = None
            return None, {}
        if code != 200:
            raise CampaignSyncError("recovery_ref_read_unconfirmed")
        head = self._git._sha(body.get("object", {}).get("sha"))
        entries = {}
        raw = self._git._run(["ls-tree", "-rz", "--full-tree", head]).stdout
        for row in raw.split(b"\0"):
            if not row:
                continue
            try:
                metadata, name = row.split(b"\t", 1)
                mode, kind, sha = metadata.decode("ascii").split()
                path = name.decode("utf-8")
            except (ValueError, UnicodeError):
                raise CampaignSyncError("recovery_tree_invalid") from None
            if (mode != "100644" or kind != "blob"
                    or path not in {BRANCH + "/" + name for name in RECEIPT_FILES}
                    or path in entries):
                raise CampaignSyncError("non_recovery_tree_refused")
            data = self._git._object(self._git._sha(sha), "blob")
            _validate_data(data)
            entries[path] = data
        self.head = head
        return head, entries

    def read(self, path):
        full = self._path(path)
        with self._mutex:
            _, entries = self._inspect()
            return entries.get(full)

    def create(self, path, data):
        full = self._path(path)
        _validate_data(data)
        with self._mutex:
            # Never publish the credential if a caller accidentally embeds it.
            token = self._git._environment.get("GITHUB_TOKEN", "").encode()
            if token and token in data:
                raise CampaignSyncError("credential_in_recovery_receipt_refused")
            head, entries = self._inspect()
            if full in entries:
                return entries[full]
            blob = self._git._sha(self._git._run(
                ["hash-object", "-w", "--stdin"], data=data).stdout.decode().strip())
            index = self._git._private / "recovery-index"
            index.unlink(missing_ok=True)
            env = {"GIT_INDEX_FILE": str(index)}
            base = self._git._run(["rev-parse", head + "^{tree}"]).stdout.decode().strip() if head else "--empty"
            self._git._run(["read-tree", base], environment=env)
            self._git._run(["update-index", "-z", "--index-info"],
                data=b"100644 " + blob.encode() + b"\t" + full.encode() + b"\0", environment=env)
            tree = self._git._sha(self._git._run(["write-tree"], environment=env).stdout.decode().strip())
            args = ["commit-tree", tree] + (["-p", head] if head else [])
            commit = self._git._sha(self._git._run(args,
                data=("Record immutable E2 recovery receipt: " + path + "\n").encode()).stdout.decode().strip())
            try:
                code, _, _ = self._git._push(commit, create=head is None)
            except CampaignSyncError:
                # Some servers report a concurrent ref lock as an ambiguous
                # push failure. A read-only confirmation can recover the one
                # persisted receipt; it never retries the write or the worker.
                _, actual = self._inspect()
                if full in actual:
                    return actual[full]
                raise
            if code not in (200, 201, 409):
                raise CampaignSyncError("recovery_receipt_push_unconfirmed")
            # A stale sibling push can only lose. Re-read the winner; do not
            # generate a new claim or retry the push with a different parent.
            _, actual = self._inspect()
            if full not in actual:
                raise CampaignConflict("recovery_receipt_head_conflict")
            if code in (200, 201) and actual[full] != data:
                raise CampaignSyncError("recovery_receipt_ack_bytes_changed")
            return actual[full]

    def close(self):
        self._git.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
