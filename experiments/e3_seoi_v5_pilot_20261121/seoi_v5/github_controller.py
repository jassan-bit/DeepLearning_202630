"""Synchronous encrypted GitHub persistence for the explicit SE-OI pilot.

Credentials are accepted only as a small JSON object on stdin. Importing this
module, setup, status and export never start a fit. AI assistance declared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time


class ControllerError(RuntimeError):
    """A fixed diagnostic code, never a credential-bearing HTTP exception."""


def require(condition, code):
    if not condition:
        raise ControllerError(code)


def credentials(stream):
    raw = stream.read(65537)
    require(len(raw) <= 65536, "credentials_input_too_large")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate_credentials_key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError):
        raise ControllerError("credentials_json_required_on_stdin") from None
    require(isinstance(value, dict) and set(value) == {"token", "backup_secret"},
            "exact_credentials_fields_required")
    require(isinstance(value["token"], str) and bool(value["token"].strip())
            and len(value["token"]) <= 4096, "github_token_required")
    require(isinstance(value["backup_secret"], str) and len(value["backup_secret"]) >= 32
            and len(value["backup_secret"]) <= 16384, "backup_secret_minimum_32_characters")
    return value


def redact(value, secrets):
    """Defense in depth for terminal output; credentials never become arguments."""
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {str(key): redact(item, secrets) for key, item in value.items()
                if str(key).lower() not in {"token", "backup_secret", "authorization"}}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    return value


def storage_code(error):
    """Retain fixed store diagnostics, never an arbitrary exception response."""
    code = getattr(error, "storage_diagnostic", None)
    if code is None:
        code = getattr(error, "code", None)
    if code is None and type(error).__name__ == "StorageError":
        code = str(error)
    if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,159}(?: status=[1-5][0-9]{2})?", code):
        return code
    return "storage_detail_unavailable"


class GithubController:
    def __init__(self, source_root, case_root, data_file, source_commit, store,
                 *, runner_module=None, monotonic=time.monotonic, redactions=()):
        if runner_module is None:
            from . import runner as runner_module
        self.runner = runner_module
        self.source_root = self.runner.safe(source_root)
        self.case_root = self.runner.safe(case_root)
        self.data_file = self.runner.safe(data_file) if data_file is not None else None
        require(isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit),
                "immutable_source_commit_required")
        self.source_commit, self.store = source_commit, store
        self.clock, self.last_success = monotonic, None
        self.redactions = tuple(redactions)

    def _durability(self, code):
        error = getattr(self.runner, "DurabilityError", ControllerError)
        return error(code)

    def _storage_failure(self, error, code):
        wrapped = self._durability(code)
        wrapped.storage_diagnostic = storage_code(error)
        return wrapped

    def _source(self):
        sources = self.runner.verify_sources(self.source_root)
        anchor = self.runner.read_json(self.source_root / "PUBLIC_SOURCE_ANCHOR.json")
        require(anchor.get("source_commit") == self.source_commit
                and anchor.get("manifest_sha256") == sources["manifest_sha256"],
                "pinned_public_source_anchor_differs")

    def _remote(self):
        try:
            remote = self.store.status()
        except Exception as error:
            raise self._storage_failure(error, "github_status_failed_local_evidence_retained") from None
        require(isinstance(remote, dict) and isinstance(remote.get("exists"), bool),
                "invalid_github_status_receipt")
        require(remote.get("source_commit") == self.source_commit, "remote_source_commit_differs")
        return remote

    def _raw(self):
        require(self.data_file is not None and self.data_file.is_file(), "pinned_data_cache_required")
        require(self.data_file.stat().st_size == self.runner.NPZ_BYTES
                and self.runner.sha(self.data_file) == self.runner.NPZ_SHA256,
                "pinned_data_cache_identity_differs")
        target = self.runner.safe(self.case_root / "inputs/cohort.npz")
        if target.exists():
            require(target.is_file() and target.stat().st_size == self.runner.NPZ_BYTES
                    and self.runner.sha(target) == self.runner.NPZ_SHA256,
                    "existing_case_raw_archive_differs")
        else:
            self.runner.atomic_bytes(target, self.data_file.read_bytes(), immutable=True)

    def _case(self, remote=None):
        plan, _, _ = self.runner.load_case(self.source_root, self.case_root, load_arrays=False)
        anchor_path = self.case_root / "PUBLIC_SOURCE_ANCHOR.json"
        require(anchor_path.is_file()
                and self.runner.read_json(anchor_path).get("source_commit") == self.source_commit,
                "case_public_source_anchor_differs")
        if remote is not None and remote["exists"]:
            require(remote.get("case_id") == plan["case_id"]
                    and remote.get("plan_sha256") == self.runner.digest(plan),
                    "remote_case_or_plan_identity_differs")
            marker = self.case_root / ".github_backup_receipt.json"
            require(marker.is_file(), "local_case_has_no_remote_receipt_preserve_and_restore_to_fresh_path")
            pending = self.runner.safe(self.case_root / ".github_pending")
            require(self.runner.read_json(marker).get("commit") == remote.get("head") or pending.exists(),
                    "local_case_is_stale_remote_head_differs")
        return plan

    def _session(self, opened, stage):
        self.runner.write_json(self.case_root / "github_session.json", {
            "session_open": bool(opened), "last_stage": stage,
            "recorded_at_utc": self.runner.utc_now(),
            "source_commit": self.source_commit,
            "recovery_scope": "An abrupt VM loss can leave unuploaded compute; a remote snapshot is not a complete physical work ledger.",
        })

    def _backup(self, stage):
        try:
            had_pending = self.runner.safe(self.case_root / ".github_pending").exists()
            receipt = self.store.backup(self.case_root, stage=stage)
            require(isinstance(receipt, dict) and receipt.get("remote_verified") is True
                    and receipt.get("source_commit") == self.source_commit,
                    "github_backup_not_verified")
            if had_pending:
                # Reconcile the previous immutable ciphertext first. Its success
                # alone need not cover newer local session or progress records.
                require(not self.runner.safe(self.case_root / ".github_pending").exists(),
                        "previous_pending_snapshot_not_settled")
                receipt = self.store.backup(self.case_root, stage=stage)
                require(isinstance(receipt, dict) and receipt.get("remote_verified") is True
                        and receipt.get("source_commit") == self.source_commit,
                        "github_latest_state_backup_not_verified")
        except Exception as error:
            raise self._storage_failure(error, "github_backup_failed_stop_before_further_fits_evidence_retained") from None
        self.last_success = self.clock()
        return receipt

    def setup(self):
        """Prepare once or restore authenticated state, then publish before any fit."""
        self._source()
        with self.runner.case_lock(self.case_root):
            remote = self._remote()
            restored = None
            if not self.case_root.exists():
                if remote["exists"]:
                    try:
                        restored = self.store.restore(self.case_root)
                    except Exception as error:
                        raise self._storage_failure(error, "github_restore_failed_no_training_started") from None
                    require(isinstance(restored, dict) and restored.get("remote_verified") is True,
                            "github_restore_not_verified")
                    self._raw()
                    self._case(remote)
                else:
                    require(self.data_file is not None, "pinned_data_cache_required")
                    self.runner.prepare(self.source_root, self.data_file, self.case_root)
                    self._case()
                    self._session(False, "prepared")
            else:
                self._raw()
                self._case(remote if remote["exists"] else None)
                require(remote["exists"] or not (self.case_root / ".github_backup_receipt.json").exists(),
                        "previously_saved_remote_branch_missing_preserve_case")
                require(not remote["exists"] or (self.case_root / ".github_backup_receipt.json").is_file(),
                        "existing_case_remote_receipt_missing")
                if not (self.case_root / "github_session.json").exists():
                    self._session(False, "existing_case_prepared")
            receipt = self._backup("restored" if restored else "prepared_or_existing_verified")
            return {"status": "github_case_ready_no_training_started", "backup": receipt,
                    "restored": restored is not None, "new_real_data_fits": 0,
                    "case": self.runner.status(self.source_root, self.case_root)}

    def _checkpoint(self, case_root, event):
        require(Path(case_root).absolute() == self.case_root, "callback_case_identity_differs")
        require(isinstance(event, dict) and event.get("kind") in {"epoch_checkpoint", "job_completed", "fit_failed"},
                "unexpected_checkpoint_callback")
        forced = event["kind"] in {"job_completed", "fit_failed"}
        if forced or self.last_success is None or self.clock() - self.last_success >= 60.0:
            receipt = self._backup(event["kind"])
            print(json.dumps(redact({"event": "github_backup_verified", "stage": event["kind"],
                                     "commit": receipt.get("commit"), "remote_verified": True}, self.redactions)), flush=True)

    def run(self):
        self._source()
        require(hasattr(self.runner, "DurabilityError"), "runner_durability_guard_required")
        with self.runner.case_lock(self.case_root):
            remote = self._remote()
            require(remote["exists"], "setup_and_verified_remote_plan_required_before_training")
            self._case(remote)
            self._session(True, "run_started")
            before = self._backup("run_started")
            try:
                result = self.runner.run(self.source_root, self.case_root, on_checkpoint=self._checkpoint)
            except self.runner.DurabilityError:
                # A pending ciphertext must remain bound to its original snapshot.
                # Do not rewrite a closed session or start another job after failure.
                raise
            except BaseException as error:
                self._session(False, "run_interrupted" if isinstance(error, KeyboardInterrupt) else "run_failed")
                self._backup("run_interrupted" if isinstance(error, KeyboardInterrupt) else "run_failed")
                raise
            self._session(False, "run_finished")
            receipt = self._backup("run_finished")
            return {"status": "run_saved_to_github", "result": result, "backup": receipt,
                    "initial_remote_plan_commit": before.get("commit")}

    def assess(self):
        self._source()
        with self.runner.case_lock(self.case_root):
            remote = self._remote(); require(remote["exists"], "verified_remote_case_required")
            self._case(remote)
            self._session(True, "assessment_started")
            self._backup("assessment_started")
            try:
                result = self.runner.assess(self.source_root, self.case_root)
            except BaseException as error:
                self._session(False, "assessment_interrupted" if isinstance(error, KeyboardInterrupt) else "assessment_failed")
                self._backup("assessment_interrupted" if isinstance(error, KeyboardInterrupt) else "assessment_failed")
                raise
            self._session(False, "assessment_finished")
            return {"status": "assessment_saved_to_github", "result": result,
                    "backup": self._backup("assessment_finished")}

    def status(self):
        self._source(); remote = self._remote()
        local = None
        if self.case_root.exists():
            self._case(remote if remote["exists"] else None)
            local = self.runner.status(self.source_root, self.case_root)
        return {"status": "github_and_local_status", "remote": remote, "local": local,
                "new_real_data_fits": 0}

    def export(self, output):
        self._source()
        with self.runner.case_lock(self.case_root):
            remote = self._remote(); require(remote["exists"], "verified_remote_case_required")
            self._case(remote)
            receipt = self.runner.export_case(self.source_root, self.case_root, output)
            return {"status": "public_export_created", "export": receipt,
                    "backup": self._backup("public_export_created"), "new_real_data_fits": 0}


def main(argv=None, *, stdin=None, stdout=None, store_factory=None, runner_module=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repository", default="jassan-bit/DeepLearning_202630")
    parser.add_argument("--branch", default="e3-seoi-v5-run-01")
    parser.add_argument("--prefix", default="runs/e3_seoi_v5/run-01")
    parser.add_argument("--output", type=Path)
    parser.add_argument("action", choices=("setup", "run", "assess", "status", "export"))
    args = parser.parse_args(argv)
    secret_values = ()
    output = stdout if stdout is not None else sys.stdout
    try:
        supplied = credentials(stdin if stdin is not None else sys.stdin)
        secret_values = (supplied["token"], supplied["backup_secret"])
        if store_factory is None:
            from .github_store import GithubStore as store_factory
        store = store_factory(token=supplied["token"], backup_secret=supplied["backup_secret"],
                              repository=args.repository, branch=args.branch,
                              source_commit=args.source_commit, prefix=args.prefix)
        controller = GithubController(args.source_root, args.case_root, args.data_file,
                                      args.source_commit, store, runner_module=runner_module,
                                      redactions=secret_values)
        if args.action == "export":
            require(args.output is not None, "export_output_required")
            result = controller.export(args.output)
        else:
            result = getattr(controller, args.action)()
        print(json.dumps(redact(result, secret_values), ensure_ascii=False, allow_nan=False), file=output)
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted_no_further_jobs_started"}), file=output)
        return 130
    except Exception as error:
        code = str(error) if isinstance(error, ControllerError) else "github_controller_stopped_evidence_retained"
        print(json.dumps(redact({"status": "stopped", "diagnostic": code,
                                 "storage_diagnostic": storage_code(error),
                                 "error_type": type(error).__name__}, secret_values)), file=output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
