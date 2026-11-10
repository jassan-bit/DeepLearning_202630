"""GitHub mirror of positively selected scientific search recovery records.

Only a dedicated ``campaigns/<id>`` orphan branch is writable. The mirror
never uploads model weights, worker files, environment captures or arbitrary
directories. Its small snapshot binds a SHA256 Merkle inventory; restoration
uses an immutable public archive without sending its write token to codeload.
Git's non-force fast-forward update is the single-writer compare-and-swap.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

VERSION = "e2-github-scientific-campaign-v1"
LEDGER_VERSION = "e2-search-session-v1"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
MAX_FILE_BYTES = 8*1024**2
MAX_ARCHIVE_BYTES = 1024**3


class CampaignSyncError(RuntimeError):
    """A safe, fixed-code failure; arbitrary API bodies are never exposed."""
    def __init__(self, code="github_sync_failed"):
        self.code = code
        super().__init__(code)


class CampaignConflict(CampaignSyncError):
    pass


class RemoteMissing(CampaignSyncError):
    pass


class CampaignRateLimited(CampaignSyncError):
    def __init__(self, retry_after_seconds=None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("github_rate_limit_pause")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def blob_sha(payload):
    return hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()


def _pairs(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise CampaignSyncError("duplicate_json_key")
        value[key] = child
    return value


def _decode(payload):
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, TypeError):
        raise CampaignSyncError("invalid_scientific_json") from None


def _keys(value, allowed, required=()):
    if type(value) is not dict or set(value)-set(allowed) or not set(required)<=set(value):
        raise CampaignSyncError("unapproved_scientific_fields")


SENSITIVE_KEYS = {"token", "github_token", "gh_token", "access_token", "refresh_token", "api_key",
                  "client_secret", "private_key", "cookies", "headers", "authorization", "credentials", "password", "secret", "env",
                  "environment", "hostname", "host_name", "machine_id", "gpu_uuid",
                  "command", "commands", "stderr", "stdout", "traceback", "source_roots",
                  "worker_profile", "raw_logs", "raw_receipts", "absolute_path", "argv", "cwd", "hostname"}


def _safe_values(value):
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CampaignSyncError("nonfinite_scientific_number")
        return
    if type(value) is str:
        if (len(value)>65536 or "\x00" in value or value.startswith(("/", "~", "file:", "http:", "https:"))
                or re.search(r"[A-Za-z]:[\\/]", value)
                or re.search(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{12,}", value)
                or re.search(r"\bBearer\s+\S+", value, re.I)
                or re.search(r"\b(?:https?|file)://", value, re.I)
                or re.search(r"(?:^|\s)/(?:content|workspace|tmp|home|root|usr|var|etc)(?:/|\b)", value)):
            raise CampaignSyncError("private_or_unapproved_text")
        return
    if type(value) is list:
        for child in value:
            _safe_values(child)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str or key.lower() in SENSITIVE_KEYS:
                raise CampaignSyncError("private_or_unapproved_fields")
            _safe_values(key)
            _safe_values(child)
        return
    raise CampaignSyncError("non_json_scientific_value")


PLAN_KEYS = {"version", "protocol_sha256", "registration_sha256", "source_commit", "source_sha256",
             "domain_sha256", "models", "origins", "k", "seed_pools", "execution", "checkpoint_policy",
             "scope", "coverage", "hardware_binding", "initial_controls", "stage_caps", "remote_storage"}
OUTCOME_KEYS = {"status", "run_id", "resource_use", "predictions_original", "prediction_dates",
                "phase_epochs", "reason", "fatal"}
RESOURCE_KEYS = {"model_parameters", "optimizer_steps", "optimizer_steps_known_lower_bound",
                 "optimization_stages", "elapsed_seconds", "phase_records", "optimizer_steps_unknown",
                 "partial_optimizer_update_may_have_occurred", "cuda_metrics", "error_code",
                 "phase_completed_count", "supervisor_elapsed_seconds", "supervisor_rss_bytes", "checkpoint_policy", "identity"}
PHASE_KEYS = {"epochs_completed", "best_epoch", "optimizer_steps", "optimizer_steps_known_lower_bound",
              "elapsed_seconds", "fixed_epochs", "selected_state_sha256", "final_state_sha256"}
STATUS_KEYS = {"version", "plan_sha256", "status", "scope", "completed_jobs", "failed_jobs",
               "known_optimizer_steps", "unknown_optimizer_step_jobs", "completed_selections", "coverage",
               "budget", "scientific_HPO_or_NAS_executed", "final_market_values_read", "final_R",
               "full_campaign_wall_hours", "remote_repository", "remote_branch", "checkpoint_policy",
               "pause_reason", "retry_after_seconds"}
EVENT_FIELDS = {
    "session_started": {"session_id", "session_budget_seconds", "campaign_budget_seconds",
        "charged_before_session_seconds", "directory_fsync_required", "directory_fsync_supported"},
    "job_started": {"job_id", "session_id", "job_payload_sha256", "reservation_seconds"},
    "job_finished": {"job_id", "session_id", "result_sha256"},
    "job_interrupted": {"job_id", "session_id", "exception_type", "known_elapsed_seconds"},
    "session_paused": {"session_id", "reason", "details"},
    "session_closed": {"session_id", "session_elapsed_seconds", "charged_overhead_seconds", "paused_reason",
        "directory_fsync_supported"},
    "campaign_budget_extended": {"previous_budget_seconds", "new_budget_seconds", "reason"},
}
EVENT_BASE = {"version", "sequence", "kind", "plan_sha256", "previous_sha256", "utc", "sha256"}
RESULT_KEYS = {"version", "plan_sha256", "job_id", "job_payload_sha256", "session_id", "started_event_sha256",
               "outcome", "outcome_sha256", "charged_wall_seconds", "callback_wall_seconds", "reservation_seconds",
               "reservation_overrun", "session_elapsed_seconds", "sha256"}
EXECUTION_KEYS = {"device","dtype","amp","tf32","deterministic","cpu_threads","interop_threads",
                  "gpu_memory_fraction","gpu_memory_cap_bytes"}
HARDWARE_KEYS = {"device","device_name","torch","torch_cuda","total_memory_bytes","effective_memory_limit_bytes",
                 "cudnn","cpu_threads","interop_threads","dtype","amp","tf32","deterministic",
                 "execution_policy_scope","cuda_kernel_probe_passed"}
COVERAGE_KEYS = {"selected_models","selected_origins","full_registry_models","full_registry_origins",
                 "partial_models","partial_origins","final_refit_included","classical_models_included","final_market_values_read"}
BUDGET_KEYS = {"version","plan_sha256","completed_jobs","charged_wall_seconds","campaign_budget_seconds",
               "remaining_campaign_seconds","storage_used_bytes","max_storage_bytes","recovered_result_commits",
               "conservative_unclosed_session_charges","directory_fsync_supported","paused_reason"}
TRIAL_KEYS = {"trial_id","recipe_id","learning_rate","weight_decay","optimization_seeds","batch_size","dropout",
              "kernel","num_kernels","moving_average","warmup_fraction","top_k","architecture_reference"}
ARCHITECTURE_KEYS = {"architecture_id","recipe_id","widths","dropout_pattern","dropout_rate","is_reference"}


def _validate_candidate(value):
    _keys(value,{"model","trial","architecture","stage"})
    if "trial" in value:
        _keys(value["trial"],TRIAL_KEYS)
        if "architecture_reference" in value["trial"]:
            _keys(value["trial"]["architecture_reference"],{"widths","dropout_pattern","dropout_rate"})
    if value.get("architecture") is not None:_keys(value["architecture"],ARCHITECTURE_KEYS)


def _validate_plan(plan):
    _keys(plan,PLAN_KEYS,("version","source_commit","protocol_sha256"))
    for key,length in (("source_commit",40),("protocol_sha256",64),("registration_sha256",64)):
        if key in plan and not isinstance(plan[key],str):raise CampaignSyncError("invalid_plan_identifier")
        if key in plan and not re.fullmatch("[0-9a-f]{%d}"%length,plan[key]):raise CampaignSyncError("invalid_plan_identifier")
    if "source_sha256" in plan:
        if type(plan["source_sha256"]) is not dict:raise CampaignSyncError("unapproved_source_hash_map")
        for path,sha in plan["source_sha256"].items():
            if (not re.fullmatch(r"(?:scripts/[a-zA-Z0-9_]+\.py|requirements-[a-zA-Z0-9_.-]+\.txt)",path)
                    or not isinstance(sha,str) or not HEX64.fullmatch(sha)):
                raise CampaignSyncError("unapproved_source_hash_map")
    if "domain_sha256" in plan:
        if type(plan["domain_sha256"]) is not dict:raise CampaignSyncError("invalid_domain_hash")
        for domains in plan["domain_sha256"].values():
            _keys(domains,{"hpo","nas"}, {"hpo","nas"})
            if any(not isinstance(v,str) or not HEX64.fullmatch(v) for v in domains.values()):raise CampaignSyncError("invalid_domain_hash")
    if "execution" in plan:_keys(plan["execution"],EXECUTION_KEYS)
    if "hardware_binding" in plan and plan["hardware_binding"] is not None:_keys(plan["hardware_binding"],HARDWARE_KEYS)
    if "coverage" in plan:_keys(plan["coverage"],COVERAGE_KEYS)
    if "seed_pools" in plan:_keys(plan["seed_pools"],{"hpo","nas1","nas2"})
    if "initial_controls" in plan:_keys(plan["initial_controls"],{"session_budget_seconds","campaign_budget_seconds","storage_bytes"})
    if "stage_caps" in plan:_keys(plan["stage_caps"],{"short_epochs","full_epochs","patience","short_seconds","full_seconds"})
    if "remote_storage" in plan:
        remote=plan["remote_storage"]
        _keys(remote,{"repository","branch","prefix"},{"repository","branch","prefix"})
        parts=remote["repository"].split("/") if isinstance(remote["repository"],str) else []
        if (len(parts)!=2 or not all(IDENTITY.fullmatch(p) for p in parts)
                or not isinstance(remote["branch"],str) or not remote["branch"].startswith("campaigns/")
                or not IDENTITY.fullmatch(remote["branch"][10:]) or remote["prefix"]!=remote["branch"]):
            raise CampaignSyncError("invalid_plan_remote_storage")


def approved_path(relative):
    return (relative in ("campaign.json", "status.json")
            or re.fullmatch(r"events/[0-9]{9}\.json", relative) is not None
            or re.fullmatch(r"results/[0-9a-f]{64}\.json", relative) is not None)


def validate_record(relative, payload):
    """Validate the positive scientific file/envelope schema without rewriting bytes."""
    if not approved_path(relative) or len(payload)>MAX_FILE_BYTES:
        raise CampaignSyncError("unapproved_campaign_file")
    value = _decode(payload)
    _safe_values(value)
    if relative=="status.json":
        _keys(value, STATUS_KEYS, ("version", "plan_sha256", "status"))
        reason=value.get("pause_reason")
        if reason is not None and (not isinstance(reason,str) or not re.fullmatch(r"[a-z][a-z0-9_]*",reason)):
            raise CampaignSyncError("unapproved_pause_code")
        retry=value.get("retry_after_seconds")
        if retry is not None and (type(retry) not in (int,float) or not math.isfinite(retry) or retry<0):
            raise CampaignSyncError("invalid_retry_interval")
        for selection in value.get("completed_selections", []):
            _keys(selection, {"model", "origin", "winner_id", "winner", "fixed_epochs", "selection_signature",
                              "attempts", "selection_rule"})
            if "winner" in selection:_validate_candidate(selection["winner"])
        if "coverage" in value:_keys(value["coverage"],COVERAGE_KEYS)
        if "budget" in value:_keys(value["budget"],BUDGET_KEYS)
        return value
    if value.get("version")!=LEDGER_VERSION:
        raise CampaignSyncError("unrecognized_ledger_version")
    copied = dict(value)
    claimed = copied.pop("sha256", None)
    if claimed!=digest(copied):
        raise CampaignSyncError("scientific_record_hash_mismatch")
    if relative=="campaign.json":
        _keys(value, {"version", "plan", "plan_sha256", "controls", "created_at_utc", "sha256"},
              ("plan", "plan_sha256", "controls"))
        _validate_plan(value["plan"])
        if digest(value["plan"])!=value["plan_sha256"]:
            raise CampaignSyncError("scientific_plan_hash_mismatch")
        _keys(value["controls"], {"campaign_budget_seconds", "max_storage_bytes", "completion_reserve_bytes", "require_directory_fsync"})
    elif relative.startswith("events/"):
        kind=value.get("kind")
        if kind not in EVENT_FIELDS:
            raise CampaignSyncError("unapproved_campaign_event")
        _keys(value, EVENT_BASE|EVENT_FIELDS[kind], EVENT_BASE|EVENT_FIELDS[kind])
        if relative!="events/%09d.json" % value["sequence"]:
            raise CampaignSyncError("event_filename_mismatch")
        if kind=="session_paused":
            _keys(value["details"], {"reservation_seconds", "remaining_seconds", "storage_used_bytes", "charged_wall_seconds"})
    else:
        _keys(value, RESULT_KEYS, RESULT_KEYS)
        if relative!="results/"+hashlib.sha256(value["job_id"].encode()).hexdigest()+".json":
            raise CampaignSyncError("result_filename_mismatch")
        _keys(value["outcome"], OUTCOME_KEYS, OUTCOME_KEYS)
        if digest(value["outcome"])!=value["outcome_sha256"]:
            raise CampaignSyncError("scientific_outcome_hash_mismatch")
        resources=value["outcome"]["resource_use"]
        _keys(resources, RESOURCE_KEYS)
        if "identity" in resources:
            _keys(resources["identity"], {"model","origin","candidate_id","payload_sha256","seed","stage","prepared_signature"})
        for phase in resources.get("phase_records", {}).values():
            _keys(phase, PHASE_KEYS)
        _keys(resources.get("cuda_metrics", {}), {"max_memory_allocated_bytes", "max_memory_reserved_bytes", "effective_limit_bytes"})
    return value


def _descriptor(path, payload):
    return {"path":path, "bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest(), "git_blob_sha":blob_sha(payload)}


class _Inventory:
    """Incremental SHA256 Merkle inventory: 256 path buckets, eight tree levels."""
    def __init__(self):
        self.records={}
        self.buckets=[{} for _ in range(256)]
        self.tree=[b""]*512
        empty=hashlib.sha256(b"leaf\0"+canonical([])).digest()
        for i in range(256,512):self.tree[i]=empty
        for i in range(255,0,-1):self.tree[i]=hashlib.sha256(b"node\0"+self.tree[2*i]+self.tree[2*i+1]).digest()
        self.bytes=0

    def set(self, path, descriptor):
        old=self.records.get(path)
        if old:self.bytes-=old["bytes"]
        bucket=int(hashlib.sha256(path.encode()).hexdigest()[:2],16)
        if descriptor is None:
            self.records.pop(path,None);self.buckets[bucket].pop(path,None)
        else:
            self.records[path]=descriptor;self.buckets[bucket][path]=descriptor;self.bytes+=descriptor["bytes"]
        index=256+bucket
        self.tree[index]=hashlib.sha256(b"leaf\0"+canonical([self.buckets[bucket][p] for p in sorted(self.buckets[bucket])])).digest()
        while index>1:
            index//=2
            self.tree[index]=hashlib.sha256(b"node\0"+self.tree[2*index]+self.tree[2*index+1]).digest()

    @property
    def root(self):return self.tree[1].hex()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def _https(method,url,headers,payload):
    limit=MAX_ARCHIVE_BYTES if urllib.parse.urlsplit(url).hostname=="codeload.github.com" else 32*1024**2
    data=None if payload is None else canonical(payload)
    request=urllib.request.Request(url,data=data,headers=headers,method=method)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request,timeout=30) as response:
            body=response.read(limit+1)
            if len(body)>limit:raise CampaignSyncError("github_response_too_large")
            return response.status,dict(response.headers),body
    except urllib.error.HTTPError as error:
        return error.code,dict(error.headers),b""
    except (urllib.error.URLError,TimeoutError,OSError):
        raise CampaignSyncError("github_request_unconfirmed") from None


class GitHubCampaignMirror:
    def __init__(self,repository,branch,prefix,*,token=None,transport=None,max_snapshot_bytes=2*1024**3):
        parts=repository.split("/") if isinstance(repository,str) else []
        if len(parts)!=2 or not all(IDENTITY.fullmatch(p) for p in parts):
            raise CampaignSyncError("invalid_github_repository")
        if not isinstance(branch,str) or not branch.startswith("campaigns/") or not IDENTITY.fullmatch(branch[10:]):
            raise CampaignSyncError("campaign_branch_required")
        if prefix!=branch:raise CampaignSyncError("campaign_prefix_must_match_branch")
        self.repository,self.branch,self.prefix=repository,branch,prefix
        self._token=token if token is not None else os.environ.get("GITHUB_TOKEN")
        if not isinstance(self._token,str) or not self._token.strip() or any(c.isspace() for c in self._token):
            raise CampaignSyncError("github_token_required")
        self._transport=transport or _https
        self.max_snapshot_bytes=int(max_snapshot_bytes)
        if self.max_snapshot_bytes<=0:raise CampaignSyncError("invalid_snapshot_limit")
        self._base="/repos/"+repository
        self._head=None;self._tree=None;self._manifest=None;self._inventory=None
        self._inspected=False;self._requires_refresh=False;self._verified_directories=set();self._remaining=None;self._reset=None

    def _call(self,method,path,payload=None,*,missing_ok=False):
        headers={"Accept":"application/vnd.github+json","Authorization":"Bearer "+self._token,
                 "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/json","User-Agent":VERSION}
        try:code,response_headers,body=self._transport(method,"https://api.github.com"+path,headers,payload)
        except CampaignSyncError:raise
        except Exception:raise CampaignSyncError("github_request_unconfirmed") from None
        h={str(k).lower():str(v) for k,v in response_headers.items()}
        try:self._remaining=int(h["x-ratelimit-remaining"])
        except (KeyError,ValueError):pass
        try:self._reset=int(h["x-ratelimit-reset"])
        except (KeyError,ValueError):pass
        if code==429 or code==403 and (h.get("x-ratelimit-remaining")=="0" or "retry-after" in h):
            try:delay=max(0,min(86400,int(float(h.get("retry-after",self._reset-time.time() if self._reset else 60)))))
            except (ValueError,TypeError):delay=None
            raise CampaignRateLimited(delay)
        if code==404 and missing_ok:return None
        if code in (409,422):raise CampaignConflict("github_campaign_head_conflict")
        if not 200<=code<300:raise CampaignSyncError("github_http_%d" % code)
        if isinstance(body,dict):return body
        value=_decode(body)
        if not isinstance(value,dict):raise CampaignSyncError("github_response_invalid")
        return value

    def check_capacity(self,required_requests=32,*,reserve_bytes=1024*1024):
        """Reserve remote metadata capacity before creating a STARTED record.

        The allowance is an admission reserve for a scientific result and its
        events, not permission to publish weights or expand the file schema.
        """
        if type(reserve_bytes) is not int or reserve_bytes<=0:
            raise CampaignSyncError("invalid_snapshot_reservation")
        if self._remaining is not None and self._remaining<int(required_requests):
            raise CampaignRateLimited(max(0,int(self._reset-time.time())) if self._reset else None)
        used=0 if self._inventory is None else self._inventory.bytes
        if used+reserve_bytes>self.max_snapshot_bytes:
            raise CampaignSyncError("github_snapshot_capacity_pause")

    @staticmethod
    def _sha(value):
        if not isinstance(value,str) or not HEX40.fullmatch(value):raise CampaignSyncError("github_object_invalid")
        return value

    def _entry(self,tree_sha,name,kind,*,only_names=None):
        obj=self._call("GET",self._base+"/git/trees/"+tree_sha)
        entries=obj.get("tree")
        if obj.get("truncated") or not isinstance(entries,list):raise CampaignSyncError("github_tree_invalid")
        if only_names is not None and {e.get("path") for e in entries}-set(only_names):raise CampaignSyncError("non_campaign_tree_refused")
        found=[e for e in entries if e.get("path")==name and e.get("type")==kind]
        if len(found)!=1:raise CampaignSyncError("github_campaign_tree_missing")
        return self._sha(found[0].get("sha"))

    def _blob(self,sha):
        obj=self._call("GET",self._base+"/git/blobs/"+sha)
        try:
            if obj.get("encoding")!="base64":raise ValueError()
            payload=base64.b64decode(obj["content"].replace("\n",""),validate=True)
            if len(payload)>MAX_FILE_BYTES or len(payload)!=obj["size"] or blob_sha(payload)!=sha:raise ValueError()
            return payload
        except (KeyError,ValueError,TypeError):raise CampaignSyncError("github_blob_hash_mismatch") from None

    def inspect_remote(self):
        ref=self._call("GET",self._base+"/git/ref/heads/"+urllib.parse.quote(self.branch,safe="/"),missing_ok=True)
        self._inspected=True;self._requires_refresh=False
        if ref is None:
            self._head=self._tree=self._manifest=None;self._inventory=_Inventory();return None
        head=self._sha(ref.get("object",{}).get("sha"))
        commit=self._call("GET",self._base+"/git/commits/"+head)
        tree=self._sha(commit.get("tree",{}).get("sha"))
        campaigns=self._entry(tree,"campaigns","tree",only_names={"campaigns"})
        branch_tree=self._entry(campaigns,self.branch[10:],"tree",only_names={self.branch[10:]})
        manifest_sha=self._entry(branch_tree,"snapshot.json","blob",only_names={"snapshot.json","campaign.json","events","results","status.json"})
        manifest=_decode(self._blob(manifest_sha))
        _keys(manifest,{"version","repository","branch","prefix","record_count","record_bytes","inventory_algorithm",
                        "inventory_root_sha256","plan_sha256","previous_commit","sha256"})
        check=dict(manifest);claimed=check.pop("sha256",None)
        if (digest(check)!=claimed or manifest.get("version")!=VERSION or manifest.get("repository")!=self.repository
                or manifest.get("branch")!=self.branch or manifest.get("prefix")!=self.prefix
                or manifest.get("inventory_algorithm")!="sha256-path-bucket-merkle-v1"
                or type(manifest.get("record_count")) is not int or manifest["record_count"]<1
                or type(manifest.get("record_bytes")) is not int or not 0<manifest["record_bytes"]<=self.max_snapshot_bytes
                or not HEX64.fullmatch(manifest.get("inventory_root_sha256", ""))):
            raise CampaignSyncError("github_snapshot_invalid")
        if head!=self._head:self._inventory=None;self._verified_directories.clear()
        self._head,self._tree,self._manifest=head,tree,manifest
        return deepcopy(manifest)

    def _archive_records(self):
        url="https://codeload.github.com/"+self.repository+"/zip/"+self._head
        try:code,headers,payload=self._transport("GET",url,{"User-Agent":VERSION},None)
        except Exception:raise CampaignSyncError("github_archive_unavailable") from None
        if code!=200 or not isinstance(payload,bytes) or len(payload)>MAX_ARCHIVE_BYTES:raise CampaignSyncError("github_archive_unavailable")
        try:archive=zipfile.ZipFile(io.BytesIO(payload))
        except (ValueError,zipfile.BadZipFile):raise CampaignSyncError("github_archive_invalid") from None
        records={};root=None;total=0;manifest_bytes=None
        with archive:
            for item in archive.infolist():
                parts=PurePosixPath(item.filename).parts
                if not parts or item.filename.startswith("/") or ".." in parts or "\\" in item.filename:
                    raise CampaignSyncError("github_archive_path_invalid")
                if root is None:root=parts[0]
                if parts[0]!=root:raise CampaignSyncError("github_archive_root_invalid")
                mode=item.external_attr>>16
                if stat.S_ISLNK(mode) or item.flag_bits&1:raise CampaignSyncError("github_archive_entry_invalid")
                if item.is_dir():continue
                relative="/".join(parts[1:])
                if not relative.startswith(self.prefix+"/"):raise CampaignSyncError("non_campaign_archive_refused")
                relative=relative[len(self.prefix)+1:]
                if item.file_size>MAX_FILE_BYTES:raise CampaignSyncError("github_archive_file_too_large")
                total+=item.file_size
                if total>self.max_snapshot_bytes+MAX_FILE_BYTES:raise CampaignSyncError("github_archive_too_large")
                try:data=archive.read(item)
                except (zipfile.BadZipFile,RuntimeError,OSError,EOFError):
                    raise CampaignSyncError("github_archive_content_invalid") from None
                if relative=="snapshot.json":
                    if manifest_bytes is not None:raise CampaignSyncError("github_archive_duplicate")
                    manifest_bytes=data;continue
                if relative in records:raise CampaignSyncError("github_archive_duplicate")
                validate_record(relative,data);records[relative]=data
        if manifest_bytes is None or _decode(manifest_bytes)!=self._manifest:raise CampaignSyncError("github_archive_snapshot_mismatch")
        inventory=_Inventory()
        for path,data in records.items():inventory.set(path,_descriptor(path,data))
        if (len(records)!=self._manifest["record_count"] or inventory.bytes!=self._manifest["record_bytes"]
                or inventory.root!=self._manifest["inventory_root_sha256"] or "campaign.json" not in records):
            raise CampaignSyncError("github_archive_inventory_mismatch")
        campaign=_decode(records["campaign.json"])
        if campaign["plan_sha256"]!=self._manifest["plan_sha256"]:raise CampaignSyncError("github_snapshot_plan_mismatch")
        self._validate_remote_binding(campaign)
        self._inventory=inventory
        return records

    def _validate_remote_binding(self,campaign):
        storage=campaign["plan"].get("remote_storage")
        if storage is not None and storage!={"repository":self.repository,"branch":self.branch,"prefix":self.prefix}:
            raise CampaignConflict("campaign_remote_binding_changed")

    @staticmethod
    def _safe_local(path):
        absolute=path.absolute()
        if any(p.is_symlink() for p in (absolute,*absolute.parents)):raise CampaignSyncError("campaign_symlink_refused")

    def restore(self,directory):
        self.inspect_remote()
        if self._head is None:return {"status":"remote_missing","restored_files":0}
        records=self._archive_records();directory=Path(directory).absolute();self._safe_local(directory)
        # Check every existing destination before publishing any missing file.
        for relative,data in records.items():
            target=directory/relative;self._safe_local(target)
            if target.exists() and (not target.is_file() or target.read_bytes()!=data) and relative!="status.json":
                raise CampaignConflict("local_remote_record_conflict")
        directory.mkdir(parents=True,exist_ok=True)
        count=0
        for relative,data in records.items():
            target=directory/relative
            if target.exists():continue  # Derived status is recomputed by the caller.
            target.parent.mkdir(parents=True,exist_ok=True)
            descriptor,temporary=tempfile.mkstemp(prefix=".restore-",dir=target.parent)
            try:
                with os.fdopen(descriptor,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
                if target.exists():raise CampaignConflict("local_restore_target_appeared")
                os.link(temporary,target)
            finally:Path(temporary).unlink(missing_ok=True)
            count+=1
        for name in ("events","results","sessions"):(directory/name).mkdir(exist_ok=True)
        self._verified_directories.add(str(directory))
        return {"status":"restored","restored_files":count,"remote_commit":self._head,"plan_sha256":self._manifest["plan_sha256"]}

    def _local_records(self,directory,paths):
        directory=Path(directory).absolute();self._safe_local(directory)
        if paths is None:
            paths=["campaign.json"]
            if (directory/"status.json").exists():paths.append("status.json")
            for name in ("events","results"):
                folder=directory/name;self._safe_local(folder)
                if folder.exists():paths.extend(str(p.relative_to(directory).as_posix()) for p in folder.iterdir() if p.is_file() and not p.name.startswith(".pending-"))
        values={}
        for relative in paths:
            if not isinstance(relative,str) or not approved_path(relative):raise CampaignSyncError("unapproved_campaign_file")
            path=directory/relative;self._safe_local(path)
            if not path.is_file() or path.stat().st_size>MAX_FILE_BYTES:raise CampaignSyncError("campaign_record_unavailable")
            payload=path.read_bytes();validate_record(relative,payload);values[relative]=payload
        return directory,values

    def push(self,directory,*,message="Save scientific campaign progress",paths=None):
        if self._requires_refresh:raise CampaignSyncError("github_refresh_required_after_failure")
        if not self._inspected:self.inspect_remote()
        if self._head is not None and self._inventory is None:self._archive_records()
        directory,values=self._local_records(directory,paths)
        if "campaign.json" in values:self._validate_remote_binding(_decode(values["campaign.json"]))
        if str(directory) not in self._verified_directories:
            # One startup scan verifies that local state did not omit remote history.
            for path,expected in self._inventory.records.items():
                if path=="status.json":continue
                target=directory/path;self._safe_local(target)
                if not target.is_file() or _descriptor(path,target.read_bytes())!=expected:
                    raise CampaignConflict("local_remote_history_missing")
            self._verified_directories.add(str(directory))
        if "campaign.json" not in self._inventory.records and "campaign.json" not in values:
            raise CampaignSyncError("campaign_record_required")
        changes={}
        for path,payload in values.items():
            descriptor=_descriptor(path,payload);old=self._inventory.records.get(path)
            if old==descriptor:continue
            if old is not None and path!="status.json":raise CampaignConflict("immutable_remote_record_changed")
            changes[path]=(payload,descriptor)
        if not changes:return {"status":"unchanged","remote_commit":self._head,"synced_paths":tuple(values)}
        _safe_values(message)
        if len(message)>200 or "\n" in message:raise CampaignSyncError("invalid_commit_message")
        before={path:self._inventory.records.get(path) for path in changes}
        try:
            for path,(_,descriptor) in changes.items():self._inventory.set(path,descriptor)
            if self._inventory.bytes>self.max_snapshot_bytes:raise CampaignSyncError("campaign_snapshot_limit")
            campaign=values.get("campaign.json")
            plan_sha=_decode(campaign)["plan_sha256"] if campaign is not None else self._manifest["plan_sha256"]
            if "status.json" in values and _decode(values["status.json"]).get("plan_sha256")!=plan_sha:
                raise CampaignSyncError("derived_status_plan_mismatch")
            manifest={"version":VERSION,"repository":self.repository,"branch":self.branch,"prefix":self.prefix,
                      "record_count":len(self._inventory.records),"record_bytes":self._inventory.bytes,
                      "inventory_algorithm":"sha256-path-bucket-merkle-v1","inventory_root_sha256":self._inventory.root,
                      "plan_sha256":plan_sha,"previous_commit":self._head}
            manifest["sha256"]=digest(manifest)
            entries=[]
            for path,(payload,descriptor) in changes.items():
                obj=self._call("POST",self._base+"/git/blobs",{"encoding":"base64","content":base64.b64encode(payload).decode()})
                if obj.get("sha")!=descriptor["git_blob_sha"]:raise CampaignSyncError("github_uploaded_blob_mismatch")
                entries.append({"path":self.prefix+"/"+path,"mode":"100644","type":"blob","sha":descriptor["git_blob_sha"]})
            snapshot=canonical(manifest)+b"\n"
            obj=self._call("POST",self._base+"/git/blobs",{"encoding":"base64","content":base64.b64encode(snapshot).decode()})
            if obj.get("sha")!=blob_sha(snapshot):raise CampaignSyncError("github_uploaded_snapshot_mismatch")
            entries.append({"path":self.prefix+"/snapshot.json","mode":"100644","type":"blob","sha":obj["sha"]})
            body={"tree":entries}
            if self._tree is not None:body["base_tree"]=self._tree
            tree=self._sha(self._call("POST",self._base+"/git/trees",body).get("sha"))
            commit_body={"message":message,"tree":tree,"parents":[] if self._head is None else [self._head]}
            commit=self._sha(self._call("POST",self._base+"/git/commits",commit_body).get("sha"))
            if self._head is None:
                response=self._call("POST",self._base+"/git/refs",{"ref":"refs/heads/"+self.branch,"sha":commit})
            else:
                response=self._call("PATCH",self._base+"/git/refs/heads/"+urllib.parse.quote(self.branch,safe="/"),{"sha":commit,"force":False})
            if response.get("object",{}).get("sha")!=commit:raise CampaignSyncError("github_ref_ack_missing")
            self._head,self._tree,self._manifest=commit,tree,manifest
            return {"status":"synced","remote_commit":commit,"synced_paths":tuple(values),"record_count":len(self._inventory.records)}
        except BaseException:
            for path,old in before.items():self._inventory.set(path,old)
            self._requires_refresh=True
            raise
