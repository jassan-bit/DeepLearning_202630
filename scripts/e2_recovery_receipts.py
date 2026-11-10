"""Create-only public receipts for the explicitly registered lost Colab attempt.

These three small records describe a new physical retry, independently of the
old sealed logical job. A fresh process may consume an observed result, but it
cannot take over a started physical attempt. Nothing here trains a model.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request

from e2_github_campaign import (CampaignSyncError, HEX40, HEX64, OUTCOME_KEYS,
    RESOURCE_KEYS, PHASE_KEYS, _keys, _safe_values, blob_sha, canonical, digest)

VERSION = "e2-manual-recovery-physical-v1"
REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "recoveries/e2-cuda-selection-v1/20261110"
BASE_COMMIT = "5648b8305174a827d1ffa0a90ce58dab1ad5b71c"
PLAN_SHA256 = "4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda"
SOURCE_COMMIT = "564ba4762f84a800b02fa4c734f5752a169f5223"
REGISTRATION_SHA256 = "78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9"
JOB_ID = "3775f65fd049666c641c1131e94726e12e98143f91a755075c07ac6761dd668b"
PAYLOAD_SHA256 = "0eb5d72e23cb399edb41275467b3ae05c6cc856999f192c4cb254d25de49bf34"
STARTED_SHA256 = "5afba2cd98646e734799c0ae7ef3bdbc86d4688eda63648354d8a2c757ead1f5"
OLD_SESSION_ID = "9bc856c8bdb34536ad8f01de065e2261"
MAX_BYTES = 8*1024**2
UNKNOWN = {"outcome":"unknown", "elapsed_seconds":None, "optimizer_steps":None,
    "optimizer_steps_known_lower_bound":0, "partial_optimizer_update_may_have_occurred":True}
COMMON = {"version", "phase", "authorization_sha256", "plan_sha256",
    "scientific_source_commit", "recovery_source_commit", "sealed_campaign_head", "job_id", "job_payload_sha256",
    "original_started_event_sha256", "old_session_id", "recovery_session_id",
    "owner_nonce", "original_unknown_attempt", "sha256"}
EXTRA = {
    "started":{"reservation_seconds", "new_session_budget_seconds", "campaign_budget_seconds",
        "old_conservative_allocation_seconds", "started_at_utc"},
    "observed":{"outcome", "outcome_sha256", "actual_fit_wall_seconds", "recovery_elapsed_seconds",
        "observed_at_utc", "started_receipt_sha256"},
    "completed":{"result_sha256", "campaign_commit", "completed_at_utc", "actual_fit_wall_seconds",
        "recovery_elapsed_seconds", "observed_receipt_sha256", "recovery_wall_seconds",
        "charged_recovery_overhead_seconds", "old_session_conservative_allocation_seconds",
        "recovery_elapsed_lower_bound_only", "current_repair_elapsed_seconds"},
}
OPTIONAL_COMPLETED = {"recovery_wall_seconds", "charged_recovery_overhead_seconds",
    "old_session_conservative_allocation_seconds", "recovery_elapsed_lower_bound_only",
    "current_repair_elapsed_seconds"}
ERROR_CODES = {"timeout","memory","resource_limit","nonfinite","geometry","missing_gradient",
    "checkpoint","checkpoint_persistence","validation_inference","state_mutation",
    "external_timeout","external_memory","worker_exception","worker_result_invalid","runtime_error"}


class RecoveryReceiptError(CampaignSyncError):
    """Only fixed public codes escape; HTTP bodies and credentials never do."""


class RecoveryReceiptConflict(RecoveryReceiptError):
    pass


class RecoveryReceiptRateLimited(RecoveryReceiptError):
    def __init__(self, retry_after_seconds=None):
        self.retry_after_seconds = retry_after_seconds
        super().__init__("recovery_receipt_rate_limit_pause")


def _fail(code):
    raise RecoveryReceiptError(code)


def _decode(payload):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:_fail("recovery_duplicate_json_key")
            result[key]=value
        return result
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, TypeError):
        _fail("recovery_invalid_json")


def _hash(value, length=64):
    if not isinstance(value,str) or not (HEX64 if length==64 else HEX40).fullmatch(value):
        _fail("recovery_invalid_hash")


def _number(value, *, integer=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or integer and type(value) is not int:
        _fail("recovery_invalid_number")


def _utc(value):
    try:
        if not isinstance(value,str) or len(value)>40:raise ValueError()
        instant=datetime.fromisoformat(value.replace("Z","+00:00"))
        if instant.tzinfo is None or instant.utcoffset()!=timezone.utc.utcoffset(instant):raise ValueError()
    except (ValueError,TypeError):_fail("recovery_invalid_utc")


def validate_outcome(value):
    """The A3 positive public TrialRun schema, without pandas/Torch imports."""
    _keys(value,OUTCOME_KEYS,OUTCOME_KEYS);_safe_values(value)
    if value["status"] not in ("completed","failed") or type(value["fatal"]) is not bool:
        _fail("recovery_invalid_outcome")
    if not isinstance(value["run_id"],str) or not re.fullmatch(r"(?:(?:worker|external)-)?[0-9a-f]{64}",value["run_id"]):
        _fail("recovery_invalid_run_id")
    if value["reason"] is not None and value["reason"] not in ERROR_CODES:_fail("recovery_invalid_reason")
    if value["status"]=="completed" and value["reason"] is not None:_fail("recovery_invalid_reason")
    resources=value["resource_use"]
    _keys(resources,RESOURCE_KEYS)
    for key,child in resources.items():
        if key in ("phase_records","cuda_metrics","identity"):continue
        if key in ("optimizer_steps_unknown","partial_optimizer_update_may_have_occurred"):
            if type(child) is not bool:_fail("recovery_invalid_outcome_boolean")
        elif key=="checkpoint_policy":
            if child!="selection_metadata_only":_fail("recovery_invalid_checkpoint_policy")
        elif key=="error_code":
            if child not in ERROR_CODES:_fail("recovery_invalid_reason")
        elif child is not None:_number(child,integer=key not in ("elapsed_seconds","supervisor_elapsed_seconds"))
    phases=resources.get("phase_records",{})
    _keys(phases,{"ae1","ae2","supervised"})
    for name,phase in phases.items():
        if name not in ("ae1","ae2","supervised"):_fail("recovery_invalid_phase")
        _keys(phase,PHASE_KEYS)
        for key,child in phase.items():
            if child is None:continue
            if key.endswith("_sha256"):_hash(child)
            else:_number(child,integer=key!="elapsed_seconds")
    metrics=resources.get("cuda_metrics",{})
    _keys(metrics,{"max_memory_allocated_bytes","max_memory_reserved_bytes","effective_limit_bytes"})
    for child in metrics.values():
        if child is not None:_number(child,integer=True)
    if "identity" in resources:
        identity=resources["identity"]
        fields={"model","origin","candidate_id","payload_sha256","seed","stage","prepared_signature"}
        _keys(identity,fields,fields)
        for name in ("model","candidate_id"):
            if not isinstance(identity[name],str) or not identity[name] or "/" in identity[name] or "\\" in identity[name]:
                _fail("recovery_invalid_identity")
        if not isinstance(identity["origin"],str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}",identity["origin"]) or identity["stage"] not in ("hpo","nas1","nas2"):
            _fail("recovery_invalid_identity")
        _number(identity["seed"],integer=True)
        for name in ("payload_sha256","prepared_signature"):_hash(identity[name])
    predictions,dates=value["predictions_original"],value["prediction_dates"]
    if predictions is not None:
        if type(predictions) is not list or len(predictions)>10000:_fail("recovery_invalid_predictions")
        for child in predictions:
            if type(child) not in (int,float) or not math.isfinite(child):_fail("recovery_invalid_predictions")
    if dates is not None:
        if type(dates) is not list or len(dates)>10000:_fail("recovery_invalid_dates")
        for child in dates:_utc(child)
    epochs=value["phase_epochs"]
    if epochs is not None:
        _keys(epochs,{"ae1","ae2","supervised"})
        if not epochs:_fail("recovery_invalid_phase_epochs")
        for child in epochs.values():
            _number(child,integer=True)
            if child<1:_fail("recovery_invalid_phase_epochs")
    if value["status"]=="completed" and (not predictions or dates is None or len(predictions)!=len(dates) or epochs is None):
        _fail("recovery_invalid_completed_outcome")
    # Re-run the actual frozen exporter as the final canonicality authority.
    # Its public_trial path uses only the standard library; date wrappers avoid
    # importing pandas simply to check already bounded ISO strings.
    from types import SimpleNamespace
    from run_e2_cuda_search import public_trial
    class PublicDate:
        def __init__(self,text):self.text=text
        def isoformat(self):return self.text
    trial=SimpleNamespace(**value)
    trial.prediction_dates=None if dates is None else [PublicDate(child) for child in dates]
    try:
        if canonical(public_trial(trial))!=canonical(value):_fail("recovery_noncanonical_public_outcome")
    except (ValueError,TypeError,AttributeError):_fail("recovery_invalid_public_outcome")
    return value


def validate_receipt(value, phase, authorization_sha):
    allowed=COMMON|EXTRA[phase]
    required=allowed-OPTIONAL_COMPLETED if phase=="completed" else allowed
    _keys(value,allowed,required);_safe_values(value)
    if value["version"]!=VERSION or value["phase"]!=phase or value["authorization_sha256"]!=authorization_sha:
        _fail("recovery_receipt_identity_mismatch")
    copied=dict(value);claimed=copied.pop("sha256")
    if claimed!=digest(copied):_fail("recovery_receipt_hash_mismatch")
    fixed={"plan_sha256":PLAN_SHA256,"scientific_source_commit":SOURCE_COMMIT,
        "sealed_campaign_head":BASE_COMMIT,"job_id":JOB_ID,"job_payload_sha256":PAYLOAD_SHA256,
        "original_started_event_sha256":STARTED_SHA256,"old_session_id":OLD_SESSION_ID,
        "original_unknown_attempt":UNKNOWN}
    if any(canonical(value[key])!=canonical(child) for key,child in fixed.items()):_fail("recovery_sealed_binding_mismatch")
    _hash(value["recovery_source_commit"],40)
    for key in ("recovery_session_id","owner_nonce"):
        if not isinstance(value[key],str) or not re.fullmatch("[0-9a-f]{32}",value[key]):_fail("recovery_invalid_owner")
    if value["recovery_session_id"]==OLD_SESSION_ID:_fail("recovery_original_session_reused")
    for key,child in value.items():
        if key.endswith("_at_utc"):_utc(child)
        elif key.endswith("_seconds"):_number(child)
    if phase=="started":
        expected={"reservation_seconds":5700,"new_session_budget_seconds":28800,
            "campaign_budget_seconds":86400,"old_conservative_allocation_seconds":28800}
        if any(value[key]!=child for key,child in expected.items()):_fail("recovery_budget_binding_mismatch")
    elif phase=="observed":
        validate_outcome(value["outcome"])
        if value["outcome_sha256"]!=digest(value["outcome"]):_fail("recovery_outcome_hash_mismatch")
        _hash(value["started_receipt_sha256"])
        if value["recovery_elapsed_seconds"]<value["actual_fit_wall_seconds"]:_fail("recovery_timing_mismatch")
    else:
        _hash(value["result_sha256"]);_hash(value["campaign_commit"],40);_hash(value["observed_receipt_sha256"])
        if value.get("old_session_conservative_allocation_seconds",28800)!=28800:_fail("recovery_budget_binding_mismatch")
        if value["recovery_elapsed_seconds"]<value["actual_fit_wall_seconds"]:_fail("recovery_timing_mismatch")
        if "recovery_elapsed_lower_bound_only" in value and type(value["recovery_elapsed_lower_bound_only"]) is not bool:
            _fail("recovery_invalid_lower_bound_flag")
        if value.get("recovery_elapsed_lower_bound_only") and "current_repair_elapsed_seconds" not in value:
            _fail("recovery_current_repair_elapsed_required")
    if len(canonical(value))>MAX_BYTES:_fail("recovery_receipt_too_large")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


def _https(method,url,headers,payload):
    request=urllib.request.Request(url,data=None if payload is None else canonical(payload),headers=headers,method=method)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request,timeout=30) as response:
            body=response.read(2*MAX_BYTES+1)
            if len(body)>2*MAX_BYTES:_fail("recovery_response_too_large")
            return response.status,dict(response.headers),body
    except urllib.error.HTTPError as error:return error.code,dict(error.headers),b""
    except (urllib.error.URLError,TimeoutError,OSError):_fail("recovery_request_unconfirmed")


class RecoveryReceiptStore:
    def __init__(self,repository,branch,authorization_sha,base_commit,*,token=None,transport=None,
                 authorization=None,authorization_bytes=None):
        if repository!=REPOSITORY or branch!=BRANCH or base_commit!=BASE_COMMIT:_fail("recovery_remote_binding_mismatch")
        _hash(authorization_sha)
        self.repository,self.branch=repository,branch
        self.authorization_sha,self.base_commit=authorization_sha,base_commit
        self._owner_nonce=secrets.token_hex(16)
        self._token=token if token is not None else os.environ.get("GITHUB_TOKEN")
        if self._token is not None and (not isinstance(self._token,str) or not self._token or any(c.isspace() for c in self._token)):
            _fail("recovery_token_required")
        self._transport=transport or _https
        self._base="/repos/"+repository
        self.prefix="recovery/"+authorization_sha
        self._authorization_verified=False
        self._claim_attempt=None;self._claimed=None
        self._verified_heads={self.base_commit}
        if authorization_bytes is not None:
            if not isinstance(authorization_bytes,bytes) or len(authorization_bytes)>MAX_BYTES:_fail("recovery_invalid_authorization")
            if hashlib.sha256(authorization_bytes).hexdigest()!=authorization_sha:_fail("recovery_authorization_hash_mismatch")
            parsed=_decode(authorization_bytes)
            if authorization is not None and parsed!=authorization:_fail("recovery_authorization_mismatch")
            self._validate_authorization(parsed)
            self._authorization_verified=True
        elif authorization is not None:self._validate_authorization(authorization)

    @property
    def owner_nonce(self):
        """Fresh process identity; a restored public nonce cannot be adopted."""
        return self._owner_nonce

    def _validate_authorization(self,value):
        fields={"version","status","registered_at_utc","registered_at_bogota","repository",
            "campaign_branch","campaign_prefix","sealed_campaign_head","plan_sha256",
            "scientific_source_commit","scientific_registration_sha256","previous_completed_jobs","target",
            "original_unknown_attempt","decision","evidence","limits","ledger_interpretation","receipt_branch","implementation_sha256"}
        _keys(value,fields,fields);_safe_values(value)
        expected={"version":"e2-manual-recovery-authorization-v1","status":"registered_before_recovery_execution",
            "repository":REPOSITORY,"campaign_branch":"campaigns/e2-cuda-selection-v1",
            "campaign_prefix":"campaigns/e2-cuda-selection-v1","sealed_campaign_head":BASE_COMMIT,
            "plan_sha256":PLAN_SHA256,"scientific_source_commit":SOURCE_COMMIT,"previous_completed_jobs":494,
            "scientific_registration_sha256":REGISTRATION_SHA256,
            "original_unknown_attempt":UNKNOWN,"receipt_branch":BRANCH,
            "decision":"manual_retry_same_sealed_logical_job_once_after_colab_local_cache_absent"}
        if any(canonical(value[key])!=canonical(child) for key,child in expected.items()):_fail("recovery_authorization_binding_mismatch")
        target={"job_id":JOB_ID,"job_payload_sha256":PAYLOAD_SHA256,"started_event_sha256":STARTED_SHA256,
            "sequence":989,"old_session_id":OLD_SESSION_ID,"reservation_seconds":5700}
        limits={"session_seconds":28800,"campaign_seconds":86400,
            "old_session_conservative_allocation_seconds":28800,"budget_extension_authorized":False}
        if value["target"]!=target or value["limits"]!=limits:_fail("recovery_authorization_binding_mismatch")
        _utc(value["registered_at_utc"]);_hash(value["scientific_registration_sha256"])
        implementations=value["implementation_sha256"]
        expected_modules={"scripts/e2_recovery_receipts.py","scripts/e2_manual_reconciliation.py","scripts/run_e2_cuda_recovery.py"}
        _keys(implementations,expected_modules,expected_modules)
        for sha in implementations.values():_hash(sha)

    def _call(self,method,path,payload=None,*,missing_ok=False,raw=False):
        headers={"Accept":"application/vnd.github.raw+json" if raw else "application/vnd.github+json",
            "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/json","User-Agent":VERSION}
        if self._token is not None:headers["Authorization"]="Bearer "+self._token
        try:code,response_headers,body=self._transport(method,"https://api.github.com"+path,headers,payload)
        except RecoveryReceiptError:raise
        except Exception:_fail("recovery_request_unconfirmed")
        h={str(key).lower():str(value) for key,value in response_headers.items()}
        if code==429 or code==403 and (h.get("x-ratelimit-remaining")=="0" or "retry-after" in h):
            try:delay=max(0,min(86400,float(h.get("retry-after",float(h["x-ratelimit-reset"])-time.time() if "x-ratelimit-reset" in h else 60))))
            except (ValueError,TypeError):delay=None
            raise RecoveryReceiptRateLimited(delay)
        if code==404 and missing_ok:return None
        if code in (409,422):raise RecoveryReceiptConflict("recovery_create_conflict")
        if not 200<=code<300:_fail("recovery_http_%d"%code)
        if raw:
            if not isinstance(body,bytes) or len(body)>MAX_BYTES:_fail("recovery_response_too_large")
            return body
        if isinstance(body,dict):return body
        value=_decode(body)
        if not isinstance(value,dict):_fail("recovery_invalid_response")
        return value

    def _ref(self):
        value=self._call("GET",self._base+"/git/ref/heads/"+self.branch,missing_ok=True)
        if value is not None:
            head=value.get("object",{}).get("sha");_hash(head,40)
            if head not in self._verified_heads:
                comparison=self._call("GET",self._base+"/compare/"+self.base_commit+"..."+head+"?per_page=1")
                if (comparison.get("status") not in ("ahead","identical")
                        or comparison.get("base_commit",{}).get("sha")!=self.base_commit
                        or comparison.get("merge_base_commit",{}).get("sha")!=self.base_commit):
                    _fail("recovery_branch_ancestry_mismatch")
                self._verified_heads.add(head)
        return value

    def _ensure_branch(self):
        if self._ref() is not None:return
        try:self._call("POST",self._base+"/git/refs",{"ref":"refs/heads/"+self.branch,"sha":self.base_commit})
        except (RecoveryReceiptConflict,RecoveryReceiptError):
            if self._ref() is None:raise

    def _read(self,phase):
        path=self._base+"/contents/"+self.prefix+"/"+phase+".json?ref="+urllib.parse.quote(self.branch,safe="")
        response=self._call("GET",path,missing_ok=True)
        if response is None:return None
        try:
            if response.get("type")!="file" or response.get("encoding") not in ("base64","none"):raise ValueError()
            if type(response.get("size")) is not int or response["size"]>MAX_BYTES:raise ValueError()
            if response["encoding"]=="none":payload=self._call("GET",path,raw=True)
            else:payload=base64.b64decode(response["content"].replace("\n",""),validate=True)
            if len(payload)!=response["size"] or blob_sha(payload)!=response.get("sha"):raise ValueError()
        except (KeyError,ValueError,TypeError):_fail("recovery_receipt_download_invalid")
        value=_decode(payload)
        validate_receipt(value,phase,self.authorization_sha)
        if canonical(value)!=payload:_fail("recovery_noncanonical_receipt")
        return value

    def inspect(self):
        if self._ref() is None:return {phase:None for phase in EXTRA}
        records={phase:self._read(phase) for phase in EXTRA}
        self._links(records)
        return records

    @staticmethod
    def _links(records):
        started,observed,completed=(records[phase] for phase in EXTRA)
        if observed is not None:
            if started is None or observed["started_receipt_sha256"]!=started["sha256"]:_fail("recovery_started_link_mismatch")
            if any(observed[key]!=started[key] for key in COMMON-{"phase","sha256"}):_fail("recovery_physical_binding_mismatch")
        if completed is not None:
            if observed is None or completed["observed_receipt_sha256"]!=observed["sha256"]:_fail("recovery_observed_link_mismatch")
            if any(completed[key]!=observed[key] for key in COMMON-{"phase","sha256"}):_fail("recovery_physical_binding_mismatch")
            if completed["actual_fit_wall_seconds"]!=observed["actual_fit_wall_seconds"]:_fail("recovery_timing_mismatch")
            if completed["recovery_elapsed_seconds"]<observed["recovery_elapsed_seconds"]:_fail("recovery_timing_mismatch")
            if (completed.get("recovery_elapsed_lower_bound_only")
                    and completed["recovery_elapsed_seconds"]!=observed["recovery_elapsed_seconds"]):
                _fail("recovery_observed_lower_bound_mismatch")

    def _seal(self,phase,record):
        value=deepcopy(record)
        value.setdefault("version",VERSION);value.setdefault("phase",phase)
        value.setdefault("authorization_sha256",self.authorization_sha)
        _safe_values(value)
        if "sha256" not in value:value["sha256"]=digest(value)
        return validate_receipt(value,phase,self.authorization_sha)

    def _create(self,phase,value):
        if not self._authorization_verified:_fail("recovery_authorization_bytes_required")
        if self._token is None:_fail("recovery_token_required")
        payload=canonical(value)
        self._ensure_branch()
        existing=self._read(phase)
        if existing is not None:
            if canonical(existing)!=payload:raise RecoveryReceiptConflict("recovery_immutable_receipt_conflict")
            return existing
        path=self._base+"/contents/"+self.prefix+"/"+phase+".json"
        try:
            self._call("PUT",path,{"message":"Record registered physical recovery "+phase,
                "content":base64.b64encode(payload).decode("ascii"),"branch":self.branch})
        except RecoveryReceiptError:
            # A lost HTTP acknowledgment is resolved by the fixed immutable path;
            # it never causes a second physical fit or an overwrite.
            existing=self._read(phase)
            if existing is None:raise
            if canonical(existing)!=payload:raise RecoveryReceiptConflict("recovery_immutable_receipt_conflict") from None
            return existing
        existing=self._read(phase)
        if existing is None or canonical(existing)!=payload:_fail("recovery_receipt_ack_unconfirmed")
        return existing

    def claim_started(self,record):
        value=self._seal("started",record)
        if value["owner_nonce"]!=self.owner_nonce:raise RecoveryReceiptConflict("recovery_started_owned_by_another_process")
        if self._claim_attempt is not None and canonical(value)!=canonical(self._claim_attempt):
            raise RecoveryReceiptConflict("recovery_claim_changed")
        records=self.inspect()
        if records["observed"] is not None:raise RecoveryReceiptConflict("recovery_already_observed_use_receipt")
        if records["started"] is not None and canonical(records["started"])!=canonical(value):
            raise RecoveryReceiptConflict("recovery_started_owned_by_another_process")
        self._claim_attempt=value
        self._claimed=self._create("started",value)
        return deepcopy(self._claimed)

    def save_observed(self,record):
        value=self._seal("observed",record)
        records=self.inspect()
        if records["observed"] is not None:
            if records["observed"]!=value:raise RecoveryReceiptConflict("recovery_immutable_receipt_conflict")
            return deepcopy(records["observed"])
        if self._claimed is None or value["owner_nonce"]!=self.owner_nonce:
            raise RecoveryReceiptConflict("recovery_live_claim_required")
        self._links({"started":records["started"],"observed":value,"completed":None})
        return deepcopy(self._create("observed",value))

    def save_completed(self,record):
        value=self._seal("completed",record)
        records=self.inspect()
        self._links({"started":records["started"],"observed":records["observed"],"completed":value})
        return deepcopy(self._create("completed",value))
