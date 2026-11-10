import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e2_recovery_receipts as receipts


class GitHub:
    """In-memory Contents API; no network credentials or production writes."""
    def __init__(self):
        self.ref=None;self.files={};self.calls=[];self.lose=None;self.reject=None;self.corrupt=False;self.unrelated=False

    def __call__(self,method,url,headers,payload):
        self.calls.append((method,url,deepcopy(headers),deepcopy(payload)))
        parsed=urlsplit(url);path=unquote(parsed.path)
        assert parsed.scheme=="https" and parsed.netloc=="api.github.com"
        base="/repos/"+receipts.REPOSITORY
        assert path.startswith(base+"/")
        if method=="GET" and path==base+"/git/ref/heads/"+receipts.BRANCH:
            return (404,{},b"") if self.ref is None else (200,{}, {"object":{"sha":self.ref}})
        if method=="POST" and path==base+"/git/refs":
            assert payload=={"ref":"refs/heads/"+receipts.BRANCH,"sha":receipts.BASE_COMMIT}
            if self.ref is not None:return 422,{},b""
            self.ref=payload["sha"];return 201,{}, {"object":{"sha":self.ref}}
        if method=="GET" and path==base+"/compare/"+receipts.BASE_COMMIT+"..."+self.ref:
            return 200,{}, {"status":"ahead","base_commit":{"sha":receipts.BASE_COMMIT},
                "merge_base_commit":{"sha":"f"*40 if self.unrelated else receipts.BASE_COMMIT}}
        assert "/contents/recovery/" in path
        name=path.split("/contents/",1)[1]
        assert name.endswith(("/started.json","/observed.json","/completed.json"))
        if method=="GET":
            assert unquote(parsed.query)=="ref="+receipts.BRANCH
            if name not in self.files:return 404,{},b""
            data=self.files[name]
            if headers["Accept"]=="application/vnd.github.raw+json":return 200,{},data
            return 200,{}, {"type":"file","encoding":"base64","size":len(data),
                "sha":"0"*40 if self.corrupt else receipts.blob_sha(data),
                "content":base64.b64encode(data).decode()}
        assert method=="PUT" and parsed.query==""
        assert set(payload)=={"message","content","branch"} and payload["branch"]==receipts.BRANCH
        if self.reject is not None:return self.reject
        if name in self.files:return 422,{},b""
        self.files[name]=base64.b64decode(payload["content"],validate=True)
        self.ref=hashlib.sha1(self.files[name]).hexdigest()
        if self.lose==name.rsplit("/",1)[1]:
            self.lose=None;raise OSError("private transport detail must not escape")
        return 201,{}, {"content":{"sha":receipts.blob_sha(self.files[name])}}


def authorization():
    value=json.loads((ROOT/"experiments/e2_recovery_20261110/authorization.json").read_text())
    value["status"]="registered_before_recovery_execution"
    value["implementation_sha256"]={name:"a"*64 for name in (
        "scripts/e2_recovery_receipts.py","scripts/e2_manual_reconciliation.py","scripts/run_e2_cuda_recovery.py")}
    data=receipts.canonical(value)
    return value,data,hashlib.sha256(data).hexdigest()


def store(remote,**kwargs):
    _,data,sha=authorization()
    return receipts.RecoveryReceiptStore(receipts.REPOSITORY,receipts.BRANCH,sha,receipts.BASE_COMMIT,
        token="test-private-token",transport=remote,authorization_bytes=data,**kwargs)


def common(client):
    return {"plan_sha256":receipts.PLAN_SHA256,"scientific_source_commit":receipts.SOURCE_COMMIT,"recovery_source_commit":"e"*40,
        "sealed_campaign_head":receipts.BASE_COMMIT,"job_id":receipts.JOB_ID,
        "job_payload_sha256":receipts.PAYLOAD_SHA256,"original_started_event_sha256":receipts.STARTED_SHA256,
        "old_session_id":receipts.OLD_SESSION_ID,"recovery_session_id":"a"*32,
        "owner_nonce":client.owner_nonce,"original_unknown_attempt":deepcopy(receipts.UNKNOWN)}


def started(client):
    return {**common(client),"reservation_seconds":5700,"new_session_budget_seconds":28800,
        "campaign_budget_seconds":86400,"old_conservative_allocation_seconds":28800,
        "started_at_utc":"2026-11-10T11:00:00+00:00"}


def outcome():
    return {"status":"completed","run_id":"b"*64,"resource_use":{
        "model_parameters":50,"optimizer_steps":3,"optimizer_steps_known_lower_bound":3,
        "optimizer_steps_unknown":False,"partial_optimizer_update_may_have_occurred":False,
        "checkpoint_policy":"selection_metadata_only","phase_records":{"supervised":{
            "epochs_completed":2,"optimizer_steps":3,"elapsed_seconds":4.0}},"cuda_metrics":{}},
        "predictions_original":[-0.5,0.8],"prediction_dates":["2024-01-01T00:00:00+00:00","2024-01-02T00:00:00+00:00"],
        "phase_epochs":{"supervised":2},"reason":None,"fatal":False}


def observed(claim):
    return {**{key:deepcopy(claim[key]) for key in receipts.COMMON-{"phase","sha256"}},
        "phase":"observed","outcome":outcome(),"outcome_sha256":receipts.digest(outcome()),
        "actual_fit_wall_seconds":7,"recovery_elapsed_seconds":10,
        "observed_at_utc":"2026-11-10T11:00:10+00:00","started_receipt_sha256":claim["sha256"]}


def completed(record):
    return {**{key:deepcopy(record[key]) for key in receipts.COMMON-{"phase","sha256"}},
        "phase":"completed","result_sha256":"c"*64,"campaign_commit":"d"*40,
        "completed_at_utc":"2026-11-10T11:00:12+00:00","actual_fit_wall_seconds":7,
        "recovery_elapsed_seconds":12,"observed_receipt_sha256":record["sha256"],
        "recovery_wall_seconds":12,"charged_recovery_overhead_seconds":5,
        "old_session_conservative_allocation_seconds":28800}


class RecoveryReceiptsTests(unittest.TestCase):
    def test_full_three_stage_create_only_branch_and_public_receipts(self):
        remote=GitHub();client=store(remote)
        self.assertEqual(client.inspect(),dict.fromkeys(("started","observed","completed")))
        claim=client.claim_started(started(client));actual=client.save_observed(observed(claim))
        finish=client.save_completed(completed(actual))
        self.assertEqual(client.inspect(),{"started":claim,"observed":actual,"completed":finish})
        writes=[c for c in remote.calls if c[0] in ("POST","PUT")]
        self.assertEqual([c[0] for c in writes],["POST","PUT","PUT","PUT"])
        self.assertEqual(len(remote.files),3)
        for name,data in remote.files.items():
            self.assertTrue(name.startswith("recovery/"+client.authorization_sha+"/"))
            self.assertNotIn(b"test-private-token",data)
            self.assertNotIn(b"evidence",data)
        self.assertIsNone(actual["original_unknown_attempt"]["elapsed_seconds"])
        self.assertIsNone(actual["original_unknown_attempt"]["optimizer_steps"])

    def test_lost_started_ack_confirms_same_immutable_claim_once(self):
        remote=GitHub();remote.lose="started.json";client=store(remote)
        claim=client.claim_started(started(client))
        self.assertEqual(client.claim_started(started(client)),claim)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)

    def test_new_process_started_without_observed_cannot_take_over(self):
        remote=GitHub();first=store(remote);claim=first.claim_started(started(first));second=store(remote)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"another_process"):
            second.claim_started(started(second))
        copied=started(first)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"another_process"):
            second.claim_started(copied)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"live_claim"):
            second.save_observed(observed(claim))
        with self.assertRaises(AttributeError):second.owner_nonce=first.owner_nonce
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)

    def test_new_process_can_complete_observed_without_new_claim_or_fit(self):
        remote=GitHub();first=store(remote);claim=first.claim_started(started(first))
        remote.lose="observed.json";actual=first.save_observed(observed(claim))
        second=store(remote)
        self.assertEqual(second.inspect()["observed"],actual)
        finish=second.save_completed(completed(actual))
        self.assertEqual(second.save_completed(completed(actual)),finish)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"already_observed"):
            second.claim_started(started(second))
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),3)

    def test_new_process_retains_prior_elapsed_as_lower_bound_and_its_own_repair_clock(self):
        remote=GitHub();first=store(remote);claim=first.claim_started(started(first))
        physical=observed(claim);physical["recovery_elapsed_seconds"]=120
        actual=first.save_observed(physical)
        second=store(remote);record=completed(actual)
        record.update(recovery_elapsed_seconds=120,recovery_elapsed_lower_bound_only=True,
            current_repair_elapsed_seconds=2,recovery_wall_seconds=28800,
            charged_recovery_overhead_seconds=28793)
        finish=second.save_completed(record)
        self.assertEqual(finish["recovery_elapsed_seconds"],120)
        self.assertEqual(finish["current_repair_elapsed_seconds"],2)
        self.assertTrue(finish["recovery_elapsed_lower_bound_only"])
        self.assertEqual(second.inspect()["observed"],actual)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),3)

    def test_lower_bound_fields_reject_invented_elapsed_and_invalid_numeric_or_flag_types(self):
        remote=GitHub();first=store(remote);claim=first.claim_started(started(first))
        actual=first.save_observed(observed(claim));second=store(remote)
        for flag,repair in ((0,2),("true",2),(True,True),(True,float("nan")),(True,-1),(True,None)):
            record=completed(actual);record.update(recovery_elapsed_seconds=actual["recovery_elapsed_seconds"],
                recovery_elapsed_lower_bound_only=flag,current_repair_elapsed_seconds=repair)
            with self.assertRaises(receipts.CampaignSyncError):second.save_completed(record)
        record=completed(actual);record.update(recovery_elapsed_seconds=11,
            recovery_elapsed_lower_bound_only=True,current_repair_elapsed_seconds=2)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"observed_lower_bound"):
            second.save_completed(record)
        record=completed(actual);record.update(recovery_elapsed_seconds=actual["recovery_elapsed_seconds"],
            recovery_elapsed_lower_bound_only=True)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"current_repair_elapsed_required"):
            second.save_completed(record)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),2)

    def test_observed_requires_confirmed_live_claim_and_valid_hash_chain(self):
        remote=GitHub();client=store(remote)
        fake=client._seal("started",started(client))
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"live_claim"):
            client.save_observed(observed(fake))
        claim=client.claim_started(started(client));record=observed(claim)
        record["started_receipt_sha256"]="f"*64
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"started_link"):
            client.save_observed(record)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)

    def test_immutable_observed_conflict_and_completed_link_rejected(self):
        remote=GitHub();client=store(remote);claim=client.claim_started(started(client))
        actual=client.save_observed(observed(claim));different=observed(claim)
        different["actual_fit_wall_seconds"]=8
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"immutable"):
            client.save_observed(different)
        final=completed(actual);final["observed_receipt_sha256"]="f"*64
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"observed_link"):
            client.save_completed(final)

    def test_authorization_raw_hash_status_and_module_manifest_bound(self):
        remote=GitHub();value,data,sha=authorization()
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_hash"):
            receipts.RecoveryReceiptStore(receipts.REPOSITORY,receipts.BRANCH,"f"*64,receipts.BASE_COMMIT,
                token="test",transport=remote,authorization_bytes=data)
        for mutate in (lambda v:v.update(status="draft_before_recovery_execution"),
                       lambda v:v["implementation_sha256"].clear(),
                       lambda v:v["target"].update(job_id="f"*64)):
            bad=deepcopy(value);mutate(bad);payload=receipts.canonical(bad)
            with self.assertRaises(receipts.CampaignSyncError):
                receipts.RecoveryReceiptStore(receipts.REPOSITORY,receipts.BRANCH,hashlib.sha256(payload).hexdigest(),
                    receipts.BASE_COMMIT,token="test",transport=remote,authorization_bytes=payload)
        readonly=receipts.RecoveryReceiptStore(receipts.REPOSITORY,receipts.BRANCH,sha,receipts.BASE_COMMIT,
            token="test",transport=remote,authorization=value)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_bytes_required"):
            readonly.claim_started(started(readonly))
        self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))

    def test_remote_and_schema_restrict_private_arbitrary_fields(self):
        remote=GitHub();client=store(remote)
        for key,value in (("token","secret"),("stdout","log"),("source_root","/content/private")):
            record=started(client);record[key]=value
            with self.assertRaises(receipts.CampaignSyncError):client.claim_started(record)
        record=started(client);record["original_unknown_attempt"]["optimizer_steps"]=0
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"sealed_binding"):
            client.claim_started(record)
        _,_,sha=authorization()
        for repo,branch,base in (("other/repo",receipts.BRANCH,receipts.BASE_COMMIT),
                (receipts.REPOSITORY,"main",receipts.BASE_COMMIT),
                (receipts.REPOSITORY,receipts.BRANCH,"f"*40)):
            with self.assertRaisesRegex(receipts.RecoveryReceiptError,"remote_binding"):
                receipts.RecoveryReceiptStore(repo,branch,sha,base,token="test",transport=remote)

    def test_outcome_positive_schema_types_and_hash_are_checked(self):
        remote=GitHub();client=store(remote);claim=client.claim_started(started(client))
        for mutate in (lambda v:v["resource_use"].update(stderr="private"),
                       lambda v:v["resource_use"].update(optimizer_steps=True),
                       lambda v:v.update(predictions_original=[float("nan")]),
                       lambda v:v.update(run_id="/content/private"),
                       lambda v:v["resource_use"].update(phase_records={"unknown":{"epochs_completed":1}})):
            value=outcome();mutate(value)
            with self.assertRaises((receipts.CampaignSyncError,ValueError)):
                receipts.validate_outcome(value)
        record=observed(claim);record["outcome_sha256"]="f"*64
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"outcome_hash"):
            client.save_observed(record)
        value=outcome();value.update(status="failed",predictions_original=None,prediction_dates=None,phase_epochs=None)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"noncanonical_public"):
            receipts.validate_outcome(value)

    def test_rate_limit_pauses_no_mutation_retry_or_error_body(self):
        remote=GitHub();client=store(remote)
        remote.reject=(429,{"Retry-After":"17"},b"private server body")
        with self.assertRaises(receipts.RecoveryReceiptRateLimited) as error:
            client.claim_started(started(client))
        self.assertEqual(error.exception.retry_after_seconds,17)
        self.assertEqual(str(error.exception),"recovery_receipt_rate_limit_pause")
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)

    def test_corrupt_remote_blob_and_oversized_records_refused(self):
        remote=GitHub();client=store(remote);client.claim_started(started(client));remote.corrupt=True
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"download_invalid"):client.inspect()
        record=started(client);record["owner_nonce"]="x"*(receipts.MAX_BYTES+1)
        with self.assertRaises(receipts.CampaignSyncError):client.claim_started(record)

    def test_existing_ref_must_descend_from_registered_base(self):
        remote=GitHub();remote.ref="d"*40;remote.unrelated=True;client=store(remote)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"ancestry_mismatch"):client.inspect()
        self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))
        remote.unrelated=False;client.inspect();client.inspect()
        self.assertEqual(sum("/compare/" in c[1] for c in remote.calls),2)

    def test_contents_raw_fallback_remains_on_authenticated_api_and_checks_blob(self):
        remote=GitHub();client=store(remote);claim=client.claim_started(started(client))
        def raw_contents(method,url,headers,payload):
            code,h,body=remote(method,url,headers,payload)
            if "/contents/" in url and method=="GET" and isinstance(body,dict):
                body=dict(body);body["encoding"]="none";body["content"]=""
            return code,h,body
        client._transport=raw_contents
        self.assertEqual(client.inspect()["started"],claim)
        self.assertTrue(any(c[2]["Accept"]=="application/vnd.github.raw+json" for c in remote.calls))
        self.assertTrue(all(urlsplit(c[1]).netloc=="api.github.com" for c in remote.calls))

    def test_public_inspect_needs_no_token_and_never_sends_auth_header(self):
        remote=GitHub();_,data,sha=authorization()
        from unittest.mock import patch
        with patch.dict("os.environ",{},clear=True):
            client=receipts.RecoveryReceiptStore(receipts.REPOSITORY,receipts.BRANCH,sha,receipts.BASE_COMMIT,
                transport=remote,authorization_bytes=data)
        client.inspect()
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"token_required"):
            client.claim_started(started(client))
        self.assertTrue(all("Authorization" not in c[2] for c in remote.calls))
        self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))

    def test_redirect_http_errors_and_transport_exceptions_are_safe(self):
        remote=GitHub();client=store(remote)
        def redirect(*args):return 302,{"Location":"https://untrusted.example"},b"private"
        client._transport=redirect
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"recovery_http_302"):client.inspect()
        def broken(*args):raise RuntimeError("test-private-token raw sensitive details")
        client._transport=broken
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"^recovery_request_unconfirmed$"):client.inspect()
        self.assertIsNone(receipts._NoRedirect().redirect_request(None,None,None,None,None))


if __name__=="__main__":unittest.main()
