"""Offline tests of authorization-bound immutable physical receipt stages."""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e2_recovery_receipts_v3 as receipts

R1={"head":"5648b8305174a827d1ffa0a90ce58dab1ad5b71c","count":494,"sequence":989,
    "job":"3775f65fd049666c641c1131e94726e12e98143f91a755075c07ac6761dd668b",
    "payload":"0eb5d72e23cb399edb41275467b3ae05c6cc856999f192c4cb254d25de49bf34",
    "start":"5afba2cd98646e734799c0ae7ef3bdbc86d4688eda63648354d8a2c757ead1f5",
    "session":"9bc856c8bdb34536ad8f01de065e2261","allocation":28800,"branch":"recoveries/e2-cuda-selection-v1/20261110-r1"}
R2={"head":"5122af43c9dab0b9a20e79622a5047bacd316db6","count":501,"sequence":1009,
    "job":"e98bca476404aa9d15236e51623ab2ee738c0cce6d43cc85513c07b8fa45fd5d",
    "payload":"a75359a62eca2f1e142857561657667fd9d5dd23b76d69c6749fb2ccc774cbb9",
    "start":"4b91d270d4499930736ca53fe0033f5826ab628f90de602cc50312acfcbc4f20",
    "session":"5b86b68a8fc84257b644cf8c95636d7e","allocation":14761.54321,"branch":"recoveries/e2-cuda-selection-v1/20261110-r2"}


def authorization(incident=R2):
    return {"version":"e2-manual-recovery-authorization-v3","status":"registered_before_recovery_execution",
        "registered_at_utc":"2026-11-10T17:00:00+00:00","registered_at_bogota":"2026-11-10T12:00:00-05:00",
        "repository":receipts.REPOSITORY,"campaign_branch":receipts.CAMPAIGN_BRANCH,"campaign_prefix":receipts.CAMPAIGN_BRANCH,
        "sealed_campaign_head":incident["head"],"plan_sha256":receipts.PLAN_SHA256,
        "scientific_source_commit":receipts.SOURCE_COMMIT,"scientific_registration_sha256":receipts.REGISTRATION_SHA256,
        "previous_completed_jobs":incident["count"],"target":{"job_id":incident["job"],
            "job_payload_sha256":incident["payload"],"started_event_sha256":incident["start"],
            "sequence":incident["sequence"],"old_session_id":incident["session"],"reservation_seconds":5700},
        "original_unknown_attempt":deepcopy(receipts.UNKNOWN),"decision":receipts.DECISION,
        "evidence":"Offline synthetic fixture: executor and output absence explicitly confirmed.",
        "limits":{"session_seconds":28800,"initial_campaign_seconds":86400,"campaign_seconds":129600,
            "budget_extension_event_sha256":receipts.BUDGET_EXTENSION_EVENT_SHA256,
            "old_session_conservative_allocation_seconds":incident["allocation"],"budget_extension_authorized":False},
        "ledger_interpretation":"New observed physical retry retains the old physical costs as unknown.",
        "receipt_branch":incident["branch"],"implementation_sha256":{name:"a"*64 for name in receipts.IMPLEMENTATION_PATHS}}


class GitHub:
    """Small Contents API mock with separate pre-existing immutable R1 files."""
    def __init__(self,auth):
        self.auth=deepcopy(auth);self.head=None;self.files={};self.calls=[]
        self.old_branch="recoveries/e2-cuda-selection-v1/20261110"
        self.old_files={"recovery/"+"a"*64+"/completed.json":b'{"fixture":"immutable-r1"}'}
        self.lose=None;self.reject=None;self.unrelated=False;self.corrupt=False
    def __call__(self,method,url,headers,payload):
        self.calls.append((method,url,deepcopy(headers),deepcopy(payload)))
        parsed=urlsplit(url);path=unquote(parsed.path);base="/repos/"+receipts.REPOSITORY
        assert parsed.scheme=="https" and parsed.netloc=="api.github.com" and path.startswith(base+"/")
        branch=self.auth["receipt_branch"];head=self.auth["sealed_campaign_head"]
        if method=="GET" and path==base+"/git/ref/heads/"+branch:
            return (404,{},b"") if self.head is None else (200,{}, {"object":{"sha":self.head}})
        if method=="GET" and path==base+"/compare/"+head+"..."+self.head:
            return 200,{}, {"status":"ahead","base_commit":{"sha":head},
                "merge_base_commit":{"sha":"f"*40 if self.unrelated else head}}
        if method=="POST" and path==base+"/git/refs":
            assert payload=={"ref":"refs/heads/"+branch,"sha":head}
            if self.head is not None:return 422,{},b""
            self.head=head;return 201,{}, {"object":{"sha":self.head}}
        assert "/contents/recovery/" in path
        name=path.split("/contents/",1)[1]
        assert name.endswith(("/started.json","/observed.json","/completed.json"))
        if method=="GET":
            assert unquote(parsed.query)=="ref="+branch
            if name not in self.files:return 404,{},b""
            data=self.files[name]
            if headers["Accept"]=="application/vnd.github.raw+json":return 200,{},data
            return 200,{}, {"type":"file","encoding":"base64","size":len(data),
                "sha":"0"*40 if self.corrupt else receipts.blob_sha(data),"content":base64.b64encode(data).decode()}
        assert method=="PUT" and parsed.query==""
        assert set(payload)=={"message","content","branch"} and payload["branch"]==branch
        if self.reject is not None:return self.reject
        if name in self.files:return 422,{},b""
        self.files[name]=base64.b64decode(payload["content"],validate=True)
        self.head=hashlib.sha1(self.files[name]).hexdigest()
        if self.lose==name.rsplit("/",1)[1]:
            self.lose=None;raise OSError("private transport text")
        return 201,{}, {"content":{"sha":receipts.blob_sha(self.files[name])}}


def store(remote,auth=None,*,raw=True,token="offline-test-token"):
    value=deepcopy(remote.auth if auth is None else auth);data=receipts.canonical(value)
    sha=hashlib.sha256(data).hexdigest()
    kwargs={"authorization_bytes":data} if raw else {"authorization":value}
    return receipts.RecoveryReceiptStore(value["repository"],value["receipt_branch"],sha,value["sealed_campaign_head"],
        token=token,transport=remote,**kwargs)


def common(client):
    a=client._authorization;t=a["target"]
    return {"plan_sha256":a["plan_sha256"],"scientific_source_commit":a["scientific_source_commit"],
        "recovery_source_commit":"e"*40,"sealed_campaign_head":a["sealed_campaign_head"],"job_id":t["job_id"],
        "job_payload_sha256":t["job_payload_sha256"],"original_started_event_sha256":t["started_event_sha256"],
        "old_session_id":t["old_session_id"],"recovery_session_id":"a"*32,"owner_nonce":client.owner_nonce,
        "original_unknown_attempt":deepcopy(receipts.UNKNOWN)}


def started(client):
    a=client._authorization;l=a["limits"]
    return {**common(client),"reservation_seconds":a["target"]["reservation_seconds"],
        "new_session_budget_seconds":l["session_seconds"],"campaign_budget_seconds":l["campaign_seconds"],
        "old_conservative_allocation_seconds":l["old_session_conservative_allocation_seconds"],
        "started_at_utc":"2026-11-10T17:00:00+00:00"}


def outcome():
    return {"status":"completed","run_id":"b"*64,"resource_use":{"optimizer_steps":3,
        "optimizer_steps_known_lower_bound":3,"optimizer_steps_unknown":False,"checkpoint_policy":"selection_metadata_only",
        "phase_records":{"supervised":{"epochs_completed":2,"optimizer_steps":3,"elapsed_seconds":4.0}},"cuda_metrics":{}},
        "predictions_original":[-0.5,0.8],"prediction_dates":["2024-01-01T00:00:00+00:00","2024-01-02T00:00:00+00:00"],
        "phase_epochs":{"supervised":2},"reason":None,"fatal":False}


def observed(claim):
    return {**{k:deepcopy(claim[k]) for k in receipts.COMMON-{"phase","sha256"}},"phase":"observed",
        "outcome":outcome(),"outcome_sha256":receipts.digest(outcome()),"actual_fit_wall_seconds":7,
        "recovery_elapsed_seconds":10,"observed_at_utc":"2026-11-10T17:00:10+00:00","started_receipt_sha256":claim["sha256"]}


def completed(client,actual):
    return {**{k:deepcopy(actual[k]) for k in receipts.COMMON-{"phase","sha256"}},"phase":"completed",
        "result_sha256":"c"*64,"campaign_commit":"d"*40,"completed_at_utc":"2026-11-10T17:00:12+00:00",
        "actual_fit_wall_seconds":7,"recovery_elapsed_seconds":12,"observed_receipt_sha256":actual["sha256"],
        "recovery_wall_seconds":12,"charged_recovery_overhead_seconds":5,
        "old_session_conservative_allocation_seconds":client._authorization["limits"]["old_session_conservative_allocation_seconds"]}


class RecoveryReceiptsV3Tests(unittest.TestCase):
    def test_both_incidents_bind_exact_distinct_authorized_targets_and_float_allocation(self):
        for incident in (R1,R2):
            with self.subTest(incident=incident["count"]):
                remote=GitHub(authorization(incident));old=deepcopy(remote.old_files);client=store(remote)
                claim=client.claim_started(started(client));actual=client.save_observed(observed(claim))
                final=client.save_completed(completed(client,actual))
                self.assertEqual(client.inspect(),{"started":claim,"observed":actual,"completed":final})
                self.assertEqual(claim["job_id"],incident["job"])
                self.assertEqual(final["old_session_conservative_allocation_seconds"],incident["allocation"])
                self.assertEqual(remote.old_files,old)
                self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),3)
                self.assertTrue(all(c[3]["branch"]==incident["branch"] for c in remote.calls if c[0]=="PUT"))

    def test_fractional_old_allocation_boundary_and_invalid_values(self):
        for allocation in (0.0001,5700.125,28800):
            a=authorization();a["limits"]["old_session_conservative_allocation_seconds"]=allocation
            client=store(GitHub(a));claim=client.claim_started(started(client))
            self.assertEqual(claim["old_conservative_allocation_seconds"],allocation)
        for allocation in (0,-1,28800.01,True,None,float("nan"),float("inf")):
            a=authorization();a["limits"]["old_session_conservative_allocation_seconds"]=allocation
            with self.assertRaises((receipts.CampaignSyncError,ValueError)):store(GitHub(a))

    def test_receipt_rejects_every_authorized_incident_binding_drift(self):
        client=store(GitHub(authorization()))
        for key,value in (("sealed_campaign_head",R1["head"]),("job_id",R1["job"]),
                ("job_payload_sha256",R1["payload"]),("original_started_event_sha256",R1["start"]),
                ("old_session_id",R1["session"]),("old_conservative_allocation_seconds",28800),
                ("plan_sha256","f"*64),("scientific_source_commit","f"*40)):
            record=started(client);record[key]=value
            with self.subTest(key=key),self.assertRaises(receipts.CampaignSyncError):client.claim_started(record)

    def test_sealed_raw_authorization_hash_and_remote_coordinates_cannot_drift(self):
        a=authorization();data=receipts.canonical(a);remote=GitHub(a);sha=hashlib.sha256(data).hexdigest()
        for branch,head in ((R1["branch"],a["sealed_campaign_head"]),(a["receipt_branch"],R1["head"])):
            with self.assertRaises(receipts.RecoveryReceiptError):
                receipts.RecoveryReceiptStore(receipts.REPOSITORY,branch,sha,head,token="test",transport=remote,authorization_bytes=data)
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_hash"):
            receipts.RecoveryReceiptStore(receipts.REPOSITORY,a["receipt_branch"],"f"*64,a["sealed_campaign_head"],
                token="test",transport=remote,authorization_bytes=data)
        changed=deepcopy(a);changed["target"]["job_id"]="f"*64
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_mismatch"):
            receipts.RecoveryReceiptStore(receipts.REPOSITORY,a["receipt_branch"],sha,a["sealed_campaign_head"],
                token="test",transport=remote,authorization_bytes=data,authorization=changed)
        self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))

    def test_fixed_science_budget_decision_and_exact_new_helper_closure(self):
        mutations=(lambda a:a.update(version="e2-manual-recovery-authorization-v1"),
            lambda a:a.update(decision="manual_retry_same_sealed_logical_job_once_after_colab_local_cache_absent"),
            lambda a:a.update(scientific_registration_sha256="f"*64),lambda a:a.update(plan_sha256="f"*64),
            lambda a:a.update(scientific_source_commit="f"*40),lambda a:a.update(campaign_branch="main"),
            lambda a:a["limits"].update(session_seconds=28800.0),lambda a:a["limits"].update(campaign_seconds=172800),
            lambda a:a["limits"].update(budget_extension_authorized=True),
            lambda a:a["implementation_sha256"].update({"scripts/e2_recovery_receipts.py":"a"*64}),
            lambda a:a["target"].update(sequence=True),lambda a:a.update(previous_completed_jobs=True))
        for mutate in mutations:
            a=authorization();mutate(a)
            with self.assertRaises(receipts.CampaignSyncError):store(GitHub(a))

    def test_existing_extension_is_bound_without_authorizing_another_extension(self):
        remote=GitHub(authorization());client=store(remote)
        claim=client.claim_started(started(client))
        actual=client.save_observed(observed(claim))
        final=client.save_completed(completed(client,actual))
        self.assertEqual(client._authorization["limits"]["initial_campaign_seconds"],86400)
        self.assertEqual(claim["campaign_budget_seconds"],129600)
        self.assertEqual(client._authorization["limits"]["budget_extension_event_sha256"],
            receipts.BUDGET_EXTENSION_EVENT_SHA256)
        self.assertIs(client._authorization["limits"]["budget_extension_authorized"],False)
        for record in (claim,actual,final):
            self.assertEqual(record["original_unknown_attempt"],receipts.UNKNOWN)
            self.assertIsNone(record["original_unknown_attempt"]["elapsed_seconds"])
            self.assertIsNone(record["original_unknown_attempt"]["optimizer_steps"])
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),3)

    def test_initial_effective_and_extension_event_bindings_fail_before_remote_write(self):
        for key,value in (("initial_campaign_seconds",129600),
                ("initial_campaign_seconds",86400.0),
                ("campaign_seconds",86400),("campaign_seconds",172800),
                ("campaign_seconds",129600.0),
                ("budget_extension_event_sha256","f"*64),
                ("budget_extension_event_sha256",None),
                ("budget_extension_authorized",True)):
            a=authorization();a["limits"][key]=value;remote=GitHub(a)
            with self.subTest(key=key,value=value),self.assertRaises(receipts.CampaignSyncError):
                store(remote)
            self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))
        for key in ("initial_campaign_seconds","budget_extension_event_sha256"):
            a=authorization();del a["limits"][key];remote=GitHub(a)
            with self.subTest(missing=key),self.assertRaises(receipts.CampaignSyncError):
                store(remote)
            self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))

    def test_started_receipt_uses_effective_budget_and_keeps_closed_schema(self):
        for key,value in (("campaign_budget_seconds",86400),
                ("campaign_budget_seconds",172800),("initial_campaign_seconds",86400),
                ("budget_extension_event_sha256",receipts.BUDGET_EXTENSION_EVENT_SHA256)):
            remote=GitHub(authorization());client=store(remote)
            record=started(client);record[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(receipts.CampaignSyncError):
                client.claim_started(record)
            self.assertFalse(any(c[0]=="PUT" for c in remote.calls))

    def test_old_branch_and_malformed_new_branch_are_read_write_ineligible(self):
        for branch in ("recoveries/e2-cuda-selection-v1/20261110","main","campaigns/e2-cuda-selection-v1",
                "recoveries/e2-cuda-selection-v1/20260230-r2","recoveries/e2-cuda-selection-v1/20261110-r0",
                "recoveries/e2-cuda-selection-v1/20261110-r2/extra"):
            a=authorization();a["receipt_branch"]=branch
            with self.assertRaises(receipts.CampaignSyncError):store(GitHub(a))

    def test_lost_start_ack_and_repeated_same_owner_are_one_claim_only(self):
        remote=GitHub(authorization());remote.lose="started.json";client=store(remote)
        claim=client.claim_started(started(client));self.assertEqual(client.claim_started(started(client)),claim)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)
        other=store(remote)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"another_process"):other.claim_started(started(other))
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"another_process"):other.claim_started(started(client))
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"live_claim"):other.save_observed(observed(claim))

    def test_observed_ack_loss_and_new_process_completion_use_no_new_claim(self):
        remote=GitHub(authorization());first=store(remote);claim=first.claim_started(started(first))
        remote.lose="observed.json";actual=first.save_observed(observed(claim));other=store(remote)
        final=other.save_completed(completed(other,actual));self.assertEqual(other.save_completed(completed(other,actual)),final)
        with self.assertRaisesRegex(receipts.RecoveryReceiptConflict,"already_observed"):other.claim_started(started(other))
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),3)

    def test_resumed_clock_lower_bound_separate_from_current_repair(self):
        remote=GitHub(authorization());first=store(remote);claim=first.claim_started(started(first))
        record=observed(claim);record["recovery_elapsed_seconds"]=120;actual=first.save_observed(record)
        other=store(remote);record=completed(other,actual)
        record.update(recovery_elapsed_seconds=120,recovery_elapsed_lower_bound_only=True,current_repair_elapsed_seconds=2,
            recovery_wall_seconds=28800,charged_recovery_overhead_seconds=28793)
        final=other.save_completed(record)
        self.assertEqual(final["current_repair_elapsed_seconds"],2)
        self.assertEqual(final["recovery_elapsed_seconds"],120)
        self.assertTrue(final["recovery_elapsed_lower_bound_only"])

    def test_hash_links_nonfinite_and_private_fields_refused_before_upload(self):
        remote=GitHub(authorization());client=store(remote);claim=client.claim_started(started(client))
        for mutate in (lambda v:v.update(started_receipt_sha256="f"*64),lambda v:v.update(outcome_sha256="f"*64),
                lambda v:v["outcome"]["resource_use"].update(stdout="private"),
                lambda v:v.update(actual_fit_wall_seconds=float("nan")),lambda v:v.update(source_root="/content/private")):
            record=observed(claim);mutate(record)
            with self.assertRaises(receipts.CampaignSyncError):client.save_observed(record)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)

    def test_public_read_only_inspect_needs_auth_mapping_but_no_token(self):
        remote=GitHub(authorization())
        with patch.dict("os.environ",{},clear=True):client=store(remote,raw=False,token=None)
        client.inspect()
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_bytes_required"):client.claim_started(started(client))
        self.assertTrue(all("Authorization" not in c[2] for c in remote.calls))
        self.assertFalse(any(c[0] in ("PUT","POST") for c in remote.calls))
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"authorization_required"):
            receipts.RecoveryReceiptStore(receipts.REPOSITORY,remote.auth["receipt_branch"],"a"*64,remote.auth["sealed_campaign_head"])

    def test_ancestry_rate_limit_redirect_and_raw_fallback_are_fail_closed(self):
        remote=GitHub(authorization());client=store(remote);remote.head="d"*40;remote.unrelated=True
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"ancestry"):client.inspect()
        remote.head=None;remote.unrelated=False;remote.reject=(429,{"Retry-After":"17"},b"private")
        with self.assertRaises(receipts.RecoveryReceiptRateLimited) as error:client.claim_started(started(client))
        self.assertEqual(error.exception.retry_after_seconds,17)
        self.assertEqual(sum(c[0]=="PUT" for c in remote.calls),1)
        remote.reject=None;client.claim_started(started(client))
        def raw(method,url,headers,payload):
            code,h,body=remote(method,url,headers,payload)
            if method=="GET" and "/contents/" in url and isinstance(body,dict):body={**body,"encoding":"none","content":""}
            return code,h,body
        client._transport=raw;self.assertIsNotNone(client.inspect()["started"])
        client._transport=lambda *args:(302,{"Location":"https://untrusted.example"},b"private")
        with self.assertRaisesRegex(receipts.RecoveryReceiptError,"^recovery_http_302$"):client.inspect()
        self.assertIsNone(receipts._NoRedirect().redirect_request(None,None,None,None,None))

    def test_r1_sources_are_unchanged_when_present(self):
        hashes={"scripts/e2_recovery_receipts.py":"b7838a9bb442ed3341ef0d99783f7a00a8bfd2e71632c4887428a472cde58761",
            "tests/test_e2_recovery_receipts.py":"7154e74641e1116d276182e41f446bb632a3f235e8e6931c871eef7172e83850"}
        for name,expected in hashes.items():
            if (ROOT/name).exists():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)


if __name__=="__main__":unittest.main()
