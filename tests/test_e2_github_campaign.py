"""Scientific-only GitHub recovery, mocked GitData/CAS/archive transport."""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.parse import urlsplit
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_github_campaign as github
from e2_search_session import SearchSession, UnresolvedAttempt


PLAN={"version":"e2-cuda-search-plan-v1","source_commit":"a"*40,"protocol_sha256":"b"*64,
      "models":["GRU32"],"origins":["2025-01-01"],"checkpoint_policy":"selection_metadata_only_refit_separate"}
OUTCOME={"status":"completed","run_id":"c"*64,"resource_use":{"optimizer_steps":4,
    "optimizer_steps_known_lower_bound":4,"phase_records":{}},"predictions_original":[.1,.2],
    "prediction_dates":["2024-12-23T00:00:00+00:00","2024-12-24T00:00:00+00:00"],
    "phase_epochs":{"supervised":2},"reason":None,"fatal":False}


class GitServer:
    def __init__(self):
        self.head=None;self.blobs={};self.trees={};self.commits={};self.calls=[]
        self.fail_next=None;self.conflict=False;self.corrupt_archive=False;self.archive_extra=None

    def tree(self,flat):
        sha=hashlib.sha1(github.canonical(flat)).hexdigest();self.trees[sha]=dict(flat);return sha

    def __call__(self,method,url,headers,payload):
        self.calls.append((method,url,dict(headers),deepcopy(payload)))
        if self.fail_next:
            failure=self.fail_next;self.fail_next=None;return failure
        if urlsplit(url).hostname=="codeload.github.com":
            sha=url.rsplit("/",1)[-1]
            flat=self.trees[self.commits[sha]["tree"]]
            output=io.BytesIO()
            with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
                for path,blob in flat.items():
                    data=self.blobs[blob]
                    if self.corrupt_archive and path.endswith("campaign.json"):data=data.replace(b'"GRU32"',b'"LSTM32"')
                    archive.writestr("repo-"+sha+"/"+path,data)
                if self.archive_extra:archive.writestr(self.archive_extra,b"bad")
            return 200,{},output.getvalue()
        path=urlsplit(url).path.split("/git/")[-1]
        response_headers={"x-ratelimit-remaining":"5000"}
        if method=="GET" and path.startswith("ref/heads/"):
            return (404,response_headers,{}) if self.head is None else (200,response_headers,{"object":{"sha":self.head}})
        if method=="GET" and path.startswith("commits/"):
            return 200,response_headers,{"tree":{"sha":self.commits[path[8:]]["tree"]}}
        if method=="GET" and path.startswith("trees/"):
            flat=self.trees[path[6:]];entries=[]
            for name in sorted({p.split("/",1)[0] for p in flat}):
                if name in flat:entries.append({"path":name,"type":"blob","sha":flat[name]})
                else:
                    child={p[len(name)+1:]:v for p,v in flat.items() if p.startswith(name+"/")}
                    entries.append({"path":name,"type":"tree","sha":self.tree(child)})
            return 200,response_headers,{"tree":entries,"truncated":False}
        if method=="GET" and path.startswith("blobs/"):
            data=self.blobs[path[6:]]
            return 200,response_headers,{"encoding":"base64","content":base64.b64encode(data).decode(),"size":len(data)}
        if method=="POST" and path=="blobs":
            data=base64.b64decode(payload["content"]);sha=github.blob_sha(data);self.blobs[sha]=data
            return 201,response_headers,{"sha":sha}
        if method=="POST" and path=="trees":
            flat=dict(self.trees.get(payload.get("base_tree"),{}))
            for entry in payload["tree"]:flat[entry["path"]]=entry["sha"]
            return 201,response_headers,{"sha":self.tree(flat)}
        if method=="POST" and path=="commits":
            sha=hashlib.sha1(github.canonical(payload)).hexdigest();self.commits[sha]=dict(payload)
            return 201,response_headers,{"sha":sha}
        if method=="POST" and path=="refs":
            if self.head is not None:return 422,response_headers,{}
            self.head=payload["sha"];return 201,response_headers,{"object":{"sha":self.head}}
        if method=="PATCH" and path.startswith("refs/heads/"):
            if self.conflict or self.commits[payload["sha"]]["parents"]!=[self.head]:return 422,response_headers,{}
            assert payload["force"] is False
            self.head=payload["sha"];return 200,response_headers,{"object":{"sha":self.head}}
        raise AssertionError((method,path))


def session(directory):
    return SearchSession(directory,PLAN,session_budget_seconds=100,campaign_budget_seconds=300,max_storage_bytes=10*1024**2)


def mirror(server,**kwargs):
    return github.GitHubCampaignMirror("jassan-bit/DeepLearning_202630","campaigns/test35","campaigns/test35",
                                      token="private-test-value",transport=server,**kwargs)


class MirrorTests(unittest.TestCase):
    def test_missing_remote_is_noop_and_does_not_create_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"campaign"
            self.assertEqual(mirror(GitServer()).restore(path)["status"],"remote_missing")
            self.assertFalse(path.exists())

    def test_initial_orphan_branch_has_only_scientific_files_and_no_main_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server)
            with session(Path(tmp)/"local") as state:
                remote.push(state.directory,paths=state.pending_record_paths)
            self.assertEqual(server.commits[server.head]["parents"],[])
            files=server.trees[server.commits[server.head]["tree"]]
            self.assertEqual(set(files),{"campaigns/test35/campaign.json","campaigns/test35/snapshot.json"})
            writes=[c for c in server.calls if c[0] in ("POST","PATCH") and "/refs" in c[1]]
            self.assertEqual(writes[0][3]["ref"],"refs/heads/campaigns/test35")
            self.assertFalse(any("main" in c[1] or "gh-pages" in c[1] for c in writes))

    def test_completed_result_round_trip_and_no_refit_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server);a=Path(tmp)/"a";b=Path(tmp)/"b"
            with session(a) as state:
                remote.push(a,paths=state.pending_record_paths);state.mark_records_synced(state.pending_record_paths)
                def callback(deadline):
                    remote.push(a,paths=state.pending_record_paths);state.mark_records_synced(state.pending_record_paths)
                    return OUTCOME
                self.assertEqual(state.run_job("job1",{"seed":1},callback,reservation_seconds=1),OUTCOME)
                remote.push(a,paths=state.pending_record_paths);state.mark_records_synced(state.pending_record_paths)
            remote.push(a,paths=state.pending_record_paths)
            restored=mirror(server);self.assertGreater(restored.restore(b)["restored_files"],0)
            with session(b) as state:
                result=state.run_job("job1",{"seed":1},lambda deadline:self.fail("fit was replayed"),reservation_seconds=1)
                self.assertEqual(result,OUTCOME)
            archive_calls=[c for c in server.calls if "codeload.github.com" in c[1]]
            self.assertTrue(archive_calls)
            self.assertTrue(all("Authorization" not in c[2] for c in archive_calls))

    def test_remote_started_without_result_blocks_instead_of_replaying(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server);a=Path(tmp)/"a"
            with session(a) as state:
                remote.push(a)
                def crash(deadline):
                    remote.push(a,paths=state.pending_record_paths)
                    raise RuntimeError("synthetic process loss")
                with self.assertRaises(UnresolvedAttempt):state.run_job("job",{},crash,reservation_seconds=1)
            b=Path(tmp)/"b";mirror(server).restore(b)
            with self.assertRaises(UnresolvedAttempt):session(b)

    def test_local_committed_result_after_remote_start_is_preserved_and_can_be_pushed(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server);a=Path(tmp)/"a"
            with session(a) as state:
                remote.push(a)
                def done(deadline):remote.push(a,paths=state.pending_record_paths);return OUTCOME
                state.run_job("job",{},done,reservation_seconds=1)
            resumed=mirror(server);resumed.restore(a)
            with session(a) as state:
                self.assertEqual(state.run_job("job",{},lambda deadline:self.fail("repeat"),reservation_seconds=1),OUTCOME)
                resumed.push(a,paths=state.pending_record_paths)
            b=Path(tmp)/"b";mirror(server).restore(b)
            with session(b) as state:self.assertEqual(state.completed_job_ids,("job",))

    def test_incremental_noop_does_not_upload_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server)
            with session(Path(tmp)/"a") as state:
                pending=state.pending_record_paths;remote.push(state.directory,paths=pending);state.mark_records_synced(pending)
                before=len(server.calls)
                self.assertEqual(remote.push(state.directory,paths=state.pending_record_paths)["status"],"unchanged")
                self.assertEqual(len(server.calls),before)

    def test_ref_compare_and_swap_conflict_never_forces_or_rebases(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();remote=mirror(server)
            with session(Path(tmp)/"a") as state:
                remote.push(state.directory);head=server.head
                state.extend_campaign_budget(400,reason="additional compute allocation")
                server.conflict=True
                with self.assertRaises(github.CampaignConflict):remote.push(state.directory,paths=state.pending_record_paths)
                self.assertEqual(server.head,head)
                with self.assertRaisesRegex(github.CampaignSyncError,"refresh_required"):
                    remote.push(state.directory,paths=state.pending_record_paths)
                patches=[c for c in server.calls if c[0]=="PATCH"]
                self.assertEqual(len(patches),1);self.assertFalse(patches[0][3]["force"])

    def test_private_fields_paths_raw_logs_and_weights_are_not_published(self):
        for bad in ({"env":{"token":"secret"}}, {"stdout":"worker output"}, {"source_commit":"/content/private"},
                    {"source_commit":"ghp_abcdefghijklmnopqrstuv"}):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"a";plan={**PLAN,**bad}
                with SearchSession(root,plan,session_budget_seconds=100,campaign_budget_seconds=300,max_storage_bytes=10*1024**2):pass
                server=GitServer()
                with self.assertRaises(github.CampaignSyncError):mirror(server).push(root)
                self.assertFalse(any(c[0]=="POST" for c in server.calls))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a"
            with session(root):pass
            (root/"weights.pt").write_bytes(b"private weights")
            (root/"worker.log").write_text("private runtime data")
            server=GitServer();mirror(server).push(root)
            self.assertFalse(any(p.endswith(("weights.pt","worker.log")) for p in server.trees[server.commits[server.head]["tree"]]))

    def test_unknown_outcome_fields_are_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a"
            with session(root) as state:
                state.run_job("job",{},lambda deadline:{**OUTCOME,"raw_report":"bad"},reservation_seconds=1)
            server=GitServer()
            with self.assertRaises(github.CampaignSyncError):mirror(server).push(root)
            self.assertFalse(any(c[0]=="POST" for c in server.calls))

    def test_unapproved_nested_plan_fields_and_source_paths_are_rejected(self):
        for field in ({"execution":{"unapproved":"arbitrary runtime text"}},
                      {"hardware_binding":{"serial_number":"sensitive"}},
                      {"source_sha256":{"private/internal.py":"d"*64}},
                      {"coverage":{"notes":"private runtime"}}):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"a"
                with SearchSession(root,{**PLAN,**field},session_budget_seconds=100,campaign_budget_seconds=300,max_storage_bytes=10*1024**2):pass
                server=GitServer()
                with self.assertRaises(github.CampaignSyncError):mirror(server).push(root)
                self.assertFalse(any(c[0]=="POST" for c in server.calls))

    def test_symlinks_and_explicit_unsupported_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a";server=GitServer()
            with session(root):pass
            (root/"alias").symlink_to(root,target_is_directory=True)
            with self.assertRaises(github.CampaignSyncError):mirror(server).push(root/"alias")
            with self.assertRaises(github.CampaignSyncError):mirror(server).push(root,paths=["weights.pt"])
            self.assertFalse(any(c[0]=="POST" for c in server.calls))

    def test_merkle_inventory_is_order_independent_and_changes_with_file_bytes(self):
        first=github._Inventory();second=github._Inventory()
        records=[("events/000000001.json",b"one"),("campaign.json",b"two"),("status.json",b"three")]
        for path,data in records:first.set(path,github._descriptor(path,data))
        for path,data in reversed(records):second.set(path,github._descriptor(path,data))
        self.assertEqual(first.root,second.root)
        second.set("status.json",github._descriptor("status.json",b"changed"))
        self.assertNotEqual(first.root,second.root)
        second.set("status.json",github._descriptor("status.json",b"three"))
        self.assertEqual(first.root,second.root)

    def test_snapshot_budget_refuses_growth_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a";server=GitServer()
            with session(root):pass
            with self.assertRaises(github.CampaignSyncError):mirror(server,max_snapshot_bytes=1).push(root)
            self.assertFalse(any(c[0]=="POST" for c in server.calls))

    def test_restore_refuses_immutable_conflict_without_publishing_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            server=GitServer();a=Path(tmp)/"a";b=Path(tmp)/"b"
            with session(a):pass
            mirror(server).push(a);b.mkdir();(b/"campaign.json").write_text("different")
            with self.assertRaises(github.CampaignConflict):mirror(server).restore(b)
            self.assertEqual((b/"campaign.json").read_text(),"different")
            self.assertEqual(list(b.iterdir()),[b/"campaign.json"])

    def test_archive_inventory_corruption_and_traversal_are_rejected(self):
        for mode in ("corrupt","traversal"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                server=GitServer();a=Path(tmp)/"a"
                with session(a):pass
                mirror(server).push(a)
                if mode=="corrupt":server.corrupt_archive=True
                else:server.archive_extra="repo/../../outside"
                with self.assertRaises(github.CampaignSyncError):mirror(server).restore(Path(tmp)/"b")
                self.assertFalse((Path(tmp)/"b").exists())

    def test_rate_limit_and_http_failures_never_expose_token_or_body(self):
        for response in ((429,{"Retry-After":"17"},b"private body"),(403,{},b"private body"),(302,{},b"redirect")):
            with self.subTest(response=response):
                server=GitServer();server.fail_next=response
                with self.assertRaises(github.CampaignSyncError) as caught:mirror(server).inspect_remote()
                self.assertNotIn("private",str(caught.exception));self.assertNotIn("Bearer",str(caught.exception))
                self.assertEqual(len(server.calls),1)
                if response[0]==429:self.assertEqual(caught.exception.retry_after_seconds,17)

    def test_low_known_quota_pauses_before_attempt_creation(self):
        server=GitServer();remote=mirror(server);remote._remaining=2
        with self.assertRaises(github.CampaignRateLimited):remote.check_capacity()
        self.assertEqual(server.calls,[])

    def test_snapshot_capacity_reservation_pauses_before_a_started_record_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a";server=GitServer();remote=mirror(server)
            with session(root):pass
            remote.push(root)
            used=remote._inventory.bytes
            remote.max_snapshot_bytes=used+100
            before=len(server.calls)
            remote.check_capacity(reserve_bytes=100)
            with self.assertRaisesRegex(github.CampaignSyncError,"snapshot_capacity_pause"):
                remote.check_capacity(reserve_bytes=101)
            self.assertEqual(len(server.calls),before)
            self.assertFalse(list((root/"events").iterdir()))

    def test_default_remote_bounds_cover_the_full_scientific_metadata_plan(self):
        remote=mirror(GitServer())
        self.assertEqual(remote.max_snapshot_bytes,2*1024**3)
        self.assertEqual(github.MAX_ARCHIVE_BYTES,1024**3)
        self.assertEqual(github.MAX_FILE_BYTES,8*1024**2)
        for bad in (0,-1,True,1.5):
            with self.subTest(bad=bad),self.assertRaises(github.CampaignSyncError):
                remote.check_capacity(reserve_bytes=bad)

    def test_only_campaign_branches_and_matching_prefix_are_allowed(self):
        for branch,prefix in (("main","main"),("gh-pages","gh-pages"),("campaigns/../bad","campaigns/../bad"),("campaigns/a","campaigns/b")):
            with self.subTest(branch=branch),self.assertRaises(github.CampaignSyncError):
                github.GitHubCampaignMirror("jassan-bit/DeepLearning_202630",branch,prefix,token="valid")

    def test_derived_status_can_change_and_restore_keeps_local_recomputed_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a";server=GitServer();remote=mirror(server)
            with session(root) as state:plan_hash=state.plan_sha256
            status={"version":"e2-cuda-search-status-v1","plan_sha256":plan_hash,"status":"paused"}
            (root/"status.json").write_text(json.dumps(status))
            remote.push(root);head=server.head
            status["status"]="complete";(root/"status.json").write_text(json.dumps(status))
            remote.push(root,paths=["status.json"])
            self.assertNotEqual(server.head,head)
            status["status"]="locally_recomputed";(root/"status.json").write_text(json.dumps(status))
            mirror(server).restore(root)
            self.assertEqual(json.loads((root/"status.json").read_text())["status"],"locally_recomputed")

    def test_derived_status_must_bind_the_campaign_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"a";server=GitServer()
            with session(root):pass
            (root/"status.json").write_text(json.dumps({"version":"e2-cuda-search-status-v1","plan_sha256":"0"*64,"status":"paused"}))
            with self.assertRaises(github.CampaignSyncError):mirror(server).push(root)
            self.assertFalse(any(c[0]=="POST" for c in server.calls))

    def test_pause_status_accepts_fixed_codes_and_finite_retry_intervals_only(self):
        base={"version":"e2-cuda-search-status-v1","plan_sha256":"a"*64,"status":"paused_github_sync",
              "pause_reason":"github_rate_limit_pause","retry_after_seconds":17}
        self.assertEqual(github.validate_record("status.json",github.canonical(base))["retry_after_seconds"],17)
        for change in ({"pause_reason":"arbitrary exception text"},{"retry_after_seconds":-1},
                       {"retry_after_seconds":True},{"retry_after_seconds":"17"}):
            with self.subTest(change=change),self.assertRaises(github.CampaignSyncError):
                github.validate_record("status.json",github.canonical({**base,**change}))

    def test_public_resource_identity_is_supported_but_private_identity_fields_are_not(self):
        for extra in ({},{"GITHUB_TOKEN":"unapprovedsecret"}):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"a";value=deepcopy(OUTCOME)
                value["resource_use"]["identity"]={"model":"GRU32","origin":"2025-01-01","candidate_id":"hpo:GRU32:hpo-000",
                    "payload_sha256":"d"*64,"seed":1,"stage":"hpo","prepared_signature":"e"*64,**extra}
                with session(root) as state:state.run_job("job",{},lambda deadline:value,reservation_seconds=1)
                server=GitServer()
                if extra:
                    with self.assertRaises(github.CampaignSyncError):mirror(server).push(root)
                else:self.assertEqual(mirror(server).push(root)["status"],"synced")

    def test_plan_binds_the_exact_remote_repository_branch_and_prefix(self):
        for branch in ("campaigns/test35","campaigns/another"):
            with self.subTest(branch=branch),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/"a";server=GitServer()
                plan={**PLAN,"remote_storage":{"repository":"jassan-bit/DeepLearning_202630","branch":branch,"prefix":branch}}
                with SearchSession(root,plan,session_budget_seconds=100,campaign_budget_seconds=300,max_storage_bytes=10*1024**2):pass
                if branch=="campaigns/test35":self.assertEqual(mirror(server).push(root)["status"],"synced")
                else:
                    with self.assertRaises(github.CampaignConflict):mirror(server).push(root)
                    self.assertFalse(any(c[0]=="POST" for c in server.calls))


if __name__=="__main__":unittest.main()
