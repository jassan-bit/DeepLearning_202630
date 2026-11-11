"""Real local Git persistence; no network, credentials or GPU execution."""
from copy import deepcopy
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_github_campaign import CampaignConflict,CampaignRateLimited,CampaignSyncError
from e2_github_git_transport_v2 import GitCampaignMirror,GitDataTransport
from e2_search_session import SearchSession,UnresolvedAttempt

PLAN={"version":"e2-cuda-search-plan-v1","source_commit":"a"*40,"protocol_sha256":"b"*64,
      "models":["GRU32"],"origins":["2025-01-01"],"checkpoint_policy":"selection_metadata_only_refit_separate",
      "remote_storage":{"repository":"jassan-bit/DeepLearning_202630","branch":"campaigns/test35","prefix":"campaigns/test35"}}


def git(*args):
    return subprocess.run(["git",*map(str,args)],check=True,capture_output=True).stdout.decode().strip()


def remote_repository(root):
    remote=Path(root)/"remote.git"
    git("init","--bare","--object-format=sha1",remote)
    return remote


def mirror(remote):
    return GitCampaignMirror("jassan-bit/DeepLearning_202630","campaigns/test35","campaigns/test35",
        token="private-test-token-12345",_test_remote=remote,_allow_local_for_tests=True)


def session(root):
    return SearchSession(root,PLAN,session_budget_seconds=100,campaign_budget_seconds=300,max_storage_bytes=10*1024**2)


def outcome(index):
    return {"status":"completed","run_id":hashlib.sha256(str(index).encode()).hexdigest(),
        "resource_use":{"optimizer_steps":4,"optimizer_steps_known_lower_bound":4,"phase_records":{}},
        "predictions_original":[.1,.2],"prediction_dates":["2024-12-23T00:00:00+00:00","2024-12-24T00:00:00+00:00"],
        "phase_epochs":{"supervised":2},"reason":None,"fatal":False}


class GitTransportTests(unittest.TestCase):
    def test_commit_hash_containing_429_is_not_an_http_rate_limit(self):
        transport=object.__new__(GitDataTransport)
        transport._known_head="a"*40
        transport._ref="refs/heads/campaigns/test35"
        result=subprocess.CompletedProcess([],1,b"! [rejected] abc429def -> campaigns/test35 (fetch first)",b"")
        with patch.object(transport,"_object",return_value=b"parent "+b"a"*40+b"\n\nmessage"), \
             patch.object(transport,"_run",return_value=result):
            self.assertEqual(transport._push("b"*40,create=False)[0],409)

    def test_explicit_http_429_still_pauses_transport(self):
        transport=object.__new__(GitDataTransport)
        transport._known_head="a"*40
        transport._ref="refs/heads/campaigns/test35"
        result=subprocess.CompletedProcess([],1,b"",b"fatal: The requested URL returned error: 429")
        with patch.object(transport,"_object",return_value=b"parent "+b"a"*40+b"\n\nmessage"), \
             patch.object(transport,"_run",return_value=result):
            with self.assertRaises(CampaignRateLimited):
                transport._push("b"*40,create=False)

    def test_real_git_three_job_roundtrip_and_resume_without_repeated_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp);first=Path(tmp)/"first";second=Path(tmp)/"second";calls=[]
            with mirror(remote) as target:
                self.assertEqual(target.restore(first)["status"],"remote_missing")
                with session(first) as state:
                    pending=state.pending_record_paths;target.push(first,paths=pending);state.mark_records_synced(pending)
                    for index in range(3):
                        def callback(deadline,index=index):
                            pending=state.pending_record_paths;target.push(first,paths=pending);state.mark_records_synced(pending)
                            calls.append(index);return outcome(index)
                        result=state.run_job("job-%d"%index,{"seed":index},callback,reservation_seconds=1)
                        self.assertEqual(result,outcome(index))
                        pending=state.pending_record_paths;target.push(first,paths=pending);state.mark_records_synced(pending)
                target.push(first,paths=state.pending_record_paths)
                initial=target._head
                self.assertEqual(target.git_transport._run(["rev-parse","refs/e2-campaign/current"]).stdout.decode().strip(),initial)
            self.assertEqual(git("--git-dir="+str(remote),"for-each-ref","--format=%(refname)","refs/heads/"),"refs/heads/campaigns/test35")
            self.assertEqual(git("--git-dir="+str(remote),"rev-list","--max-parents=0",initial).count("\n"),0)
            with mirror(remote) as restored:
                self.assertGreater(restored.restore(second)["restored_files"],0)
                with session(second) as state:
                    for index in range(3):
                        cached=state.run_job("job-%d"%index,{"seed":index},lambda deadline:self.fail("fit repeated"),reservation_seconds=1)
                        self.assertEqual(cached,outcome(index))
            self.assertEqual(calls,[0,1,2])

    def test_non_fast_forward_race_rejects_second_writer_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp);a=Path(tmp)/"a";b=Path(tmp)/"b"
            with mirror(remote) as first,mirror(remote) as second:
                first.restore(a)
                with session(a) as setup:first.push(a,paths=setup.pending_record_paths)
                second.restore(b)
                with session(a) as state_a,session(b) as state_b:
                    def first_callback(deadline):first.push(a,paths=state_a.pending_record_paths);return outcome(1)
                    state_a.run_job("one",{},first_callback,reservation_seconds=1)
                    first.push(a,paths=state_a.pending_record_paths)
                    head=first._head;worked=[]
                    def second_callback(deadline):
                        second.push(b,paths=state_b.pending_record_paths)
                        worked.append(True);return outcome(2)
                    with self.assertRaises(UnresolvedAttempt) as caught:
                        state_b.run_job("two",{},second_callback,reservation_seconds=1)
                    self.assertIsInstance(caught.exception.__cause__,CampaignConflict)
                    self.assertEqual(worked,[])
                    self.assertEqual(git("--git-dir="+str(remote),"rev-parse","refs/heads/campaigns/test35"),head)

    def test_remote_started_state_blocks_restored_campaign_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp);a=Path(tmp)/"a";b=Path(tmp)/"b"
            with mirror(remote) as target:
                target.restore(a)
                with session(a) as state:
                    target.push(a,paths=state.pending_record_paths)
                    def interrupted(deadline):
                        target.push(a,paths=state.pending_record_paths)
                        raise RuntimeError("synthetic interruption")
                    with self.assertRaises(UnresolvedAttempt):state.run_job("incomplete",{},interrupted,reservation_seconds=1)
            with mirror(remote) as target:target.restore(b)
            with self.assertRaises(UnresolvedAttempt):session(b)

    def test_token_is_only_in_private_askpass_environment_never_argv_script_or_config(self):
        token="private-test-token-12345";calls=[];start=subprocess.Popen
        def observed(argv,**kwargs):
            calls.append((list(argv),dict(kwargs.get("env",{}))))
            return start(argv,**kwargs)
        with patch.dict(os.environ,{"GIT_TRACE":"/tmp/unapproved-trace","GIT_CONFIG_COUNT":"1",
                                    "GIT_CONFIG_KEY_0":"credential.helper","GIT_CONFIG_VALUE_0":"unapproved"}):
            with patch("e2_github_git_transport.subprocess.Popen",side_effect=observed):
                target=GitCampaignMirror("jassan-bit/DeepLearning_202630","campaigns/test35","campaigns/test35",token=token)
                transport=target.git_transport;private=transport._private
                config=(transport._cache/"config").read_text();askpass=(private/"askpass.sh").read_text()
                self.assertNotIn(token,config);self.assertNotIn(token,askpass)
                self.assertIn("https://github.com/jassan-bit/DeepLearning_202630.git",config)
                self.assertIn('$GITHUB_TOKEN',askpass)
                self.assertTrue(all(token not in " ".join(argv) for argv,env in calls))
                self.assertTrue(all(env.get("GIT_TERMINAL_PROMPT")=="0" for argv,env in calls))
                self.assertTrue(all("GIT_TRACE" not in env for argv,env in calls))
                self.assertTrue(all("GIT_CONFIG_COUNT" not in env for argv,env in calls))
                target.close()
                self.assertFalse(private.exists())
                self.assertNotIn("GITHUB_TOKEN",transport._environment)

    def test_protocol_restrictions_are_preserved_and_git_locale_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp)
            with patch.dict(os.environ,{"GIT_ALLOW_PROTOCOL":"file","GIT_PROTOCOL_FROM_USER":"0","LC_ALL":"unavailable_locale"}):
                with mirror(remote) as target:
                    env=target.git_transport._environment
                    self.assertEqual(env["GIT_ALLOW_PROTOCOL"],"file")
                    self.assertEqual(env["GIT_PROTOCOL_FROM_USER"],"0")
                    self.assertEqual(env["LC_ALL"],"C")
                    self.assertEqual(target.restore(Path(tmp)/"empty")["status"],"remote_missing")
            with patch.dict(os.environ,{"GIT_ALLOW_PROTOCOL":"https"}):
                with mirror(remote) as target:
                    with self.assertRaises(CampaignSyncError):target.restore(Path(tmp)/"blocked")

    def test_timeout_terminates_the_git_process_group_and_helpers(self):
        class HungGit:
            pid=987654321
            stdout=stderr=None
            def __init__(self):self.calls=0
            def communicate(self,**kwargs):
                self.calls+=1
                if self.calls==1:raise subprocess.TimeoutExpired("git",.01)
                return b"",b""
        with tempfile.TemporaryDirectory() as tmp:
            with mirror(remote_repository(tmp)) as target:
                process=HungGit()
                with patch("e2_github_git_transport.subprocess.Popen",return_value=process) as start:
                    with patch("e2_github_git_transport.os.killpg") as kill:
                        with self.assertRaisesRegex(CampaignSyncError,"git_request_unconfirmed"):
                            target.git_transport._run(["status"],timeout=.01)
                self.assertTrue(start.call_args.kwargs["start_new_session"])
                kill.assert_called_once_with(process.pid,signal.SIGKILL)
                self.assertEqual(process.calls,2)

    def test_periodic_cache_maintenance_follows_acknowledged_local_ref_and_has_no_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp);root=Path(tmp)/"state"
            with mirror(remote) as target:
                target.restore(root)
                with session(root) as state:
                    target.push(root,paths=state.pending_record_paths)
                    state.mark_records_synced(state.pending_record_paths)
                    state.extend_campaign_budget(400,reason="additional compute allocation")
                    target.git_transport._pushes=127
                    calls=[];run=target.git_transport._run
                    def observed(args,**kwargs):calls.append((list(args),kwargs));return run(args,**kwargs)
                    with patch.object(target.git_transport,"_run",side_effect=observed):
                        target.push(root,paths=state.pending_record_paths)
                    ref=next(i for i,(args,_) in enumerate(calls) if args[:2]==["update-ref","refs/e2-campaign/current"])
                    gc=next(i for i,(args,_) in enumerate(calls) if args==["-c","gc.autoDetach=false","gc","--auto"])
                    self.assertLess(ref,gc)
                    self.assertEqual(calls[gc][1]["environment"]["GITHUB_TOKEN"],"")
                    self.assertEqual(target.git_transport._run(["rev-parse","refs/e2-campaign/current"]).stdout.decode().strip(),target._head)
                    self.assertFalse(target.git_transport._maintenance_failed)

    def test_local_remote_requires_explicit_private_test_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp)
            with self.assertRaises(CampaignSyncError):
                GitCampaignMirror("jassan-bit/DeepLearning_202630","campaigns/test35","campaigns/test35",token="test",_test_remote=remote)

    def test_inherited_git_directory_namespace_and_helpers_cannot_modify_another_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp);outside=Path(tmp)/"outside.git"
            git("init","--bare","--object-format=sha1",outside)
            def snapshot():
                return {p.relative_to(outside).as_posix():p.read_bytes() for p in outside.rglob("*") if p.is_file()}
            before=snapshot()
            overrides={"GIT_COMMON_DIR":str(outside),"GIT_DIR":str(outside),"GIT_WORK_TREE":str(outside),
                "GIT_NAMESPACE":"unapproved","GIT_TEMPLATE_DIR":str(outside),"GIT_EXEC_PATH":"/unapproved-helper",
                "GIT_CONFIG":str(outside/"config"),"GIT_OBJECT_DIRECTORY":str(outside/"objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES":str(outside/"objects"),"GIT_INDEX_FILE":str(outside/"index")}
            with patch.dict(os.environ,overrides):
                with mirror(remote) as target:
                    actual=target.git_transport._run(["rev-parse","--git-common-dir"]).stdout.decode().strip()
                    self.assertEqual(Path(actual).resolve(),target.git_transport._cache.resolve())
                    self.assertFalse(any(key in target.git_transport._environment for key in overrides))
                    self.assertEqual(target.restore(Path(tmp)/"fresh")["status"],"remote_missing")
            self.assertEqual(snapshot(),before)

    def test_transport_refuses_foreign_hosts_and_main_or_force_ref_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            remote=remote_repository(tmp)
            with mirror(remote) as target:
                transport=target.git_transport
                for method,url,payload in (
                    ("GET","https://example.com/repos/other/repo/git/ref/heads/main",None),
                    ("PATCH","https://api.github.com/repos/jassan-bit/DeepLearning_202630/git/refs/heads/main",{"sha":"a"*40,"force":False}),
                    ("PATCH","https://api.github.com/repos/jassan-bit/DeepLearning_202630/git/refs/heads/campaigns/test35",{"sha":"a"*40,"force":True})):
                    with self.subTest(url=url),self.assertRaises(CampaignSyncError):transport(method,url,{},payload)


if __name__=="__main__":unittest.main()
