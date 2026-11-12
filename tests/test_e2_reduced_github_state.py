"""Synthetic control-CAS tests; no credentials, network or model training."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.append('/workspace/work/e2architecture/scripts')
from e2_reduced_github_backup import BackupPause,canonical,git_blob_sha
from e2_reduced_github_state import GitHubState,POINTER,_RemoteBudget


class Client:
    max_conflicts=3
    def __init__(self):
        self.head='a'*40;self.blobs={};self.trees={'t':{}};self.commits={self.head:{'tree':{'sha':'t'},'parents':[]}}
        self.number=0;self.publications=[]
    def _branch_head(self):return self.head
    def _content(self,path,head):
        tree=self.trees[self.commits[head]['tree']['sha']]
        if path not in tree:return [] if any(k.startswith(path+'/') for k in tree) else None
        sha=tree[path];return {'sha':sha,'content':base64.b64encode(self.blobs[sha]).decode(),'encoding':'base64'}
    def _request(self,method,path,payload=None):
        if method=='GET' and path.startswith('/git/commits/'):return self.commits[path.rsplit('/',1)[1]]
        if method=='POST' and path=='/git/blobs':
            raw=base64.b64decode(payload['content']);sha=git_blob_sha(raw);self.blobs[sha]=raw;return {'sha':sha}
        if method=='POST' and path=='/git/trees':
            tree=dict(self.trees[payload['base_tree']]);tree.update({x['path']:x['sha'] for x in payload['tree']})
            sha=hashlib.sha1(canonical(tree)).hexdigest();self.trees[sha]=tree;return {'sha':sha}
        if method=='POST' and path=='/git/commits':
            self.number+=1;sha=f'{self.number:040x}';self.commits[sha]={'tree':{'sha':payload['tree']},'parents':payload['parents']};return {'sha':sha}
        if method=='PATCH':
            if self.commits[payload['sha']]['parents']!=[self.head]:raise BackupPause('conflict',status=422)
            self.head=payload['sha'];self.publications.append(self.head);return {'sha':self.head}
        raise AssertionError((method,path))


def row(kind,**extra):
    value={'sequence':0,'previous_sha256':'0'*64,'kind':kind,**extra}
    value['sha256']=hashlib.sha256(canonical(value)).hexdigest();return canonical(value)+b'\n'


class Budget:
    def __init__(self,root):self.root=Path(root);self.root.mkdir(parents=True)
    def reserve(self,identity,amount,metadata=None):
        (self.root/'pilot_budget.jsonl').write_bytes(row('reserved',attempt_id=identity,reservation_seconds=amount))
    def close(self):pass


class StateTests(unittest.TestCase):
    def test_remote_lease_exists_before_worker_can_start_and_blocks_fresh_vm(self):
        client=Client()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'campaign';state=GitHubState(root,client);state.restore()
            facade=_RemoteBudget(Budget(root/'shared_budget'),state)
            facade.reserve('attempt',660)
            self.assertIn(POINTER,client.trees[client.commits[client.head]['tree']['sha']])
            # Simulate loss of disposable content while retaining the same path.
            import shutil;shutil.rmtree(root)
            with self.assertRaisesRegex(BackupPause,'unclosed_remote_lease'):
                GitHubState(root,client).restore()

    def test_control_compare_and_swap_rejects_other_session_without_merging_lease(self):
        client=Client()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'campaign';a=GitHubState(root,client);b=GitHubState(root,client)
            a.restore();b.restore();a.snapshot('owner-A')
            previous=client.head
            with self.assertRaisesRegex(BackupPause,'another_session'):
                b.snapshot('owner-B')
            self.assertEqual(client.head,previous)

    def test_metadata_roundtrip_excludes_private_assets_and_refuses_local_changes(self):
        client=Client()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'campaign';state=GitHubState(root,client);state.restore()
            (root/'config.json').write_text('{"seed":1}')
            (root/'private-known-data').mkdir();(root/'private-known-data/raw.csv').write_text('open,high,low,close,volume\n1,2,1,1,1')
            state.snapshot('metadata')
            (root/'config.json').write_text('local-newer')
            with self.assertRaisesRegex(BackupPause,'local_control_diverges'):
                GitHubState(root,client).restore()
            self.assertEqual((root/'config.json').read_text(),'local-newer')
            import shutil;shutil.rmtree(root)
            with patch('e2_reduced_github_restore.restore_results',return_value={'files_restored':0}):
                report=GitHubState(root,client).restore()
            self.assertTrue(report['restored']);self.assertEqual((root/'config.json').read_text(),'{"seed":1}')
            self.assertFalse((root/'private-known-data').exists())

    def test_old_scientific_runs_without_ledger_cannot_obtain_fresh_budget(self):
        client=Client();raw=b'{}';sha=git_blob_sha(raw);client.blobs[sha]=raw;client.trees['t']['runs/old/manifest.json']=sha
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(BackupPause,'existing_results_without_control'):
                GitHubState(root,client).restore()

    def test_final_supervisor_distinct_result_and_lease_ids_close_by_hash_binding(self):
        client=Client()
        with tempfile.TemporaryDirectory() as root:
            state=GitHubState(root,client);state.restore()
            result={'status':'completed','worker_spec_sha256':'spec','attempt_id':'result-id-different-from-lease'}
            raw=canonical(result)+b'\n';Path(root,'result.json').write_bytes(raw)
            state._pending['lease-id']={'worker_spec_sha256':'spec','result_sha256':hashlib.sha256(raw).hexdigest()}
            order=[]
            def backed(directory,receipt):
                order.append('artifacts_uploaded')
                return {'manifest_path':'runs/result/manifest.json','manifest_sha256':'a'*64}
            client.persist_result=backed
            state.persist_result(root,{'result':result,'budget_receipt':{'attempt_id':'lease-id'}})
            self.assertFalse(state._pending)
            self.assertEqual(order,['artifacts_uploaded'])
            state.snapshot('next_admission_allowed')

    def test_completed_unbacked_lease_cannot_be_published_as_released(self):
        client=Client()
        with tempfile.TemporaryDirectory() as root:
            state=GitHubState(root,client);state.restore();state._pending['a']={'worker_spec_sha256':'x','result_sha256':'y'}
            with self.assertRaisesRegex(BackupPause,'requires_backup'):
                state.snapshot('unsafe-close')
            with self.assertRaisesRegex(BackupPause,'pending_cost_binding'):
                state.persist_result(root,{'status':'completed','attempt_id':'wrong'})


if __name__=='__main__':unittest.main()
