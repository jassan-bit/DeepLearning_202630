"""Synthetic reconciliation tests; no market data, credentials or optimizer."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
# Local task checkout; a normal repository checkout already has these modules.
if (ROOT.parent/'work/e2architecture/scripts').is_dir():
    sys.path.append(str(ROOT.parent/'work/e2architecture/scripts'))
from e2_reduced_github_backup import BackupPause,canonical,git_blob_sha
from e2_reduced_github_state import GitHubState
from reconcile_e2_github_completed import reconcile_completed_local


def sha(value):return hashlib.sha256(value).hexdigest()


class Client:
    max_conflicts=3
    def __init__(self):
        self.head='a'*40;self.blobs={};self.trees={'t':{}};self.commits={self.head:{'tree':{'sha':'t'},'parents':[]}}
        self.number=0;self.backups=0;self.fail=False
    def _branch_head(self):return self.head
    def _content(self,path,head):
        tree=self.trees[self.commits[head]['tree']['sha']]
        if path not in tree:return None
        key=tree[path];return {'sha':key,'content':base64.b64encode(self.blobs[key]).decode(),'encoding':'base64'}
    def _request(self,method,path,payload=None):
        if method=='GET'and path.startswith('/git/commits/'):return self.commits[path.rsplit('/',1)[1]]
        if method=='POST'and path=='/git/blobs':
            raw=base64.b64decode(payload['content']);key=git_blob_sha(raw);self.blobs[key]=raw;return {'sha':key}
        if method=='POST'and path=='/git/trees':
            tree=dict(self.trees[payload['base_tree']]);tree.update({x['path']:x['sha'] for x in payload['tree']})
            key=hashlib.sha1(canonical(tree)).hexdigest();self.trees[key]=tree;return {'sha':key}
        if method=='POST'and path=='/git/commits':
            self.number+=1;key=f'{self.number:040x}';self.commits[key]={'tree':{'sha':payload['tree']},'parents':payload['parents']};return {'sha':key}
        if method=='PATCH':
            if self.commits[payload['sha']]['parents']!=[self.head]:raise BackupPause('conflict',status=422)
            self.head=payload['sha'];return {'sha':self.head}
        raise AssertionError((method,path))
    def persist_result(self,directory,receipt):
        self.backups+=1
        if self.fail:raise BackupPause('transport_failed')
        # Publication of the closed ledger must still be pending at upload.
        self.last_before_backup=GitHubState(Path(directory).parents[1],self)._current(self.head)['reason']
        for relative,expected in receipt['artifacts_sha256'].items():
            assert sha((Path(directory)/relative).read_bytes())==expected
        return {'status':'backed_up','manifest_path':'runs/attempt/manifest.json','manifest_sha256':'f'*64}


def append(rows,kind,**extra):
    value={'version':'e2-amended-pilot-budget-v1','sequence':len(rows),
        'previous_sha256':rows[-1]['sha256']if rows else'0'*64,'reconciliation_sha256':'c'*64,
        'kind':kind,**extra}
    value['sha256']=sha(canonical(value));rows.append(value)


def setup(root,client,*,close=True):
    root=Path(root);directory=root/'jobs/attempt';directory.mkdir(parents=True)
    (root/'shared_budget').mkdir();(root/'config.json').write_text('{"unchanged":true}')
    spec={'output':str(directory),'fixture':'synthetic'};spec_sha=sha(canonical(spec))
    (directory/'worker_spec.json').write_bytes(canonical(spec)+b'\n')
    rows=[];append(rows,'reserved',attempt_id='attempt',reservation_seconds=660,
        metadata={'worker_spec_sha256':spec_sha})
    ledger=root/'shared_budget/pilot_budget.jsonl';ledger.write_bytes(b''.join(canonical(x)+b'\n'for x in rows))
    state=GitHubState(root,client);state._restored=True;state.snapshot('lease_reserved_before_worker')
    old_head=client.head
    if close:
        (directory/'weights.pt').write_bytes(b'synthetic-weights-not-a-model')
        result={'status':'completed','attempt_id':'attempt','worker_spec_sha256':spec_sha,
            'optimizer_steps':7,'artifacts_sha256':{'weights.pt':sha((directory/'weights.pt').read_bytes())}}
        (directory/'result.json').write_bytes(canonical(result)+b'\n')
        append(rows,'charged',attempt_id='attempt',reservation_seconds=660,elapsed_seconds=12.5,
            status='completed',reservation_overrun=False,
            resource_use={'worker_spec_sha256':spec_sha,'result_sha256':sha((directory/'result.json').read_bytes()),
                'optimizer_steps':7,'automatic_retry':False})
        append(rows,'overhead',elapsed_seconds=3.)
        ledger.write_bytes(b''.join(canonical(x)+b'\n'for x in rows))
    return directory,ledger,old_head


class ReconciliationTests(unittest.TestCase):
    def test_existing_completion_is_uploaded_before_lease_close_without_changing_cost(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)/'output';client=Client();directory,ledger,old=setup(root,client)
            prior=ledger.read_bytes()
            report=reconcile_completed_local(root,client,proc_root=Path(temp)/'emptyproc')
            self.assertTrue(report['reconciled']);self.assertEqual(report['existing_elapsed_seconds'],12.5)
            self.assertEqual(ledger.read_bytes(),prior);self.assertEqual(client.backups,1)
            self.assertEqual(client.last_before_backup,'lease_reserved_before_worker')
            current=GitHubState(root,client)._current(client.head)
            self.assertEqual(current['reason'],'completed_result_uploaded')
            self.assertIn('jobs/attempt',current['result_backups']);self.assertNotEqual(client.head,old)

    def test_unknown_or_divergent_local_ledger_cannot_release_or_create_credit(self):
        for close in(False,True):
            with self.subTest(close=close),tempfile.TemporaryDirectory()as temp:
                root=Path(temp)/'output';client=Client();directory,ledger,old=setup(root,client,close=close)
                if close:ledger.write_bytes(b'x'+ledger.read_bytes())
                with self.assertRaises(BackupPause):reconcile_completed_local(root,client,proc_root=Path(temp)/'emptyproc')
                self.assertEqual(client.head,old);self.assertEqual(client.backups,0)

    def test_changed_weight_or_old_metadata_blocks_before_backup(self):
        for target in('weights.pt','config.json'):
            with self.subTest(target=target),tempfile.TemporaryDirectory()as temp:
                root=Path(temp)/'output';client=Client();directory,ledger,old=setup(root,client)
                path=directory/target if target=='weights.pt'else root/target;path.write_bytes(b'changed')
                with self.assertRaises((BackupPause,ValueError)):reconcile_completed_local(root,client,proc_root=Path(temp)/'emptyproc')
                self.assertEqual(client.head,old);self.assertEqual(client.backups,0)

    def test_failed_upload_preserves_remote_lease_and_existing_closed_local_charge(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)/'output';client=Client();directory,ledger,old=setup(root,client)
            prior=ledger.read_bytes();client.fail=True
            with self.assertRaises(BackupPause):reconcile_completed_local(root,client,proc_root=Path(temp)/'emptyproc')
            self.assertEqual(ledger.read_bytes(),prior);self.assertEqual(client.head,old)
            self.assertEqual(GitHubState(root,client)._current(old)['reason'],'lease_reserved_before_worker')
            client.fail=False
            self.assertTrue(reconcile_completed_local(root,client,proc_root=Path(temp)/'emptyproc')['reconciled'])

    def test_existing_controller_blocks_reconciliation(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp)/'output';client=Client();directory,ledger,old=setup(root,client)
            proc=Path(temp)/'proc/999999';proc.mkdir(parents=True)
            (proc/'cmdline').write_bytes(b'python\0run_e2_reduced_selection.py\0')
            with self.assertRaises(BackupPause):reconcile_completed_local(root,client,proc_root=proc.parent)
            self.assertEqual(client.head,old);self.assertEqual(client.backups,0)


if __name__=='__main__':unittest.main()
