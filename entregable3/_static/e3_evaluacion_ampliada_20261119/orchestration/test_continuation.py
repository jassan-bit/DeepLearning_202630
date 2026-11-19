"""Meaningful byte/chain recovery checks; no network or scientific models."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote
import zipfile

HERE=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
C=load(HERE/'continuation.py','continuation_under_test')
A=load(C.ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','old_api_for_continuation_tests')
SOURCE='a'*40
MASTER='20261118T150000000000Z'
TOKEN='synthetic_token_for_offline_tests'
PROTOCOL=b'{"frozen":true}\n'
PROTO=hashlib.sha256(PROTOCOL).hexdigest()


def checkpoint(value):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('checkpoint/data.pkl',value)
    return stream.getvalue()


class FakeGitHub:
    REPOSITORY=C.REPOSITORY
    StorageError=A.StorageError
    StorageConflict=A.StorageConflict
    _token=staticmethod(A._token)
    _canonical=staticmethod(A._canonical)
    _git_sha=staticmethod(A._git_sha)
    _safe_relative=staticmethod(A._safe_relative)
    _regular_path=staticmethod(A._regular_path)
    def __init__(self):
        self.blobs={};self.trees={};self.commits={};self.refs={};self.calls=[]
        self.corrupt_blob=None;self.fail_ref_reads=0;self.arm_after_ref=False
        self.mutate_tree=False
        original=self.blob(b'original source bytes\n')
        root=self.make_tree({'README.md':{'sha':original,'mode':'100644'},
                             'src/frozen.py':{'sha':original,'mode':'100644'}})
        self.commits[SOURCE]={'sha':SOURCE,'tree':{'sha':root},'parents':[]}
        self.source_tree=root
    def validate_credentials(self,token):
        assert token==TOKEN
        return {'authenticated':True}
    def blob(self,raw):
        sha=self._git_sha(raw);self.blobs[sha]=raw;return sha
    def make_tree(self,flat):
        nested={}
        for path,row in flat.items():
            cur=nested;parts=path.split('/')
            for component in parts[:-1]:cur=cur.setdefault(component,{})
            cur[parts[-1]]=dict(row,type='blob')
        def make(node):
            rows=[]
            for name,value in sorted(node.items()):
                if value.get('type')=='blob':row={'path':name,**value}
                else:row={'path':name,'type':'tree','mode':'040000','sha':make(value)}
                rows.append(row)
            sha=hashlib.sha1(json.dumps(rows,sort_keys=True).encode()).hexdigest()
            self.trees[sha]=rows;return sha
        return make(nested)
    def flat(self,sha):
        result={}
        def visit(tree,prefix=''):
            for row in self.trees[tree]:
                name=prefix+row['path']
                if row['type']=='tree':visit(row['sha'],name+'/')
                else:result[name]={'sha':row['sha'],'mode':row['mode']}
        visit(sha);return result
    def raw_blob(self,sha,size,token):
        return b'corrupt' if sha==self.corrupt_blob else self.blobs[sha]
    def _request(self,method,path,token,payload=None,allow_missing=False):
        self.calls.append((method,path,copy.deepcopy(payload)))
        prefix='/repos/'+self.REPOSITORY
        if not path.startswith(prefix):raise AssertionError(path)
        path=path[len(prefix):]
        if method=='GET' and path.startswith('/git/commits/'):
            sha=path.rsplit('/',1)[1]
            if sha not in self.commits:raise self.StorageError('missing commit')
            return copy.deepcopy(self.commits[sha])
        if method=='GET' and path.startswith('/git/trees/'):
            sha=path.split('/git/trees/',1)[1].split('?')[0]
            if '?recursive=1' in path:
                rows=[]
                def visit(current,prefix=''):
                    for row in self.trees[current]:
                        rows.append({**row,'path':prefix+row['path']})
                        if row['type']=='tree':visit(row['sha'],prefix+row['path']+'/')
                visit(sha)
            else:rows=self.trees[sha]
            return {'sha':sha,'truncated':False,'tree':copy.deepcopy(rows)}
        if method=='GET' and path.startswith('/git/matching-refs/heads/'):
            return [copy.deepcopy(r) for r in reversed(list(self.refs.values()))]
        if method=='GET' and path.startswith('/git/ref/heads/'):
            if self.fail_ref_reads:
                self.fail_ref_reads-=1;raise self.StorageError('synthetic connection lost')
            key='refs/heads/'+path.split('/git/ref/heads/',1)[1]
            if key not in self.refs:
                if allow_missing:return None
                raise self.StorageError('missing ref')
            return copy.deepcopy(self.refs[key])
        if method=='GET' and path.startswith('/contents/'):
            name,sha=path[len('/contents/'):].rsplit('?ref=',1);name=unquote(name)
            commit=self.commits[sha];flat=self.flat(commit['tree']['sha'])
            if name not in flat:raise self.StorageError('missing contents')
            blob=flat[name]['sha'];raw=self.blobs[blob]
            return {'type':'file','path':name,'sha':blob,'size':len(raw),
                    'encoding':'base64','content':base64.b64encode(raw).decode()}
        if method=='POST' and path=='/git/blobs':
            return {'sha':self.blob(base64.b64decode(payload['content']))}
        if method=='POST' and path=='/git/trees':
            flat=self.flat(payload['base_tree'])
            for row in payload['tree']:
                flat[row['path']]={'sha':row['sha'],'mode':row['mode']}
            if self.mutate_tree:flat.pop('src/frozen.py')
            return {'sha':self.make_tree(flat)}
        if method=='POST' and path=='/git/commits':
            sha=hashlib.sha1(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            self.commits[sha]={'sha':sha,'tree':{'sha':payload['tree']},
                               'parents':[{'sha':p} for p in payload['parents']]}
            return {'sha':sha}
        if method=='POST' and path=='/git/refs':
            if payload['ref'] in self.refs:raise self.StorageConflict('ref exists')
            self.refs[payload['ref']]={'ref':payload['ref'],'object':{'type':'commit','sha':payload['sha']}}
            if self.arm_after_ref:self.fail_ref_reads=1;self.arm_after_ref=False
            return copy.deepcopy(self.refs[payload['ref']])
        raise AssertionError((method,path))
    def fork(self,head):
        original=C._read_part(self,TOKEN,head['head_commit'],head['head_part'])
        part=C._stamp(original['part_id']);prefix=C.PREFIX+part
        old=original['receipt'];receipt={**old,'part_id':part}
        parent=old['previous_part']['commit_sha'] if old['previous_part'] else SOURCE
        parent_tree=self.commits[parent]['tree']['sha']
        flat=self.flat(parent_tree)
        members={'stage_receipt.json':self._canonical(receipt)}
        oldfiles=self.flat(original['tree_sha'])
        for name in receipt['files']:
            members['payload/'+name]=self.blobs[oldfiles[original['prefix']+'/payload/'+name]['sha']]
        manifest={'kind':'e3_closure_storage_manifest','source_commit':SOURCE,'run_id':part,'profile':'audit',
                  'files':{name:C._meta(raw,self) for name,raw in members.items()}}
        members['storage_manifest.json']=self._canonical(manifest)
        for name,raw in members.items():flat[prefix+'/'+name]={'sha':self.blob(raw),'mode':'100644'}
        tree=self.make_tree(flat);sha=hashlib.sha1((part+'fork').encode()).hexdigest()
        self.commits[sha]={'sha':sha,'tree':{'sha':tree},'parents':[{'sha':parent}]}
        self.refs['refs/heads/'+C.BRANCH+part]={'ref':'refs/heads/'+C.BRANCH+part,'object':{'type':'commit','sha':sha}}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.output=self.root/'output';self.output.mkdir();self.api=FakeGitHub()
        (self.output/'protocol.json').write_bytes(PROTOCOL)
    def tearDown(self):self.temp.cleanup()
    def job(self,name,value):
        folder=self.output/'jobs'/name
        (folder/'supervised').mkdir(parents=True,exist_ok=True)
        (folder/'fit_request.json').write_text('{"seed":1}')
        marker=folder/'supervised/phase_started.json'
        if not marker.exists():marker.write_text('{"started":true}')
        (folder/'supervised/checkpoint.pt').write_bytes(checkpoint(value))
    def save(self):return C.save_changed(self.output,TOKEN,SOURCE,MASTER,PROTO,self.api)
    def test_new_session_restores_three_partial_jobs_and_appends_without_retraining(self):
        self.job('a',b'a epoch one');first=self.save()
        self.job('a',b'a epoch two');self.job('b',b'b epoch one');second=self.save()
        self.job('c',b'c epoch one')
        (self.output/'controller_status.json').write_text('{"stage_status":"completed","analysis_complete":true}')
        third=self.save();self.assertEqual(third['saved_parts'][0]['sequence'],3)
        runs=C.discover_runs(TOKEN,self.api,SOURCE)
        self.assertEqual(len(runs),1);self.assertEqual(runs[0]['parts'],3)
        self.assertTrue(runs[0]['analysis_complete']);self.assertEqual(runs[0]['stage_status'],'completed')
        fresh=self.root/'fresh'
        result=C.restore(fresh,TOKEN,SOURCE,PROTO,self.api,run_url=first['saved_parts'][0]['run_url'])
        self.assertEqual(result['head_commit'],third['head_commit'])
        for p in self.output.rglob('*'):
            if p.is_file():self.assertEqual(p.read_bytes(),(fresh/p.relative_to(self.output)).read_bytes())
        (fresh/'jobs/c/supervised/checkpoint.pt').write_bytes(checkpoint(b'c epoch two'))
        saved=C.save_changed(fresh,TOKEN,SOURCE,MASTER,PROTO,self.api)
        self.assertEqual(saved['saved_parts'][0]['new_real_data_fits'],0)
        self.assertEqual(saved['saved_parts'][0]['sequence'],4)
        latest=self.api.commits[saved['head_commit']]
        self.assertEqual(latest['parents'],[{'sha':third['head_commit']}])
        self.assertEqual(self.api.flat(latest['tree']['sha'])['src/frozen.py'],
                         self.api.flat(self.api.source_tree)['src/frozen.py'])
        recursive=[p for m,p,v in self.api.calls if m=='GET' and '/git/trees/' in p and '?recursive=1' in p]
        self.assertNotIn('/repos/'+C.REPOSITORY+'/git/trees/'+self.api.source_tree+'?recursive=1',recursive)
    def test_unchanged_retry_and_lost_ack_do_not_duplicate_fit_or_parts(self):
        self.job('a',b'first');self.api.arm_after_ref=True
        with self.assertRaises(A.StorageError):self.save()
        self.assertEqual(len(self.api.refs),1)
        self.job('a',b'second')
        result=self.save()
        self.assertEqual(len(result['saved_parts']),2)
        self.assertEqual([p['new_real_data_fits'] for p in result['saved_parts']],[1,0])
        refs=len(self.api.refs);again=self.save()
        self.assertEqual(again['saved_parts'],[]);self.assertEqual(len(self.api.refs),refs)
    def test_restored_workdir_can_persist_through_a_stable_view_with_shared_state(self):
        self.job('a',b'first completed checkpoint')
        shared_state=self.root/'controller'/'run-state.json'
        first=C.save_changed(self.output,TOKEN,SOURCE,MASTER,PROTO,self.api,statefile=shared_state)
        working=self.root/'restored-worker'
        C.restore(working,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER,statefile=shared_state)
        view=self.root/'stable-persistence-view'
        shutil.copytree(working,view)
        unchanged=C.save_changed(view,TOKEN,SOURCE,MASTER,PROTO,self.api,statefile=shared_state)
        self.assertEqual(unchanged['saved_parts'],[])
        self.assertEqual(unchanged['head_commit'],first['head_commit'])
        path=working/'jobs/a/supervised/checkpoint.pt'
        path.write_bytes(checkpoint(b'later completed checkpoint'))
        marker=working/'jobs/b/supervised/phase_started.json'
        marker.parent.mkdir(parents=True)
        marker.write_bytes(b'{"started":true}')
        (working/'jobs/b/fit_request.json').write_bytes(b'{"seed":2}')
        status=working/'controller_status.json'
        status.write_bytes(b'{"stage_status":"running","analysis_complete":false}')
        shutil.copytree(working,view,dirs_exist_ok=True)
        second=C.save_changed(view,TOKEN,SOURCE,MASTER,PROTO,self.api,statefile=shared_state)
        self.assertEqual(second['saved_parts'][0]['new_real_data_fits'],1)
        self.assertEqual(second['saved_parts'][0]['sequence'],2)
        again=self.root/'another-new-session'
        C.restore(again,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertEqual((again/'jobs/a/supervised/checkpoint.pt').read_bytes(),path.read_bytes())
        self.assertEqual((again/'jobs/b/supervised/phase_started.json').read_bytes(),marker.read_bytes())
        self.assertEqual(C.discover_runs(TOKEN,self.api)[0]['stage_status'],'running')
    def test_svg_figures_roundtrip_without_mutating_the_historical_validator(self):
        historical=load(C.ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py',
                        'unchanged_historical_store_for_svg_test')
        self.assertNotIn('.svg',historical.ALLOWED)
        folder=self.output/'analysis/figures';folder.mkdir(parents=True)
        figure=folder/'paired_gain.svg'
        figure.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 1L2 3"/></svg>\n')
        self.save()
        destination=self.root/'with-figure'
        C.restore(destination,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertEqual((destination/'analysis/figures/paired_gain.svg').read_bytes(),figure.read_bytes())
        self.assertNotIn('.svg',historical.ALLOWED)
    def test_local_user_edit_conflict_is_not_overwritten(self):
        self.job('a',b'first');one=self.save();fresh=self.root/'fresh'
        C.restore(fresh,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.job('a',b'second');self.save()
        path=fresh/'jobs/a/supervised/checkpoint.pt';path.write_bytes(b'my local edit')
        with self.assertRaisesRegex(A.StorageError,'local_edits_conflict'):
            C.restore(fresh,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertEqual(path.read_bytes(),b'my local edit')
    def test_pristine_older_local_state_advances_and_commit_url_selects_chain(self):
        self.job('a',b'first');one=self.save();fresh=self.root/'fresh'
        url='https://github.com/'+C.REPOSITORY+'/tree/'+one['head_commit']+'/'+C.PREFIX+one['head_part']
        C.restore(fresh,TOKEN,SOURCE,PROTO,self.api,run_url=url)
        self.job('a',b'second');two=self.save()
        result=C.restore(fresh,TOKEN,SOURCE,PROTO,self.api,run_url=url)
        self.assertEqual(result['head_commit'],two['head_commit'])
        self.assertEqual((fresh/'jobs/a/supervised/checkpoint.pt').read_bytes(),
                         (self.output/'jobs/a/supervised/checkpoint.pt').read_bytes())
    def test_deleted_ancestor_ref_does_not_drop_reachable_checkpoint_history(self):
        self.job('a',b'first');one=self.save();self.job('b',b'second');two=self.save()
        del self.api.refs['refs/heads/'+C.BRANCH+one['head_part']]
        result=C.restore(self.root/'restore',TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertEqual(result['parts'],2)
        self.assertTrue((self.root/'restore/jobs/a/supervised/checkpoint.pt').is_file())
    def test_remote_symlink_payload_is_rejected_before_restore(self):
        self.job('a',b'first');one=self.save()
        commit=self.api.commits[one['head_commit']];flat=self.api.flat(commit['tree']['sha'])
        path=C.PREFIX+one['head_part']+'/payload/jobs/a/supervised/checkpoint.pt'
        flat[path]['mode']='120000';commit['tree']['sha']=self.api.make_tree(flat)
        with self.assertRaisesRegex(A.StorageError,'remote_symlink_or_mode'):
            C.restore(self.root/'unsafe',TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertFalse((self.root/'unsafe').exists())
    def test_changed_source_protocol_and_foreign_url_cannot_select_a_run(self):
        self.job('a',b'first');self.save()
        for kwargs in [{'source':'b'*40},{'protocol_sha256':'b'*64},
                       {'run_url':'https://github.com/other/repo/tree/x'}]:
            values={'output':self.root/'mismatch','token':TOKEN,'source':SOURCE,
                    'protocol_sha256':PROTO,'api':self.api,'master_run_id':MASTER,**kwargs}
            with self.assertRaises(A.StorageError):C.restore(**values)
        self.assertFalse((self.root/'mismatch').exists())
    def test_lost_local_state_cannot_reuse_master_with_different_protocol(self):
        self.job('a',b'first');one=self.save()
        Path(one['statefile']).unlink()
        with self.assertRaisesRegex(A.StorageError,'state_identity_mismatch'):
            C.save_changed(self.output,TOKEN,SOURCE,MASTER,'b'*64,self.api)
        self.assertEqual(len(self.api.refs),1)
    def test_corrupt_blob_or_missing_chain_leaves_destination_unwritten(self):
        self.job('a',b'first');one=self.save();self.job('a',b'second');two=self.save()
        part=C._read_part(self.api,TOKEN,two['head_commit'],two['head_part'])
        self.api.corrupt_blob=part['receipt']['files']['jobs/a/supervised/checkpoint.pt']['git_blob_sha']
        with self.assertRaisesRegex(A.StorageError,'blob_integrity'):
            C.restore(self.root/'bad',TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertFalse((self.root/'bad').exists())
        self.api.corrupt_blob=None;del self.api.commits[one['head_commit']]
        with self.assertRaises(A.StorageError):
            C.restore(self.root/'missing',TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertFalse((self.root/'missing').exists())
    def test_ambiguous_fork_is_not_silently_selected(self):
        self.job('a',b'first');self.save();self.job('a',b'second');head=self.save()
        self.api.fork(head)
        with self.assertRaisesRegex(A.StorageError,'ambiguous_chain_heads'):
            C.discover_runs(TOKEN,self.api)
    def test_source_tree_mutation_is_rejected_before_ref_publication(self):
        self.job('a',b'first');self.api.mutate_tree=True
        with self.assertRaisesRegex(A.StorageError,'source_tree_preservation'):
            self.save()
        self.assertEqual(self.api.refs,{})
    def test_secrets_symlinks_and_removed_files_fail_before_new_part(self):
        (self.output/'secret.json').write_text(TOKEN)
        with self.assertRaisesRegex(A.StorageError,'secret'):self.save()
        (self.output/'secret.json').unlink();(self.output/'link.json').symlink_to(self.output/'protocol.json')
        with self.assertRaisesRegex(A.StorageError,'symlink'):self.save()
        (self.output/'link.json').unlink();self.save();(self.output/'protocol.json').unlink()
        with self.assertRaisesRegex(A.StorageError,'file_removed'):self.save()
    def test_shards_are_bounded_and_attempts_count_once_per_job(self):
        self.job('a',b'largeish first');self.job('b',b'largeish second')
        with patch.object(C,'MAX_PAYLOAD',210),patch.object(C,'MAX_FILES',2):result=self.save()
        self.assertGreater(len(result['saved_parts']),1)
        self.assertEqual(sum(p['new_real_data_fits'] for p in result['saved_parts']),2)
        for ref in self.api.refs.values():
            part_id=ref['ref'].split(C.BRANCH,1)[1]
            part=C._read_part(self.api,TOKEN,ref['object']['sha'],part_id)
            self.assertLessEqual(len(part['receipt']['files']),2)
        restored=self.root/'split-restored'
        C.restore(restored,TOKEN,SOURCE,PROTO,self.api,master_run_id=MASTER)
        self.assertEqual((restored/'jobs/a/supervised/checkpoint.pt').read_bytes(),
                         (self.output/'jobs/a/supervised/checkpoint.pt').read_bytes())


if __name__=='__main__':unittest.main()
