"""Real local Git remote integration with the unchanged frozen publisher.

No HTTP, external Git service, CUDA work or scientific fitting is performed.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'


def scientific_checkout():
    configured = os.environ.get('E3_SCIENTIFIC_CHECKOUT')
    candidates = ([Path(configured)] if configured else
                  [*HERE.parents, Path('/content/e3-extended-source-' + SOURCE), Path('/workspace/e3_review9')])
    for folder in candidates:
        if (folder/'experiments/e3_evaluacion_ampliada_20261119/orchestration/continuation.py').is_file():
            return folder
    raise RuntimeError('Set E3_SCIENTIFIC_CHECKOUT to the pinned scientific checkout')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCIENCE = scientific_checkout()
C = load(SCIENCE/'experiments/e3_evaluacion_ampliada_20261119/orchestration/continuation.py', 'pack_integration_continuation')
A = load(SCIENCE/'experiments/e3_remediacion_20261118/evaluation/colab_github.py', 'pack_integration_legacy')
from test_git_stage_transport import Fixture, git, TOKEN


class LegacyProxy:
    REPOSITORY = C.REPOSITORY
    StorageError = A.StorageError
    StorageConflict = A.StorageConflict
    _token = staticmethod(A._token)
    _canonical = staticmethod(A._canonical)
    _git_sha = staticmethod(A._git_sha)
    _safe_relative = staticmethod(A._safe_relative)
    _regular_path = staticmethod(A._regular_path)

    def __init__(self, fixture=None):
        self.calls = []
        self._opener = SimpleNamespace(open=lambda *args, **kwargs:None)
        self.fixture = fixture

    def _inputs(self, method, path, token, payload):
        assert token == TOKEN
        if payload is not None:
            assert token.encode() not in self._canonical(payload)
        return token, None if payload is None else self._canonical(payload)

    def validate_credentials(self, token):
        assert token == TOKEN
        return {'authenticated':True}

    def raw_blob(self, sha, size, token):
        raise AssertionError('Unexpected REST blob request in real-Git integration')

    def _request(self, method, path, token, payload=None, **kwargs):
        self.calls.append((method, path, payload, kwargs))
        root='/repos/'+self.REPOSITORY
        if method=='GET' and self.fixture is not None and path.startswith(root):
            route=path[len(root):]
            if route.startswith('/git/commits/'):
                sha=route.rsplit('/',1)[1]
                lines=git(self.fixture.remote,'cat-file','commit',sha).split('\n\n',1)[0].splitlines()
                return {'sha':sha,'tree':{'sha':next(v[5:] for v in lines if v.startswith('tree '))},
                        'parents':[{'sha':v[7:]} for v in lines if v.startswith('parent ')]}
            if route.startswith('/git/trees/'):
                sha=route.split('/git/trees/',1)[1].split('?')[0]
                recursive='?recursive=1' in route
                args=['ls-tree','-z']+(['-r','-t'] if recursive else [])+[sha]
                raw=git(self.fixture.remote,*args)
                rows=[]
                for value in raw.split('\0'):
                    if not value:continue
                    header,name=value.split('\t',1);mode,kind,obj=header.split(' ')
                    row={'path':name,'mode':mode,'type':kind,'sha':obj,
                         'url':'https://api.github.com/old-rest-object/'+obj}
                    if kind=='blob':row['size']=int(git(self.fixture.remote,'cat-file','-s',obj))
                    rows.append(row)
                return {'sha':sha,'truncated':False,'tree':rows}
            if route.startswith('/git/ref/heads/'):
                ref='refs/heads/'+route.split('/git/ref/heads/',1)[1]
                result=git(self.fixture.remote,'show-ref','--verify','--hash',ref,check=False)
                if result.returncode:
                    if kwargs.get('allow_missing'):return None
                    raise self.StorageError('missing simulated REST ref')
                return {'ref':ref,'object':{'type':'commit','sha':result.stdout.decode().strip()}}
            if route.startswith('/contents/'):
                name,commit=route[len('/contents/'):].rsplit('?ref=',1);name=unquote(name)
                response=git(self.fixture.remote,'show',commit+':'+name,check=False)
                if response.returncode:raise self.StorageError('missing simulated REST contents')
                raw=response.stdout
                # The engine adapter should serve all newly published contents;
                # this fallback exists only for already-readable old metadata.
                return {'type':'file','path':name,'sha':self._git_sha(raw),'size':len(raw),
                        'encoding':'base64','content':base64.b64encode(raw).decode()}
        raise AssertionError('Unexpected REST request in local real-Git integration: ' + method + ' ' + path)


def pending_fixture(module, base, git_fixture, *, sequence=63, part='20261119T170007583476Z', parent=None):
    """A real byte-preserved pending snapshot extending a real Git parent."""
    parent = parent or git_fixture.parent
    protocol_raw = b'{"local_offline_protocol":true}\n'
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    output = base/('e3-extended-audit-' + module.MASTER)
    output.mkdir(parents=True, exist_ok=True)
    (output/'protocol.json').write_bytes(protocol_raw)
    job = 'jobs/D03/MLP16__final__20261118'
    payload = {job+'/fit_request.json':b'{"seed":20261118}\n',
               job+'/supervised/checkpoint.pt':b'unchanged real Git checkpoint\x00\xff'*128,
               job+'/supervised/phase_started.json':b'{"deadline":123456.0}\n',
               job+'/result.json':b'{"status":"completed"}\n',
               'resources.csv':b'job,seconds\nMLP16,120\n'}
    for name, raw in payload.items():
        p=output/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(raw)
    state=C._new_state(module.SOURCE, protocol_hash, module.MASTER)
    state.update(head_commit=parent, head_part='20261119T165424079731Z',
                 head_receipt_sha256='b'*64, sequence=sequence-1)
    snapshot=base/('e3-audit-part-'+part+'-fixture');snapshot.mkdir()
    receipt={'kind':'e3_closure_stage_receipt','profile':'audit','schema_version':1,
             'repository':C.REPOSITORY,'source_commit':module.SOURCE,
             'protocol_sha256':protocol_hash,'master_run_id':module.MASTER,
             'part_id':part,'sequence':sequence,
             'previous_part':{'commit_sha':parent,'part_id':state['head_part'],
                              'receipt_sha256':state['head_receipt_sha256']},
             'files':{name:C._meta(raw,LegacyProxy()) for name,raw in payload.items()},
             'new_attempted_jobs':[job],'new_real_data_fits':1,
             'R1_accredited':False,'R3_accredited':False,'main_updated':False,
             'controller_status':{'stage_status':'failed','analysis_complete':False}}
    for name, raw in payload.items():
        p=snapshot/'payload'/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(raw)
    (snapshot/'stage_receipt.json').write_bytes(A._canonical(receipt))
    state['pending']={'part_id':part,'snapshot':str(snapshot)}
    statepath=base/('e3-extended-continuation-'+module.MASTER+'.json')
    statepath.write_bytes(A._canonical(state))
    return SimpleNamespace(state=state,statepath=statepath,snapshot=snapshot,receipt=receipt,
        output=output,protocol_hash=protocol_hash,part=part,
        original_snapshot={p.relative_to(snapshot).as_posix():p.read_bytes() for p in snapshot.rglob('*') if p.is_file()},
        original_output={p.relative_to(output).as_posix():p.read_bytes() for p in output.rglob('*') if p.is_file()})


class GitPublisherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.f=Fixture(self.base)
        self.launcher=load(HERE/'launcher_part.py','real_git_pack_launcher_tested')
        import git_api_provider as provider
        self.provider=provider
        self.pending=pending_fixture(self.launcher,self.base,self.f)
        self.protocol_patch=patch.object(provider,'PROTOCOL',self.pending.protocol_hash)
        self.protocol_patch.start()
        self.launcher.PROTOCOL=self.pending.protocol_hash
        self.context=provider.GitPackContext(self.base,TOKEN,identity={'name':'Offline integration','email':'offline@example.invalid'},repo_url=self.f.url,emit=lambda value:None)
        self.legacy=LegacyProxy(self.f)
        self.api=self.launcher.install_api(self.legacy,TOKEN,self.context)
        git(self.f.remote,'update-ref','refs/heads/main',self.f.parent)
        self.original_refs=git(self.f.remote,'show-ref')

    def tearDown(self):
        self.protocol_patch.stop();self.temp.cleanup()

    def save(self):
        with patch('sys.stdout',io.StringIO()):
            return self.launcher.finish_pending(self.base,TOKEN,self.api,C,self.context)

    def assert_unchanged(self,pending=None):
        pending=pending or self.pending
        for name,raw in pending.original_snapshot.items():self.assertEqual((pending.snapshot/name).read_bytes(),raw)
        for name,raw in pending.original_output.items():self.assertEqual((pending.output/name).read_bytes(),raw)
        self.assertEqual(git(self.f.remote,'rev-parse','refs/heads/main'),self.f.parent)

    def test_original_frozen_publisher_publishes_exact_pending_and_preserves_old_bytes(self):
        result=self.save();state=json.loads(self.pending.statepath.read_bytes())
        self.assertEqual(result['sequence'],63);self.assertIsNone(state['pending'])
        self.assertEqual(state['head_part'],self.pending.part)
        actual=git(self.f.remote,'rev-parse','refs/heads/e3-extended-audit-'+self.pending.part)
        self.assertEqual(actual,state['head_commit'])
        self.assertEqual(git(self.f.remote,'rev-parse',actual+'^'),self.f.parent)
        self.assertEqual(state['files'],self.pending.receipt['files'])
        self.assertTrue(all(method=='GET' for method,path,payload,kwargs in self.legacy.calls))
        self.assertFalse(self.context.stores[self.f.parent]._has(self.f.old_sha,'blob'))
        self.assert_unchanged()

    def test_next_parent64_preserves63_and_uses_current_head(self):
        first=self.save();old=json.loads(self.pending.statepath.read_bytes())
        later=pending_fixture(self.launcher,self.base,self.f,sequence=64,part='20261119T190000000000Z',parent=old['head_commit'])
        state=json.loads(later.statepath.read_bytes())
        state.update(files=old['files'],counted_jobs=old['counted_jobs'],head_part=old['head_part'],head_receipt_sha256=old['head_receipt_sha256'])
        receipt=later.receipt;receipt['previous_part']={'commit_sha':old['head_commit'],'part_id':old['head_part'],'receipt_sha256':old['head_receipt_sha256']}
        (later.snapshot/'stage_receipt.json').write_bytes(A._canonical(receipt));later.statepath.write_bytes(A._canonical(state))
        later.original_snapshot['stage_receipt.json']=(later.snapshot/'stage_receipt.json').read_bytes()
        with patch('sys.stdout',io.StringIO()):result=self.launcher.finish_pending(self.base,TOKEN,self.api,C,self.context)
        self.assertEqual(result['sequence'],64)
        now=json.loads(later.statepath.read_bytes());self.assertIsNone(now['pending'])
        self.assertEqual(git(self.f.remote,'rev-parse',now['head_commit']+'^'),old['head_commit'])
        self.assertEqual(git(self.f.remote,'rev-parse','refs/heads/e3-extended-audit-'+self.pending.part),old['head_commit'])
        self.assertIn(old['head_commit'],self.context.stores)
        self.assertTrue(set(old['files']).issubset(now['files']))
        self.assert_unchanged(later)

    def test_changed_snapshot_hash_refuses_before_git_publication(self):
        (self.pending.snapshot/'payload/resources.csv').write_bytes(b'changed')
        before=self.pending.statepath.read_bytes()
        with self.assertRaisesRegex(ValueError,'Pending payload differs'):self.save()
        self.assertEqual(self.pending.statepath.read_bytes(),before)
        self.assertEqual(git(self.f.remote,'show-ref'),self.original_refs)

    def test_other_protocol_and_source_refused_before_publication(self):
        for field,value in [('protocol_sha256','a'*64),('source_commit','a'*40),('master_run_id','20260101T000000000000Z')]:
            state=copy_state=dict(self.pending.state);state[field]=value
            self.pending.statepath.write_bytes(A._canonical(state));before=self.pending.statepath.read_bytes()
            with self.subTest(field=field),self.assertRaises((A.StorageError,ValueError)):self.save()
            self.assertEqual(self.pending.statepath.read_bytes(),before)
            self.assertEqual(git(self.f.remote,'show-ref'),self.original_refs)
        self.pending.statepath.write_bytes(A._canonical(self.pending.state))

    def test_concurrent_conflicting_audit_branch_never_overwritten(self):
        original=self.context.reference
        def conflict(payload):
            git(self.f.remote,'update-ref',payload['ref'],self.f.parent)
            return original(payload)
        before=self.pending.statepath.read_bytes()
        with patch.object(self.context,'reference',side_effect=conflict),self.assertRaisesRegex(ValueError,'Conflicting remote'):
            self.save()
        ref='refs/heads/e3-extended-audit-'+self.pending.part
        self.assertEqual(git(self.f.remote,'rev-parse',ref),self.f.parent)
        self.assertEqual(self.pending.statepath.read_bytes(),before)
        self.assert_unchanged()

    def test_published_raw_blob_is_read_by_git_with_identity_checks(self):
        self.save();name='jobs/D03/MLP16__final__20261118/supervised/checkpoint.pt'
        meta=self.pending.receipt['files'][name]
        raw=self.api.raw_blob(meta['git_blob_sha'],meta['bytes'],TOKEN)
        self.assertEqual(raw,(self.pending.output/name).read_bytes())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['sha256'])
        with self.assertRaises(ValueError):self.api.raw_blob(meta['git_blob_sha'],meta['bytes']+1,TOKEN)

    def test_combined_script_import_and_actual_pending_publication(self):
        combined=load(HERE/'resume_cell.py','combined_git_pack_launcher_tested')
        combined.PROTOCOL=self.pending.protocol_hash
        combined.GitPackContext.head.__globals__['PROTOCOL']=self.pending.protocol_hash
        context=combined.GitPackContext(self.base,TOKEN,identity={'name':'Offline integration','email':'offline@example.invalid'},repo_url=self.f.url,emit=lambda value:None)
        api=combined.install_api(LegacyProxy(self.f),TOKEN,context)
        with patch('sys.stdout',io.StringIO()):result=combined.finish_pending(self.base,TOKEN,api,C,context)
        self.assertEqual(result['sequence'],63)
        self.assertIsNone(json.loads(self.pending.statepath.read_bytes())['pending'])
        self.assert_unchanged()

    def test_main_publishes_pending_before_fixed_original_worker(self):
        seed_source=b"def seed_view(base,required=False):\n    (base/'seed.called').write_text('ready')\ndef _expected_view_error(error,base):\n    return False\n"
        calls=[]
        def worker(source,token,**kwargs):
            self.assertEqual(source,SOURCE);self.assertEqual(token,TOKEN)
            self.assertEqual(kwargs,{'base':self.base,'start_new':False,'master_run_id':self.launcher.MASTER})
            self.assertTrue((self.base/'seed.called').is_file())
            state=json.loads(self.pending.statepath.read_bytes());self.assertIsNone(state['pending']);self.assertEqual(state['sequence'],63)
            self.assertEqual(git(self.f.remote,'rev-parse','refs/heads/e3-extended-audit-'+self.pending.part),state['head_commit'])
            calls.append('worker');return {'status':'offline_worker_not_executed'}
        controller=SimpleNamespace(run=worker,load=lambda *args:None)
        def modules(path,name):return controller if Path(path).name=='controller.py' else C
        namespace={'_E3_GITHUB_TOKEN':TOKEN,'E3_EXTENDED_MASTER_RUN':'wrong_prior_master','E3_EXTENDED_START_NEW':True}
        with patch.dict(os.environ,{},clear=True),patch.object(self.launcher.subprocess,'check_output',return_value='NVIDIA A100-SXM4-80GB\n'), \
             patch.object(self.launcher,'checkout_source',return_value=SCIENCE), \
             patch.object(self.launcher,'GitPackContext',return_value=self.context), \
             patch.object(self.launcher,'prepare_api',return_value=self.api), \
             patch.object(self.launcher,'load',side_effect=modules), \
             patch.object(self.launcher,'SEED_SHA256',hashlib.sha256(seed_source).hexdigest()), \
             patch.object(self.launcher.urllib.request,'urlopen',return_value=io.BytesIO(seed_source)), \
             patch('sys.stdout',io.StringIO()):
            result=self.launcher.main(namespace,base=self.base)
        self.assertEqual(result,{'status':'offline_worker_not_executed'});self.assertEqual(calls,['worker'])
        self.assertEqual(namespace['_E3_EXTENDED_AUDIT_RESULT'],result)
        self.assert_unchanged()


if __name__=='__main__':
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    report={'kind':'e3_real_git_pack_frozen_publisher_integration','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'actual_frozen_finish_pending_exercised':True,'real_local_bare_git_remote':True,'external_HTTP_requests':0,'external_Git_operations':0,'GPU_fits':0,'checkpoint_deadline_snapshot_output_preserved':True,'audit_state_advanced_by_original_publisher_only':True,'main_branch_unchanged':True,'combined_resume_cell_sha256':hashlib.sha256((HERE/'resume_cell.py').read_bytes()).hexdigest(),'details':stream.getvalue()}
    (HERE/'integration_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
