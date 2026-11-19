"""CPU stage contracts using fabricated receipts; no HTTP or market model."""
import hashlib
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


stage = load(HERE/'colab.py', 'admission_stage_under_test')
SOURCE = 'a'*40
OLD_SOURCE = 'b'*40
RUN = '20261119T022547490732Z'
REUSE_URL = 'https://github.com/'+stage.REPOSITORY+'/tree/e3-closure-cpu-'+RUN+'/runs/e3_closure_cpu/'+RUN


def fixture(output, source, admitted=3):
    output = Path(output)
    (output/'datasets').mkdir(parents=True,exist_ok=True)
    rows=[]
    for i, name in enumerate(stage.EXPECTED_IDS):
        row={'id':name, 'admitted':i<admitted, 'target_values_computed':False}
        if row['admitted']:
            raw=b'date,open,high,low,close,volume\n2020-01-01,1,2,1,2,10\n'
            path=output/'datasets'/f'{name}.csv';path.write_bytes(raw)
            row.update(path=f'datasets/{name}.csv',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        else:
            row.update(error='Synthetic source unavailable, not a network observation')
        rows.append(row)
    report={'kind':'e3_additional_OHLCV_admission','source_commit':source,
            'datasets_required':3,'datasets_admitted':admitted,'datasets':rows,
            'datasets_evaluated_by_models':0,'new_real_data_fits':0,
            'new_market_predictions':0,'new_statistical_contrasts':0,'target_values_computed':False,
            'protocol_frozen':False,'real_campaign_allowed':False}
    (output/'admission.json').write_text(json.dumps(report))
    return report


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.output=self.base/'evidence'

    def tearDown(self):
        self.temp.cleanup()

    def failed(self,folder=None,admitted=2):
        folder=folder or self.base/'failed'
        stage.run_stage(self.base,folder,OLD_SOURCE,
            collector=lambda output,source:fixture(output,source,admitted=admitted))
        for name,raw in [('responses/D02/archive.zip',b'raw original archive fixture'),
                         ('responses/D02/archive.zip.CHECKSUM',b'official checksum fixture'),
                         ('responses/D02/download_receipts.json',b'{"first_http_observation":true}'),
                         ('sources/terms.md',b'pinned source attribution fixture')]:
            path=folder/'data'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        files={p.relative_to(folder/'data').as_posix():{'bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in (folder/'data').rglob('*') if p.is_file()}
        (folder/'data/raw_manifest.json').write_bytes(stage._canonical({
            'kind':'admission_raw_file_manifest','source_commit':OLD_SOURCE,'files':files}))
        return folder

    def remote(self,folder,storage_mutation=None):
        api=load(ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','reuse_fixture_api')
        files={p.relative_to(folder).as_posix():p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        metadata={n:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                     'git_blob_sha':api._git_sha(raw)} for n,raw in files.items()}
        storage={'kind':'e3_closure_storage_manifest','profile':'cpu','source_commit':OLD_SOURCE,
                 'run_id':RUN,'files':metadata}
        if storage_mutation:storage_mutation(storage)
        files['storage_manifest.json']=stage._canonical(storage)
        blobs={api._git_sha(raw):raw for raw in files.values()};calls=[]
        commit='c'*40;tree_ids=['d'*40,'e'*40,'f'*40,'1'*40]
        root='/repos/'+stage.REPOSITORY
        def request(method,path,token,*args,**kwargs):
            self.assertEqual(method,'GET');self.assertEqual(token,'synthetic_reuse_token');calls.append(path)
            if path==root+'/git/ref/heads/e3-closure-cpu-'+RUN:
                return {'ref':'refs/heads/e3-closure-cpu-'+RUN,'object':{'type':'commit','sha':commit}}
            if path==root+'/git/commits/'+commit:
                return {'sha':commit,'parents':[{'sha':OLD_SOURCE}],'tree':{'sha':tree_ids[0]}}
            for i,part in enumerate(('runs','e3_closure_cpu',RUN)):
                if path==root+'/git/trees/'+tree_ids[i]:
                    return {'sha':tree_ids[i],'truncated':False,
                        'tree':[{'path':part,'type':'tree','mode':'040000','sha':tree_ids[i+1]}]}
            if path==root+'/git/trees/'+tree_ids[-1]+'?recursive=1':
                return {'sha':tree_ids[-1],'truncated':False,'tree':[
                    {'path':n,'type':'blob','mode':'100644','sha':api._git_sha(raw),'size':len(raw)}
                    for n,raw in files.items()]}
            if path.startswith(root+'/git/blobs/'):
                sha=path.rsplit('/',1)[1];raw=blobs[sha]
                return {'sha':sha,'encoding':'base64','content':base64.b64encode(raw).decode()}
            self.fail('Unexpected read '+path)
        api._request=request
        return api,calls

    def test_complete_physical_admission_cannot_accredit_or_train(self):
        result=stage.run_stage(self.base,self.output,SOURCE,collector=fixture)
        self.assertEqual(result['datasets_admitted'],3)
        self.assertEqual(result['status'],'all_three_files_admitted')
        for field in ['R1_accredited','R3_accredited','real_campaign_allowed','Render_checked','GPU_required']:
            self.assertIs(result[field],False)
        self.assertEqual(result['new_real_data_fits'],0)
        self.assertEqual(result['datasets_evaluated_by_models'],0)

    def test_incomplete_downloads_remain_incomplete_and_saved(self):
        result=stage.run_stage(self.base,self.output,SOURCE,
            collector=lambda output,source:fixture(output,source,admitted=1))
        self.assertEqual(result['datasets_admitted'],1)
        self.assertEqual(result['status'],'incomplete_admission')
        self.assertTrue((self.output/'data/admission.json').is_file())

    def test_completed_retry_preserves_bytes_without_recollecting(self):
        result=stage.run_stage(self.base,self.output,SOURCE,collector=fixture)
        before={str(p):p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
        def forbidden(*args):
            raise AssertionError('No repeat collection after a completed immutable stage')
        self.assertEqual(stage.run_stage(self.base,self.output,SOURCE,collector=forbidden),result)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.output.rglob('*') if p.is_file()})

    def test_interrupted_receipt_publication_leaves_no_final_marker_and_retries(self):
        with patch.object(stage.os,'link',side_effect=KeyboardInterrupt('synthetic publication interruption')):
            with self.assertRaises(KeyboardInterrupt):
                stage.run_stage(self.base,self.output,SOURCE,collector=fixture)
        self.assertFalse((self.output/'stage_receipt.json').exists())
        self.assertEqual(list(self.output.glob('.partial-cache-*')),[])
        data_before=(self.output/'data/datasets/D02.csv').read_bytes()
        receipt=stage.run_stage(self.base,self.output,SOURCE,collector=fixture)
        self.assertEqual(receipt['status'],'all_three_files_admitted')
        self.assertEqual(json.loads((self.output/'stage_receipt.json').read_bytes()),receipt)
        self.assertEqual((self.output/'data/datasets/D02.csv').read_bytes(),data_before)

    def test_forged_scope_count_or_file_bytes_refused(self):
        mutations=[lambda report:report.update(new_real_data_fits=1),
                   lambda report:report.update(datasets_admitted=2),
                   lambda report:report['datasets'][0].update(sha256='0'*64),
                   lambda report:report['datasets'][0].update(path='../unrelated.csv')]
        for i,mutation in enumerate(mutations):
            with self.subTest(case=i):
                output=self.base/str(i)
                def fake(folder,source):
                    report=fixture(folder,source);mutation(report);return report
                with self.assertRaises(ValueError):
                    stage.run_stage(self.base,output,SOURCE,collector=fake)
                self.assertFalse((output/'stage_receipt.json').exists())

    def test_source_mismatch_and_symlink_refused(self):
        stage.run_stage(self.base,self.output,SOURCE,collector=fixture)
        with self.assertRaises(ValueError):stage.run_stage(self.base,self.output,'b'*40)
        link=self.base/'linked';link.symlink_to(self.output,target_is_directory=True)
        with self.assertRaises(ValueError):stage.run_stage(self.base,link,SOURCE)

    def test_local_failed_cache_reused_without_redownload_or_old_stage_mutation(self):
        old=self.failed();before={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
        info=stage.inspect_failed_stage(old)
        self.assertEqual(info['source_commit'],OLD_SOURCE);self.assertEqual(info['datasets_admitted'],2)
        api=SimpleNamespace(_request=lambda *args:self.fail('Local cache must avoid remote downloads'))
        receipt=stage.seed_cache(self.base,self.output,SOURCE,previous_output=old,reuse_url=REUSE_URL,
                                 api=api,token='synthetic_reuse_token')
        self.assertEqual(receipt['origin']['source_commit'],OLD_SOURCE)
        self.assertEqual(receipt['copied_files'],6)
        self.assertEqual((self.output/'data/datasets/D02.csv').read_bytes(),before['data/datasets/D02.csv'])
        for name in ('stage_receipt.json','data/admission.json','data/raw_manifest.json','data/run_config.json'):
            self.assertFalse((self.output/name).exists())
        self.assertEqual(before,{p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()})
        self.assertEqual(stage.seed_cache(self.base,self.output,SOURCE,previous_output=old),receipt)

    def test_cache_source_with_scientific_activity_or_changed_bytes_is_refused(self):
        old=self.failed();stage_path=old/'stage_receipt.json';saved=stage_path.read_bytes()
        for field,value in [('new_real_data_fits',1),('new_market_predictions',1),
                            ('new_statistical_contrasts',1),('target_values_computed',True),
                            ('protocol_frozen',True),('datasets_admitted',3)]:
            with self.subTest(field=field):
                receipt=json.loads(saved);receipt[field]=value;stage_path.write_bytes(stage._canonical(receipt))
                with self.assertRaises(ValueError):stage.seed_cache(self.base,self.output,SOURCE,previous_output=old)
                self.assertFalse(self.output.exists())
        stage_path.write_bytes(saved)
        (old/'data/responses/D02/archive.zip').write_bytes(b'corrupted source bytes')
        with self.assertRaisesRegex(ValueError,'byte/hash manifest'):
            stage.inspect_failed_stage(old)
        self.assertFalse(self.output.exists())

    def test_cache_destination_conflict_preserves_all_existing_bytes(self):
        old=self.failed();target=self.output/'data/datasets/D02.csv';target.parent.mkdir(parents=True)
        target.write_bytes(b'preexisting user evidence')
        with self.assertRaisesRegex(ValueError,'no cache file was replaced'):
            stage.seed_cache(self.base,self.output,SOURCE,previous_output=old)
        self.assertEqual(target.read_bytes(),b'preexisting user evidence')
        self.assertFalse((self.output/'cache_reuse.json').exists())
        self.assertFalse((self.output/'data/datasets/D01.csv').exists())

    def test_remote_failed_stage_is_pinned_and_hash_verified_before_cache_copy(self):
        old=self.failed();api,calls=self.remote(old)
        receipt=stage.seed_cache(self.base,self.output,SOURCE,previous_output=self.base/'session-gone',
            reuse_url=REUSE_URL,token='synthetic_reuse_token',api=api)
        self.assertEqual(receipt['origin']['results_commit'],'c'*40)
        self.assertEqual(receipt['origin']['source_commit'],OLD_SOURCE)
        self.assertTrue((self.output/'data/responses/D02/archive.zip').is_file())
        self.assertFalse((self.output/'stage_receipt.json').exists())
        self.assertGreater(len(calls),8)
        self.assertFalse(any(b'synthetic_reuse_token' in p.read_bytes() for p in self.output.rglob('*') if p.is_file()))

    def test_complete_three_file_admission_can_be_reused_before_a_protocol(self):
        old=self.failed(admitted=3)
        before={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
        proof=stage.inspect_preprotocol_stage(old)
        self.assertEqual(proof['datasets_admitted'],3)
        self.assertEqual(proof['new_real_data_fits'],0)
        with self.assertRaisesRegex(ValueError,'already complete'):stage.inspect_failed_stage(old)
        local=stage.seed_cache(self.base,self.output,SOURCE,previous_output=old)
        self.assertTrue((self.output/'data/datasets/D03.csv').is_file())
        self.assertEqual(local['new_real_data_fits'],0);self.assertFalse(local['protocol_frozen'])
        api,_=self.remote(old)
        recovered=stage.seed_cache(self.base,self.base/'remote-complete',SOURCE,
            reuse_url=REUSE_URL,token='synthetic_reuse_token',api=api)
        self.assertEqual(recovered['origin']['source_commit'],OLD_SOURCE)
        self.assertEqual(recovered['new_market_predictions'],0)
        self.assertEqual(before,{p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()})

    def test_complete_admission_rejects_scientific_scope_and_inconsistent_status(self):
        old=self.failed(admitted=3);path=old/'stage_receipt.json';saved=json.loads(path.read_bytes())
        for field,value in [('status','incomplete_admission'),('protocol_frozen',True),
                            ('target_values_computed',True),('new_real_data_fits',1),
                            ('new_market_predictions',1),('new_statistical_contrasts',1)]:
            with self.subTest(field=field):
                record={**saved,field:value};path.write_bytes(stage._canonical(record))
                with self.assertRaises(ValueError):stage.inspect_preprotocol_stage(old)
                self.assertFalse(self.output.exists())

    def test_remote_corrupted_storage_hash_and_foreign_url_are_refused(self):
        old=self.failed()
        def corrupt(storage):storage['files']['data/datasets/D02.csv']['sha256']='0'*64
        api,calls=self.remote(old,corrupt)
        with self.assertRaisesRegex(ValueError,'byte/hash manifest'):
            stage.seed_cache(self.base,self.output,SOURCE,reuse_url=REUSE_URL,
                token='synthetic_reuse_token',api=api)
        self.assertFalse(self.output.exists())
        before=len(calls)
        for url in (REUSE_URL.replace('jassan-bit','foreign'),REUSE_URL+'?ref=other',
                    REUSE_URL.replace('https:','http:'),REUSE_URL+'/extra'):
            with self.assertRaisesRegex(ValueError,'exact GitHub URL'):
                stage.seed_cache(self.base,self.output,SOURCE,reuse_url=url,
                    token='synthetic_reuse_token',api=api)
        self.assertEqual(len(calls),before)

    def test_controller_checks_access_then_uses_existing_hardened_store(self):
        api=load(ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','admission_fixture_api')
        store=load(ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py','admission_fixture_store')
        events=[];token='synthetic_admission_token_not_persisted'
        def wrap(value):
            self.assertIs(value,api);events.append('wrapped');return value
        def validate(value):
            self.assertEqual(events,['wrapped'])
            self.assertEqual(value,token);events.append('validated')
        def save(output,value,source,run_id,actual_api,profile):
            self.assertEqual(profile,'cpu');self.assertIs(actual_api,api)
            files=store.snapshot(output,value,source,run_id,actual_api,profile)
            self.assertIn('data/datasets/D03.csv',files)
            self.assertIn('stage_receipt.json',files)
            self.assertFalse(any(token.encode() in raw for raw in files.values()))
            events.append('saved')
            return {'run_url':'https://github.com/fixture/results','commit_sha':'b'*40,'saved':True}
        real_run_stage=stage.run_stage
        def run_stage(checkout,output,source):
            self.assertEqual(events,['wrapped','validated'])
            return real_run_stage(checkout,output,source,collector=fixture)
        def loader(path,name):
            if Path(path).name=='colab_github.py':return api
            if Path(path).name=='github_transport.py':return SimpleNamespace(wrap=wrap)
            return SimpleNamespace(save=save)
        with patch.object(stage,'checkout_source',return_value=ROOT), \
             patch.object(stage,'_load',side_effect=loader), \
             patch.object(api,'validate_credentials',side_effect=validate), \
             patch.object(stage,'run_stage',side_effect=run_stage):
            result=stage.run(SOURCE,token,base=self.base,resume_output=self.output)
        self.assertEqual(events,['wrapped','validated','saved'])
        self.assertTrue(result['storage']['saved'])


if __name__=='__main__':
    unittest.main()
