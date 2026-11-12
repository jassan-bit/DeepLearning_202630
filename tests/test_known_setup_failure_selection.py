"""CPU-only controller tests; no data access, fit, retry, or GitHub writes."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_known_setup_failure_selection as handler
E = handler.original


class KnownFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.domains = E.current_domains()
        self.jobs = E.job_plan(['TimesNet'], domains=self.domains)
        self.job = self.jobs[0]
        self.assertEqual(self.job['logical_job_id'], handler.KNOWN_JOB)
        self.directory = self.root / 'jobs' / handler.KNOWN_JOB
        self.directory.mkdir(parents=True)
        self.source = {'frozen-engine': 'unchanged'}
        self.spec = {'job': self.job, 'source_sha256': self.source}
        E.write_json(self.directory/'worker_spec.json', self.spec)
        E.write_json(self.directory/'request.json', {'setup': True})
        worker = {'status':'failed','error_type':'FileNotFoundError',
            'reason':handler.KNOWN_REASON,'resource_receipt':{},'automatic_retry':False}
        E.write_json(self.directory/'worker_failure.json', worker)
        (self.directory/'worker.log').write_text(json.dumps(worker))
        spec_sha = E.digest(self.spec)
        self.charge = {'kind':'charged','attempt_id':handler.KNOWN_JOB,'status':'failed',
            'reservation_overrun':False,'resource_use':{'worker_spec_sha256':spec_sha,
            'failure':'worker_exit_2','automatic_retry':False}}
        self.charge['sha256'] = E.digest(self.charge)
        self.failure = {'status':'failed','attempt_id':handler.KNOWN_JOB,'job':self.job,
            'automatic_retry':False,'worker_spec_sha256':spec_sha,
            'supervision':{'worker_spec_sha256':spec_sha,'failure':'worker_exit_2',
                'automatic_retry':False,'budget_receipt':self.charge},
            'artifacts_sha256':{p.name:E.file_sha(p) for p in self.directory.iterdir()}}
        E.write_json(self.directory/'failure_result.json', self.failure)
        self.budget = SimpleNamespace(_inner=SimpleNamespace(_events=[self.charge]),
            remaining_seconds=30000,stage_wall_seconds=600,
            report=lambda:{'remaining_seconds':30000})
        for key,value in [('KNOWN_SPEC',spec_sha),('KNOWN_COST',self.charge['sha256']),
                ('KNOWN_FAILURE_FILE',E.file_sha(self.directory/'worker_failure.json'))]:
            p = patch.object(handler,key,value);p.start();self.addCleanup(p.stop)

    def test_closed_failure_is_not_retried_and_all_eighteen_rows_select(self):
        runtime=self.root/'runtime.json'
        E.write_json(runtime,{'fit_execution_policy_verified':True,'runtime':{}})
        args=SimpleNamespace(output=self.root,runtime_profile=runtime,
            models=['TimesNet'],data_root='unopened',source_roots='unopened')
        cache=[]
        for job in self.jobs[1:]:
            d=self.root/'jobs'/job['logical_job_id'];d.mkdir(parents=True)
            result={'rmse_original':.01+int(job['configuration_id'][-1])*.001,
                'phase_best_epochs':{'supervised':2},'artifacts_sha256':{},
                'cache_reuse':True,'optimizer_steps':0}
            E.write_json(d/'result.json',result)
            E.write_json(d/'selection_row.json',E._row(job,result,E.file_sha(d/'result.json')))
            cache.append({'model_id':'TimesNet','origin':job['fold_origin'],
                'configuration_id':job['configuration_id'],'reuse_approved':True})
        cuda=ModuleType('run_e2_cuda_search');cuda.prepare_panel=lambda _: (None,None)
        temporal=ModuleType('e2_temporal_training')
        temporal.prepare_b=lambda *a:SimpleNamespace(inner_validation=SimpleNamespace(dates=list(range(83))))
        failure_bytes=(self.directory/'failure_result.json').read_bytes()
        with patch.dict(sys.modules,{'run_e2_cuda_search':cuda,'e2_temporal_training':temporal}), \
                patch.object(E,'verify_reduced_registration'),patch.object(E,'source_identity',return_value=self.source), \
                patch.object(E,'supervise_command',side_effect=AssertionError('No retry permitted')) as run:
            first=handler.run_selection(args,self.budget,{'fits':cache},'unopened')
            second=handler.run_selection(args,self.budget,{'fits':cache},'unopened')
        self.assertEqual(first,second)
        self.assertEqual(failure_bytes,(self.directory/'failure_result.json').read_bytes())
        self.assertFalse(run.called)
        selected=first['selections']['TimesNet']
        self.assertEqual(selected['winner_configuration_id'],'C02')
        self.assertEqual(len(selected['candidate_ledger']),6)
        self.assertEqual(selected['candidate_ledger'][0]['status'],'failed')
        self.assertEqual(sum(r['status']=='completed' for r in selected['candidate_ledger']),5)
        row=json.loads((self.directory/'selection_row.json').read_text())
        self.assertEqual(row['predictions_produced'],0)
        self.assertFalse(row['resource_use']['cache_reuse'])

    def test_tampered_failure_or_phase_artifact_is_rejected(self):
        (self.directory/'phase.pt').write_bytes(b'not a setup-only attempt')
        with self.assertRaisesRegex(ValueError,'Unexpected output'):
            handler.known_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        (self.directory/'phase.pt').unlink()
        (self.directory/'worker.log').write_text('changed')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            handler.known_failure_row(self.job,self.directory,self.budget,self.source,range(83))

    def test_other_attempt_is_not_accepted_as_known_failure(self):
        with self.assertRaisesRegex(ValueError,'restricted'):
            handler.known_failure_row(self.jobs[1],self.directory,self.budget,self.source,range(83))

    def test_exact_github_transport_receipt_is_allowed_and_bound(self):
        # Build with frozen E's real GitHub manifest serialization, without I/O.
        import e2_reduced_github_backup as backup_engine
        prefix='runs/'+handler.KNOWN_JOB+'/'
        artifacts=[]
        for relative,sha in sorted(self.failure['artifacts_sha256'].items()):
            raw=(self.directory/relative).read_bytes()
            artifacts.append({'path':relative,'storage':'git_blob',
                'remote_path':prefix+'artifacts/'+relative,'sha256':sha,'bytes':len(raw),
                'git_blob_sha':backup_engine.git_blob_sha(raw)})
        manifest={'version':backup_engine.VERSION,'attempt_id':handler.KNOWN_JOB,
            'fit_status':'failed','receipt_sha256':backup_engine.digest_bytes(backup_engine.canonical(self.failure)),
            'artifacts':artifacts,'retry_scientific_fit':False,'raw_market_data_included':False}
        receipt={'version':backup_engine.VERSION,'status':'backed_up','attempt_id':handler.KNOWN_JOB,
            'repository':backup_engine.REPOSITORY,'branch':backup_engine.BRANCH,
            'release_tag':backup_engine.RELEASE_TAG,'remote_commit':'a'*40,
            'manifest_path':prefix+'manifest.json',
            'manifest_sha256':backup_engine.digest_bytes(backup_engine.canonical(manifest)+b'\n'),
            'artifact_count':4,'existing_identical_result':True,'retry_scientific_fit':False}
        E.write_json(self.directory/'github_backup_receipt.json',receipt)
        row,_=handler.known_failure_row(self.job,self.directory,self.budget,self.source,range(83))
        self.assertEqual(row['status'],'failed')
        for key,value in [('manifest_sha256','0'*64),('repository','elsewhere/repo'),
                ('manifest_path','runs/another/manifest.json')]:
            changed={**receipt,key:value}
            (self.directory/'github_backup_receipt.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'transport receipt changed'):
                handler.known_failure_row(self.job,self.directory,self.budget,self.source,range(83))

    def test_other_new_worker_uses_the_original_frozen_entrypoint(self):
        job=self.jobs[1]
        runtime=self.root/'runtime.json'
        E.write_json(runtime,{'fit_execution_policy_verified':True,'runtime':{}})
        args=SimpleNamespace(output=self.root,runtime_profile=runtime,
            models=['TimesNet'],data_root='unopened',source_roots='unopened')
        cuda=ModuleType('run_e2_cuda_search');cuda.prepare_panel=lambda _: (None,None)
        temporal=ModuleType('e2_temporal_training');temporal.prepare_b=lambda *a:None
        commands=[]
        def worker(command,budget,attempt,directory,spec_sha,**kwargs):
            commands.append(command)
            self.assertEqual(attempt,job['logical_job_id'])
            self.assertEqual(command[2],Path(E.__file__))
            spec=json.loads((directory/'worker_spec.json').read_text())
            self.assertEqual(spec['source_sha256'],self.source)
            result={'rmse_original':.02,'phase_best_epochs':{'supervised':2},
                'artifacts_sha256':{},'optimizer_steps':5}
            E.write_json(directory/'result.json',result)
            return {'result':result}
        with patch.dict(sys.modules,{'run_e2_cuda_search':cuda,'e2_temporal_training':temporal}), \
                patch.object(E,'verify_reduced_registration'), \
                patch.object(E,'source_identity',return_value=self.source), \
                patch.object(E,'job_plan',return_value=[job]), \
                patch.object(E,'supervise_command',side_effect=worker), \
                patch.object(E,'select_reduced_hpo',return_value={'winner_configuration_id':'C02'}) as selector, \
                patch.object(handler,'known_failure_row',side_effect=AssertionError('Foreign job')):
            receipt=handler.run_selection(args,self.budget,{},'unopened')
        self.assertEqual(len(commands),1)
        rows=selector.call_args.args[0]
        self.assertEqual(rows[0]['status'],'completed')
        self.assertEqual(rows[0]['resource_use']['optimizer_steps'],5)
        self.assertTrue(receipt['selections']['TimesNet']['new_training_performed'])


if __name__=='__main__':
    unittest.main()
