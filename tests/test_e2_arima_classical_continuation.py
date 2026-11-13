"""Synthetic orchestration/Git fixtures; no market data or model training."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import e2_arima_classical_continuation as c
import e2_reduced_final as final


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='arima-parent-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'campaign/final'
        self.budget = SimpleNamespace(root=self.output.parent / 'shared_budget',
            stage_wall_seconds=1800, reserve=Mock())
        self.budget.root.mkdir(parents=True)
        self.directory = self.output / c.FAILED_RELATIVE
        self.directory.mkdir(parents=True)
        self.source = {'source_commit': c.ENGINE_COMMIT, 'files_sha256': {'example.py': 'a'*64}}
        self.runtime = {'device': 'cuda:0', 'device_name': 'NVIDIA A100-SXM4-80GB'}
        self.base = final._base_spec(self.root / 'data', self.source, self.runtime, None, 'cuda:0')
        spec = {**self.base, 'kind': 'classical', 'mode': 'classical-cv', 'model': 'ARIMA',
            'parameters': {'order': [0, 0, 0]}, 'fold_origin': '2024-01-01',
            'output': str(self.directory.resolve()), 'fit_wall_seconds': 1800}
        spec_sha = c.digest(spec)
        spec['worker_spec_sha256'] = spec_sha
        spec_raw = c.canonical(spec) + b'\n'
        self.spec_path = self.directory / 'worker_spec.json'
        self.spec_path.write_bytes(spec_raw)
        failure = {'status': 'failed', 'error_type': 'AttributeError',
            'reason': c.FAILURE_REASON, 'resource_receipt': {}, 'automatic_retry': False}
        failure_raw = c.canonical(failure) + b'\n'
        self.failure_path = self.directory / 'worker_failure.json'
        self.failure_path.write_bytes(failure_raw)
        job = c.digest({'version': final.VERSION, 'spec_sha256': spec_sha})
        reserve = {'sequence': 0, 'previous_sha256': '0'*64, 'kind': 'reserved',
            'attempt_id': job, 'reservation_seconds': 1860,
            'metadata': {'worker_spec_sha256': spec_sha}}
        reserve['sha256'] = c.digest(reserve)
        charge = {'sequence': 1, 'previous_sha256': reserve['sha256'], 'kind': 'charged',
            'attempt_id': job, 'reservation_seconds': 1860, 'status': 'failed',
            'reservation_overrun': False, 'elapsed_seconds': c.FAILED_SECONDS,
            'resource_use': {'worker_spec_sha256': spec_sha, 'failure': 'worker_exit_2',
                'optimizer_steps_unknown': True, 'automatic_retry': False}}
        charge['sha256'] = c.digest(charge)
        self.ledger = self.budget.root / 'pilot_budget.jsonl'
        self.ledger.write_bytes(c.canonical(reserve)+b'\n'+c.canonical(charge)+b'\n')
        for name, value in [('FAILED_SPEC', spec_sha), ('FAILED_JOB', job),
                ('FAILED_SPEC_FILE', hashlib.sha256(spec_raw).hexdigest()),
                ('FAILED_WORKER_FILE', hashlib.sha256(failure_raw).hexdigest()),
                ('FAILED_CHARGE', charge['sha256'])]:
            item = patch.object(c, name, value); item.start(); self.addCleanup(item.stop)
        self.policy = c.policy_template()
        self.git_root = self.root / 'registration'
        self.git_root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.git_root)], check=True)
        for relative, raw in [(c.SOURCE_PATH, (ROOT/c.SOURCE_PATH).read_bytes()),
                (c.POLICY_PATH, c.canonical(self.policy)+b'\n')]:
            path = self.git_root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        files = {r:hashlib.sha256((self.git_root/r).read_bytes()).hexdigest()
            for r in [c.SOURCE_PATH, c.POLICY_PATH]}
        record = {'engine_commit': c.ENGINE_COMMIT, 'protocol_id': 'e2-reduced-delivery-v1',
            'files_sha256': files, 'policy_sha256': self.policy['sha256'],
            'policy_file_sha256': files[c.POLICY_PATH]}
        path = self.git_root / c.REGISTRATION_PATH; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(c.canonical(record)+b'\n')
        subprocess.run(['git','-C',str(self.git_root),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.git_root),'-c','user.name=Synthetic fixture',
            '-c','user.email=fixture@example.invalid','commit','-qm','Synthetic repair fixture'],check=True)
        self.commit = subprocess.check_output(['git','-C',str(self.git_root),'rev-parse','HEAD'],text=True).strip()
        self.physical_supervisor = Mock(name='prospective repair supervisor')
        self.run = c.make_run_classical(final, self.policy, registration_commit=self.commit,
            registration_git_root=self.git_root, supervisor=self.physical_supervisor)

    def known(self, persist=None):
        proof = c._registration(self.policy, self.commit, self.git_root)
        return c._implementation_failure(final, self.directory, self.base,
            self.budget, proof, self.policy, persist)

    def test_actual_registered_policy_is_not_a_retry_or_budget_extension(self):
        policy = c.validate_policy(self.policy)
        self.assertFalse(policy['scientific_retry_allowed'])
        self.assertFalse(policy['failed_fit_repeated'])
        self.assertEqual(policy['planned_classical_cv_opportunities'], 63)
        changed = dict(policy, new_budget_authorization=True)
        with self.assertRaises(ValueError): c.validate_policy(changed)
        with self.assertRaises(ValueError):
            c.make_run_classical(final, policy, registration_commit='f'*40,
                registration_git_root=self.git_root)

    def test_failure_disposition_is_terminal_and_does_not_create_a_result(self):
        before = {p:p.read_bytes() for p in [self.spec_path,self.failure_path,self.ledger]}
        persisted=[]
        row=self.known(lambda directory,receipt:persisted.append((directory,receipt)))
        self.assertEqual(row['candidate_status'],'implementation_failed')
        self.assertIsNone(row['RMSE']);self.assertIsNone(row['result_sha256'])
        self.assertEqual(row['completed_validation_count'],0)
        self.assertFalse((self.directory/'result.json').exists())
        self.assertFalse((self.directory/'predictor.pkl').exists())
        self.assertEqual(persisted[0][1]['status'],'failed')
        self.assertTrue(persisted[0][1]['optimizer_steps_unknown'])
        self.assertIsNone(persisted[0][1]['optimizer_steps'])
        self.budget.reserve.assert_not_called()
        for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)

    def test_changed_failure_or_cost_never_creates_a_disposition(self):
        original=self.failure_path.read_bytes()
        self.failure_path.write_bytes(original.replace(b'raw_value',b'other_key'))
        with self.assertRaises(ValueError):self.known()
        self.assertFalse((self.directory/'failure_disposition.json').exists())
        self.failure_path.write_bytes(original)
        rows=[json.loads(x)for x in self.ledger.read_bytes().splitlines()]
        rows[-1]['status']='completed';rows[-1].pop('sha256');rows[-1]['sha256']=c.digest(rows[-1])
        self.ledger.write_bytes(b''.join(c.canonical(r)+b'\n'for r in rows))
        with self.assertRaises(ValueError):self.known()

    def test_existing_success_or_recovered_predictor_requires_reconciliation(self):
        for filename in ['result.json','predictor.pkl']:
            path=self.directory/filename;path.write_bytes(b'existing')
            with self.assertRaises(ValueError):self.known()
            path.unlink()

    def test_full_calendar_preserves_11_cached_results_and_selects_only_3_valid_folds(self):
        cached={};calls=[];created=[]
        def body(spec):
            score=.001 if spec.get('parameters')=={'order':[0,0,0]} else .01
            return {'status':'completed','candidate_status':'completed','rmse_original':score,
                'validation_count':83,'model':spec['model']}
        for name in ['Persistence','SeasonalNaive']:
            configs=[]if name=='Persistence'else[{'period':p}for p in(7,14,28)]
            for config in configs:
                for fold in final.FOLDS:
                    directory=self.output/'classical-cv'/final.slug(name)/c.digest(config)[:16]/fold
                    path=directory/'result.json';final.write_json(path,body({'model':name,'parameters':config}));cached[path]=path.read_bytes()
            path=self.output/'classical-final'/final.slug(name)/'result.json'
            final.write_json(path,body({'model':name,'parameters':{}}));cached[path]=path.read_bytes()
        def dispatch(spec,directory,budget,**kwargs):
            self.assertIs(kwargs['supervisor'],self.physical_supervisor)
            self.assertNotEqual(Path(directory),self.directory)
            calls.append((spec,Path(directory)))
            path=Path(directory)/'result.json'
            if path.exists():return json.loads(path.read_bytes())
            value=body(spec);final.write_json(path,value);created.append(path);return value
        with patch.object(final,'_dispatch',dispatch):
            results=self.run(self.root/'data',self.output,self.budget,
                source_identity=self.source,runtime_identity=self.runtime)
        self.assertEqual(len(results),3);self.assertEqual(len(calls),65);self.assertEqual(len(created),54)
        self.assertEqual(sum(spec['mode']=='classical-cv'for spec,_ in calls),62)
        for path,raw in cached.items():self.assertEqual(path.read_bytes(),raw)
        selected=json.loads((self.output/'classical-selections'/('6c395251bcaf9062.json')).read_text())
        self.assertNotEqual(selected['parameters'],{'order':[0,0,0]})
        self.assertEqual(len(selected['ledger']),54)
        first=selected['ledger'][0];self.assertEqual(first['candidate_status'],'implementation_failed')
        self.assertIsNone(first['RMSE']);self.assertEqual(first['planned_validation_count'],83)
        self.assertFalse((self.directory/'result.json').exists())

    def test_repeated_accounting_keeps_the_same_disposition_and_no_extra_charge(self):
        self.known();path=self.directory/'failure_disposition.json';raw=path.read_bytes()
        self.known();self.assertEqual(path.read_bytes(),raw);self.budget.reserve.assert_not_called()


if __name__=='__main__':unittest.main()
