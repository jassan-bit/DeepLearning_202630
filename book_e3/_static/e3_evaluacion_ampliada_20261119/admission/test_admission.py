"""CPU stage contracts using fabricated receipts; no HTTP or market model."""
import hashlib
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
            'new_market_predictions':0,'target_values_computed':False,
            'real_campaign_allowed':False}
    (output/'admission.json').write_text(json.dumps(report))
    return report


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.base=Path(self.temp.name)
        self.output=self.base/'evidence'

    def tearDown(self):
        self.temp.cleanup()

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

    def test_controller_checks_access_then_uses_existing_hardened_store(self):
        api=load(ROOT/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','admission_fixture_api')
        store=load(ROOT/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py','admission_fixture_store')
        events=[];token='synthetic_admission_token_not_persisted'
        def validate(value):
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
            self.assertEqual(events,['validated'])
            return real_run_stage(checkout,output,source,collector=fixture)
        def loader(path,name):
            return api if Path(path).name=='colab_github.py' else SimpleNamespace(save=save)
        with patch.object(stage,'checkout_source',return_value=ROOT), \
             patch.object(stage,'_load',side_effect=loader), \
             patch.object(api,'validate_credentials',side_effect=validate), \
             patch.object(stage,'run_stage',side_effect=run_stage):
            result=stage.run(SOURCE,token,base=self.base,resume_output=self.output)
        self.assertEqual(events,['validated','saved'])
        self.assertTrue(result['storage']['saved'])


if __name__=='__main__':
    unittest.main()
