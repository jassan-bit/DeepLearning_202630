"""Documentary exports must never turn partial/unverified work into completion."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_publish_posthoc_evidence as export


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(export.canonical(value)+b'\n')


def record(folder,member,*,remote=False):
    base = folder/'artifacts' if remote else folder
    write_json(base/'diagnostics.json',member)
    receipt = {'version':export.DIAGNOSTIC_VERSION,'status':'completed','run_id':member['run_id'],
               'attempt_id':'posthoc-'+member['model']+'-'+str(member['seed']), 'training_performed':False,
               'artifacts_sha256':{'diagnostics.json':export.file_hash(base/'diagnostics.json')}}
    write_json(folder/('receipt.json' if remote else 'result.json'),receipt)
    if remote:
        write_json(folder/'manifest.json',{'receipt_sha256':export.digest(export.canonical(receipt)),
            'attempt_id':receipt['attempt_id'],'fit_status':'completed','artifacts':[{
                'path':'diagnostics.json','sha256':receipt['artifacts_sha256']['diagnostics.json'],
                'bytes':(base/'diagnostics.json').stat().st_size}]})
    return receipt


def member(seed=271801,*,explain=False,model='GRU32'):
    item = {'model':model,'seed':seed,'run_id':'frozen-actual-run', 'checkpoint_sha256':'a'*64,
        'baseline_projection_passed':True,'training_performed':False,'max_absolute_error':0.,
        'baseline_predictions':[.01]*608,
        'perturbations':[{'kind':kind,'severity':level,'predictions':[.012]*608}
                         for kind,levels in export.SEVERITIES.items() for level in levels],
        'explanation':None,'profile':{'parameters':3777,'batch_size':1,'latency_gpu_median_ms':.5,
            'latency_gpu_p95_ms':.7,'throughput_serial_forecasts_per_second':1900.,'peak_allocated_bytes':1024,
            'latency_scope':'recorded_GPU_forward_only'},'elapsed_wall_seconds':10.}
    if explain:
        item['explanation'] = {'mean_absolute_effect':list(range(1,36)),
            'signed_effects':[list(range(1,36)),list(range(1,36))], 'baseline_predictions':[10.,20.],
            'fractions':[0.,1.], 'deletion_predictions':[[10.,20.],[0.,0.]],
            'insertion_predictions':[[0.,0.],[10.,20.]],
            'random_deletion_predictions':[[10.,20.],[0.,0.]],
            'random_insertion_predictions':[[0.,0.],[10.,20.]],
            'input_scope':'standardized_D7_coordinates_training_reference'}
    return item


def csv_rows(path):
    with path.open() as stream: return list(csv.DictReader(stream))


class PublishEvidenceTests(unittest.TestCase):
    def test_partial_actual_receipts_export_profiles_and_fidelity_without_fake_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir()
            record(source/'robustness/predictors/first',member(explain=True))
            record(source/'robustness/predictors/second',member(271802,explain=True))
            before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            result=export.export_evidence(source,root/'static')
            self.assertEqual(result['export_status'],'partial_evidence')
            self.assertEqual(result['diagnostics']['verified_completed_predictors'],2)
            self.assertFalse(result['diagnostics']['completed'])
            self.assertEqual(result['reproduction']['verified_completed_fits'],0)
            self.assertEqual(result['ablations']['verified_completed_fits'],0)
            self.assertEqual(csv_rows(root/'static/robustness.csv'),[])
            self.assertEqual(len(csv_rows(root/'static/explanation_cells.csv')),70)
            stability=csv_rows(root/'static/explanation_stability.csv')[0]
            self.assertEqual(stability['R_available'],'2')
            self.assertEqual(stability['all_planned_seeds_available'],'False')
            self.assertAlmostEqual(float(stability['mean_spearman']),1.)
            fidelity=csv_rows(root/'static/fidelity_prediction_response.csv')
            self.assertTrue(all(row['RMSE']=='' for row in fidelity))
            self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError,'NEW'):
                export.export_evidence(source,root/'static')

    def test_orphan_and_missing_claimed_artifact_cannot_establish_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir()
            write_json(source/'orphan/diagnostics.json',member())
            folder=source/'incomplete'; receipt=record(folder,member())
            receipt['artifacts_sha256']['started.json']='b'*64
            write_json(folder/'result.json',receipt)
            status=export.export_evidence(source,root/'static')
            self.assertEqual(status['diagnostics']['verified_completed_predictors'],0)
            self.assertEqual(status['nonadmitted_receipts'][0]['missing_artifacts'],['started.json'])
            self.assertEqual(len(status['orphan_diagnostics_without_result_receipt']),1)
            with self.assertRaisesRegex(ValueError,'Complete pipeline receipts'):
                export.export_evidence(source,root/'complete',require_complete=True)
            self.assertFalse((root/'complete').exists())

    def test_modified_bytes_and_path_escape_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; source.mkdir()
            folder=source/'predictor';receipt=record(folder,member())
            (folder/'diagnostics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                export.export_evidence(source,root/'static')
            self.assertFalse((root/'static').exists())
            receipt['artifacts_sha256']={'../outside.json':'a'*64};write_json(folder/'result.json',receipt)
            with self.assertRaisesRegex(ValueError,'escapes'):
                export.export_evidence(source,root/'static')

    def test_restored_github_manifest_binds_receipt_and_exact_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            record(source/'runs/first',member(),remote=True)
            status=export.export_evidence(source,root/'static')
            self.assertEqual(status['diagnostics']['verified_completed_predictors'],1)
            evidence=json.loads((root/'static/source_evidence.json').read_bytes())
            self.assertTrue(evidence['verified_source_receipts'][0]['remote_manifest']['manifest_sha256'])
            path=source/'runs/first/manifest.json'; value=json.loads(path.read_bytes());value['receipt_sha256']='0'*64;write_json(path,value)
            with self.assertRaisesRegex(ValueError,'does not bind'):
                export.export_evidence(source,root/'bad')

    def test_root_summary_metrics_are_exported_as_recorded_but_missing_members_stay_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();record(source/'robustness/predictors/first',member())
            summary={'run_id':'frozen-actual-run','predictor_records':79,'all_baseline_projections_passed':True,
                'robustness':[{'model':'GRU32','R':6,'kind':'noise','severity':.1,'RMSE_degradation':.001,
                               'paired_date_block_CI95':[.0001,.002]}],
                'normalized_degradation_areas':[],'interpretability':[]}
            config={'version':export.DIAGNOSTIC_VERSION,'configuration_commit':'c'*40}
            run_id='posthoc-'+export.digest(export.canonical(config))[:24]
            value=member();value['run_id']=run_id
            # Replace the synthetic receipt with the actual registration-bound identity.
            record(source/'robustness/predictors/first',value)
            summary['run_id']=run_id
            write_json(source/'robustness/summary.json',summary)
            write_json(source/'robustness/registration.json',config)
            (source/'robustness/robustness.csv').write_text('model,R\nGRU32,6\n')
            write_json(source/'robustness/result.json',{'version':export.DIAGNOSTIC_VERSION,'status':'completed',
                'attempt_id':'root','run_id':run_id,'artifacts_sha256':{
                    name:export.file_hash(source/'robustness'/name) for name in ('summary.json','registration.json','robustness.csv')}})
            status=export.export_evidence(source,root/'static')
            self.assertFalse(status['diagnostics']['completed'])
            self.assertEqual(status['diagnostics']['verified_completed_predictors'],1)
            self.assertEqual(csv_rows(root/'static/robustness.csv')[0]['RMSE_degradation'],'0.001')

    def test_reproduction_and_ablation_require_actual_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            for kind,condition in (('reproduction','full'),('ablation','remove_second_encoder')):
                folder=source/'reproduction/jobs'/kind;write_json(folder/'test_predictions.json',{'prediction_original':[.01]*608})
                receipt={'version':export.FIT_VERSION,'status':'completed','kind':kind,'condition':condition,
                    'model':export.SAE,'seed':271801,'attempt_id':kind,'configuration_commit':'c'*40,
                    'scientific_commit':export.ENGINE_COMMIT,'original_freeze_unchanged':True,
                    'new_winner_selection_performed':False,
                    'metadata':{'clean_reproduction_passed':True,'fresh_fit':True},'rmse_original':.003,
                    'artifacts_sha256':{'test_predictions.json':export.file_hash(folder/'test_predictions.json')}}
                write_json(folder/'result.json',receipt)
            status=export.export_evidence(source,root/'static')
            self.assertEqual(status['reproduction']['verified_completed_fits'],1)
            self.assertFalse(status['reproduction']['all_81_recorded_checks_passed'])
            self.assertEqual(status['ablations']['verified_completed_fits'],1)
            self.assertEqual(csv_rows(root/'static/ablation_comparisons.csv'),[])

    def test_same_source_reproduces_every_exported_byte_without_pipeline_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();record(source/'first',member(explain=True))
            export.export_evidence(source,root/'one')
            original_import=__import__
            def guard(name,*args,**kwargs):
                if name in ('torch','e2_frozen_posthoc_gpu','e2_posthoc_fit_reproduction'):
                    raise AssertionError('Documentary export imported a model execution module')
                return original_import(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=guard):
                export.export_evidence(source,root/'two')
            a={str(p.relative_to(root/'one')):p.read_bytes() for p in (root/'one').rglob('*') if p.is_file()}
            b={str(p.relative_to(root/'two')):p.read_bytes() for p in (root/'two').rglob('*') if p.is_file()}
            self.assertEqual(a,b)

    def test_full_completion_requires_all_79_diagnostics_99_fits_and_both_root_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            config={'version':export.DIAGNOSTIC_VERSION,'configuration_commit':'c'*40}
            run_id='posthoc-'+export.digest(export.canonical(config))[:24]
            for index,(model,seed) in enumerate(sorted(export.EXPECTED_DIAGNOSTICS,key=lambda x:(x[0],str(x[1])))):
                value=member(seed,model=model);value['run_id']=run_id
                if model=='ARIMA': value.update(perturbations=[],profile=None)
                record(source/'robustness/predictors'/str(index),value)
            summary={'run_id':run_id,'predictor_records':79,'all_baseline_projections_passed':True,
                'robustness':[{'model':model,'R':6,'kind':kind,'severity':level,'block_length':block}
                    for model in export.NEURAL for kind,levels in export.SEVERITIES.items() for level in levels for block in (7,14,28)]}
            write_json(source/'robustness/summary.json',summary)
            write_json(source/'robustness/registration.json',config)
            (source/'robustness/robustness.csv').write_text('recorded_producer_table\n')
            write_json(source/'robustness/result.json',{'version':export.DIAGNOSTIC_VERSION,'status':'completed',
                'attempt_id':run_id,'run_id':run_id,'artifacts_sha256':{
                    name:export.file_hash(source/'robustness'/name) for name in ('summary.json','registration.json','robustness.csv')}})
            planned=[('reproduction',model,seed,'full') for model,seed in export.EXPECTED_REPRODUCTION]
            planned += [('ablation',export.SAE,seed,condition) for condition,seed in export.EXPECTED_ABLATIONS]
            for index,(kind,model,seed,condition) in enumerate(planned):
                folder=source/'reproduction/jobs'/str(index)
                write_json(folder/'test_predictions.json',{'prediction_original':[.01]*608})
                write_json(folder/'result.json',{'version':export.FIT_VERSION,'status':'completed','kind':kind,
                    'model':model,'seed':seed,'condition':condition,'attempt_id':str(index),'configuration_commit':'c'*40,
                    'scientific_commit':export.ENGINE_COMMIT,'original_freeze_unchanged':True,'new_winner_selection_performed':False,
                    'metadata':{'fresh_fit':True,'clean_reproduction_passed':True},
                    'artifacts_sha256':{'test_predictions.json':export.file_hash(folder/'test_predictions.json')}})
            # All leaves alone are insufficient evidence that the aggregate stage closed.
            with self.assertRaisesRegex(ValueError,'Complete pipeline receipts'):
                export.export_evidence(source,root/'without_root',require_complete=True)
            folder=source/'reproduction/summary-actual'
            write_json(folder/'summary.json',{'status':'completed','completed_reproduction_fits':81,'completed_ablation_fits':18})
            write_json(folder/'receipt.json',{'version':export.FIT_VERSION,'status':'completed','attempt_id':'fit-summary',
                'artifacts_sha256':{'summary.json':export.file_hash(folder/'summary.json')}})
            result=export.export_evidence(source,root/'complete',require_complete=True)
            self.assertEqual(result['export_status'],'completed_evidence')
            self.assertTrue(result['diagnostics']['completed'])
            self.assertTrue(result['reproduction']['all_81_recorded_checks_passed'])
            self.assertTrue(result['ablations']['completed'])


if __name__=='__main__': unittest.main()
