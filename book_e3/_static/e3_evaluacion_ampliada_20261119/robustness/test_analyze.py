"""Generated fixtures only: alignment, calibration isolation and estimands."""
import csv
from datetime import date,timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('extended_robustness_analysis',HERE/'analyze.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture(folder):
    family=['candidate','rival','Persistence','floor_only','floor_plus_mean_excess']
    protocol={'kind':'e3_extended_evaluation_protocol','mode':'synthetic_contract',
        'family':family,'candidate':'candidate','seeds':[11,12],'datasets':{'D01':{}},
        'analysis':{'block_lengths':[3,5,7],'bootstrap_repetitions':199,
                    'scenarios':list(audit.SCENARIOS)}}
    protocol_path=folder/'protocol.json';protocol_path.write_bytes(audit.canonical(protocol))
    rows=[];offset=0
    for split,size in [('fit',12),('validation',12),('calibration',20),('test',42)]:
        for i in range(size):
            label=(date(2020,1,1)+timedelta(days=offset+i)).isoformat();y=.1+.005*np.sin(i*.31)
            for scenario_index,scenario in enumerate(audit.SCENARIOS if split=='test' else ['unperturbed']):
                floor=.05 if scenario_index==0 else .10+scenario_index*.025
                for model in family:
                    seeds=[-1] if model in {'Persistence','floor_only','floor_plus_mean_excess'} else [11,12]
                    for seed in seeds:
                        if model=='candidate':raw=y+.007*np.cos(i*.23+seed)
                        elif model=='rival':raw=y+.025*np.cos(i*.4+seed)
                        elif model=='Persistence':raw=.13
                        elif model=='floor_only':raw=.05
                        else:raw=.07
                        if scenario_index and model not in {'Persistence','floor_only','floor_plus_mean_excess'}:
                            raw+=scenario_index*.002
                        rows.append({'dataset':'D01','split':split,'scenario':scenario,'model':model,
                            'seed':seed,'date':label,'target':float(y),'raw_prediction':float(raw),
                            'known_floor':floor,'prediction':max(raw,floor),
                            'classical_raw_forecast_invariant_by_input_contract':model=='Persistence' and scenario_index>0})
        offset+=size+7
    path=folder/'panel.csv';path.write_bytes(audit.csv_bytes(rows))
    return protocol,protocol_path,path,rows


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.protocol,self.protocol_path,self.panel,self.rows=fixture(self.folder)

    def tearDown(self):self.tmp.cleanup()

    def save(self,rows):self.panel.write_bytes(audit.csv_bytes(rows))

    def test_complete_analysis_uses_individual_rmse_not_ensemble_and_outputs_all_models(self):
        cfg,panels,proof=audit.read_panel(self.panel,self.protocol)
        result=audit.analyze(panels,cfg)
        self.assertEqual(proof['rows'],len(self.rows))
        rows=[r for r in result['model_metrics'] if r['split']=='test' and r['scenario']=='unperturbed']
        self.assertEqual(len(rows),5)
        candidate=panels[('D01','test','unperturbed','candidate')]
        individual=np.sqrt(((candidate['p']-candidate['y'])**2).mean(1)).mean()
        ensemble=np.sqrt(((candidate['p'].mean(0)-candidate['y'])**2).mean())
        reported=next(r['RMSE_mean_individual'] for r in rows if r['model']=='candidate')
        self.assertAlmostEqual(reported,individual)
        self.assertGreater(abs(reported-ensemble),1e-5)
        self.assertTrue(all(r['family_size']==4 for r in result['paired_effects'] if r['scenario']=='unperturbed'))
        self.assertTrue(all(r['family_size']==12 for r in result['noise_contrasts']))

    def test_test_target_change_cannot_change_a_clean_calibrator(self):
        cfg,base,_=audit.read_panel(self.panel,self.protocol)
        before=audit.analyze(base,cfg)['calibrators']
        changed=[dict(r,target=r['target']*8) if r['split']=='test' else dict(r) for r in self.rows]
        self.save(changed);_,after,_=audit.read_panel(self.panel,self.protocol)
        self.assertEqual(before,audit.analyze(after,cfg)['calibrators'])
        self.assertTrue(all(r['fit_only_on_clean_calibration'] for r in before))

    def test_missing_seed_date_and_changed_scenario_truth_are_rejected(self):
        self.save(self.rows[:-1])
        with self.assertRaisesRegex(ValueError,'Incomplete'):audit.read_panel(self.panel,self.protocol)
        changed=[dict(r) for r in self.rows];changed[-1]['target']+=.001;self.save(changed)
        with self.assertRaisesRegex(ValueError,'Targets differ'):audit.read_panel(self.panel,self.protocol)

    def test_calibration_noise_and_duplicate_predictions_are_rejected(self):
        rows=[dict(r) for r in self.rows];rows[0]['scenario']='ohlcv_observation_noise_1';self.save(rows)
        with self.assertRaisesRegex(ValueError,'held-out test'):audit.read_panel(self.panel,self.protocol)
        self.save([*self.rows,self.rows[0]])
        with self.assertRaisesRegex(ValueError,'Duplicate'):audit.read_panel(self.panel,self.protocol)

    def test_raw_classical_invariance_and_common_projection_are_separate(self):
        _,panels,_=audit.read_panel(self.panel,self.protocol)
        a=panels[('D01','test','unperturbed','Persistence')]
        b=panels[('D01','test','ohlcv_observation_noise_3','Persistence')]
        self.assertTrue(np.array_equal(a['raw_p'],b['raw_p']))
        self.assertFalse(np.array_equal(a['p'],b['p']))
        rows=[dict(r) for r in self.rows]
        chosen=next(r for r in rows if r['model']=='Persistence' and r['scenario']=='ohlcv_observation_noise_3')
        chosen['raw_prediction']+=.001;self.save(rows)
        with self.assertRaisesRegex(ValueError,'invariant classical raw'):audit.read_panel(self.panel,self.protocol)

    def test_finite_sample_calibration_rank_and_constant_target_r2(self):
        q,rank=audit.quantile(np.zeros(9),np.arange(9),.1)
        self.assertEqual((q,rank),(8.,9))
        with self.assertRaisesRegex(ValueError,'too small'):audit.quantile(np.zeros(3),np.ones(3),.1)
        self.assertIsNone(audit.metrics(np.ones(3),np.ones(3))['R2'])

    def test_all_hpo_trials_are_described_without_reselecting_on_test(self):
        cfg,panels,_=audit.read_panel(self.panel,self.protocol)
        rows=[]
        for model in ['candidate','rival']:
            base=panels[('D01','test','unperturbed',model)]
            for trial,rate in enumerate([.0003,.001,.003]):
                for date,target in zip(base['dates'],base['y']):
                    rows.append({'dataset':'D01','model':model,'seed':20261117,
                        'split':'test','scenario':'unperturbed','date':date,'target':target,
                        'prediction':float(target)+trial*.01,'hpo_trial':trial,
                        'learning_rate':rate,'validation_RMSE':[.03,.01,.02][trial],
                        'selected_by_validation':trial==1})
        path=self.folder/'hpo.csv';path.write_bytes(audit.csv_bytes(rows))
        receipt,metrics=audit.hpo_sensitivity(path,panels,cfg,self.protocol)
        self.assertEqual(receipt['groups'],6)
        self.assertTrue(all(r['selected_by_validation']==(r['hpo_trial']==1) for r in metrics))
        # Trial zero deliberately has lower TEST RMSE; it must not replace
        # the prespecified winner of validation.
        self.assertTrue(all(r['RMSE']==0 for r in metrics if r['hpo_trial']==0))
        wrong=[dict(r,selected_by_validation=r['hpo_trial']==0) for r in rows]
        path.write_bytes(audit.csv_bytes(wrong))
        with self.assertRaisesRegex(ValueError,'validation argmin'):
            audit.hpo_sensitivity(path,panels,cfg,self.protocol)

    def test_hash_binding_refuses_a_different_forecast_panel(self):
        _,_,proof=audit.read_panel(self.panel,self.protocol)
        manifest={'protocol_sha256':audit.digest(self.protocol_path.read_bytes()),'panel_sha256':proof['sha256']}
        self.assertEqual(audit.bindings(self.protocol_path,proof,manifest)['status'],'hash_bindings_checked')
        manifest['panel_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'panel hash'):audit.bindings(self.protocol_path,proof,manifest)

    def test_sharded_manifest_matches_flat_panel_and_rejects_missing_or_corrupt_payload(self):
        records=[]
        for index,model in enumerate(self.protocol['family']):
            rows=[r for r in self.rows if r['model']==model]
            path=self.folder/'panels'/f'model{index}.csv';path.parent.mkdir(exist_ok=True)
            raw=audit.csv_bytes(rows);path.write_bytes(raw)
            records.append({'path':path.relative_to(self.folder).as_posix(),'sha256':audit.digest(raw),
                'bytes':len(raw),'rows':len(rows),'dataset':'D01','model':model,
                'observed_seeds':sorted({r['seed'] for r in rows})})
        manifest={'kind':'e3_extended_panel_manifest','schema_version':1,
            'family':self.protocol['family'],'seeds':self.protocol['seeds'],
            'files':records,'total_rows':len(self.rows),'single_flat_CSV_materialized':False}
        path=self.folder/'panel_manifest.json';path.write_bytes(audit.canonical(manifest))
        _,flat,_=audit.read_panel(self.panel,self.protocol)
        _,sharded,proof=audit.read_panel(path,self.protocol)
        self.assertEqual(proof['rows'],len(self.rows))
        self.assertTrue(proof['all_shard_hashes_verified'])
        for key in flat:self.assertTrue(np.array_equal(flat[key]['p'],sharded[key]['p']))
        target=self.folder/records[0]['path'];before=target.read_bytes();target.write_bytes(before+b'corrupt')
        with self.assertRaisesRegex(ValueError,'bytes differ'):audit.read_panel(path,self.protocol)
        target.write_bytes(before)
        manifest['files']=records[1:];path.write_bytes(audit.canonical(manifest))
        with self.assertRaisesRegex(ValueError,'Incomplete'):audit.read_panel(path,self.protocol)

    def test_real_audit_requires_nineteen_systems_four_datasets_and_ten_seeds(self):
        protocol=dict(self.protocol);protocol.pop('mode')
        with self.assertRaisesRegex(ValueError,'17 historical'):audit.spec(protocol)
        protocol.update(family=list(audit.FAMILY),candidate=audit.CANDIDATE,
                        datasets={d:{} for d in ('D00','D01','D02','D03')})
        protocol['analysis']={}
        with self.assertRaisesRegex(ValueError,'ten neural'):audit.spec(protocol)
        # The worker places the candidate after its sixteen historical rivals;
        # this valid protocol order must not be mistaken for a missing family.
        family=[m for m in audit.FAMILY if m!=audit.CANDIDATE]
        family.insert(16,audit.CANDIDATE)
        protocol.update(family=family,seeds=list(range(10)))
        self.assertEqual(audit.spec(protocol)['family'],family)

    def test_full_runner_never_declares_scientific_or_unobserved_robustness_complete(self):
        result=audit.run(self.panel,self.protocol_path,self.folder/'output')
        self.assertFalse(result['R3_accredited']);self.assertFalse(result['robustness_component_complete'])
        self.assertFalse(result['test_targets_used_for_calibration_or_selection'])
        self.assertIsNone(result['CRPS']);self.assertIsNone(result['PIT'])
        self.assertEqual(result['new_model_fits'],0)
        self.assertTrue((self.folder/'output/output_manifest.json').is_file())
        before=(self.folder/'output/analysis_receipt.json').read_bytes()
        with patch.object(audit,'analyze',side_effect=AssertionError('Completed analysis must be reused')):
            again=audit.run(self.panel,self.protocol_path,self.folder/'output')
        self.assertEqual(result,again)
        self.assertEqual((self.folder/'output/analysis_receipt.json').read_bytes(),before)
        (self.folder/'output/model_metrics.csv').write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'failed its manifest'):
            audit.run(self.panel,self.protocol_path,self.folder/'output')

    def curve_fixture(self):
        cfg,panels,_=audit.read_panel(self.panel,self.protocol);cfg['learning_comparator']='rival'
        records=[];rows=[]
        for model in ('candidate','rival'):
            base=panels[('D01','test','unperturbed',model)]
            for j,seed in enumerate(cfg['seeds']):
                for f in (.25,.5,1.):
                    fit_id=f'{model}-{seed}-{f}'
                    receipt=self.folder/(fit_id+'.json');receipt.write_text('{"generated_fixture":true,"steps":10}')
                    records.append({'dataset':'D01','model':model,'seed':seed,'train_fraction':f,
                        'fit_id':fit_id,'training_rows':int(40*f),'full_training_rows':40,
                        'scaler_fit_rows':int(40*f),'optimizer_steps':10,
                        'actual_new_fit':f<1,'reused_base_fit':f==1,
                        'fit_receipt_path':receipt.name,'fit_receipt_sha256':audit.digest(receipt.read_bytes())})
                    for k,label in enumerate(base['dates']):
                        rows.append({'dataset':'D01','split':'test','scenario':'unperturbed','model':model,
                            'seed':seed,'train_fraction':f,'date':label,'target':float(base['y'][k]),
                            'prediction':float(base['p'][j,k]+.01*(1-f))})
        panel=self.folder/'curves.csv';panel.write_bytes(audit.csv_bytes(rows))
        index=self.folder/'curve_index.json';index.write_bytes(audit.canonical({'fits':records}))
        return cfg,panels,panel,index,records

    def test_sample_curves_require_distinct_real_fit_receipts_and_clean_scalers(self):
        cfg,panels,panel,index,records=self.curve_fixture()
        result,metrics=audit.learning_curves(panel,index,panels,cfg)
        self.assertEqual(result['new_fit_groups'],8)
        self.assertEqual(result['base_reused_groups'],4)
        self.assertEqual(len(metrics),12)
        self.assertTrue(result['prefix_calendar_span_confounded'])
        records[0]['fit_id']=records[1]['fit_id'];index.write_bytes(audit.canonical({'fits':records}))
        with self.assertRaisesRegex(ValueError,'relabelled'):audit.learning_curves(panel,index,panels,cfg)

    def test_sample_curves_cannot_reuse_full_data_scaler_or_omit_fits(self):
        cfg,panels,panel,index,records=self.curve_fixture()
        records[0]['scaler_fit_rows']=40;index.write_bytes(audit.canonical({'fits':records}))
        with self.assertRaisesRegex(ValueError,'full-data scaler'):audit.learning_curves(panel,index,panels,cfg)
        index.write_bytes(audit.canonical({'fits':records[1:]}))
        with self.assertRaisesRegex(ValueError,'All predeclared'):audit.learning_curves(panel,index,panels,cfg)


if __name__=='__main__':unittest.main()
