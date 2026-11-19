"""Synthetic orchestration and real temporal-control/data contracts; no market fits."""
import csv
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
def module(name):
    s=importlib.util.spec_from_file_location('test_extended_eval_'+name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module('data');c=module('classical');r=module('run')

def panel(dataset='D01',n=300):
    rng=np.random.default_rng(85)
    close=np.exp(4+np.cumsum(rng.normal(0,.02,n)));op=close*np.exp(rng.normal(0,.001,n))
    raw=np.column_stack([op,np.maximum(op,close)*1.01,np.minimum(op,close)*.99,close,rng.integers(100,1000,n)])
    days=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-01-01')+n).astype('U10')
    a=d.old_geometry().windows(raw);geometry=d.old_geometry().propose_split_rows(n)
    pos={k:np.arange(v['window_start'],v['window_stop_exclusive']) for k,v in geometry['parts'].items()}
    wd=np.stack([days[t-7:t] for t in range(7,n)])
    return d.package_panel(a['raw_X'],a['y'],a['A'],a['U'],days[7:],pos,dataset=dataset,raw_sha='b'*64,
        window_dates=wd,history=dict(zip(days[7:].tolist(),a['y'].tolist())),
        geometry={**geometry,'clock':'UTC_daily','units':'native'})

def proto():
    return {'schema_version':1,'kind':'e3_extended_evaluation_protocol','source_commit':'a'*40,
        'family':list(r.FAMILY),'candidate':r.common.LEGACY,'seeds':list(range(10,20)),'hpo_seed':9,
        'datasets':{k:'b'*64 for k in r.DATASETS},'split':{'ratios':[.55,.15,.1],'purge':7},
        'training':{'lr_grid':[.0003,.001,.003],'max_epochs':1,'patience':1,'batch_size':8,
                    'gradient_clip':1.,'per_fit_seconds':10.},'evidence_mode':'prespecified_exploratory_audit'}

class DataContracts(unittest.TestCase):
    def test_purges_separate_all_eight_close_supports(self):
        data=panel()
        for left,right in zip(('fit','validation','calibration'),('validation','calibration','test')):
            self.assertLess(int(data['positions'][left][-1])+7,int(data['positions'][right][0]))
        self.assertTrue(np.allclose(data['all_y']**2,data['all_A']+(6/49)*data['all_U']**2,atol=1e-15))

    def test_fit_statistics_unchanged_by_reserved_rows_and_curve_scaler_refitted(self):
        a=panel();b=panel();b['raw_X'][b['positions']['test'],:,4]*=1e6
        b=d.package_panel(b['raw_X'],b['all_y'],b['all_A'],b['all_U'],b['dates'],b['positions'],dataset='D01',raw_sha='b'*64,
            window_dates=b['window_dates'],history=b['history'],geometry=b['geometry'])
        self.assertEqual(a['scaler'],b['scaler'])
        reduced=r._curve_data(a,.25)
        self.assertEqual(len(reduced['scaler']['fit_positions']),int(len(a['train_X'])*.25))
        self.assertNotEqual(reduced['scaler']['mean'],a['scaler']['mean'])
        self.assertTrue(np.array_equal(reduced['test_y_original'],a['test_y_original']))

    def test_fit_diagnostic_mask_is_fixed_availability_for_every_model(self):
        a=panel();missing=str(a['dates'][40]);a['history'].pop(missing)
        indexes,excluded=c.fit_diagnostic_mask(a)
        excluded_dates={row['date'] for row in excluded}
        self.assertIn(str(a['dates'][41]),excluded_dates)
        self.assertIn(str(a['dates'][47]),excluded_dates)
        self.assertNotIn(41,indexes)
        self.assertEqual(len(a['train_X']),len(a['positions']['fit']))

    def test_annotated_freeze_binding_and_all19_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'protocol.json';f=Path(tmp)/'freeze.json';p.write_text(json.dumps(proto()))
            receipt={'verified':True,'source_commit':'a'*40,'protocol_sha256':r.sha(p),'annotated_tag':'fixture-annotated','tag_object_sha':'c'*40}
            f.write_text(json.dumps(receipt));r.protocol(p,f,'a'*40)
            changed=proto();changed['family']=changed['family'][:-1];p.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'family'):r.protocol(p,f,'a'*40)

class TemporalContracts(unittest.TestCase):
    def test_session_seasonal_lag_is_seven_observations_not_calendar_days(self):
        a=panel();a['geometry']['clock']='session'
        a['history']={day:v for i,(day,v) in enumerate(a['history'].items()) if i%7 not in (5,6)}
        dates=np.asarray(sorted(a['history']));a['dates']=dates
        a['positions']={'fit':np.arange(100),'validation':np.arange(107,125),'calibration':np.arange(132,145),'test':np.arange(152,170)}
        a['train_dates']=dates[:100];a['val_dates']=dates[107:125];a['calibration_dates']=dates[132:145];a['test_dates']=dates[152:170]
        with tempfile.TemporaryDirectory() as tmp:
            values,meta=c.predictions('SeasonalNaive',a,tmp,source_commit='a'*40,request_sha256='b'*64)
        target=str(a['test_dates'][0]);i=dates.tolist().index(target)
        self.assertEqual(values[target],a['history'][str(dates[i-7])]);self.assertEqual(meta['lag_observations'],7)

    def test_real_ARIMA_keeps_missing_Kalman_days_and_replays_without_refit(self):
        a=panel(n=400);rng=np.random.default_rng(49);v=2.
        values=[]
        for i in range(len(a['dates'])):
            v=2+.6*(v-2)+rng.normal(0,.04);values.append(v)
        a['history']=dict(zip(a['dates'].tolist(),values));gap=list(a['history'])[80:88]
        for day in gap:a['history'].pop(day)
        with tempfile.TemporaryDirectory() as tmp:
            first,meta=c.predictions('ARIMA',a,tmp,source_commit='a'*40,request_sha256='b'*64)
            second,again=c.predictions('ARIMA',a,tmp,source_commit='a'*40,request_sha256='b'*64)
            self.assertEqual(first,second)
            self.assertEqual(len(meta['missing_positions']),8)
            self.assertEqual(meta['fit_labels']-meta['fit_observed_labels'],8)
            log=[json.loads(line) for line in (Path(tmp)/'causal_forecasts.log').read_text().splitlines()]
            self.assertEqual(len({row['date'] for row in log}),len(log))
            self.assertTrue(all(row['written_before_observation'] for row in log))

    def test_missing_required_lag_aborts_instead_of_zero_or_date_removal(self):
        a=panel();day=str(a['val_dates'][0]);axis=c._axis(a);a['history'].pop(axis[axis.index(day)-1])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Unavailable'):c.predictions('Persistence',a,tmp,source_commit='a'*40,request_sha256='b'*64)

    def test_ARIMA_budget_exhaustion_does_not_create_a_replacement_fit(self):
        a=panel(n=400)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(c.time,'perf_counter',side_effect=[0.,2.]):
            with self.assertRaisesRegex(RuntimeError,'budget expired'):
                c.predictions('ARIMA',a,tmp,source_commit='a'*40,request_sha256='b'*64,per_fit_seconds=1.)
            self.assertTrue((Path(tmp)/'arima_fit_started.json').is_file())
            self.assertFalse((Path(tmp)/'arima_fit.zip').exists())
            self.assertFalse((Path(tmp)/'causal_forecasts.log').exists())

class PanelContracts(unittest.TestCase):
    def test_classical_raw_noise_invariant_but_common_projection_can_change(self):
        data=panel();perturb=r.load(HERE.parent/'robustness/perturb.py','test_panel_perturb')
        profile=r._profile(data,perturb)
        values={str(day):0. for day in data['dates']}
        def frozen(raw,indexes):
            _,A=r.fit_api.transform(raw,data['scaler']);floor=np.sqrt(A);rawp=np.zeros(len(indexes))
            return rawp,np.maximum(rawp,floor),floor
        rows,_=r._panel_rows(data,'Persistence',-1,frozen,perturb,profile)
        test=[row for row in rows if row['split']=='test']
        self.assertEqual({row['scenario'] for row in test},{'unperturbed',*[f'ohlcv_observation_noise_{i}' for i in (1,2,3)]})
        self.assertTrue(all(row['target']==float(data['all_y'][data['dates'].tolist().index(row['date'])]) for row in test))
        self.assertTrue(all(row['raw_prediction']==0 for row in test))
        self.assertTrue(all(row['prediction']==max(row['raw_prediction'],row['known_floor']) for row in test))
        base=[row['prediction'] for row in test if row['scenario']=='unperturbed'];stress=[row['prediction'] for row in test if row['scenario']=='ohlcv_observation_noise_3']
        self.assertFalse(np.array_equal(base,stress))

    def test_full_coordinator_19_systems_4_domains_shards_and_reuses_jobs(self):
        """Model workers are mocked here; phase/factory fits have their own tests."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'protocol.json';f=root/'freeze.json';p.write_text(json.dumps(proto()))
            f.write_text(json.dumps({'verified':True,'source_commit':'a'*40,'protocol_sha256':r.sha(p),
                'annotated_tag':'fixture-annotated','tag_object_sha':'c'*40}))
            fits=[]
            def fake_fit(name,seed,data,directory,config,device,rows,roots,identity):
                fits.append((name,seed,identity['kind'],identity['dataset']))
                model={'name':name,'seed':seed};directory.mkdir(parents=True,exist_ok=True)
                (directory/'selected_model.pt').write_text(json.dumps(model))
                (directory/'fit_request.json').write_text('{}')
                phase={'measured_completed_epoch_seconds':.01,'optimizer_steps':1,'budget_seconds':10.,
                    'deadline_overrun_seconds':0.,'stop_reason':'fixture',
                    'active_compute_lost_in_an_unclosed_crash_segment_is_unknown':False}
                meta={'validation_projected_RMSE':abs(config['learning_rate']-.001)+.01,
                    'optimizer_steps':1,'selected_model_sha256':r.sha(directory/'selected_model.pt'),
                    'parameters':1,'phase_resources':{'supervised':phase},'peak_GPU_allocated_bytes':None}
                (directory/'fit_metadata.json').write_text(json.dumps(meta))
                return model,meta
            def restore(name,path,data,device,rows,roots):
                return json.loads(path.read_text()),json.loads(path.with_name('fit_metadata.json').read_text())
            def predict(model,name,raw,scaler,device):
                _,A=r.fit_api.transform(raw,scaler);floor=np.sqrt(A)
                y=floor+scaler['d_fit']*(1+model['seed']*.00001)
                return y,np.maximum(y,floor),floor
            def controls(name,data,directory,**kwargs):
                dates=set(str(d) for split in ('val','calibration','test') for d in data[split+'_dates'])
                positions,_=c.fit_diagnostic_mask(data);dates|={str(data['dates'][i]) for i in positions}
                return {day:data['history'][day]*.9 for day in dates},{'real_data_fits':0,'training_seconds':0.,'parameters':0}
            out=root/'run'
            with patch.object(r.common,'_package',return_value=(root,{},None,None)), \
                 patch.object(r.data_api,'btc',side_effect=lambda *a,**k:panel('D00',200)), \
                 patch.object(r.data_api,'additional',side_effect=lambda a,dataset,*rest,**k:panel(dataset,200)), \
                 patch.object(r.fit_api,'fit',side_effect=fake_fit), \
                 patch.object(r.fit_api,'load_selected',side_effect=restore), \
                 patch.object(r.fit_api,'predict',side_effect=predict), \
                 patch.object(r.classical,'predictions',side_effect=controls),redirect_stdout(io.StringIO()):
                first=r.run(out,'a'*40,root/'admission.json',p,f,root/'unused.npz',device='cpu',allow_cpu_for_synthetic_tests=True)
                count=len(fits)
                second=r.run(out,'a'*40,root/'admission.json',p,f,root/'unused.npz',device='cpu',allow_cpu_for_synthetic_tests=True)
            self.assertEqual(first['status'],'completed',first['failures'])
            self.assertEqual(second['status'],'completed',second['failures'])
            self.assertEqual(first['additional_datasets_evaluated'],3)
            self.assertEqual(first['new_real_data_fits'],0)
            self.assertEqual(len(fits),count,'A retry must load selected jobs rather than refit')
            manifest=json.loads((out/'panel_manifest.json').read_text())
            self.assertEqual(len(manifest['files']),4*19)
            self.assertFalse((out/'predictions.csv').exists())
            self.assertTrue(all(item['bytes']<48*1024**2 for item in manifest['files']))
            index=json.loads((out/'sample_curve_jobs.json').read_text())
            self.assertEqual(len(index),4*2*3*3)
            self.assertEqual(sum(entry['actual_new_fit'] for entry in index),48)
            selection=json.loads((out/'HPOselection.json').read_text())
            self.assertTrue(all(v['trial_index']==1 for models in selection.values() for v in models.values()))
            with (out/'HPO_sensitivity.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),4*14*3*len(panel('D00',200)['test_dates']))
            # An acknowledged job must retain its seed/identity, not merely a
            # global audit hash that also belongs to the other seed replicas.
            first_job=out/'jobs'/'D00'/(r.slug(r.common.NEURAL[0])+'__hpo_lr0__9')/'result.json'
            record=json.loads(first_job.read_bytes());record['seed']+=100;first_job.write_text(json.dumps(record))
            name=record['model'];kind=record['kind'];dataset=record['dataset']
            with self.assertRaisesRegex(ValueError,'identity or frozen bytes'):
                r._job(out,dataset,name,kind,record['seed']-100,
                    json.loads((out/'request.json').read_bytes()),panel(dataset,200),
                    {},'cpu',None,None)

if __name__=='__main__':unittest.main()
