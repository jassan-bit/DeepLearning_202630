import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_reduced_final as final
import e2_reduced_final_analysis as analysis
from e2_temporal_training import DatedPanel
from e2_published_registry import registered_trials


def panel():
    dates=pd.date_range('2023-01-01','2024-12-30',tz='UTC')
    v=np.linspace(100,160,len(dates))
    x=np.column_stack((v,v+2,v-2,v+1,np.ones(len(dates))*1000))
    return DatedPanel(dates,x,np.linspace(.01,.025,len(dates)))


def selection(name='FCN-1D'):
    trial=registered_trials(name)[5]
    return {'model_id':name,'all_eighteen_slots_accounted_for':True,'run_artifact_bytes_verified':True,
        'source_current_recipe_verified':True,'selection_ledger_sha256':'a'*64,'winner_trial':trial,
        'winner_parent_trial_id':trial['trial_id'],'fixed_phase_epochs':{'supervised':2}}


class FinalExecutorTests(unittest.TestCase):
    def test_calibration_changes_cannot_change_final_fit_or_scaler(self):
        a=panel();x=a.features;y=a.targets
        x[a.calendar>pd.Timestamp('2024-10-31',tz='UTC')]*=2
        y[a.calendar>pd.Timestamp('2024-10-31',tz='UTC')]=999
        b=DatedPanel(a.calendar,x,y)
        first,second=final.final_training_split(a),final.final_training_split(b)
        self.assertEqual(first.signature,second.signature)
        self.assertEqual(first.dates[-1].strftime('%Y-%m-%d'),'2024-10-31')
        self.assertEqual(first.scaler.record(),second.scaler.record())

    def test_final_refit_never_receives_calibration_validation(self):
        data=panel();calls=[]
        class Fake(nn.Module):
            def __init__(self):super().__init__();self.linear=nn.Linear(35,1)
            def forward(self,x):return self.linear(x.flatten(1)).squeeze(-1)
        def fit(name,trial,seed,train,validation,output,**kwargs):
            calls.append((train,validation,kwargs))
            output=Path(output);output.mkdir(parents=True)
            model=Fake();p=output/'selected_model.pt'
            torch.save({'state_dict':model.state_dict(),'scaler':train.scaler.record()},p)
            return {'selected_checkpoint':{'path':p.name,'sha256':final.file_sha(p)},
                'optimizer_steps':7,'phase_records':{},'architecture':{}}
        with tempfile.TemporaryDirectory() as temp, mock.patch('e2_development_data.load_development',return_value=(data,{})), \
                mock.patch('e2_reduced_fit.fit_reduced_job',side_effect=fit), \
                mock.patch.object(final,'_trial_builder',side_effect=lambda *a:Fake()), \
                mock.patch('e2_published_pilot_training.predict_pilot',return_value=torch.ones(53)):
            spec={'model':'FCN-1D','seed':271801,'selection':selection(),'data_root':'unused',
                'source_identity':{},'runtime_identity':{},'fit_wall_seconds':600}
            result=final.neural_final_worker(spec,temp)
            self.assertIsNone(calls[0][1])
            self.assertEqual(calls[0][2]['fixed_epochs'],{'supervised':2})
            self.assertEqual(result['calibration_dates'],final.CALIBRATION_DATES)
            calibration=final.read_json(Path(temp)/'calibrator.json')
            self.assertEqual(calibration['predictor_sha256'],result['checkpoint_sha256'])
            self.assertTrue(result['no_refit_after_calibration'])

    def test_unverified_selection_and_changed_recipe_are_rejected(self):
        item=selection();item['run_artifact_bytes_verified']=False
        with self.assertRaises(ValueError):final.validate_selection('FCN-1D',item)
        item=selection();item['winner_trial']=dict(item['winner_trial'],learning_rate=.9)
        with self.assertRaises(ValueError):final.validate_selection('FCN-1D',item)

    def test_backup_failure_stops_before_any_second_fit(self):
        calls=[]
        class Budget:stage_wall_seconds=600
        def supervisor(command,budget,logical_id,output,spec_sha,**kwargs):
            calls.append(logical_id)
            return {'result':{'status':'completed'}}
        def backup(*args):raise RuntimeError('backup unavailable')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError,'backup unavailable'):
                final.run_final_fits('unused',{'FCN-1D':selection()},temp,Budget(),
                    source_identity={},runtime_identity={},models=['FCN-1D'],persist_result=backup,
                    supervisor=supervisor)
            self.assertEqual(len(calls),1)

    def test_known_daily_network_is_unreachable_before_full_freeze(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch('e2_reduced_final_data.validate_final_freeze',side_effect=ValueError('not frozen')), \
                mock.patch('urllib.request.urlopen') as download:
            with self.assertRaisesRegex(ValueError,'not frozen'):
                final._materialize_known_daily(temp,temp,{},source_commit='a'*40)
            download.assert_not_called()

    def test_complete_freeze_requires_real_bound_calibrators_and81_predictors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            final.write_json(root/'pilot_power_MDE.json',{'status':'development_pilot_completed_power_reported'})
            schedule=[(name,seed,root/'final-fit'/final.slug(name)/str(seed)) for name in final.NEURAL for seed in final.FINAL_SEEDS]
            schedule.extend((name,'deterministic',root/'classical-final'/final.slug(name)) for name in final.CLASSICAL)
            for name,seed,directory in schedule:
                directory.mkdir(parents=True);(directory/'weights.bin').write_bytes(f'{name}-{seed}'.encode())
                final.write_json(directory/'scaler.json',{'fit_cutoff':'2024-10-31'})
                calibrator=analysis.residual_calibrator(np.ones(53),np.ones(53),dates=final.CALIBRATION_DATES,
                    predictor_sha256=final.file_sha(directory/'weights.bin'))
                final.write_json(directory/'calibrator.json',calibrator)
                final.write_json(directory/'result.json',{'status':'completed','model':name,'seed':seed,
                    'no_refit_after_calibration':True,'checkpoint':'weights.bin','checkpoint_sha256':final.file_sha(directory/'weights.bin'),
                    'scaler':'scaler.json','scaler_sha256':final.file_sha(directory/'scaler.json'),
                    'calibrator':'calibrator.json','calibrator_sha256':final.file_sha(directory/'calibrator.json')})
            result=final.freeze_final(root)
            self.assertEqual(len(result['models']),81)
            self.assertFalse(result['prospective_blinding_recovered'])
            (schedule[0][2]/'weights.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):final.freeze_final(root)


if __name__=='__main__':unittest.main()
