"""Numerical equivalence and no-replay/budget/provenance checks, no market data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import torch
import e2_timesnet_exact_crop as crop
import e2_nas_spectral as spectral
import run_e2_optimized_selection_v2 as optimized

torch.set_num_threads(2)


def cuda_profile():
    rows=[{"width":w,"batch":b,"num_kernels":6,"parameters":10,
        "original_seconds":2.,"cropped_seconds":1.,"speedup":2.,
        "forward_and_gradient_parity":True} for w in (16,32,64,128,256) for b in (16,32,64)]
    return {"version":crop.VERSION,"device":"cuda:0","torch":"2.8.0+cu126",
        "device_name":"test_device","repeats":5,"synthetic_only":True,"market_data_read":False,
        "optimizer_included":False,"end_to_end_training_speedup_measured":False,
        "rows":rows,"fast_cases":[[w,b] for w in (16,32,64,128,256) for b in (16,32,64)],"cuda_gate_passed":True}


class NumericalTests(unittest.TestCase):
    def test_every_grid_and_kernel_count_preserves_outputs_and_all_parameter_gradients(self):
        for shape in ((1,8),(2,4),(4,2)):
            for kernels in (1,2,4,6):
                with self.subTest(shape=shape,kernels=kernels):
                    torch.manual_seed(101)
                    a=spectral._OddKernelInception(3,5,kernels).double()
                    b=deepcopy(a)
                    x=torch.randn(2,3,*shape,dtype=torch.float64,requires_grad=True)
                    y=x.detach().clone().requires_grad_()
                    before=a(x)
                    w,bias,pad=crop.effective_parameters(b,*shape)
                    after=torch.nn.functional.conv2d(y,w,bias,padding=pad)
                    torch.testing.assert_close(after,before,rtol=1e-12,atol=1e-12)
                    grad=torch.randn_like(before)
                    (before*grad).sum().backward();(after*grad).sum().backward()
                    torch.testing.assert_close(y.grad,x.grad,rtol=1e-12,atol=1e-12)
                    for p,q in zip(a.parameters(),b.parameters()):
                        torch.testing.assert_close(q.grad,p.grad,rtol=1e-12,atol=1e-12)

    def test_full_period_blocks_all_topk_and_patterns(self):
        for k in (2,3,4):
            for values in (torch.zeros(3,8,16,dtype=torch.float64),torch.randn(3,8,16,dtype=torch.float64)):
                torch.manual_seed(200+k)
                a=spectral._NASPeriodBlock(16,k,6).double();b=deepcopy(a)
                before=a(values);after=crop.period_forward(b,values)
                torch.testing.assert_close(after,before,rtol=1e-11,atol=1e-11)
                gradient=torch.randn_like(before)
                (before*gradient).sum().backward();(after*gradient).sum().backward()
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(q.grad,p.grad,rtol=1e-9,atol=1e-10)

    def test_complete_model_with_dropout_and_three_adamw_updates(self):
        from e2_nas_common import NASConfig
        for pattern in ((0,0,0),(1,1,1)):
            torch.manual_seed(303)
            a=spectral.TimesNetNAS(NASConfig((16,32,16),pattern),top_k=4,num_kernels=6)
            b=deepcopy(a)
            for block in b.blocks:
                block.forward=types.MethodType(crop.period_forward,block)
            self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
            x=torch.randn(4,7,5);truth=torch.rand(4)
            opt_a=torch.optim.AdamW(a.parameters(),lr=1e-3)
            opt_b=torch.optim.AdamW(b.parameters(),lr=1e-3)
            for step in range(3):
                opt_a.zero_grad();opt_b.zero_grad()
                torch.manual_seed(404+step);before=a(x)
                torch.manual_seed(404+step);after=b(x)
                torch.testing.assert_close(after,before,rtol=3e-5,atol=3e-6)
                ((before-truth)**2).mean().backward();((after-truth)**2).mean().backward()
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(q.grad,p.grad,rtol=3e-4,atol=3e-5)
                opt_a.step();opt_b.step()
            # Floating point trajectories may diverge near tiny gradients;
            # assertions above bound observed forward/gradient differences.

    def test_dispatch_keeps_unselected_width_on_original_function(self):
        original=spectral._NASPeriodBlock.forward
        old_capture,old_widths=crop._ORIGINAL_FORWARD,crop._FAST_CASES
        try:
            a=spectral._NASPeriodBlock(16,4,6)
            x=torch.randn(2,8,16);before=a(x)
            crop.install([(64,32)])
            self.assertTrue(torch.equal(a(x),before))
        finally:
            spectral._NASPeriodBlock.forward=original
            crop._ORIGINAL_FORWARD,crop._FAST_CASES=old_capture,old_widths

    def test_nonstandard_convolution_cannot_use_cropping(self):
        a=spectral._OddKernelInception(3,5,2)
        a.kernels[0].stride=(2,2)
        with self.assertRaisesRegex(ValueError,"unsupported_convolution"):
            crop.effective_parameters(a,1,8)


class PolicyTests(unittest.TestCase):
    def test_previous_optimization_cannot_reset_the_shared_budget(self):
        optimized.require_no_previous_optimization(types.SimpleNamespace(inspect_remote=lambda:None))
        with self.assertRaisesRegex(ValueError,'budget_and_cache_migration'):
            optimized.require_no_previous_optimization(types.SimpleNamespace(inspect_remote=lambda:{'exists':True}))
    def synthetic_plan(self):
        profile=cuda_profile()
        return {'version':optimized.VERSION,'source_commit':optimized.ENGINE,
            'protocol_sha256':'a'*64,'registration_sha256':'b'*64,
            'origins':list(optimized.ORIGINS),'models':['TimesNet'],'k':40,
            'hardware_binding':{'device_name':'test_device'},
            'initial_controls':{'session_budget_seconds':28800.,'campaign_budget_seconds':172700.,
                                'storage_bytes':10*1024**3},
            'remote_storage':{'repository':optimized.REPOSITORY,'branch':optimized.BRANCH,'prefix':optimized.BRANCH},
            'optimization':{'profile':profile,'profile_sha256':optimized.digest(profile),
                            'fast_cases':profile['fast_cases']},
            'legacy_cache':{'commit':'c'*40,'plan_sha256':optimized.LEGACY_PLAN,'completed_jobs':1802,
                            'spent_seconds':100.,'limit_seconds':172800.}}

    def test_schema_binds_original_budget_and_rejects_hidden_origin_cuts(self):
        import e2_github_campaign as public
        optimized.install_public_schema(public)
        plan=self.synthetic_plan();public._validate_plan(plan)
        for mutation in (lambda p:p['initial_controls'].update(campaign_budget_seconds=172800.),
                         lambda p:p.update(origins=['2025-01-01']),
                         lambda p:p['optimization'].update(fast_cases=[])):
            wrong=deepcopy(plan);mutation(wrong)
            with self.assertRaises(ValueError):public._validate_plan(wrong)

    def test_real_ledger_and_public_record_roundtrip_accept_new_closed_schema(self):
        import e2_github_campaign as public
        from e2_search_session import SearchSession
        optimized.install_public_schema(public)
        with tempfile.TemporaryDirectory() as tmp:
            with SearchSession(tmp,self.synthetic_plan(),session_budget_seconds=100,
                    campaign_budget_seconds=172700.,max_storage_bytes=10*1024**3) as session:
                value={'status':'completed','run_id':'f'*64,'resource_use':{'optimizer_steps':4},
                    'predictions_original':[.1],'prediction_dates':['2024-01-01T00:00:00+00:00'],
                    'phase_epochs':{'supervised':1},'reason':None,'fatal':False}
                session.run_job('one',{'seed':1},lambda deadline:value,reservation_seconds=1)
            for path in Path(tmp).rglob('*.json'):
                public.validate_record(path.relative_to(tmp).as_posix(),path.read_bytes())
            report={'version':'v1','plan_sha256':optimized.digest(self.synthetic_plan()),
                    'status':'paused_campaign_budget','reused_jobs':1,'total_available_unique_jobs':1803,
                    'combined_charged_seconds':101.}
            public.validate_record('status.json',optimized.canonical(report))
            report['reused_jobs']=-1
            with self.assertRaises(ValueError):public.validate_record('status.json',optimized.canonical(report))

    def test_admission_clock_preserves_setup_time_without_sleep(self):
        clock=optimized.AdmissionClock(100.)
        with patch.object(optimized.time,'monotonic',return_value=125.):
            self.assertEqual(clock(),100.)
            self.assertEqual(clock(),125.)

    def test_cuda_width_profile_is_complete_and_dispatch_is_measured(self):
        p=cuda_profile();optimized.validate_profile(p)
        for mutation in (lambda p:p.update(device="cpu"),
                         lambda p:p.update(fast_cases=[]),
                         lambda p:p["rows"].pop(),
                         lambda p:p["rows"][0].update(forward_and_gradient_parity=False),
                         lambda p:p["rows"][0].update(cropped_seconds=float('nan'))):
            wrong=deepcopy(p);mutation(wrong)
            with self.assertRaises(ValueError):optimized.validate_profile(wrong)

    def test_old_cache_identity_must_match_and_no_new_session_is_touched(self):
        job={"seed":1}
        runner=types.SimpleNamespace(job_payload=lambda *args:job,
            admission=types.SimpleNamespace(digest=optimized.digest),identity=lambda j:{"seed":j["seed"]})
        expected=optimized.digest({"plan_sha256":optimized.LEGACY_PLAN,"job":job})
        class Cache:
            def get_result(self,identity,payload):
                if identity!=expected or payload!=job:raise AssertionError('wrong cache lookup')
                return {"resource_use":{"identity":{"seed":1}}}
        identity,result=optimized.cached_outcome(Cache(),{},runner,None,'model',None,1)
        self.assertEqual(identity,expected)
        self.assertEqual(result['resource_use']['identity'],{'seed':1})
        class WrongCache:
            def get_result(self,*args):return {"resource_use":{"identity":{"seed":2}}}
        with self.assertRaisesRegex(ValueError,'identity_changed'):
            optimized.cached_outcome(WrongCache(),{},runner,None,'model',None,1)

    def test_coverage_wave_preserves_all_nas_candidates_and_reuses_without_new_work(self):
        order=[]
        plan={'models':['NAS-MODEL','FIXED-MODEL'],'origins':list(optimized.ORIGINS),
              'seed_pools':{'hpo':[1,2],'nas1':[3],'nas2':[4,5,6,7,8]}}
        def prepare(panel,origin,end):
            split=types.SimpleNamespace(signature=origin)
            return types.SimpleNamespace(origin=origin,inner_train=split,inner_validation=split)
        def job(full,prepared,model,candidate,seed):
            return {'model':model,'origin':prepared.origin,'seed':seed,'stage':'hpo'}
        runner=types.SimpleNamespace(job_payload=job,identity=lambda j:dict(j),
            admission=types.SimpleNamespace(digest=optimized.digest,
                planner=types.SimpleNamespace(enumerate_architectures=lambda *a:list(range(1000)))),
            restore_trial=lambda result:result)
        class Cache:
            def get_result(self,identity,payload):return {'resource_use':{'identity':dict(payload)}}
        class NoNewWork:
            def __getattr__(self,name):raise AssertionError('new ledger touched for reused fit '+name)
        def select(prepared,name,adapter,**kw):
            order.append((name,prepared.origin))
            self.assertEqual(len(kw['hpo_trials']),18)
            self.assertEqual(kw['k'],40)
            if name=='NAS-MODEL':self.assertEqual(len(kw['architectures']),1000)
            adapter.fit_candidate(None,1,prepared.inner_train,prepared.inner_validation)
            return types.SimpleNamespace(winner=types.SimpleNamespace(candidate_id='winner',payload={}),
                fixed_epochs={},signature='s',attempts=36,selection_rule='mean_rmse'),{}
        modules={'e2_training_models':types.SimpleNamespace(MODEL_IDS={'NAS-MODEL':'id'},
                     registered_trials=lambda name:list(range(18)),protocol=lambda:{'contract':{'delta':.01}}),
                 'e2_training_bindings':types.SimpleNamespace(run_neural_selection=select),
                 'e2_temporal_training':types.SimpleNamespace(prepare_b=prepare)}
        progress=[];reused=set()
        with patch.dict(sys.modules,modules):
            optimized.run_selection(plan,{},Cache(),NoNewWork(),None,runner,
                lambda *a:self.fail('GPU fit repeated'),None,progress,reused)
        self.assertEqual(order,[(name,o) for name in ('FIXED-MODEL','NAS-MODEL') for o in optimized.ORIGINS])
        self.assertEqual(len(reused),22)
        self.assertEqual(len(progress),22)

    def test_registration_does_not_cut_nas_or_reset_budget(self):
        registration,sha=optimized.read_registration()
        self.assertEqual(registration['nas_families'],6)
        self.assertEqual(registration['nas_architectures_per_family_origin'],1000)
        self.assertEqual(registration['nas_confirmation_seeds'],5)
        self.assertEqual(registration['k'],40)
        self.assertEqual(registration['upper_bound_attempts'],85074)
        self.assertIn('only unspent',registration['cost_limit'])

    def test_plan_only_does_not_import_torch_or_read_credentials(self):
        code="""
import builtins,runpy,sys
old=builtins.__import__
def guarded(name,*a,**kw):
    if name.split('.')[0] in ('torch','pandas','numpy') or name=='urllib.request':
        raise AssertionError('forbidden planning import '+name)
    return old(name,*a,**kw)
builtins.__import__=guarded
sys.argv=['run_e2_optimized_selection_v2.py','--plan-only']
runpy.run_path('scripts/run_e2_optimized_selection_v2.py',run_name='__main__')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['training_started'])


if __name__=='__main__':unittest.main()
