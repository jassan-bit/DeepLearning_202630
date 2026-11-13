import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_clean_worker_bootstrap as m
import e2_posthoc_environment_resume as resume

class BootstrapTests(unittest.TestCase):
    def check_worker(self,model):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);own=root/'scripts/e2_posthoc_clean_worker_bootstrap.py';own.parent.mkdir();own.write_text('bound-bootstrap')
            controller=root/'original/worker.py';controller.parent.mkdir();controller.write_text('unchanged-controller')
            checksum=hashlib.sha256(controller.read_bytes()).hexdigest()
            reg=root/m.REGISTRATION;reg.parent.mkdir(parents=True);reg.write_text(json.dumps({'files_sha256':{str(own.relative_to(root)):hashlib.sha256(own.read_bytes()).hexdigest()}}))
            spec=root/'worker_spec.json';spec.write_text(json.dumps({'controller_source_sha256':checksum,
                'clean_runtime':{'clean_source':str(root/'clean')},'model':model,'kind':'reproduction'}))
            old=SimpleNamespace(worker=Mock(return_value={'completed':True}),immutable_json=Mock())
            shim=SimpleNamespace(install_prediction_raw_alias=Mock(return_value={'numeric_predictions_changed':False}))
            def load(path,name):return shim if 'arima' in name else old
            with patch.object(m,'__file__',str(own)),patch.object(m,'OLD_SHA',checksum), \
                 patch.object(m,'verified_module',side_effect=load),patch.object(sys,'path',list(sys.path)):
                result=m.execute(controller,checksum,spec,root/'output')
            self.assertEqual(result,{'completed':True});old.worker.assert_called_once_with(spec,root/'output')
            if model=='ARIMA':
                shim.install_prediction_raw_alias.assert_called_once_with(root/'clean');old.immutable_json.assert_called_once()
            else:shim.install_prediction_raw_alias.assert_not_called()
    def test_ARIMA_uses_existing_name_adapter_before_same_worker(self):self.check_worker('ARIMA')
    def test_neural_worker_is_forwarded_without_ARIMA_shim(self):self.check_worker('FCN-1D')
    def test_bad_original_preflight_source_fails_before_torch_or_worker(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);runtime=root/'runtime.json';runtime.write_text('{}');old=root/'old.py';old.write_text('changed')
            with self.assertRaisesRegex(ValueError,'bytes differ'):m.preflight(old,'0'*64,runtime,root)
    def test_interpreter_delegates_normal_python_arguments_exactly(self):
        with patch.object(resume.os,'execv',side_effect=RuntimeError('delegated')) as call:
            with self.assertRaisesRegex(RuntimeError,'delegated'):resume.main(['-I','-c','print(1)'])
        call.assert_called_once_with(str(resume.REAL_PYTHON),[str(resume.REAL_PYTHON),'-I','-c','print(1)'])
    def test_interpreter_bundle_root_preserved_in_source_and_executable_layouts(self):
        base=Path('/tmp/environment-recovery-test')
        self.assertEqual(resume.bundle_root(base/'python'),base)
        self.assertEqual(resume.bundle_root(base/'scripts/e2_posthoc_environment_resume.py'),base)
    def test_notebook_dry_run_never_downloads_or_starts_work(self):
        spec=importlib.util.spec_from_file_location('e2_recovery_notebook',ROOT/'notebooks/resume_e2_clean_setup_colab.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module,'get',side_effect=AssertionError('network forbidden')):
            value=module.resume_e2_clean_setup_colab('0'*40,execute=False)
        self.assertEqual(value['authorized_total_hours'],80);self.assertFalse(value['retry_scientific_fit'])
if __name__=='__main__':unittest.main()
