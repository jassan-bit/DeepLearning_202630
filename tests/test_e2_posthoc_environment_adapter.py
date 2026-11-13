"""Synthetic environment transport and worker routing; no market/network/GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_posthoc_environment_adapter as adapter


class EnvironmentAdapterTests(unittest.TestCase):
    def make(self, root, *, fail=False):
        original=root/'original';original.mkdir()
        old_path=root/'old.py';old_path.write_text('# immutable registered controller\n')
        bootstrap=root/'extension/scripts/e2_posthoc_clean_worker_bootstrap.py'
        bootstrap.parent.mkdir(parents=True);bootstrap.write_text('# bound bootstrap\n')
        module=SimpleNamespace(__file__=str(old_path),ENGINE_COMMIT=adapter.ENGINE_COMMIT)
        calls=[]; writes=[]
        def bounded(command,**kwargs):
            words=[str(v) for v in command];calls.append((words,dict(kwargs)))
            if words[:2]==['git','init']:
                (Path(words[-1])/'.git/info').mkdir(parents=True)
            if fail and 'fetch' in words:
                raise RuntimeError('synthetic remote failure')
            return .01
        def immutable(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            writes.append((str(path),dict(value)));path.write_text(json.dumps(value))
        module.run_bounded=bounded;module.immutable_json=immutable
        def prepare(source,python,destination,*,deadline):
            destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
            if (destination/'clean_runtime.json').exists():
                return json.loads((destination/'clean_runtime.json').read_text())
            module.immutable_json(destination/'setup_started.json',{'started':True})
            clean=destination/'source'
            module.run_bounded(['git','clone','--shared','--no-checkout',source,clean],timeout=120,log=destination/'checkout.log')
            module.run_bounded(['git','-C',clean,'sparse-checkout','init','--no-cone'],timeout=60)
            (clean/'.git/info/sparse-checkout').write_text('/scripts/\n/experiments/\n')
            module.run_bounded(['git','-C',clean,'checkout','--detach',adapter.ENGINE_COMMIT],timeout=120)
            lock=destination/'environment.lock.txt';lock.write_text('torch==2.8.0+cu126\nnumpy==2.1.3\n')
            module.run_bounded(['/bin/uv','pip','install','--python',python,'--requirements',lock,
                                '--index-url',adapter.PYPI,'--extra-index-url',adapter.TORCH_INDEX],timeout=180)
            receipt={'immutable_git_objects_shared':True,'source_tracked_status':'clean','setup_completed':True,
                     'clean_python':str(python)}
            module.immutable_json(destination/'clean_runtime.json',receipt)
            return receipt
        module.prepare_clean_runtime=prepare
        return module,original,bootstrap,calls,writes

    def test_remote_checkout_sparse_exact_lock_and_truthful_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,writes=self.make(root)
            old_bytes=Path(old.__file__).read_bytes()
            original_prepare=old.prepare_clean_runtime;original_run=old.run_bounded;original_write=old.immutable_json
            info=adapter.install_environment_adapter(old,source,root/'extension')
            worker_route=old.run_bounded
            deadline=time.monotonic()+30
            value=old.prepare_clean_runtime(source,root/'python',root/'clean',deadline=deadline)
            commands=[r[0]for r in calls]
            self.assertFalse(any(cmd[:2]==['git','clone']for cmd in commands))
            self.assertEqual(commands[0],['git','init','--quiet',str(root/'clean/source')])
            self.assertEqual(commands[1][-2:],['origin',adapter.REPOSITORY])
            self.assertEqual(commands[2][-5:],['fetch','--depth','1','origin',adapter.ENGINE_COMMIT])
            self.assertTrue(all(0<options['timeout']<=30 for _,options in calls))
            patterns=(root/'clean/source/.git/info/sparse-checkout').read_text().splitlines()
            self.assertTrue(set(adapter.EXTRA_SPARSE).issubset(patterns));self.assertIn('/scripts/',patterns)
            self.assertEqual(commands[-1][-2:],['--index-strategy','unsafe-best-match'])
            self.assertFalse(value['immutable_git_objects_shared'])
            self.assertEqual(value['source_checkout_method'],'remote_exact_commit')
            self.assertEqual(value['environment_adapter_sha256'],info['environment_adapter_sha256'])
            self.assertEqual(json.loads((root/'clean/clean_runtime.json').read_text()),value)
            self.assertIs(old.run_bounded,worker_route);self.assertIs(old.immutable_json,original_write)
            self.assertIs(old._e2_environment_original_prepare,original_prepare)
            self.assertIs(old._e2_environment_original_run_bounded,original_run)
            self.assertEqual(Path(old.__file__).read_bytes(),old_bytes)
            self.assertEqual(writes[0][1],{'started':True})

    def test_only_exact_bound_worker_command_is_routed_and_all_execution_options_survive(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root)
            expected_sha=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()
            adapter.install_environment_adapter(old,source,root/'extension')
            command=[root/'clean/venv/bin/python',Path(old.__file__),'--worker-spec',root/'spec.json','--worker-output',root/'job']
            options={'timeout':123,'cwd':root,'log':root/'worker.log','env':{'TEST':'value'}}
            old.run_bounded(command,**options)
            routed,actual_options=calls[-1]
            self.assertEqual(routed,[str(command[0]),str(bootstrap),'--old-controller',old.__file__,
                '--old-controller-sha256',expected_sha,*map(str,command[2:])])
            self.assertEqual(actual_options,options)
            other=['git','status','--porcelain'];old.run_bounded(other,timeout=5)
            self.assertEqual(calls[-1][0],other)
            another=[root/'python',root/'another.py','--worker-spec',root/'spec','--worker-output',root/'job']
            old.run_bounded(another,timeout=5);self.assertEqual(calls[-1][0],list(map(str,another)))

    def test_temporary_hooks_restore_after_failure_and_originals_are_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root,fail=True)
            immutable=old.immutable_json
            adapter.install_environment_adapter(old,source,root/'extension');route=old.run_bounded
            with self.assertRaisesRegex(RuntimeError,'synthetic remote failure'):
                old.prepare_clean_runtime(source,root/'python',root/'failed_clean',deadline=time.monotonic()+30)
            self.assertIs(old.run_bounded,route);self.assertIs(old.immutable_json,immutable)
            self.assertFalse((root/'failed_clean/clean_runtime.json').exists())
            self.assertTrue((root/'failed_clean/setup_started.json').exists())
            self.assertEqual(list(source.iterdir()),[])

    def test_uv_adapter_never_rewrites_other_indexes_or_nonexact_locks(self):
        with tempfile.TemporaryDirectory() as temp:
            lock=Path(temp)/'environment.lock.txt';lock.write_text('torch==2.8.0+cu126\n')
            other=['uv','pip','install','--requirements',str(lock),'--index-url','https://example.invalid',
                   '--extra-index-url',adapter.TORCH_INDEX]
            self.assertIs(adapter._uv_exact_official(other,lock),other)
            official=[*other];official[official.index('--index-url')+1]=adapter.PYPI
            lock.write_text('torch>=2.8\n')
            with self.assertRaisesRegex(ValueError,'exact package lock'):
                adapter._uv_exact_official(official,lock)

    def test_expired_setup_deadline_does_not_start_remote_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root)
            adapter.install_environment_adapter(old,source,root/'extension');route=old.run_bounded
            with self.assertRaises(TimeoutError):
                old.prepare_clean_runtime(source,root/'python',root/'clean',deadline=time.monotonic()-1)
            self.assertEqual(calls,[]);self.assertIs(old.run_bounded,route)

    def test_changed_controller_or_bootstrap_cannot_start_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root)
            adapter.install_environment_adapter(old,source,root/'extension')
            bootstrap.write_text('# modified bootstrap\n')
            command=[root/'python',Path(old.__file__),'--worker-spec',root/'spec','--worker-output',root/'job']
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                old.run_bounded(command,timeout=5)
            self.assertEqual(calls,[])

    def test_bounded_synthetic_preflight_is_separate_from_immutable_runtime_and_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root)
            output=root/'original_output';output.mkdir()
            underlying=old.run_bounded
            def runner(command,**kwargs):
                if '--preflight' not in list(map(str,command)):
                    return underlying(command,**kwargs)
                words=list(map(str,command));calls.append((words,dict(kwargs)))
                runtime=Path(words[words.index('--clean-runtime')+1])
                acceptance={'version':adapter.ACCEPTANCE_VERSION,'status':'passed','model_count':13,
                    'clean_runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),
                    'old_controller_sha256':hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
                    'bootstrap_sha256':hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                    'market_data_read':False,'training_performed':False,'optimizer_steps':0}
                (runtime.parent/'synthetic_acceptance.json').write_text(json.dumps(acceptance))
                return .01
            old.run_bounded=runner
            adapter.install_environment_adapter(old,source,root/'extension',original_output=output)
            route=old.run_bounded;immutable=old.immutable_json
            result=old.prepare_clean_runtime(source,root/'clean-python',root/'clean',deadline=time.monotonic()+30)
            command,options=calls[-1]
            self.assertEqual(command[0],str(root/'clean-python'));self.assertEqual(command[1],str(bootstrap))
            self.assertIn('--preflight',command);self.assertEqual(command[command.index('--original-output')+1],str(output))
            self.assertTrue(0<options['timeout']<=30)
            self.assertEqual(options['cwd'],root/'extension')
            persisted=json.loads((root/'clean/clean_runtime.json').read_text())
            self.assertNotIn('synthetic_acceptance_sha256',persisted)
            self.assertEqual(result['synthetic_acceptance_sha256'],hashlib.sha256((root/'clean/synthetic_acceptance.json').read_bytes()).hexdigest())
            count=len(calls)
            reused=old.prepare_clean_runtime(source,root/'clean-python',root/'clean',deadline=time.monotonic()+30)
            self.assertEqual(len(calls),count);self.assertEqual(reused,result)
            self.assertIs(old.run_bounded,route);self.assertIs(old.immutable_json,immutable)
            evidence=json.loads((root/'clean/synthetic_acceptance.json').read_text());evidence['optimizer_steps']=1
            (root/'clean/synthetic_acceptance.json').write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError,'execution scope'):
                old.prepare_clean_runtime(source,root/'clean-python',root/'clean',deadline=time.monotonic()+30)
            self.assertEqual(len(calls),count)

    def test_missing_preflight_receipt_prevents_success_and_restores_hooks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,source,bootstrap,calls,_=self.make(root)
            output=root/'original_output';output.mkdir()
            adapter.install_environment_adapter(old,source,root/'extension',original_output=output)
            route,immutable=old.run_bounded,old.immutable_json
            with self.assertRaisesRegex(ValueError,'without its separate acceptance'):
                old.prepare_clean_runtime(source,root/'python',root/'clean',deadline=time.monotonic()+30)
            self.assertIs(old.run_bounded,route);self.assertIs(old.immutable_json,immutable)
            self.assertTrue((root/'clean/clean_runtime.json').exists())


if __name__=='__main__':unittest.main()
