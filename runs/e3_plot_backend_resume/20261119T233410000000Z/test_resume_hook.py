"""Control-flow regression: completed evaluation must never execute again."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('plot_resume', HERE / 'resume_part.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.source = self.base / 'source'
        self.output = self.base / ('e3-extended-audit-' + resume.MASTER)
        self.output.mkdir()
        self.actual_source = Path(os.environ.get('E3_SCIENTIFIC_SOURCE', '/workspace/e3_review9'))
        for name in ('evaluation/run.py', 'robustness/analyze.py', 'robustness/plots.py'):
            target = self.source / resume.SCOPE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.actual_source / resume.SCOPE / name).read_bytes())
        (self.output / 'protocol.json').write_bytes(b'protocol')
        self.patchers = [patch.object(resume, 'PROTOCOL', hashlib.sha256(b'protocol').hexdigest())]
        self.analysis = self.output / 'analysis'
        self.analysis.mkdir()
        table = b'dataset,value\nD00,1\n'
        (self.analysis / 'model_metrics.csv').write_bytes(table)
        manifest = {'protocol_sha256': hashlib.sha256(b'protocol').hexdigest(),
                    'files': {'model_metrics.csv': {'bytes': len(table), 'sha256': hashlib.sha256(table).hexdigest()}}}
        manifest_raw = raw_json(manifest)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        (self.analysis / 'output_manifest.json').write_bytes(manifest_raw)
        receipt = {'source_commit': resume.SOURCE, 'protocol_sha256': manifest['protocol_sha256'],
                   'output_manifest_sha256': manifest_sha, 'analysis_code_sha256': resume.ANALYZE_SHA256,
                   'new_model_fits': 0, 'new_model_inference': 0, 'R3_accredited': False}
        receipt_raw = raw_json(receipt)
        receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
        (self.analysis / 'analysis_receipt.json').write_bytes(receipt_raw)
        self.patchers += [patch.object(resume, 'ANALYSIS_RECEIPT_SHA256', receipt_sha),
                          patch.object(resume, 'ANALYSIS_MANIFEST_SHA256', manifest_sha)]
        for item in self.patchers:
            item.start()
            self.addCleanup(item.stop)
        self.calls = []
        self.saves = []
        self.verifications = []
        self.views = []
        self.env = {'MPLBACKEND': 'module://matplotlib_inline.backend_inline', 'PYTHONNOUSERSITE': '1'}
        self.analysis_action = None

        def original(command, cwd, env, log, on_job=None):
            self.calls.append(([str(v) for v in command], dict(env)))
            if str(command[1]).endswith('/evaluation/run.py'):
                self.fail('THE COMPLETED EVALUATION WORKER MUST NEVER RUN')
            if str(command[1]).endswith('/robustness/analyze.py') and self.analysis_action:
                self.analysis_action()
            if str(command[1]).endswith('/robustness/plots.py'):
                self.assertEqual(env['MPLBACKEND'], 'Agg')
                self.write_fake_figures()
            return 0

        def verifier(output, protocol, source):
            self.verifications.append((output, protocol, source))
            return {'status': 'published_completed_evaluation_verified', 'completed_job_results': 796}

        def stable_view(output, view, *, finished=False):
            self.views.append((output, view, finished))
            return view

        self.controller = SimpleNamespace(stream=original, stable_view=stable_view)
        resume.install_plot_finalization(self.controller, self.source, self.base, verifier)

    def command(self, name, *args):
        return ['isolated-python', self.source / resume.SCOPE / name, *args]

    def worker(self):
        return self.controller.stream(self.command('evaluation/run.py', '--output', self.output,
            '--source-commit', resume.SOURCE, '--protocol', self.output / 'protocol.json'),
            self.source, self.env, self.output / 'worker.log', lambda: self.saves.append(True))

    def analysis_call(self):
        return self.controller.stream(self.command('robustness/analyze.py', '--output', self.analysis,
            '--panel', self.output / 'panel_manifest.json', '--protocol', self.output / 'protocol.json'),
            self.source, self.env, self.output / 'analysis.log')

    def plot_call(self):
        return self.controller.stream(self.command('robustness/plots.py', '--analysis-dir', self.analysis,
            '--output', self.output / 'figures'), self.source, self.env, self.output / 'figures.log')

    def write_fake_figures(self):
        directory = self.output / 'figures'
        directory.mkdir(exist_ok=True)
        names = {kind + '_' + data for kind in ('rmse', 'noise', 'coverage', 'learning')
                 for data in ('D00', 'D01', 'D02', 'D03')}
        files = {}
        for name in names:
            for ext in ('.png', '.svg'):
                raw = (name + ext).encode()
                (directory / (name + ext)).write_bytes(raw)
                files[name + ext] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        manifest = {'inputs': {'analysis_receipt_sha256': resume.ANALYSIS_RECEIPT_SHA256,
                               'analysis_manifest_sha256': resume.ANALYSIS_MANIFEST_SHA256},
                    'plot_code_sha256': resume.PLOTS_SHA256, 'synthetic_fixture': False,
                    'new_model_fits': 0, 'R3_accredited': False,
                    'descriptions': dict.fromkeys(names, 'CONTROL-FLOW FIXTURE'), 'files': files}
        (directory / 'figures_manifest.json').write_bytes(raw_json(manifest))

    def test_finish_uses_cached_evaluation_original_analysis_and_agg(self):
        self.assertEqual(self.worker(), 796)
        self.assertEqual(len(self.verifications), 1)
        self.assertEqual(self.saves, [True])
        self.assertEqual(self.calls, [])
        self.analysis_call()
        self.plot_call()
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(self.calls[0][0][1].endswith('/robustness/analyze.py'))
        self.assertEqual(self.calls[0][1], self.env)
        self.assertEqual(self.env['MPLBACKEND'], 'module://matplotlib_inline.backend_inline')
        declaration = json.loads((self.output / 'plot_backend_recovery/recovery_v1.json').read_bytes())
        self.assertFalse(declaration['evaluation_worker_executed'])
        self.assertEqual(declaration['new_model_fits'], 0)

    def test_missing_final_analysis_stops_before_worker_or_save(self):
        (self.analysis / 'analysis_receipt.json').unlink()
        with self.assertRaises(ValueError): self.worker()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.saves, [])

    def test_changed_table_stops_before_worker_or_save(self):
        (self.analysis / 'model_metrics.csv').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.worker()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.saves, [])

    def test_noncompleted_evaluation_rejected_without_fallback_training(self):
        rejected = SimpleNamespace(stream=lambda *a: self.fail('No subprocess allowed'), stable_view=self.controller.stable_view)
        def verifier(*args): raise ValueError('incomplete receipt')
        resume.install_plot_finalization(rejected, self.source, self.base, verifier)
        with self.assertRaisesRegex(ValueError, 'incomplete receipt'):
            rejected.stream(self.command('evaluation/run.py', '--output', self.output,
                '--source-commit', resume.SOURCE, '--protocol', self.output / 'protocol.json'),
                self.source, self.env, self.output / 'worker.log', lambda: None)

    def test_analysis_cannot_run_before_completed_evaluation(self):
        with self.assertRaises(ValueError): self.analysis_call()
        self.assertEqual(self.calls, [])

    def test_plots_cannot_run_before_cached_analysis_validation(self):
        self.worker()
        with self.assertRaises(ValueError): self.plot_call()
        self.assertEqual(self.calls, [])

    def test_analysis_receipt_rewrite_rejected(self):
        self.worker()
        self.analysis_action = lambda: (self.analysis / 'analysis_receipt.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            self.analysis_call()
        self.assertEqual(len(self.calls), 1)

    def test_unrelated_command_delegated_unchanged(self):
        command = ['python', self.source / 'unrelated.py']
        self.controller.stream(command, self.source, self.env, self.output / 'unrelated.log')
        self.assertEqual(self.calls, [([str(v) for v in command], self.env)])

    def test_source_tamper_refused(self):
        (self.source / resume.SCOPE / 'robustness/plots.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            resume.install_plot_finalization(SimpleNamespace(stream=lambda *a: None, stable_view=lambda *a, **k: None), self.source, self.base, lambda *a: {})

    def test_recovery_declaration_immutable_and_idempotent(self):
        self.worker()
        path = self.output / 'plot_backend_recovery/recovery_v1.json'
        raw = path.read_bytes()
        self.worker()
        self.assertEqual(path.read_bytes(), raw)
        path.write_bytes(b'{}')
        with self.assertRaises(ValueError): self.worker()

    def test_incomplete_figure_set_refused(self):
        self.write_fake_figures()
        (self.output / 'figures/rmse_D00.png').unlink()
        with self.assertRaises(ValueError): resume.verify_figures(self.output)

    def test_figure_hash_tamper_refused(self):
        self.write_fake_figures()
        (self.output / 'figures/rmse_D00.svg').write_bytes(b'changed')
        with self.assertRaises(ValueError): resume.verify_figures(self.output)

    def test_running_save_uses_verified_complete_manifest_view(self):
        view = self.base / ('e3-extended-persist-view-' + resume.MASTER)
        self.assertEqual(self.controller.stable_view(self.output, view, finished=False), view)
        self.assertEqual(self.views, [(self.output, view, True)])
        self.assertEqual(len(self.verifications), 1)

    def test_changed_analysis_blocks_view_publication(self):
        (self.analysis / 'analysis_receipt.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            self.controller.stable_view(self.output, self.base / ('e3-extended-persist-view-' + resume.MASTER), finished=False)
        self.assertEqual(self.views, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
