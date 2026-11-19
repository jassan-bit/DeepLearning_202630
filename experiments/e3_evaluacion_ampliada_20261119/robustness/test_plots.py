"""Real PNG/SVG export from synthetic bound tables; no market results."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location('plot_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
plots=load('plots');fixtures=load('test_analyze')


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        _,protocol,panel,_=fixtures.fixture(cls.root)
        cls.analysis=cls.root/'analysis'
        fixtures.audit.run(panel,protocol,cls.analysis)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_actual_export_has_error_bar_scope_watermark_and_byte_verified_retry(self):
        output=self.root/'figures'
        result=plots.run(self.analysis,output)
        self.assertEqual(len(result['descriptions']),3)
        self.assertEqual(len(result['files']),6)
        self.assertTrue(result['synthetic_fixture'])
        self.assertIn('nominal calendar-block',result['RMSE_error_bars'])
        self.assertEqual((output/'rmse_D01.png').read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        self.assertIn('DATOS SINTÉTICOS',(output/'rmse_D01.svg').read_text())
        self.assertEqual(plots.run(self.analysis,output),result)

    def test_changed_input_table_cannot_produce_a_new_unbound_figure(self):
        path=self.analysis/'calendar_CIs.csv';raw=path.read_bytes()
        try:
            path.write_bytes(raw+b'changed')
            with self.assertRaisesRegex(ValueError,'differs from its manifest'):
                plots.run(self.analysis,self.root/'rejected')
        finally:path.write_bytes(raw)


if __name__=='__main__':unittest.main()
