"""Nonexperimental guards; only reaggregate already saved public points."""
import json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import run_current as current

HERE=Path(__file__).resolve().parent
class Guard(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_real_saved_points_recompute(self):
        source=HERE/'private_case_current/source';out=self.root/'out';out.mkdir()
        profile=json.loads((HERE/'SOFTWARE_PROFILE.json').read_text())
        r=current.regenerate_V4(source,out,'software-validation',profile)
        self.assertEqual((r['saved_predictions'],r['trajectories'],r['recipes']),(1210,22,6))
    def test_changed_saved_points_rejected(self):
        prefix=self.root/'experiments/e3_next_evidence_20261121/development/cpu_pilot_results';prefix.mkdir(parents=True)
        (prefix/'assessment_predictions.csv').write_bytes(b'altered')
        with self.assertRaises(ValueError):current.regenerate_V4(self.root,self.root,'software-validation',{})
    def profile_guard(self,update):
        request=json.loads((HERE/'SOFTWARE_PROFILE.json').read_text());request.update(update)
        (self.root/'SOFTWARE_PROFILE.json').write_text(json.dumps(request));fake=Mock()
        fake.validate_request.return_value=None
        with patch.object(current,'HERE',self.root),patch.object(current,'load_context',return_value=fake):
            with self.assertRaises(ValueError):current.run(self.root/'case','software-validation')
        fake.run.assert_not_called()
    def test_null_V4_tolerance_before_execution(self):self.profile_guard({'V4_tolerance':{'absolute':None,'relative':None}})
    def test_wrong_edition_before_execution(self):self.profile_guard({'source_commit':'f5c960eefedb5fc49b313dd795b1ec84e784bd78'})
    def test_frozen_manifest_tamper_rejected(self):
        (self.root/'kit').mkdir();(self.root/'addon').mkdir();(self.root/'kit/MANIFEST.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):current.load_context(self.root)
if __name__=='__main__':unittest.main()
