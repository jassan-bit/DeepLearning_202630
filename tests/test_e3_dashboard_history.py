import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from e3_dashboard.history import DEFAULT_HISTORY_DIR, load_training_history


class HistoricalCurves(unittest.TestCase):
    def test_actual_curve_is_complete_and_best_epoch_is_validation_minimum(self):
        record = load_training_history()
        self.assertEqual(len(record['history']), 40)
        self.assertEqual(record['seed'], 271801)
        self.assertEqual(min(record['history'], key=lambda r:r['validation_RMSE_original'])['epoch'],38)
        self.assertEqual(record['new_training_runs'],0)
        self.assertIn('not full-fit RMSE',record['units']['training_loss_normalized'])

    def test_tampered_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            copied=Path(path)/'history';shutil.copytree(DEFAULT_HISTORY_DIR,copied)
            file=copied/'history.json';raw=bytearray(file.read_bytes());raw[-1]^=1;file.write_bytes(raw)
            with self.assertRaises(ValueError):load_training_history(copied)

    def test_finite_and_order_guards_hold_even_with_a_recomputed_file_hash(self):
        for alteration in ('nan','epoch'):
            with tempfile.TemporaryDirectory() as path:
                copied=Path(path)/'history';shutil.copytree(DEFAULT_HISTORY_DIR,copied)
                record=json.loads((copied/'history.json').read_bytes())
                if alteration=='nan':record['history'][0]['training_loss_normalized']=float('nan')
                else:record['history'][0]['epoch']=2
                raw=json.dumps(record).encode();(copied/'history.json').write_bytes(raw)
                manifest=json.loads((copied/'manifest.json').read_bytes());manifest.update(history_sha256=hashlib.sha256(raw).hexdigest(),history_bytes=len(raw))
                (copied/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):load_training_history(copied)


if __name__=='__main__':unittest.main()
