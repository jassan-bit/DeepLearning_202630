"""Real snapshot audit and fail-closed regression tests; no model training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import audit_e2_real_data as audit
from dl_data import load_development


class RealDataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_development(ROOT)
        cls.frame = pd.read_csv(ROOT/audit.SOURCE_PATHS[0])

    def test_complete_audit_without_optimizer_or_model_training(self):
        import torch
        with mock.patch.object(torch.optim.AdamW, '__init__', side_effect=AssertionError('optimizer forbidden')), \
             mock.patch.object(torch.nn.Module, 'train', side_effect=AssertionError('model training forbidden')):
            result = audit.audit(ROOT)
        self.assertEqual(result['model_fits'], 0)
        self.assertEqual(result['quality']['eligible_D7_count'], 716)
        self.assertEqual(sum(f['evaluation_count_available'] for f in result['CV_and_pilot_folds'][:10]), 305)
        self.assertEqual(result['CV_and_pilot_folds'][-1]['evaluation_count_available'], 60)
        self.assertEqual(result['final_training_support']['train_count'], 709)
        self.assertEqual(result['proposed_pilot_support']['evaluation_count_available'], 53)
        self.assertEqual(result['proposed_pilot_support']['last_train_day'], '2024-10-31')
        self.assertEqual(set(result['source_date_checks']), set(audit.SOURCE_PATHS))
        self.assertTrue(result['pipeline']['real_input_rejected'])
        self.assertEqual(result, audit.audit(ROOT))
        json.dumps(result, allow_nan=False)

    def test_hash_mismatch_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = root/audit.PROTOCOL
            protocol.parent.mkdir(parents=True)
            protocol.write_bytes((ROOT/audit.PROTOCOL).read_bytes())
            for name in audit.SOURCE_PATHS:
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                audit.verify_sources(root)

    def test_missing_snapshot_reports_exact_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); protocol = root/audit.PROTOCOL
            protocol.parent.mkdir(parents=True); protocol.write_bytes((ROOT/audit.PROTOCOL).read_bytes())
            with self.assertRaisesRegex(ValueError, 'daily_market_development.csv'):
                audit.verify_sources(root)

    def test_target_calendar_and_changed_label_rejected(self):
        target_frame = pd.read_csv(ROOT/audit.SOURCE_PATHS[1])
        self.assertEqual(audit.inspect_frame(target_frame)['rows'], 731)
        target_frame.iloc[1, 0] = target_frame.iloc[0, 0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.inspect_frame(target_frame)
        data = copy.deepcopy(self.data)
        data['targets'][400] += 0.001
        with self.assertRaises(AssertionError):
            audit.check_arrays(data)

    def test_duplicate_dates_rejected(self):
        frame = self.frame.copy(); frame.iloc[1, 0] = frame.iloc[0, 0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.inspect_frame(frame)

    def test_compressed_gap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing calendar'):
            audit.inspect_frame(self.frame.drop(index=82))

    def test_naive_and_nonmidnight_dates_rejected(self):
        frame = self.frame.copy()
        frame['market_day_utc'] = pd.to_datetime(frame['market_day_utc']).dt.tz_localize(None).astype(str)
        with self.assertRaisesRegex(ValueError, 'UTC'):
            audit.inspect_frame(frame)
        frame = self.frame.copy(); frame.iloc[0, 0] = '2023-01-01 01:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'midnight'):
            audit.inspect_frame(frame)

    def test_sample_sd_target_rejected(self):
        data = copy.deepcopy(self.data); data['targets'] *= np.sqrt(7/6)
        with self.assertRaises(AssertionError):
            audit.check_arrays(data)

    def test_contemporaneous_window_rejected(self):
        data = copy.deepcopy(self.data)
        data['windows_by_history'][7][40] = data['features'][34:41]
        with self.assertRaises(AssertionError):
            audit.check_arrays(data)

    def test_wrong_gap_mask_rejected(self):
        data = copy.deepcopy(self.data); data['eligible_by_history'][7][83] = True
        with self.assertRaises(AssertionError):
            audit.check_arrays(data)

    def test_ohlcv_bounds_rejected(self):
        data = copy.deepcopy(self.data); data['features'][20, 1] = 1
        with self.assertRaisesRegex(ValueError, 'OHLCV'):
            audit.check_arrays(data)

    def test_future_training_label_rejected(self):
        split = audit.split_for_origin(self.data, 7, '2024-01-01')
        split['train_indices'] = np.append(split['train_indices'], 364)
        with self.assertRaises(AssertionError):
            audit.fold_audit(self.data, '2024-01-01', '2024-01-31', split)


if __name__ == '__main__':
    unittest.main()
