import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dataset_audit_contracts',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def row(day,close,adjusted='999999'):
    return {'Ticker':'AAPL','Date':day,'Open':str(close),'High':str(close+1),
            'Low':str(close-1),'Close':str(close),'Volume':'100','Adj Close':adjusted}

class DatasetInspectionContracts(unittest.TestCase):
    def test_adjusted_close_does_not_change_raw_price_checks(self):
        left=[row('2000-01-03',100,'1'),row('2000-01-04',101,'100000')]
        right=[row('2000-01-03',100,'bad'),row('2000-01-04',101,'bad')]
        self.assertEqual(audit.inspect_aapl(left),audit.inspect_aapl(right))
        self.assertFalse(audit.inspect_aapl(left)['adjusted_close_used'])

    def test_large_price_move_is_reported_without_removing_a_row(self):
        result=audit.inspect_aapl([row('2000-01-03',100),row('2000-01-04',48)])
        self.assertEqual(result['rows'],2)
        self.assertEqual(result['rows_removed'],0)
        self.assertEqual(len(result['close_ratio_flags']),1)
        self.assertTrue(result['flags_are_not_automatic_split_labels'])
        self.assertFalse(result['target_values_or_model_metrics_computed'])

    def test_impossible_ohlc_and_duplicate_dates_are_visible(self):
        bad=row('2000-01-03',100);bad['High']='90'
        result=audit.inspect_aapl([bad,row('2000-01-03',100)])
        self.assertEqual(result['invalid_OHLCV_rows'],1)
        self.assertEqual(result['unique_dates'],1)
        self.assertFalse(result['strictly_sorted'])
        self.assertFalse(result['trading_calendar_verified'])

if __name__=='__main__':
    unittest.main(verbosity=2)
