"""History/calendar/source contracts; no HTTP, model or additional test data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('d00_history_under_test',HERE/'d00_history.py')
history=importlib.util.module_from_spec(spec);spec.loader.exec_module(history)


class D00HistoryTests(unittest.TestCase):
    def test_actual_pinned_cache_keeps_missing_day_and_complete_calendar(self):
        def forbidden(*args,**kwargs):raise AssertionError('Cached verification must not use HTTP')
        paths,metadata=history.acquire(HERE/'d00_inputs',opener=forbidden)
        rows=history.history(paths);labels=history.sigma7(rows)
        self.assertEqual(len(rows),1339)
        self.assertEqual([r['date'] for r in rows if r['missing']],['2023-03-24'])
        self.assertTrue(all(r['cached'] for r in metadata))
        self.assertEqual(sum(v is not None for v in labels.values()),1324)
        for day in history._days('2023-03-24','2023-03-31'):
            self.assertIsNone(labels[day])
        self.assertIsNotNone(labels['2023-04-01'])
        self.assertIsNotNone(labels['2024-11-07'])  # needed across the NPZ purge
        self.assertIsNotNone(labels['2024-12-31'])  # omitted from NPZ partitions

    def test_changed_cached_bytes_refused_before_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/history.SOURCES[0]['file']
            path.write_bytes(b'corrupt_source_that_must_be_preserved')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'Pinned D00'):
                history.acquire(temp,opener=lambda *a,**k:self.fail('No fallback request for a changed cache'))
            self.assertEqual(path.read_bytes(),before)

    def test_partial_bar_nonfinite_price_and_calendar_duplicates_refused(self):
        source={'header':('market_day_utc',*history.FIELDS),'rows':2,
                'start':'2020-01-01','end':'2020-01-02'}
        header='market_day_utc,open,high,low,close,volume\n'
        valid=header+'2020-01-01 00:00:00+00:00,1,2,1,2,10\n2020-01-02 00:00:00+00:00,2,3,1,2,10\n'
        self.assertEqual(len(history.parse_daily(valid.encode(),source)),2)
        for raw in [valid.replace(',1,2,1,2,10',',1,,1,2,10'),
                    valid.replace(',1,2,1,2,10',',1,inf,1,2,10'),
                    valid.replace('2020-01-02','2020-01-01')]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):history.parse_daily(raw.encode(),source)

    def test_irregular_eight_observation_window_is_not_eight_daily_returns(self):
        days=history._days('2020-01-01','2020-01-09')
        rows=[{'date':d,'values':(1.,2.,1.,2.,10.),'missing':False} for d in days if d!='2020-01-05']
        with self.assertRaisesRegex(ValueError,'eight consecutive'):
            history.sigma7(rows)


if __name__=='__main__':unittest.main()
