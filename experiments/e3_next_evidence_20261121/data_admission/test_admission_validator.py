"""Software fixtures only: no market observations, data acquisition or models."""
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('admission', ROOT/'admission_validator.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec2 = importlib.util.spec_from_file_location('calendar_metadata', ROOT/'calendar_design.py')
c = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(c)


class Guards(unittest.TestCase):
    def case(self):
        return (json.loads((ROOT/'concrete_panels.DRAFT.json').read_text()),
                json.loads((ROOT/'AUTHOR_ACCESS_DECLARATION.template.json').read_text()))

    def test_actual_four_proposals_are_blocked_without_market_reads(self):
        registry, declaration = self.case(); result = m.validate(registry,declaration)
        self.assertEqual(result['status'],'BLOCKED_UNADMITTED_FUTURE_PROPOSALS')
        self.assertEqual(result['concrete_instrument_proposals'],4)
        self.assertEqual(result['datasets_admitted'],0)
        self.assertFalse(result['confirmation_enabled'])
        self.assertGreater(result['blocked_gate_count'],30)
        self.assertEqual(result['market_values_read'],0)

    def test_clock_horizon_cannot_be_declared_equivalent_by_renaming(self):
        registry, declaration = self.case(); registry['panels'][1]['task_equivalence_review']['accepted']=True
        result=m.validate(registry,declaration)
        self.assertIn('F01:task_clock',{x['gate'] for x in result['blocked_gates']})

    def test_no_forward_fill_policy_relaxation(self):
        registry, declaration=self.case(); registry['panels'][1]['missing_session_policy']='fill_weekends'
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_source_or_clock_not_counted_as_second_guide_axis(self):
        registry,declaration=self.case(); registry['panels'][1]['guide_difference_axes'][1]['axis']='provider'
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_known_exposed_data_hash_cannot_be_relabelled(self):
        registry,declaration=self.case();registry['panels'][1]['future_file_metadata']=[
            {'sha256':next(iter(m.KNOWN_EXPOSED)),'bytes':100,'source_URL':'https://example.org/metadata','source_version':'fixture'}]
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_market_arrays_and_credentials_are_rejected(self):
        for field in ['close','ohlcv','predictions']:
            registry,declaration=self.case();registry[field]=[1.0]
            with self.subTest(field=field),self.assertRaises(ValueError):m.validate(registry,declaration)
        registry,declaration=self.case();registry['panels'][1]['source_URL']=registry['panels'][1]['source_URL'].replace('{API_KEY}','not-a-real-key-fixture')
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_raw_policy_and_original_guard_cannot_change(self):
        registry,declaration=self.case();registry['original_fixed_calendar_guard_unchanged']=False
        with self.assertRaises(ValueError):m.validate(registry,declaration)
        registry,declaration=self.case();registry['panels'][2]['raw_price_policy']='adjusted_close'
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_duplicate_JSON_keys_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metadata.json';path.write_text('{"id":1,"id":2}')
            with self.assertRaises(ValueError):m.load(path)

    def test_late_freeze_still_blocked(self):
        registry,declaration=self.case();declaration['panel_access']['F00']['real_confirmation_freeze_utc']='2027-03-16T00:00:00Z'
        result=m.validate(registry,declaration)
        self.assertIn('F00:prospective_freeze',{x['gate'] for x in result['blocked_gates']})

    def test_calendar_metadata_counts_and_variable_horizon(self):
        btc=c.describe('2027-03-15','2028-03-14','UTC_calendar_daily')
        nyse=c.describe('2027-03-15','2028-03-14','NYSE_scheduled_sessions')
        self.assertEqual(btc['scheduled_endpoint_count'],366)
        self.assertEqual(nyse['scheduled_endpoint_count'],253)
        self.assertEqual(set(btc['elapsed_calendar_day_gaps']),{'1'})
        self.assertIn('4',nyse['elapsed_calendar_day_gaps'])
        self.assertFalse(nyse['admission_or_power_sufficiency_claimed'])
        with self.assertRaises(ValueError):c.schedule('2027-03-15','2028-03-14','Nasdaq_assumed_equal_to_NYSE')

    def test_complete_SYNTHETIC_metadata_is_only_review_ready(self):
        registry,declaration=self.case()
        declaration.update(authentic_external_declaration=True)
        declaration['declared_at_utc']='2026-11-21T00:00:00Z'
        declaration['operator']={'name':'EXPLICIT SOFTWARE FIXTURE, NOT A HUMAN','role':'fixture'}
        for panel in registry['panels']:
            panel['licensing']['specific_use_reviewed_by_owner']=True
            for flag in ['provider_symbol_metadata_verified','public_repository_and_external_reproducer_access_reviewed',
                         'origin_availability_contract_verified','seven_endpoint_boundary_purge_checked','power_dimensioning_accepts_R_and_T']:
                panel[flag]=True
            panel['task_equivalence_review']={'accepted':True,'choice':'native_observation_ahead_separate_elapsed_time',
                'reviewer':'EXPLICIT SOFTWARE FIXTURE','rationale':'guard fixture only'}
            panel['future_venue_calendar']['pinned_and_reviewed']=True
            for axis in panel['guide_difference_axes']:axis['established']=True
            panel['future_file_metadata']=[{'sha256':hashlib.sha256(panel['id'].encode()).hexdigest(),'bytes':100,'source_URL':'https://example.org/fixture','source_version':'software-fixture'}]
            panel['confirmation_endpoint_count_verified']=100
            declaration['panel_access'][panel['id']]={'outcomes_seen_or_used_for_selection':False,
                'declaration_before_outcome_access':True,'custodian_access_log_sha256':'b'*64,
                'real_confirmation_freeze_utc':'2027-02-28T00:00:00Z'}
        result=m.validate(registry,declaration)
        self.assertEqual(result['status'],'DATA_METADATA_READY_FOR_EXTERNAL_ADMISSION_REVIEW')
        self.assertFalse(result['confirmation_enabled']);self.assertEqual(result['datasets_admitted'],0)
        self.assertFalse(result['human_identity_authenticated']);self.assertFalse(result['rights_clearance_inferred'])

    def test_reject_identical_raw_files_across_instruments(self):
        registry,declaration=self.case()
        metadata=[{'sha256':'e'*64,'bytes':10,'source_URL':'https://example.org/fixture','source_version':'fixture'}]
        registry['panels'][1]['future_file_metadata']=deepcopy(metadata)
        registry['panels'][2]['future_file_metadata']=deepcopy(metadata)
        with self.assertRaises(ValueError):m.validate(registry,declaration)

    def test_cli_preserves_previous_receipt_and_builder_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'receipt.json'
            command=[sys.executable,str(ROOT/'admission_validator.py'),'--registry',str(ROOT/'concrete_panels.DRAFT.json'),
                '--declaration',str(ROOT/'AUTHOR_ACCESS_DECLARATION.template.json'),'--output',str(out)]
            first=subprocess.run(command,capture_output=True,text=True,timeout=5)
            self.assertEqual(first.returncode,2);before=out.read_bytes()
            second=subprocess.run(command,capture_output=True,text=True,timeout=5)
            self.assertNotEqual(second.returncode,0);self.assertEqual(out.read_bytes(),before)
            build=subprocess.run([sys.executable,str(ROOT/'build_registry.py'),'--output-root',folder],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(build.returncode,0);self.assertEqual(out.read_bytes(),before)


if __name__=='__main__':unittest.main()
