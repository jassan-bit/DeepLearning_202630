"""Integrity checks for literature-derived core choice and common-input capacities."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_e1_core_selection.py'
spec=importlib.util.spec_from_file_location('core_selection_builder',SCRIPT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def fixture(counts):
    records=['p'+str(i) for i in range(10)]
    return {'denominator':10,'records':records,'family_counts':{
        key:{'count':n,'denominator':10,'experiments':records[:n]} for key,n in counts.items()}}


class CoreSelectionIntegrity(unittest.TestCase):
    def test_selection_follows_actual_frequency_not_hardcoded_trio(self):
        counts={'LSTM':10,'MLP':4,'GRU':2,'TCN':1,'RNN_simple':1}
        ranked=mod.ranked_candidates(fixture(counts),list(counts),3)
        self.assertEqual([r['family'] for r in ranked if r['selected_current_core']],['LSTM','MLP','GRU'])
        changed={**counts,'TCN':3,'GRU':1}
        ranked=mod.ranked_candidates(fixture(changed),list(changed),3)
        self.assertEqual([r['family'] for r in ranked if r['selected_current_core']],['LSTM','MLP','TCN'])

    def test_boundary_tie_cannot_be_silently_resolved_for_preferred_model(self):
        counts={'LSTM':10,'MLP':4,'GRU':2,'TCN':2}
        with self.assertRaisesRegex(ValueError,'Tie'):
            mod.ranked_candidates(fixture(counts),list(counts),3)

    def test_duplicates_do_not_inflate_comparator_frequency(self):
        counts={'LSTM':10,'MLP':4,'GRU':2,'TCN':1}
        data=fixture(counts);data['family_counts']['MLP']['experiments']=['p0','p0','p1','p2']
        with self.assertRaisesRegex(ValueError,'Invalid family count'):
            mod.ranked_candidates(data,list(counts),3)
        with self.assertRaisesRegex(ValueError,'must be unique'):
            mod.ranked_candidates(fixture(counts),list(counts)+['MLP'],3)

    def test_common_input_capacity_is_not_historical_mlp_recipe(self):
        self.assertEqual(mod.parameters('MLP',32),1185)
        self.assertEqual(mod.parameters('MLP',128),4737)
        self.assertEqual(mod.parameters('MLP',16),593)
        self.assertNotEqual(mod.parameters('MLP',32,days=14),1185)
        self.assertEqual(mod.parameters('GRU',16),1121)
        self.assertEqual(mod.parameters('GRU',32),3777)
        self.assertEqual(mod.parameters('LSTM',16),1489)
        self.assertEqual(mod.parameters('LSTM',32),5025)

    def test_frozen_documentary_source_tampering_stops_compilation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'evidence.json';p.write_bytes(b'original')
            inputs={'source_sha256':{'evidence.json':mod.digest(b'original')}}
            mod.verify_sources(inputs,root)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'):
                mod.verify_sources(inputs,root)

    def test_real_compilation_declares_selection_without_inventing_fits(self):
        outputs=mod.build();summary=mod.json.loads(outputs['summary.json'])
        self.assertEqual(summary['selected_current_core_in_frequency_order'],['LSTM','MLP','GRU'])
        self.assertIsNone(summary['selected_hyperparameters_and_new_scores'])
        self.assertEqual(summary['new_model_fits_performed_by_this_builder'],0)
        self.assertTrue(summary['reserve_was_already_observed'])
        self.assertFalse(summary['historical_selection_and_metrics_rewritten'])
        manifest=mod.json.loads(outputs['manifest.json'])
        self.assertEqual(set(manifest['approved_public_files']),set(outputs)-{'manifest.json'})
        for name,record in manifest['files'].items():
            self.assertEqual(record['sha256'],mod.digest(outputs[name]))
            self.assertEqual(record['bytes'],len(outputs[name]))


if __name__=='__main__':unittest.main()
