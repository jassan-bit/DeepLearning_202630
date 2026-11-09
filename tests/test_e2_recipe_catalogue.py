"""Synthetic-only integration of section 22, with explicit external source skips."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_current_models as current
import e2_current_pipeline as pipeline
import e2_recipe_catalogue as recipes
import smoke_e2_recipe_catalogue as cli


class RecipeCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_catalogue_fixed_and_noncompact(self):
        self.assertEqual(recipes.MODEL_NAMES, current.MODEL_NAMES)
        self.assertEqual(len(recipes.RECIPE_IDS), 11)
        cards = json.loads((Path(__file__).resolve().parents[1]/'book_e2/_static/recipe_cards_20261109/cards.json').read_text())
        self.assertEqual(recipes.RECIPE_IDS, {c['model']: c['recipe_id'] for c in cards['cards']})
        self.assertEqual(recipes.build_model('FCN-1D').__class__.__name__, 'FCNRecipe')
        with self.assertRaises(ValueError):
            recipes.build_model('FCN1D')
        for name in ['TSMixer', 'TiDE', current.HYBRID_NAME]:
            torch.manual_seed(5)
            actual = recipes.build_model(name)
            torch.manual_seed(5)
            expected = current.build_model(name, current.recipe_for(name))
            self.assertEqual(pipeline._state_sha256(actual), pipeline._state_sha256(expected))
            self.assertEqual(actual.recipe['preset'], 'provisional_default')

    def test_synthetic_guard_and_caps(self):
        with self.assertRaises(ValueError):
            recipes.run_synthetic_smoke(synthetic_only=False)
        for names in [[], ['FCN-1D', 'FCN-1D'], ['not-a-model']]:
            with self.assertRaises(ValueError):
                recipes.run_synthetic_smoke(model_names=names)
        with self.assertRaises(ValueError):
            recipes.run_synthetic_smoke(config=pipeline.SmokeConfig(supervised_steps=4))

    def test_auxiliary_scale_training_only(self):
        x = torch.ones(4,7,5)*3
        self.assertEqual(recipes.fit_auxiliary_scale(x), 3.)
        validation = torch.ones(4,7,5)*100
        self.assertEqual(recipes.fit_auxiliary_scale(x), 3.)
        self.assertGreater(float((validation/recipes.fit_auxiliary_scale(x)).max()), 1.)
        with self.assertRaises(ValueError):
            recipes.fit_auxiliary_scale(torch.full_like(x, float('nan')))

    def test_four_full_recipes_and_phase_contract(self):
        receipt = recipes.run_synthetic_smoke(model_names=list(recipes.NEW_NAMES))
        self.assertEqual(receipt['executed_model_count'], 4)
        self.assertFalse(receipt['market_data_read'])
        self.assertFalse(receipt['nas_executed'])
        self.assertEqual(receipt['verified_author_file_count'], 0)
        for row in receipt['models']:
            self.assertEqual(row['phases']['supervised']['steps'], 1)
            self.assertTrue(row['phases']['supervised']['all_gradients_finite'])
            self.assertTrue(all(x >= 0 for x in row['synthetic_predictions_original_units']))
        ae = receipt['models'][-1]
        self.assertEqual(set(ae['phases']), {'ae1','ae2','supervised'})
        self.assertTrue(all(p['fresh_optimizer'] for p in ae['phases'].values()))
        self.assertTrue(ae['ae_protocol']['encoder1_frozen_during_phase2'])
        self.assertFalse(ae['ae_protocol']['decoder_parameters_in_predictor'])
        self.assertTrue(ae['ae_protocol']['transfer_encoder1_exact'])
        self.assertTrue(ae['ae_protocol']['transfer_encoder2_exact'])
        self.assertNotIn('/workspace/', json.dumps(receipt))
        self.assertNotIn('/tmp/', json.dumps(receipt))

    def test_all_eleven_and_70_pinned_files(self):
        base = os.environ.get('E2_CURRENT_AUTHOR_SOURCES')
        if not base:
            self.skipTest('Set E2_CURRENT_AUTHOR_SOURCES for full pinned author integration')
        roots = {n: str(Path(base)/n) for n in current.EXTERNAL_NAMES}
        receipt = recipes.run_synthetic_smoke(source_roots=roots)
        self.assertEqual(receipt['executed_model_count'], 11)
        self.assertEqual(receipt['skipped_model_count'], 0)
        self.assertEqual(receipt['verified_author_file_count'], 70)
        self.assertNotIn(base, json.dumps(receipt))

    def test_cli_refuses_overwrite_and_market_path(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)/'synthetic.json'
            target.write_text('historical')
            with self.assertRaises(SystemExit):
                cli.main(['--output', str(target)])
            self.assertEqual(target.read_text(), 'historical')
            with self.assertRaises(SystemExit):
                cli.main(['--market-data', 'anything'])
            with mock.patch.object(cli, 'run_synthetic_smoke', return_value={'synthetic_only': True}):
                target = Path(root)/'new_synthetic.json'
                self.assertEqual(cli.main(['--output',str(target)]), 0)
                self.assertTrue(json.loads(target.read_text())['synthetic_only'])

if __name__ == '__main__':
    unittest.main()
