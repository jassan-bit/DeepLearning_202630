import json,re,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=['FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor','TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer']
class CatalogueTests(unittest.TestCase):
 def test_machine_catalogue(self):
  s=json.loads((ROOT/'references/e2_current_catalogue_20261108.json').read_text())
  self.assertEqual(s['models'],MODELS);self.assertFalse(s['superiority_demonstrated']);self.assertFalse(s['full_current_benchmark_executed'])
 def test_intro_exact_order(self):
  s=(ROOT/'book_e2/intro.md').read_text();names=re.findall(r'^\d+\. \*\*(.*?)\*\*\.',s,re.M)
  self.assertEqual(names,MODELS);self.assertIn('superioridad no está demostrada',s)
 def test_no_retired_selection_rows(self):
  for p in (ROOT/'book_e2/sections').glob('*.md'):
   for line in p.read_text().splitlines():
    if line.startswith('|'):
     self.assertFalse(any(x in line for x in ['Informer','Autoformer','Crossformer','S-Mamba']),str(p)+line)
 def test_no_obsolete_config_counts(self):
  for p in [ROOT/'book_e2/intro.md',*(ROOT/'book_e2/sections').glob('*.md')]:
   self.assertNotRegex(p.read_text(),r'(?i)(veinte|20) configuraciones|(quince|15) (familias |recetas )?ajustables')
 def test_prototype_not_claimed_complete(self):
  s=(ROOT/'book_e2/intro.md').read_text();self.assertIn('No está listo para entrenamiento real',s);self.assertIn('sin entrenamiento integral',s)
if __name__=='__main__':unittest.main()
