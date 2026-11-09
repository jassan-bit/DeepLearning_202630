"""Validate the synthetic recipe receipt and mirror an explicit public allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/e2_recipe_synthetic_20261109'
DEST = ROOT / 'book_e2/_static/recipe_implementation_20261109'
FILES = ('synthetic_receipt.json', 'summary.json', 'README.md')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
def require(condition, message):
    if not condition:
        raise ValueError(message)
def validate(receipt):
    cards=json.loads((ROOT/'references/e2_recipe_cards_20261109/cards.json').read_text())['cards']
    require(receipt['synthetic_only'] is True, 'Synthetic receipt required')
    for key in ('market_data_read','market_training_executed','hpo_executed','nas_executed','full_benchmark_executed','generalization_or_superiority_demonstrated'):
        require(receipt[key] is False, key)
    require(receipt['nas_adjudication'] is None, 'NAS remains pending')
    require(receipt['executed_model_count']==11 and receipt['skipped_model_count']==0, 'All eleven required')
    require([r['recipe_id'] for r in receipt['models']]==[r['recipe_id'] for r in cards], 'Recipe IDs differ')
    require(receipt['verified_author_file_count']==70, 'Expected pinned source subset')
    for name, sha in receipt['source_sha256'].items():
        require(Path(name).name==name and digest(ROOT/'scripts'/name)==sha, 'Source hash: '+name)
    require(receipt['training_contract']['structural_resize_for_smoke'] is False, 'Full recipe dimensions required')
    require(receipt['fold']['validation_used_for_optimization_or_selection'] is False, 'Validation must not train')
    for row in receipt['models']:
        require(row['phases']['supervised']['steps']==1, 'One supervised smoke step')
        for phase in row['phases'].values():
            require(phase['all_gradients_finite'] and phase['fresh_optimizer'], 'Phase audit')
            require(phase['initial_state_sha256']!=phase['final_state_sha256'], 'No update')
    ae=receipt['models'][3]
    require(set(ae['phases'])=={'ae1','ae2','supervised'}, 'Two auxiliary phases')
    for key in ('encoder1_frozen_during_phase2','transfer_encoder1_exact','transfer_encoder2_exact'):
        require(ae['ae_protocol'][key],key)
    require(ae['ae_protocol']['decoder_parameters_in_predictor'] is False,'Discard decoders')
    raw=json.dumps(receipt)
    for private in ('/workspace/', '/tmp/', 'source_roots'):
        require(private not in raw, 'Private path/config in receipt')
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.receipt:
        require(not args.check,'Choose import or check')
        receipt=validate(json.loads(args.receipt.read_text()))
        SOURCE.mkdir(parents=True,exist_ok=True)
        # Public synthetic-only receipt, imported once; no historical overwrite.
        with (SOURCE/'synthetic_receipt.json').open('xb') as stream:stream.write(encoded(receipt))
    receipt=validate(json.loads((SOURCE/'synthetic_receipt.json').read_text()))
    summary=json.loads((SOURCE/'summary.json').read_text())
    for name,sha in summary['verified_code_sha256'].items():
        require(digest(ROOT/name)==sha,'Evidence source changed: '+name)
    require(summary['new_market_fits']==0 and summary['nas_adjudication'] is None,'Scope')
    manifest={'allowlist':list(FILES),'files':{name:{'sha256':digest(SOURCE/name),'bytes':(SOURCE/name).stat().st_size} for name in FILES}}
    if args.check:require((SOURCE/'manifest.json').read_bytes()==encoded(manifest),'Manifest differs')
    else:(SOURCE/'manifest.json').write_bytes(encoded(manifest))
    DEST.mkdir(parents=True,exist_ok=True)
    for name in (*FILES,'manifest.json'):
        if args.check:require((DEST/name).read_bytes()==(SOURCE/name).read_bytes(),'Mirror differs')
        else:(DEST/name).write_bytes((SOURCE/name).read_bytes())
    require({p.name for p in DEST.iterdir()}==set((*FILES,'manifest.json')),'Unexpected public file')
    print('PASS: eleven full-dimensional synthetic recipes, controls evidence, hashes and public mirror')
if __name__=='__main__':main()
