"""Check saved public acceptance evidence; no network, models or training."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED='feb83dbc08197bde7076857bd8ff475de197efd2'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    home=root/'experiments/e3_render_public_acceptance_20261120'
    manifest=json.loads((home/'manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
        path=home/p
        if path.is_symlink():raise ValueError('Symlink payload')
        raw=path.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('Payload mismatch:'+relative)
        mirror=root/'book_e3/_static/e3_render_public_acceptance_20261120'/p
        if mirror.read_bytes()!=raw:raise ValueError('Mirror mismatch:'+relative)
    report=json.loads((home/'accepted_attempt/combined_result.json').read_text())
    five=json.loads((home/'accepted_attempt/five_client.json').read_text())
    fourteen=json.loads((home/'accepted_attempt/fourteen_client.json').read_text())
    assert report['status']=='passed' and report['public_Render_acceptance_passed']
    assert report['source_commit']==EXPECTED
    assert five['scope']=='public_Render_browser_smoke' and five['render_public_smoke_passed']
    assert len(five['models'])==5 and len(five['pages'])==3 and not five['errors']
    assert fourteen['scope']=='public_acceptance' and fourteen['public_Render_acceptance_passed']
    assert len(fourteen['API_predictions'])==len(fourteen['browser_predictions'])==14
    assert len(fourteen['OOD_cases'])==len(fourteen['interval_date_guards'])==14
    assert len(fourteen['invalid_requests'])==10
    assert not fourteen['errors'] and not fourteen['browser_errors'] and not fourteen['console_errors']
    health=fourteen['health']['body']
    assert health['source_commit']==EXPECTED and health['source_commit_source']=='RENDER_GIT_COMMIT'
    assert health['profile_predictors']==14 and health['original_UI_predictors']==5 and not health['V3_enabled']
    captures=json.loads((home/'accepted_attempt/capture_index.json').read_text())['files']
    assert len(captures)==6
    for entry in captures:
        raw=(home/'accepted_attempt'/entry['path']).read_bytes()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
    print(json.dumps({'status':'public_acceptance_packet_verified','payloads':len(manifest['files']),
                      'public_five_models':5,'public_fourteen_API':14,'public_fourteen_browser':14,
                      'captures_verified':6,'new_training_runs':0,'R3_accredited':False}))

if __name__=='__main__':main()
