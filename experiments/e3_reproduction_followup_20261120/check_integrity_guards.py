"""Exercise guards with corrupted temporary copies of real accepted inputs."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from hydrate_accepted import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(__file__).resolve().parent
    manifest=json.loads((root/'accepted_input_manifest.json').read_text())
    verify(args.inputs,manifest);results=[]
    with tempfile.TemporaryDirectory() as d:
        copied=Path(d)/'inputs';shutil.copytree(args.inputs,copied)
        for item in manifest['files']:
            path=copied/item['path'];original=path.read_bytes()
            damaged=bytearray(original);damaged[-1]^=1;path.write_bytes(damaged)
            try:
                verify(copied,manifest)
            except ValueError:
                results.append({'input':item['path'],'same_length_one_bit_corruption_rejected':True})
            else:raise AssertionError('Corruption accepted: '+item['path'])
            path.write_bytes(original)
        verify(copied,manifest)
    result={'status':'passed','AI_assistance_declared':True,'checks':results,'original_inputs_unchanged':True,
            'model_forward_calls':0,'new_model_fits':0}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
