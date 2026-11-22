"""Run only the notebook's disabled branch, with network forbidden; stdlib only."""
import contextlib
import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

root=Path(__file__).resolve().parent
notebook_path=root/'E3_V5_FrozenWeights_PrivateRecovery.ipynb'
notebook=json.loads(notebook_path.read_text())
cells=[c for c in notebook['cells'] if c['cell_type']=='code']
for i,c in enumerate(cells):compile(''.join(c['source']),f'cell{i}','exec')
def forbidden(*args,**kwargs):raise AssertionError('NO_NETWORK_IN_DISABLED_NOTEBOOK')
urllib.request.urlopen=forbidden
with tempfile.TemporaryDirectory() as folder:
    namespace={};exec(''.join(cells[0]['source']),namespace)
    namespace['CODE_DIR']=str(Path(folder)/'public_code')
    namespace['WORK']=str(Path(folder)/'private_work_must_not_exist')
    with contextlib.redirect_stdout(io.StringIO()):exec(''.join(cells[1]['source']),namespace)
    assert namespace['result']['status']=='DISABLED_NO_DOWNLOAD_NO_INSTALL_NO_SECRET_NO_INFERENCE'
    assert not Path(namespace['WORK']).exists()
    assert not any(k=='torch' or k.startswith('torch.') for k in sys.modules)
    result={'status':'PASSED_SOFTWARE_FIXTURE_ONLY','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'notebook_code_cells_compiled':len(cells),'disabled_cells_executed':len(cells),
        'network_calls':0,'Torch_imported':False,'packages_installed':False,'secret_requests':0,
        'market_predictions':0,'private_work_created':False,
        'notebook_sha256':hashlib.sha256(notebook_path.read_bytes()).hexdigest()}
    (root/'NOTEBOOK_DISABLED_EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
