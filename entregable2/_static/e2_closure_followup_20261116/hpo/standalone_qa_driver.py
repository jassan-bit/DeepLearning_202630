"""Run the descriptive ZIP outside the repository with the standard library."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BASE=Path(__file__).resolve().parent
ZIP=BASE/'reanalisis_hpo_descriptivo_v1.zip'
with tempfile.TemporaryDirectory(prefix='e2-hpo-standalone-',dir='/tmp') as directory:
    destination=Path(directory)
    with zipfile.ZipFile(ZIP) as archive:
        assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in archive.namelist())
        archive.extractall(destination)
    environment=dict(os.environ);environment.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-I','replay_descriptive.py'],cwd=destination,env=environment,
                          text=True,capture_output=True,timeout=12)
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    scientific=json.loads((destination/'replay_result.json').read_text())
    assert scientific['status']=='passed'
    receipt={'version':'hpo-descriptive-standalone-QA-v1','at_utc':datetime.now(timezone.utc).isoformat(),
             'zip_name':ZIP.name,'zip_bytes':ZIP.stat().st_size,
             'zip_sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'returncode':result.returncode,
             'outside_repository':True,'isolated_python':True,'PYTHONPATH_removed':True,
             'requires_network_repository_Torch_or_market_data':False,'result':scientific}
    with (BASE/'standalone_qa.json').open('x') as f:
        json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(receipt,ensure_ascii=False))
