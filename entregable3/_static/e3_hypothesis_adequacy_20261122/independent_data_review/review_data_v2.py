"""Two independent adversarial fixtures. No market requests or final data.

Read-only with respect to the implementation and historical source A.
Generated temporary fixtures are always removed. AI assistance declared.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

BASE=Path(__file__).resolve().parent.parent
DATA=BASE/'data'
OLD=Path('/workspace/e3_confirmatory_engine_20261121/engine/data.py')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(require_fixed=False):
    tested_sha=sha(DATA/'engine_data_v2.py')
    old_sha=sha(OLD)
    assert old_sha=='3851281f00fc7b43bfb681d10be988f0efc5fb3d879e6b4faab2fe4798a9d917'
    t=load('independent_crypto_data_fixture',DATA/'test_crypto_data_v2.py')
    old=load('independent_source_A_data',OLD)
    case=t.CryptoDataTests();case.setUp()
    try:
        config=case.config(provider='binance_vision_um',symbol='BTCUSDT')
        manifest=case.collect(config)
        manifest['instrument']='ETHUSDT'
        manifest.pop('manifest_sha256')
        manifest['manifest_sha256']=t.data.identity(manifest)
        t.dump(case.root/'raw/raw_manifest.json',manifest)
        try:
            result=t.data.prepare_development(case.root/'raw/raw_manifest.json',case.split(config),case.root/'panel',
                rights=case.rights,exposure=case.exposure)
            first=dict(case='BTCUSDT archives mislabeled as ETHUSDT',rejected=False,
                observed_instrument=result['provenance']['instrument'],status=result['status'])
        except t.data.AdmissionError as exc:
            first=dict(case='BTCUSDT archives mislabeled as ETHUSDT',rejected=True,reason=str(exc))
    finally:case.tearDown()
    case=t.CryptoDataTests();case.setUp()
    try:
        config=case.config(provider='binance_vision',symbol='BTCUSDT',policy='daily_only_v1')
        get,_=case.fake_get(config)
        raw,target=case.root/'raw',case.root/'panel'
        with patch.object(old,'_network_get',side_effect=get),contextlib.redirect_stdout(io.StringIO()):
            old.collect_development(config,raw,rights_receipt=case.rights,exposure_receipt=case.exposure,
                enable_market_acquisition=True)
            admission=old.prepare_development(raw/'raw_manifest.json',case.split(config),target,
                rights=case.rights,exposure=case.exposure)
        originals={name:(target/name).read_bytes() for name in
            (*admission['partition_paths'].values(),'admission.json','recovery_metadata.json')}
        original_manifest=(raw/'raw_manifest.json').read_bytes()
        bundle=case.root/'saved-original-A.json';bundle.write_bytes(originals['recovery_metadata.json'])
        shutil.rmtree(raw);shutil.rmtree(target)
        try:
            with patch.object(t.data,'_network_get',side_effect=get),contextlib.redirect_stdout(io.StringIO()):
                result=t.data.reacquire_development(bundle,target,raw,case.root,enable_market_acquisition=True)
            identical=(raw/'raw_manifest.json').read_bytes()==original_manifest and all(
                (target/name).read_bytes()==value for name,value in originals.items())
            second=dict(case='Exact replay of source A legacy Spot admission',restored=True,
                original_bytes_identical=identical,status=result['status'])
        except t.data.AdmissionError as exc:
            second=dict(case='Exact replay of source A legacy Spot admission',restored=False,reason=str(exc))
    finally:case.tearDown()
    assert sha(DATA/'engine_data_v2.py')==tested_sha and sha(OLD)==old_sha
    report=dict(verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),tested_extension_sha256=tested_sha,
        source_A_sha256=old_sha,implementation_files_modified=False,network_requests=0,
        real_market_admissions=0,final_data_opened=False,AI_assistance_declared=True,cases=[first,second])
    print(json.dumps(report,indent=2))
    if require_fixed:
        assert first['rejected'] and second['restored'] and second['original_bytes_identical']


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--require-fixed',action='store_true')
    args=parser.parse_args()
    with patch('urllib.request.urlopen',side_effect=AssertionError('NETWORK_FORBIDDEN_IN_REVIEW')):
        main(args.require_fixed)
