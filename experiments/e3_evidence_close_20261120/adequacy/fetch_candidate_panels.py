"""Read only four immutable saved prediction CSVs; never download model weights.

AI assistance declared. Optional reproduction input acquisition; not run by CI.
Paths were obtained from a metadata-only shallow filtered Git tree, then bound
to the already-published panel_manifest SHA-256 and byte counts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

RESULT_COMMIT="16a4504f6fb9208c7d6f89b65017194fa20470da"
FILES={
 "D00":{"path":"runs/e3_extended_audit/20261119T214649469972Z/payload/panels/D00/CNN-LSTM-atencion-aditiva_E3.csv","bytes":5236448,"sha256":"1d566076e0bffc87e4890a386092c72b7ce7cc4315132b555018fa0f66d13237"},
 "D01":{"path":"runs/e3_extended_audit/20261119T045416335160Z/payload/panels/D01/CNN-LSTM-atencion-aditiva_E3.csv","bytes":8485724,"sha256":"bd7411e7ba721ae5eab7b979c666bc535298563b0204ab1e41f22ece694af0bd"},
 "D02":{"path":"runs/e3_extended_audit/20261119T190741248865Z/payload/panels/D02/CNN-LSTM-atencion-aditiva_E3.csv","bytes":3669440,"sha256":"56bc1a39cf5f63c204901e1c4c3bd99b96aba7e6371b9111fd08be17ecc6ea0d"},
 "D03":{"path":"runs/e3_extended_audit/20261119T200056308873Z/payload/panels/D03/CNN-LSTM-atencion-aditiva_E3.csv","bytes":4016526,"sha256":"2f4791c822e281234e830320338f3ef579b26a2423fb63b0c9e2669f76db2121"}}


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for dataset,record in FILES.items():
        path=output/(dataset+".csv")
        if not path.exists():
            url="https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"+RESULT_COMMIT+"/"+record["path"]
            with urllib.request.urlopen(url,timeout=60) as response:
                raw=response.read(record["bytes"]+1)
            assert len(raw)==record["bytes"] and hashlib.sha256(raw).hexdigest()==record["sha256"],dataset
            path.write_bytes(raw)
        assert path.stat().st_size==record["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"],dataset
    print(json.dumps({"saved_prediction_panels_verified":len(FILES),"bytes":sum(r["bytes"] for r in FILES.values()),
                      "model_weights_downloaded":False,"new_forecast_model_fits":0,"new_forecast_model_inference":0}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
