"""Extract clean-test points from four hash-verified saved prediction panels.

AI assistance declared. No model imports, fitting, inference or market download.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from fetch_candidate_panels import FILES,RESULT_COMMIT


def main(raw_dir,archive):
    raw_dir,archive=Path(raw_dir),Path(archive)
    manifest=json.loads((archive/"input_manifest.json").read_text())
    sources={}
    for dataset,record in FILES.items():
        path=raw_dir/(dataset+".csv");raw=path.read_bytes()
        assert len(raw)==record["bytes"] and hashlib.sha256(raw).hexdigest()==record["sha256"]
        rows=list(csv.DictReader(io.StringIO(raw.decode(),newline="")))
        selected=[r for r in rows if r["scenario"]=="unperturbed" and r["split"]=="test"]
        assert all(r["dataset"]==dataset and r["model"]=="CNN-LSTM-atencion-aditiva E3" for r in rows)
        output=[]
        for row in selected:
            output.append({"dataset":dataset,"seed":row["seed"],"date":row["date"],"target_sigma7":row["target"],
                           "prediction_sigma7":row["prediction"],"known_floor":row["known_floor"]})
        data=io.StringIO(newline="");writer=csv.DictWriter(data,list(output[0]));writer.writeheader();writer.writerows(output)
        payload=data.getvalue().encode();name="extended_candidate_clean_test_"+dataset+".csv.gz"
        (archive/name).write_bytes(gzip.compress(payload,compresslevel=9,mtime=0))
        sources[dataset]={**record,"result_commit":RESULT_COMMIT,"original_rows":len(rows),"extracted_rows":len(output),
                          "archive_file":name,"extracted_uncompressed_sha256":hashlib.sha256(payload).hexdigest(),
                          "observed_seeds":sorted({int(r["seed"]) for r in output}),
                          "filter":{"split":"test","scenario":"unperturbed"},
                          "prediction_field":"prediction (common primary projection)",
                          "no_ensembling":True}
        manifest["files_sha256"][name]=hashlib.sha256((archive/name).read_bytes()).hexdigest()
    manifest["provenance"]["extended_candidate_residual_panels"]=sources
    (archive/"input_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps({"original_panels_verified":len(sources),"selected_clean_test_rows":sum(r["extracted_rows"] for r in sources.values()),
                      "new_forecast_model_fits":0,"new_forecast_model_inference":0}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--raw-dir",type=Path,required=True);parser.add_argument("--archive",type=Path,required=True)
    args=parser.parse_args();main(args.raw_dir,args.archive)
