"""Collect already-saved metrics; no network, forecast fitting or inference.

AI assistance declared. This is retrospective evidence assembly, not preregistration.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import gzip
import json
from pathlib import Path
import shutil

CANDIDATE="CNN-LSTM-atencion-aditiva E3"
HISTORICAL_SHA="b82ba0a8a0371ea5643d9ff666a3ac5a1a4e91018b95a640bd88b1be50bb5133"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(source,historical,out):
    source,historical,out=map(Path,(source,historical,out))
    out.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/"analysis/output_manifest.json").read_text())
    for name,record in original["files"].items():
        path=source/"analysis"/name
        assert path.stat().st_size==record["bytes"] and sha(path)==record["sha256"],name
    assert sha(historical)==HISTORICAL_SHA,"Wrong historical prediction cohort"
    original_hashes={name:record["sha256"] for name,record in original["files"].items()}
    for name in ("model_metrics.csv","resources_supplied.csv"):
        shutil.copyfile(source/"analysis"/name,out/name)
    extraction={}
    for name,target in (("per_seed_metrics.csv","per_seed_candidate_clean.csv"),
                        ("interval_metrics.csv","interval_candidate_clean.csv"),
                        ("quarter_metrics.csv","quarter_candidate_clean.csv")):
        with (source/"analysis"/name).open(newline="",encoding="utf-8") as stream:
            reader=csv.DictReader(stream)
            fields=reader.fieldnames
            rows=[r for r in reader if r["model"]==CANDIDATE and r["scenario"]=="unperturbed"]
        assert rows
        with (out/target).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
        extraction[target]={"original":name,"original_sha256":original_hashes[name],"rows":len(rows),
                            "filter":{"model":CANDIDATE,"scenario":"unperturbed"}}
    for name,target in (("analysis/output_manifest.json","original_output_manifest.json"),
                        ("analysis/analysis_receipt.json","original_analysis_receipt.json")):
        shutil.copyfile(source/name,out/target)
    original_run=json.loads((source/"run_manifest.json").read_text())
    keys=("kind","source_commit","protocol_sha256","panel_sha256","protocol_tag","evidence_mode",
          "test_targets_used_for_selection","additional_datasets_used_for_mechanism_redesign",
          "primary_common_projection","sample_curve_fit_policy","R1_accredited","R3_accredited",
          "prospective_power_certified","family","seeds")
    excerpt={"is_excerpt_not_full_run_manifest":True,"original_sha256":sha(source/"run_manifest.json"),
             "fields":{key:original_run[key] for key in keys}}
    (out/"original_run_identity_excerpt.json").write_text(json.dumps(excerpt,indent=2,sort_keys=True)+"\n")
    (out/"historical_e3_predictions.csv.gz").write_bytes(gzip.compress(historical.read_bytes(),compresslevel=9,mtime=0))
    assert hashlib.sha256(gzip.decompress((out/"historical_e3_predictions.csv.gz").read_bytes())).hexdigest()==HISTORICAL_SHA
    files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="input_manifest.json"}
    manifest={"kind":"retrospective_existing_prediction_adequacy_inputs","files_sha256":files,
              "original_12_analysis_tables_verified":len(original_hashes),"extractions":extraction,
              "provenance":{
                  "published_book_commit":"29f9a26f1fd05484c6581ea6538ad7cb5a4781cb",
                  "extended_audit":{"master_run_id":"20261119T031104515520Z",
                                    "completed_result_commit":"16a4504f6fb9208c7d6f89b65017194fa20470da",
                                    "source_commit":"5a07d6157a9796ca54416dc5dea1e1527e731597",
                                    "analysis_result_commit":"3571b3e119c5453e70d4b7bb7289b6da87787ee1",
                                    "book_static_prefix":"book_e3/_static/e3_audit_completed_20261119T234320461006Z/",
                                    "original_analysis_output_sha256":sha(source/"analysis/output_manifest.json")},
                  "historical_residual_predictions":{"commit":"510c3bddfdc9926322e3210ac61ee1f5eeffad2e",
                                                     "path":"runs/estabilidad_aditiva_v1/evaluation_predictions.csv",
                                                     "sha256":HISTORICAL_SHA,
                                                     "rows":6080,"dates":608,"seeds":10,
                                                     "not_the_new_796_audit_weights":True},
                  "AI_assistance":"Evidence assembly, script and interpretation prepared with declared AI assistance.",
                  "new_forecast_model_fits":0,"new_forecast_model_inference":0}}
    (out/"input_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps({"source_tables_verified":len(original_hashes),"archived_files":len(files),"extracted_files":extraction}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--historical",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();main(args.source,args.historical,args.out)
