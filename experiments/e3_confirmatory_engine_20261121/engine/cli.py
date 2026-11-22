"""Explicit stage commands for the real R3 engine; no training on import.

Remote credentials are received via stdin JSON or private interactive input.
They are never command arguments, source files or an environment requirement.
"""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse

from .common import require, read_json, sha, write_json, verify_sources, now, canonical, digest

ROOT=Path(__file__).resolve().parents[1]


def panel_map(path):
    result=read_json(path)
    require(set(result)=={"F00","F01","F02","F03"},"Panel map must specify F00,F01,F02,F03")
    require(all(isinstance(v,str) and v for v in result.values()),"Each panel needs its private directory")
    return result


def private_credentials(stream=None):
    stream=sys.stdin if stream is None else stream
    if stream.isatty():
        return {"token":getpass.getpass("Token GitHub (Contents read/write): "),
                "backup_secret":getpass.getpass("Clave independiente del respaldo (conservar): ")}
    obj=json.loads(stream.read())
    require(isinstance(obj,dict) and set(obj)=={"token","backup_secret"},"Private stdin JSON requires token and backup_secret")
    return obj


def redact(value, secrets):
    text=str(value)
    for secret in secrets:
        if isinstance(secret,str) and secret:
            text=text.replace(secret,"[REDACTED]").replace(urllib.parse.quote(secret,safe=""),"[REDACTED]")
    return text


def method_report(path,settings=None):
    """Use the same file/replicate verification required by final training."""
    from .controller import _method_admission
    require(isinstance(settings,dict),"Exact final statistical settings are required for method admission")
    _method_admission(ROOT,{"path":str(Path(path).absolute()),"sha256":sha(path)},settings)
    return read_json(path)


def freeze_protocol(base_path,power_path,method_path,test_spec_path,output):
    """Create a new immutable final protocol from actual admitted evidence.

    This command never turns the distributed development draft into a historical
    preregistration and never acquires, trains or opens a test payload.
    """
    protocol=read_json(base_path);power=read_json(power_path)
    require(power.get("status")=="DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP",
            "Power has not dimensioned a validated full-family design")
    R,T,B=(power.get("final_R"),power.get("final_T"),power.get("final_B"))
    require(type(R) is int and 10<=R<=4091 and type(T) is int and T>0 and type(B) is int and B>=99,
            "Power receipt lacks final R/T/B")
    require(power.get("methods_sha256")==sha(ROOT/"engine/statistics.py"),
            "Power receipt belongs to another method implementation")
    test_spec=read_json(test_spec_path)
    require(set(test_spec)=={"F00","F01","F02","F03"},"Final test specification needs four panels")
    for pid,spec in test_spec.items():
        require(isinstance(spec,dict) and spec.get("count")==T,"Final test count must equal dimensioned T: "+pid)
        require(isinstance(spec.get("start"),str) and isinstance(spec.get("end"),str) and spec["start"]<=spec["end"],
                "Frozen test dates required: "+pid)
        require(spec.get("reserve_mode") in ("documented_historical_reserve","external_custodian","post_protocol"),"Authentic reserve mode required")
        require(spec.get("source_identity_sha256") and spec.get("exposure_evidence_sha256"),
                "Test sources and exposure evidence must be bound before closure")
    from .controller import _admit_final,default_stress_plans
    design=read_json(ROOT/"configs/power_design.json")
    require(power.get("design_sha256")==digest(design),"Power receipt does not match the frozen design source")
    settings=dict(protocol.get("statistical_settings",{}))
    settings.update(bootstrap_reps=B,block_length=design["inner_block_length"],hac_lags=design["hac_lags"],
                    horizon=design["horizon"],margin=design["margin_relative_rmse"],
                    status="FINAL_FROZEN_WITH_EXECUTED_METHOD_AND_POWER_EVIDENCE")
    settings.update({k:design[k] for k in ("reference_reps","reference_seed","alpha") if k in design})
    protocol["statistical_settings"]=settings
    protocol["robustness_stress_plans"]=protocol.get("robustness_stress_plans") or default_stress_plans()
    adequacy=dict(protocol.get("adequacy",{}))
    adequacy_path=Path(adequacy.get("config_path","configs/adequacy.json"))
    if not adequacy_path.is_absolute():adequacy_path=ROOT/adequacy_path
    require(adequacy_path.is_file(),"Frozen adequacy configuration is required")
    adequacy["config_sha256"]=sha(adequacy_path)
    protocol["adequacy"]=adequacy
    protocol["version"]="e3-r3-temporal-engine-final-1"
    protocol["closed_at_utc"]=now()
    protocol["final"]={"closed":True,"power_receipt":{"path":str(Path(power_path).absolute()),"sha256":sha(power_path)},
                       "method_admission":{"path":str(Path(method_path).absolute()),"sha256":sha(method_path)},
                       "R":R,"T":T,"B":B,"test_panels":test_spec,"calibration_required":True}
    protocol["closure_scope"]="Prospective protocol identity; no R3 result, academic acceptance or executed final training"
    protocol["R3_accredited"]=False
    # Do all substantive admission work before installing a closed protocol.
    prospective_sha=hashlib.sha256(canonical(protocol)+b"\n").hexdigest()
    _admit_final(ROOT,protocol,prospective_sha,power_path)
    write_json(output,protocol,immutable=True)
    return {"status":"new_final_protocol_frozen","path":str(output),"sha256":sha(output),"R":R,"T":T,"B":B,
            "requires_public_readback_before_final_training":True,"new_market_fits":0,"test_opened":False}


def check(root):
    from . import models
    integrity=verify_sources(root)
    models.verify_sources()
    config=read_json(Path(root)/"configs/protocol.json")
    val=config.get("method_validation",{})
    validation_path=Path(root)/val.get("path","qa/statistical_validation/verdict_v2.json")
    validation=read_json(validation_path) if validation_path.is_file() else {"admitted":False}
    return {"status":"source_and_engine_ready_for_explicit_commands","integrity":integrity,
            "systems":len(models.METHODS),"neural_systems":len(models.NEURAL),
            "method_validation_admitted":validation.get("admitted") is True,
            "final_protocol_closed":config.get("final",{}).get("closed") is True,
            "new_market_fits":0,"reserved_test_opened":False,"R3_accredited":False}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=ROOT)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("check")
    for command in ("prepare","run","predict-planning","seal-training","evaluate","restore","status","export","sync"):
        s=sub.add_parser(command)
        s.add_argument("--case",type=Path,required=True)
        s.add_argument("--source-commit",required=command!="export")
        s.add_argument("--panels",type=Path,required=command in ("prepare","run","predict-planning","seal-training","evaluate"))
        s.add_argument("--repository",default="jassan-bit/DeepLearning_202630")
        s.add_argument("--branch",default="e3-r3-engine-planning-run-01")
        s.add_argument("--prefix",default="runs/e3_r3_engine/planning-run-01")
        if command=="restore":
            s.add_argument("--workspace-root",type=Path,required=True)
            s.add_argument("--planning-branch")
            s.add_argument("--planning-prefix")
        if command=="prepare":
            s.add_argument("--stage",choices=("planning","final"),required=True)
            s.add_argument("--protocol",type=Path,default=ROOT/"configs/protocol.json")
            s.add_argument("--power-receipt",type=Path)
        if command in ("run","evaluate"):
            s.add_argument("--activate",action="store_true")
        if command=="export":s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("archive-evidence")
    s.add_argument("--case",type=Path,required=True)
    s.add_argument("--source-commit",required=True)
    s.add_argument("--evidence",type=Path,required=True)
    s.add_argument("--workspace-root",type=Path,required=True)
    s=sub.add_parser("dimension")
    s.add_argument("--design",type=Path,default=ROOT/"configs/power_design.json")
    s.add_argument("--predictions",type=Path,nargs="+",required=True)
    s.add_argument("--receipts",type=Path,nargs="+",required=True)
    s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("freeze-protocol")
    s.add_argument("--base",type=Path,default=ROOT/"configs/protocol.json")
    s.add_argument("--power",type=Path,required=True)
    s.add_argument("--method-report",type=Path,required=True)
    s.add_argument("--test-spec",type=Path,required=True)
    s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("assess")
    s.add_argument("--case",type=Path,required=True)
    s.add_argument("--protocol",type=Path,required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv);secrets=[]
    try:
        if args.command=="check":result=check(args.source_root)
        elif args.command=="freeze-protocol":
            result=freeze_protocol(args.base,args.power,args.method_report,args.test_spec,args.output)
        elif args.command=="dimension":
            from .power import dimension
            result=dimension({"predictions":args.predictions,"receipts":args.receipts},read_json(args.design),output=args.output)
        elif args.command=="assess":
            from .assessment import assess_case
            result=assess_case(args.case,args.protocol,output_path=args.case/"metrics/assessment.json")
            write_json(args.case/"analysis_receipt.json",{"status":"SAVED_PREDICTION_ANALYSIS_COMPLETED","protocol_sha256":sha(args.protocol),
                       "assessment_file":"metrics/assessment.json","assessment_sha256":sha(args.case/"metrics/assessment.json"),
                       "r3_scientific_criteria_satisfied":result["r3_scientific_criteria_satisfied"],"new_model_forwards":0,
                       "requires_sync_before_remote_durability_claim":True},immutable=True)
        else:
            from . import controller
            if args.command=="prepare":
                result=controller.prepare_case(args.source_root,args.case,panel_map(args.panels),args.stage,
                                               args.source_commit,args.protocol,args.power_receipt)
            elif args.command=="archive-evidence":
                result=controller.archive_evidence(args.source_root,args.case,args.evidence,args.workspace_root)
            elif args.command=="status":
                result=controller.status_case(args.source_root,args.case)
            elif args.command=="export":
                result=controller.export_case(args.source_root,args.case,args.output)
            else:
                if args.command in ("run","evaluate"):
                    require(args.activate,"Training/test opening is disabled; explicitly activate this stage")
                creds=private_credentials();secrets=[creds["token"],creds["backup_secret"]]
                from .storage import GithubStore
                store=GithubStore(creds["token"],creds["backup_secret"],args.repository,args.branch,
                                  source_commit=args.source_commit,prefix=args.prefix)
                if args.command=="restore":
                    require(bool(args.planning_branch)==bool(args.planning_prefix),"Planning restore needs both branch and prefix")
                    planning=GithubStore(creds["token"],creds["backup_secret"],args.repository,args.planning_branch,
                                         source_commit=args.source_commit,prefix=args.planning_prefix) if args.planning_branch else None
                    result=controller.restore_case(args.source_root,args.case,store,args.workspace_root,planning_store=planning)
                elif args.command=="sync":
                    result=controller.sync_case(args.source_root,args.case,store,panel_dirs=panel_map(args.panels) if args.panels else None)
                elif args.command=="run":
                    require(args.activate,"Training is disabled; explicitly activate this stage")
                    result=controller.run_case(args.source_root,args.case,panel_map(args.panels),store,activate=True)
                elif args.command=="predict-planning":
                    result=controller.predict_planning(args.source_root,args.case,panel_map(args.panels),store)
                elif args.command=="seal-training":
                    result=controller.seal_training(args.source_root,args.case,panel_map(args.panels),store)
                else:
                    require(args.activate,"Reserved evaluation is disabled; explicitly activate this stage")
                    result=controller.evaluate_once(args.source_root,args.case,panel_map(args.panels),store,activate=True)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        if isinstance(result,dict) and result.get("status")=="NOT_DIMENSIONED":return 2
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"status":"interrupted_evidence_retained","new_jobs_started_after_interrupt":False}))
        return 130
    except Exception as error:
        print(json.dumps({"status":"blocked_or_stopped","error_type":type(error).__name__,
                          "message":redact(error,secrets),"evidence_retained":True,"R3_accredited":False},ensure_ascii=False))
        return 2
    finally:
        secrets.clear()


if __name__=="__main__":raise SystemExit(main())
