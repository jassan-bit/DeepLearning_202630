"""Independently check hypothetical planning scenarios, without project helpers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/e1_audit_improvements_20261104/planning"
PROTOCOL=ROOT/"experiments/e1_audit_improvements_20261104/protocol.json"
REPORT=ROOT/"references/audits/revision_e1_advanced_planning_20261104.json"


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_t_power(count,delta,sd,alpha):
    if count < 2 or sd <= 0:raise ValueError("At least two pairs and positive difference SD required")
    critical=stats.t.ppf(1-alpha/2,count-1)
    noncentrality=delta*math.sqrt(count)/sd
    # Symmetry expresses both tails with survival functions, avoiding SciPy's
    # cancellation/NaN branch for CDF(-critical, positive large noncentrality).
    return float(stats.nct.sf(critical,count-1,noncentrality)+stats.nct.sf(critical,count-1,-noncentrality))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--report",type=Path,default=REPORT)
    args=parser.parse_args()
    if not np.isclose(paired_t_power(10,0.,1.,.025),.025,rtol=0,atol=1e-12):
        raise AssertionError("Known null-power calibration failed")
    protocol=json.loads(PROTOCOL.read_text());freeze_path=ROOT/"experiments/filipe_multihistory_20261104/freeze.json"
    freeze=json.loads(freeze_path.read_text());summary=json.loads((OUT/"summary.json").read_text())
    frame=pd.read_csv(OUT/"advanced_margin_scenarios.csv");targets=pd.read_csv(OUT/"operational_threshold_scenarios.csv")
    baseline=freeze["selection"]["baseline"];anchor=baseline["cv_rmse_mean"];sd0=baseline["cv_rmse_sample_sd"]
    reference=freeze["selection"]["persistence_cv_rmse"]
    assert summary["protocol_sha256"]==digest(PROTOCOL) and summary["source_freeze_sha256"]==digest(freeze_path)
    assert baseline["model"]=="MLP" and baseline["history_days"]==7 and summary["applied_margin_closed"] is False
    assert len(frame)==36 and len(targets)==4
    hashes=0
    for name,expected in summary["output_sha256"].items():
        assert digest(OUT/name)==expected,name;hashes+=1
    maximum=0.;minimum_checks=0
    def near(a,b,message):
        nonlocal maximum
        maximum=max(maximum,abs(float(a)-float(b)))
        assert np.isclose(a,b,rtol=2e-9,atol=1e-13),message
    plan=protocol["advanced_planning"]
    expected_grid={(r,s,c) for r in plan["relative_margin_scenarios"] for s in plan["advanced_sd_to_baseline_sd"]
                   for c in plan["paired_score_correlation_scenarios"]}
    assert set(zip(frame.relative_reduction,frame.advanced_sd_to_baseline_sd,frame.assumed_paired_correlation))==expected_grid
    for _,row in frame.iterrows():
        alpha=plan["planning_alpha"];target=plan["power_target"]
        delta=row.relative_reduction*anchor
        sd=sd0*math.sqrt(1+row.advanced_sd_to_baseline_sd**2-2*row.assumed_paired_correlation*row.advanced_sd_to_baseline_sd)
        critical=stats.norm.ppf(1-alpha/2)+stats.norm.ppf(target)
        normal=max(2,math.ceil((critical*sd/delta)**2))
        count=int(row.R_noncentral_t_normal_difference)
        near(row.delta_rmse,delta,"CV anchored delta")
        near(row.baseline_seed_sd,sd0,"CV seed SD anchor")
        near(row.assumed_difference_sd,sd,"Paired covariance scenario")
        near(row.planning_alpha_two_sided,alpha,"Planning alpha")
        near(row.power_target,target,"Power target")
        near(row.planned_alternative_difference,delta,"Hypothetical alternative")
        near(row.null_difference,0.,"Null difference")
        assert row.R_normal_approximation_ceiling==normal
        probability=paired_t_power(count,delta,sd,alpha)
        near(row.scenario_power_at_minimum_R,probability,"Noncentral t power")
        assert probability>=target
        if count>2:assert paired_t_power(count-1,delta,sd,alpha)<target
        # Independently recover the minimal integer via monotone binary search.
        low,high=2,count
        while low<high:
            middle=(low+high)//2
            if paired_t_power(middle,delta,sd,alpha)>=target:high=middle
            else:low=middle+1
        assert low==count
        # The published R10 MDE inverts the same finite-sample noncentral-t
        # power, rather than substituting the normal-approximation formula.
        lower,upper=0.,3*sd
        while paired_t_power(10,upper,sd,alpha)<target:upper*=2
        for _ in range(80):
            midpoint=(lower+upper)/2
            if paired_t_power(10,midpoint,sd,alpha)>=target:upper=midpoint
            else:lower=midpoint
        near(row.MDE_zero_at_R10_under_scenario,(lower+upper)/2,"Finite-sample R10 MDE")
        minimum_checks+=1
    for _,row in targets.iterrows():
        ratio=row.relative_reduction
        for key,wanted in {"delta_rmse_vs_neural_CV":ratio*anchor,"neural_CV_anchor_rmse":anchor,
                           "neural_target_rmse":anchor*(1-ratio),"persistence_CV_reference_rmse":reference,
                           "persistence_target_at_same_fraction":reference*(1-ratio)}.items():near(row[key],wanted,key)
    result={"status":"PASS","verified_at_utc":datetime.now(timezone.utc).isoformat(),
            "verifier":"scripts/verify_e1_advanced_planning.py","independence":"No planning/project analyzer imported; covariance identity and noncentral-t powers independently evaluated.",
            "scenario_SD_normal_R_and_MDE_rows_checked":36,"minimum_noncentral_t_R_rows_checked":minimum_checks,
            "operational_threshold_rows_checked":4,"output_hashes_checked":hashes,
            "maximum_numeric_absolute_difference":maximum,"protocol_sha256":digest(PROTOCOL),
            "source_freeze_sha256":digest(freeze_path),"verifier_sha256":digest(Path(__file__)),
            "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_market_archives_opened":0,
            "interpretation":"Hypothetical paired normal score scenarios, not achieved power, applied utility validation, or a retrospective pilot. A theoretical R below5 does not supersede the guide minimum5 (or6 for its rank-test recommendation)."}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    args.report.with_suffix(".md").write_text("# Verificación independiente de escenarios de mejora avanzada\n\nEstado: **PASS**.\n\nSe comprobaron36 escenarios de covarianza, R normal, R mínimo de t no central, potencia y MDE; cuatro umbrales operativos y los hashes.\n\n"+result["interpretation"]+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
