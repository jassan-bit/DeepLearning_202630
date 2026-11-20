"""Independent library checks of retrospective diagnostics, no forecaster fitting.

Uses statsmodels implementations on the same frozen point-prediction records.
Auxiliary diagnostic OLS is not training a delivered forecast model.
"""
import csv
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.multitest import multipletests


def verify_group(rows,observed,label):
    checks=[]
    for seed in sorted({r["seed"] for r in rows}):
        part=sorted([r for r in rows if r["seed"]==seed],key=lambda r:r["date"])
        y=np.array([float(r["target_sigma7"]) for r in part]);p=np.array([float(r["prediction_sigma7"]) for r in part]);e=y-p
        record=[r for r in observed if r["seed"]==seed]
        lb=acorr_ljungbox(e,lags=[7,14],return_df=True)
        for lag in (7,14):
            row=next(r for r in record if r["diagnostic"]=="Ljung_Box" and int(r["lag"])==lag)
            np.testing.assert_allclose(float(row["statistic"]),lb.loc[lag,"lb_stat"],rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(float(row["raw_asymptotic_p"]),lb.loc[lag,"lb_pvalue"],rtol=1e-11,atol=1e-14)
            checks.append({"seed":seed,"diagnostic":"Ljung_Box","lag":lag})
        lm,lmp,_,_=het_arch(e,nlags=7,ddof=0)
        row=next(r for r in record if r["diagnostic"]=="ARCH_LM")
        np.testing.assert_allclose(float(row["statistic"]),lm,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(float(row["raw_asymptotic_p"]),lmp,rtol=1e-10,atol=1e-18)
        checks.append({"seed":seed,"diagnostic":"ARCH_LM","lag":7})
        mz=sm.OLS(y,sm.add_constant(p)).fit(cov_type="HAC",cov_kwds={"maxlags":14,"use_correction":True})
        w=mz.wald_test((np.eye(2),np.array([0.,1.])),use_f=False,scalar=True)
        row=next(r for r in record if r["diagnostic"]=="Mincer_Zarnowitz_joint_HAC")
        np.testing.assert_allclose([float(row["MZ_alpha"]),float(row["MZ_beta"])],mz.params,rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(float(row["statistic"]),w.statistic,rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(float(row["raw_asymptotic_p"]),w.pvalue,rtol=1e-9,atol=1e-12)
        checks.append({"seed":seed,"diagnostic":"Mincer_Zarnowitz_joint_HAC","lag":14})
        jb,unused_iid_p,skew,kurtosis=jarque_bera(e)
        np.testing.assert_allclose([float(row["Jarque_Bera_shape_statistic"]),float(row["skewness_population_moment"]),float(row["kurtosis_Pearson_population_moment"])],
                                   [jb,skew,kurtosis],rtol=1e-11,atol=1e-11)
        checks.append({"seed":seed,"diagnostic":"Jarque_Bera_shape_only_no_p_interpretation","lag":None})
    for check in checks:check["scope"]=label
    return checks


def main(root):
    with gzip.open(root/"archive/historical_e3_predictions.csv.gz","rt",newline="",encoding="utf-8") as stream:
        legacy=list(csv.DictReader(stream))
    observed=list(csv.DictReader((root/"results/historical_residual_diagnostics.csv").open()))
    checks=verify_group(legacy,observed,"historical_510c3bdd")
    extended=list(csv.DictReader((root/"results/extended_candidate_residual_diagnostics.csv").open()))
    for dataset in ("D00","D01","D02","D03"):
        with gzip.open(root/("archive/extended_candidate_clean_test_"+dataset+".csv.gz"),"rt",newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        checks.extend(verify_group(rows,[r for r in extended if r["dataset"]==dataset],"extended_796_"+dataset))
    assert len(checks)==250
    for records,key in ((observed,"Holm40_asymptotic_diagnostic_p"),(extended,"Holm160_asymptotic_diagnostic_p")):
        adjusted=multipletests([float(r["raw_asymptotic_p"]) for r in records],method="holm")[1]
        np.testing.assert_allclose([float(r[key]) for r in records],adjusted,rtol=1e-12,atol=1e-15)
    report={"status":"passed_independent_library_numeric_checks","checks":checks,"checks_passed":len(checks),
            "Holm_adjusted_values_verified":200,"diagnostic_families":{"historical":40,"extended":160},
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "new_forecast_model_fits":0,"new_forecast_model_inference":0,"new_statistical_resamples":0,
            "statistical_assumptions_or_formal_R3_certified":False}
    (root/"results/reference_verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"checks_passed":len(checks)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).parent,
                        help="Folder containing archive/ and results/; permits one shared static data copy")
    main(parser.parse_args().root)
