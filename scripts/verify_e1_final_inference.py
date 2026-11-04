"""Independent verification of the registered final-test inference supplement.

Only saved forecasts and metadata are read. No project analysis/training helper,
model framework, raw market archive or model inference function is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/e1_audit_improvements_20261104/inference"
PROTOCOL=ROOT/"experiments/e1_audit_improvements_20261104/protocol.json"
REPORT=ROOT/"references/audits/revision_e1_final_inference_20261104.json"
MODELS=("MLP","LSTM","CNN1D","Persistencia")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    values=np.asarray(values,float)
    order=np.argsort(values,kind="stable")
    corrected=np.empty_like(values)
    maximum=0.
    for index,original in enumerate(order):
        maximum=max(maximum,float(values[original]*(len(values)-index)))
        corrected[original]=min(1.,maximum)
    return corrected


def paired_estimates(differences):
    values=np.asarray(differences,float)
    n=len(values);mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/math.sqrt(n)
    statistic=mean/se;critical=stats.t.ppf(.975,n-1)
    shapiro=stats.shapiro(values)
    return {"n_pairs":n,"mean_difference":mean,"sd_difference":sd,
            "ci95_lower":mean-critical*se,"ci95_upper":mean+critical*se,
            "t_statistic":statistic,"df":n-1,"d_z":mean/sd,
            "shapiro_statistic":float(shapiro.statistic),"shapiro_p_value":float(shapiro.pvalue),
            "p_value":float(2*stats.t.sf(abs(statistic),n-1))}


def manual_dm(loss_difference,lag,horizon=1):
    values=np.asarray(loss_difference,float);n=len(values)
    center=values-values.mean()
    variance=float(center@center/n)
    for offset in range(1,lag+1):
        covariance=float(center[offset:]@center[:-offset]/n)
        variance+=2*(1-offset/(lag+1))*covariance
    statistic=float(values.mean()/np.sqrt(variance/n))
    correction=math.sqrt((n+1-2*horizon+horizon*(horizon-1)/n)/n)
    adjusted=statistic*correction
    return {"n_dates":n,"HAC_lag":lag,"forecast_horizon":horizon,
            "mean_loss_difference":float(values.mean()),"long_run_variance":variance,
            "DM":statistic,"HLN_factor":correction,"DM_HLN":adjusted,
            "reference_df":n-1,"p_value":float(2*stats.t.sf(abs(adjusted),n-1))}


def manual_ljung_box(residuals,lag):
    values=np.asarray(residuals,float);n=len(values);center=values-values.mean()
    denominator=float(center@center)
    total=0.
    for offset in range(1,lag+1):
        correlation=float(center[offset:]@center[:-offset]/denominator)
        total+=correlation**2/(n-offset)
    statistic=n*(n+2)*total
    return {"n_dates":n,"lag":lag,"statistic":statistic,"model_df":0,"reference_df":lag,
            "p_value":float(stats.chi2.sf(statistic,lag))}


class Audit:
    def __init__(self):
        self.counts={};self.checks=[];self.maximum_absolute_difference=0.

    def count(self,name,amount=1):self.counts[name]=self.counts.get(name,0)+amount

    def require(self,condition,message):
        if not condition:raise AssertionError(message)

    def near(self,actual,expected,message,atol=2e-12,rtol=2e-9):
        a,b=np.asarray(actual,float),np.asarray(expected,float)
        self.require(a.shape==b.shape,message+" (shape)")
        if np.any(np.isfinite(a)&np.isfinite(b)):
            difference=float(np.max(np.abs(a[np.isfinite(a)&np.isfinite(b)]-b[np.isfinite(a)&np.isfinite(b)])))
            self.maximum_absolute_difference=max(self.maximum_absolute_difference,difference)
        self.require(np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True),message)
        self.count("numeric_values_checked",max(1,a.size))

    def row(self,row,expected,message):
        for key,value in expected.items():self.near(float(row[key]),value,message+": "+key)

    def probabilities(self,table,expected,message):
        self.near(table.p_value.to_numpy(float),expected,message+" raw p",atol=1e-300,rtol=5e-8)
        expected_adjusted=holm(expected)
        self.near(table.p_holm.to_numpy(float),expected_adjusted,message+" Holm",atol=1e-300,rtol=5e-8)
        self.near(table.log_p_value.to_numpy(float),np.log(expected),message+" log p",atol=2e-10,rtol=2e-10)
        self.near(table.log_p_holm.to_numpy(float),np.log(expected_adjusted),message+" log Holm",atol=2e-10,rtol=2e-10)

    def verify(self):
        plan=json.loads(PROTOCOL.read_text());configuration=plan["final_inference"]
        self.require(plan["new_project_training_allowed"] is False and plan["new_checkpoint_forward_calls_allowed"] is False
                     and plan["raw_market_archive_reads_allowed"] is False,"Supplement must not fit or execute forecasts")
        for name,checksum in plan["source_sha256"].items():
            self.require(digest(ROOT/name)==checksum,"Registered source changed: "+name);self.count("source_hashes_checked")
        summary=json.loads((OUT/"summary.json").read_text())
        self.require(summary["protocol_sha256"]==digest(PROTOCOL),"Supplement used another protocol")
        self.require(summary["analysis_code_sha256"]==digest(ROOT/"scripts/analyze_e1_final_inference.py"),"Inference source code changed")
        for name,checksum in summary["source_sha256"].items():
            self.require(digest(ROOT/name)==checksum,"Inference source changed: "+name);self.count("source_hashes_checked")
        for name,checksum in summary["output_sha256"].items():
            self.require(digest(OUT/name)==checksum,"Inference output changed: "+name);self.count("output_hashes_checked")
        freeze=json.loads((ROOT/"experiments/filipe_multihistory_20261104/freeze.json").read_text())
        self.require(freeze["selection"]["baseline"]["model"]=="MLP" and freeze["selection"]["baseline"]["history_days"]==7,
                     "Pre-test baseline was changed")
        self.require(summary["baseline_selection_retained"]=="MLP/D7" and summary["prospective_confirmation"] is False,
                     "The supplement must preserve selection and post-hoc scope")
        self.require(summary["new_project_fits"]==summary["checkpoint_forward_calls"]==summary["raw_market_archives_opened"]==0,
                     "Supplement claims additional model/raw access")
        frame=pd.read_csv(ROOT/"results/filipe_multihistory_20261104/test/predictions.csv")
        frame=frame.loc[frame.history_days.eq(7)].copy()
        expected_dates=pd.date_range("2025-01-01","2026-08-31").strftime("%Y-%m-%d").tolist()
        seeds=configuration["seeds"]
        self.require(len(frame)==608*3*10 and set(frame.model)==set(MODELS[:-1]),"Complete three-model D7 forecast grid required")
        errors={};y=None;p=None
        for model in MODELS[:-1]:
            paths=[]
            for seed in seeds:
                group=frame.loc[frame.model.eq(model)&frame.seed.eq(seed)].sort_values("outcome_day")
                self.require(group.outcome_day.tolist()==expected_dates,"Unequal or incomplete test dates")
                if y is None:y=group.y_true.to_numpy(float);p=group.persistence_pred.to_numpy(float)
                self.near(group.y_true.to_numpy(float),y,"Common test labels")
                self.near(group.persistence_pred.to_numpy(float),p,"Common deterministic reference")
                paths.append(group.y_true.to_numpy(float)-group.y_pred.to_numpy(float))
            errors[model]=np.asarray(paths)
        errors["Persistencia"]=(y-p)[None,:]
        rmse={name:np.sqrt(np.mean(error**2,axis=1)) for name,error in errors.items()}
        mae={name:np.mean(np.abs(error),axis=1) for name,error in errors.items()}
        squared_loss={name:np.mean(error**2,axis=0) for name,error in errors.items()}
        pairs=list(itertools.combinations(MODELS,2))
        paired=pd.read_csv(OUT/"paired_comparisons.csv")
        signed=pd.read_csv(OUT/"wilcoxon_sensitivity.csv")
        sensitivity=pd.read_csv(OUT/"design_sensitivity.csv")
        self.require(len(paired)==len(signed)==len(sensitivity)==6,"Six prespecified seed comparisons required")
        effect_indices=np.random.default_rng(configuration["effect_bootstrap_seed"]).integers(0,len(seeds),size=(configuration["bootstrap_replicates"],len(seeds)))
        paired_p=[];signed_p=[]
        for a,b in pairs:
            difference=rmse[a]-rmse[b]
            row=paired.loc[paired.model_a.eq(a)&paired.model_b.eq(b)].iloc[0]
            estimates=paired_estimates(difference);self.row(row,estimates,f"Paired RMSE {a}/{b}")
            paired_p.append(estimates["p_value"])
            self.require(bool(row.normality_flag_p_lt_0_05)==bool(estimates["shapiro_p_value"]<.05),"Incorrect Shapiro warning")
            resampled=difference[effect_indices]
            mean=resampled.mean(axis=1);sd=resampled.std(axis=1,ddof=1)
            valid=np.isfinite(sd)&(sd>0)
            bounds=np.quantile(mean[valid]/sd[valid],[.025,.975],method="linear")
            self.row(row,{"d_z_ci95_lower":bounds[0],"d_z_ci95_upper":bounds[1],
                          "d_z_bootstrap_defined":int(valid.sum()),"d_z_bootstrap_replicates":len(effect_indices)},"Seed effect bootstrap")
            signed_row=signed.loc[signed.model_a.eq(a)&signed.model_b.eq(b)].iloc[0]
            self.require(not np.any(difference==0) and len(np.unique(abs(difference)))==len(difference),"Actual exact Wilcoxon conditions changed")
            test=stats.wilcoxon(difference,alternative="two-sided",zero_method="wilcox",method="exact",correction=False)
            self.row(signed_row,{"statistic":test.statistic,"p_value":test.pvalue,"n_pairs":len(difference),
                                 "zero_differences":0,"minimum_possible_two_sided_p":2/2**len(difference)},"Wilcoxon location sensitivity")
            signed_p.append(float(test.pvalue))
            mde=sensitivity.loc[sensitivity.model_a.eq(a)&sensitivity.model_b.eq(b)].iloc[0]
            alpha=.05/6
            expected_mde=(stats.norm.ppf(1-alpha/2)+stats.norm.ppf(.8))*difference.std(ddof=1)/math.sqrt(len(difference))
            self.row(mde,{"n_seed_pairs":10,"sd_paired_seed_RMSE":difference.std(ddof=1),"planning_alpha":alpha,
                          "planning_family_size":6,"hypothetical_power_target":.8,"normal_approximation_MDE0":expected_mde},"Retrospective MDE")
            self.count("paired_tests_checked");self.count("seed_effect_bootstrap_intervals_checked")
            self.count("wilcoxon_sensitivities_checked");self.count("retrospective_MDE_rows_checked")
        self.probabilities(paired,np.asarray(paired_p),"Paired seed family")
        self.probabilities(signed,np.asarray(signed_p),"Signed-rank sensitivity family")
        dm=pd.read_csv(OUT/"dm_tests.csv")
        self.require(len(dm)==18,"Eighteen bandwidth/pair DM rows required")
        for _,row in dm.iterrows():
            expected=manual_dm(squared_loss[row.model_a]-squared_loss[row.model_b],int(row.HAC_lag),1)
            self.row(row,expected,"Manual HAC/HLN DM")
            self.count("DM_HAC_HLN_rows_checked")
        primary=dm.loc[dm.primary].copy()
        expected=np.array([manual_dm(squared_loss[r.model_a]-squared_loss[r.model_b],14,1)["p_value"] for _,r in primary.iterrows()])
        self.require(len(primary)==6 and primary.HAC_lag.eq(14).all(),"Primary DM family changed")
        self.probabilities(primary,expected,"Primary temporal family")
        self.require(dm.loc[~dm.primary,"p_holm"].isna().all(),"Sensitivity bandwidths are not extra adjusted primary families")
        lb=pd.read_csv(OUT/"ljung_box.csv")
        self.require(len(lb)==9 and lb.seed.eq(4101).all(),"Nine designated-seed residual diagnostics required")
        lb_expected=[]
        for _,row in lb.iterrows():
            expected=manual_ljung_box(errors[row.model][0],int(row.lag))
            self.row(row,expected,"Manual Ljung–Box")
            lb_expected.append(expected["p_value"]);self.count("manual_LjungBox_rows_checked")
        self.probabilities(lb,np.asarray(lb_expected),"Nine residual diagnostics")
        date=pd.read_csv(OUT/"date_bootstrap.csv")
        self.require(len(date)==36,"Thirty-six date-bootstrap rows required")
        n=608;replicates=configuration["bootstrap_replicates"]
        for length in configuration["bootstrap_lengths"]:
            generator=np.random.default_rng(configuration["bootstrap_seed"]+length)
            starts=generator.integers(0,n,size=(replicates,math.ceil(n/length)))
            indices=((starts[:,:,None]+np.arange(length))%n).reshape(replicates,-1)[:,:n]
            counts=np.zeros((replicates,n),dtype=float)
            for index,row_indices in enumerate(indices):counts[index]=np.bincount(row_indices,minlength=n)/n
            sampled={"rmse":{},"mae":{}}
            for model in MODELS:
                sampled["rmse"][model]=np.sqrt((errors[model]**2)@counts.T).mean(axis=0)
                sampled["mae"][model]=(abs(errors[model])@counts.T).mean(axis=0)
            for metric,point_scores in [("rmse",rmse),("mae",mae)]:
                for a,b in pairs:
                    row=date.loc[date.model_a.eq(a)&date.model_b.eq(b)&date.metric.eq(metric)&date.block_length.eq(length)].iloc[0]
                    differences=sampled[metric][a]-sampled[metric][b]
                    bounds=np.quantile(differences,[.025,.975],method="linear")
                    gain=100*(sampled[metric][b]-sampled[metric][a])/sampled[metric][b]
                    gain_bounds=np.quantile(gain,[.025,.975],method="linear")
                    estimate=float(point_scores[a].mean()-point_scores[b].mean())
                    expected={"estimate_difference":estimate,"ci95_lower":bounds[0],"ci95_upper":bounds[1],
                              "relative_gain_pct":float(100*(point_scores[b].mean()-point_scores[a].mean())/point_scores[b].mean()),
                              "ci95_relative_gain_lower":gain_bounds[0],"ci95_relative_gain_upper":gain_bounds[1],
                              "relative_defined_replicates":replicates,"bootstrap_replicates":replicates}
                    self.row(row,expected,"Independent paired circular date bootstrap")
                    self.count("date_bootstrap_intervals_checked")
        relative=pd.read_csv(OUT/"relative_differences.csv")
        self.require(len(relative)==24,"Twenty-four fixed-reference descriptive differences required")
        for _,row in relative.iterrows():
            scores=rmse if row.metric=="rmse" else mae
            model_mean=float(scores[row.model].mean());reference_mean=float(scores[row.reference].mean())
            self.row(row,{"mean_model_score":model_mean,"mean_reference_score":reference_mean,
                          "absolute_model_minus_reference":model_mean-reference_mean,
                          "relative_error_reduction_pct":100*(reference_mean-model_mean)/reference_mean},"Fixed-reference error difference")
            self.count("relative_difference_rows_checked")
        for key,wanted in [("paired_rejections_Holm05",int((paired.p_holm<.05).sum())),
                           ("paired_Shapiro_flags_05",int((paired.shapiro_p_value<.05).sum())),
                           ("Wilcoxon_sensitivity_rejections_Holm05",int((signed.p_holm<.05).sum())),
                           ("DM_primary_rejections_Holm05",int((primary.p_holm<.05).sum())),
                           ("LjungBox_rejections_Holm05",int((lb.p_holm<.05).sum()))]:
            self.require(summary[key]==wanted,"Summary count mismatch: "+key)
        self.checks.extend([
            "Registered source files, output hashes and the unchanged pre-test MLP/D7 selection verified.",
            "Six paired t/Shapiro comparisons, six exact signed-rank sensitivities, their Holm families, six seed-bootstrap effect intervals and six retrospective MDE scenarios reproduced.",
            "Eighteen DM values reconstructed with manual Bartlett HAC/HLN; nine Ljung–Box values reconstructed from centered residual autocorrelations.",
            "All36 RMSE/MAE paired circular-date intervals and relative-gain intervals reproduced with the registered random schedule.",
            "Twenty-four descriptive differences retain references fixed from CV; no model training, raw archive or checkpoint forward call executed."])


def self_checks():
    if not np.allclose(holm([.03,.01,.2]),[.06,.03,.2]):raise AssertionError("Known Holm correction failed")
    dm=manual_dm([1.,2.,3.,4.],1)
    if not np.isclose(dm["long_run_variance"],1.5625) or not np.isclose(dm["DM"],4.):raise AssertionError("Known Bartlett HAC failed")
    lb=manual_ljung_box([1.,2.,3.,4.],2)
    if not np.isclose(lb["statistic"],1.58):raise AssertionError("Known Ljung–Box failed")
    if not np.isclose(paired_estimates([1.,2.,3.,4.])["t_statistic"],math.sqrt(15)):raise AssertionError("Known paired statistic failed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=REPORT)
    args=parser.parse_args();self_checks();audit=Audit();failure=None
    report={"verified_at_utc":datetime.now(timezone.utc).isoformat(),"verifier":Path(__file__).name,
            "independence":"No project analyzer/helper, torch or training module imported; manual HAC and Ljung–Box formulas; stored forecasts only.",
            "raw_market_archives_opened":0,"new_project_fits":0,"checkpoint_forward_calls":0}
    try:audit.verify();report["status"]="PASS"
    except Exception as exc:failure=exc;report.update({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"})
    report.update({"checks":audit.checks,"counters":audit.counts,"maximum_numeric_absolute_difference":audit.maximum_absolute_difference,
                   "protocol_sha256":digest(PROTOCOL),"verifier_sha256":digest(Path(__file__)),
                   "interpretation":"This confirms computation/integrity of an explicitly post-hoc supplement, not prospective confirmation, achieved power or independent corroboration of the several estimands."})
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    text=["# Verificación independiente de la inferencia final","",f"Estado: **{report['status']}**.","",report["independence"],""]
    text += ["- "+entry for entry in audit.checks]
    text += ["","| Comprobación | Cantidad |","|---|---:|"]+[f"| {key} | {value} |" for key,value in audit.counts.items()]
    text += ["",report["interpretation"],""]
    if failure:text += ["Fallo: "+report["error"],""]
    args.report.with_suffix(".md").write_text("\n".join(text))
    print(json.dumps({"status":report["status"],"counters":audit.counts,"error":report.get("error")}))
    if failure:raise failure


if __name__=="__main__":main()
