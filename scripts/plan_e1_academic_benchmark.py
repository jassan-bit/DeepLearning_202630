"""Illustrative power for a future academic 5% improvement boundary."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/"experiments/e1_audit_improvements_20261104/academic_margin_policy.json"
FREEZE=ROOT/"experiments/filipe_multihistory_20261104/freeze.json"
OUT=ROOT/"results/e1_audit_improvements_20261104/planning"


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directional_power(repetitions,separation,sd,alpha):
    if repetitions<2 or separation<=0 or sd<=0 or not 0<alpha<1:
        raise ValueError("Positive alternative separation, SD and valid design required")
    critical=stats.t.ppf(1-alpha,repetitions-1)
    return float(stats.nct.sf(critical,repetitions-1,separation/sd*np.sqrt(repetitions)))


def required_repetitions(separation,sd,alpha,power):
    if not alpha<power<1:raise ValueError("Valid power target required")
    high=2
    while directional_power(high,separation,sd,alpha)<power:
        high*=2
        if high>2_000_000:raise ValueError("Scenario exceeds planning bound")
    low=2
    while low<high:
        middle=(low+high)//2
        if directional_power(middle,separation,sd,alpha)>=power:high=middle
        else:low=middle+1
    return low


def main():
    policy=json.loads(POLICY.read_text());freeze=json.loads(FREEZE.read_text())
    baseline=freeze["selection"]["baseline"]
    b,sb=baseline["cv_rmse_mean"],baseline["cv_rmse_sample_sd"]
    p=freeze["selection"]["persistence_cv_rmse"];q=policy["minimum_relative_improvement"]
    rows=[];alpha,target=policy["planning_alpha_each"],policy["planning_power_each"]
    for alt in policy["illustrative_alternative_gain_vs_persistence"]:
        advanced_mean=(1-alt)*p
        gap_b=(1-q)*b-advanced_mean;gap_p=(1-q)*p-advanced_mean
        for ratio in policy["advanced_sd_to_neural_baseline_sd"]:
            sa=ratio*sb
            for rho in policy["correlations_with_neural_baseline"]:
                sd_b=float(np.sqrt(sa**2+(1-q)**2*sb**2-2*(1-q)*rho*sa*sb))
                rb=required_repetitions(gap_b,sd_b,alpha,target)
                rp=required_repetitions(gap_p,sa,alpha,target)
                repetitions=max(policy["execution_floor"],rb,rp)
                rows.append({"boundary_relative_gain":q,"assumed_alternative_gain_vs_persistence":alt,
                    "CV_neural_reference_rmse":b,"CV_persistence_reference_rmse":p,
                    "assumed_advanced_mean_rmse":advanced_mean,"advanced_SD_to_baseline_SD":ratio,
                    "assumed_correlation_advanced_neural":rho,"mean_separation_above_neural_boundary":gap_b,
                    "mean_separation_above_persistence_boundary":gap_p,"assumed_boundary_difference_SD_neural":sd_b,
                    "assumed_boundary_difference_SD_persistence":sa,"one_sided_alpha_each":alpha,
                    "power_target_each":target,"R_neural_theoretical":rb,"R_persistence_theoretical":rp,
                    "R_joint_with_guide_min5":repetitions,"normal_assumption_power_neural_at_R":directional_power(repetitions,gap_b,sd_b,alpha),
                    "normal_assumption_power_persistence_at_R":directional_power(repetitions,gap_p,sa,alpha),
                    "scope":"Future hypothetical normal paired seed differences; not achieved power, not a prospective pilot"})
    OUT.mkdir(parents=True,exist_ok=True)
    filename=OUT/"academic_joint_power_scenarios.csv"
    if filename.exists():raise FileExistsError("Preserve the recorded planning version")
    pd.DataFrame(rows).to_csv(filename,index=False,lineterminator="\n")
    text="""# Criterio operativo del benchmark académico

El contexto confirmado es académico. Se fija para comparaciones futuras una reducción relativa de RMSE superior al **5 % frente a MLP/D7 y frente a persistencia**, evaluadas sobre las mismas fechas de una nueva evaluación independiente. El 5 % es una convención operativa transparente, coherente con el criterio previo contra persistencia y acompañada de sensibilidad 1/2,5/5/10 %; no es una tolerancia financiera ni un margen previo al test ya observado.

Para afirmar la mejora no basta el promedio: se deben considerar incertidumbre y las dos comparaciones declaradas. La planificación ilustrativa usa α=0,025 por contraste unilateral y potencia marginal 0,90; bajo los supuestos, la cota de la unión limita a 0,20 la probabilidad de fallar alguna comparación. No se presume independencia entre los dos contrastes.

Las 27 filas suponen un mismo promedio avanzado, elegido como 7,5/10/15 % menor que la persistencia histórica de CV. Se compara con ambas fronteras de mejora del 5 %, con tres DE futuras hipotéticas y tres correlaciones. Se calcula R mediante t no central sobre diferencias normales centradas en la frontera práctica, imponiendo al menos cinco ejecuciones. Es una ilustración de dimensionamiento; falta estimar la dispersión real en un piloto prospectivo y elegir otra evaluación independiente. Las incertidumbres de mercado y de semillas no son equivalentes.

Los valores absolutos de 2024 son anclas ilustrativas. En la futura evaluación se comparan errores relativos sobre una misma cohorte, sin trasladar un umbral absoluto de un régimen a otro. La selección actual MLP/D7 y sus resultados permanecen intactos.
"""
    (OUT/"academic_margin_README.md").write_text(text)
    report={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"scope":"Future academic operational definition and hypothetical boundary planning",
        "policy_sha256":digest(POLICY),"freeze_sha256":digest(FREEZE),"code_sha256":digest(__file__),
        "scenarios":len(rows),"minimum_relative_gain":q,"academic_context_confirmed":True,
        "financial_utility_validated":False,"current_test_pre_registration_claim":False,
        "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archive_reads":0,
        "output_sha256":{p.name:digest(p) for p in [filename,OUT/"academic_margin_README.md"]}}
    (OUT/"academic_planning_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"academic_criterion_recorded":True,"boundary_gain":q,"hypothetical_scenarios":len(rows),"new_model_fits":0}))


if __name__=="__main__":main()
