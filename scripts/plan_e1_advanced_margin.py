"""Scenario-based planning for a future advanced model, not achieved power."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
FREEZE = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
OUT = ROOT / "results/e1_audit_improvements_20261104/planning"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_sd(baseline_sd, advanced_sd_ratio, correlation):
    if (not np.isfinite([baseline_sd, advanced_sd_ratio, correlation]).all()
            or baseline_sd <= 0 or advanced_sd_ratio <= 0 or not -1 <= correlation <= 1):
        raise ValueError("Positive finite SD/ratio and a correlation within [-1,1] required")
    variance = baseline_sd**2 * (advanced_sd_ratio**2 + 1 - 2*correlation*advanced_sd_ratio)
    if variance <= 0:
        raise ValueError("A nondegenerate hypothetical difference is required")
    return float(np.sqrt(variance))


def normal_difference_power(repetitions, difference, difference_sd, alpha):
    """Power of a two-sided paired t under a hypothetical normal difference."""
    if repetitions < 2 or int(repetitions) != repetitions or difference_sd <= 0 or not 0 < alpha < 1:
        raise ValueError("At least two repetitions and valid SD/alpha required")
    critical = stats.t.ppf(1-alpha/2, repetitions-1)
    ncp = float(difference/difference_sd*np.sqrt(repetitions))
    # Symmetry avoids a NaN in extreme negative-tail CDF evaluations in some
    # SciPy/Boost builds: -T_ncp has the distribution T_(-ncp).
    return float(stats.nct.sf(critical, repetitions-1, ncp) + stats.nct.sf(critical, repetitions-1, -ncp))


def minimum_repetitions(difference, difference_sd, alpha, target=.8):
    if difference <= 0 or difference_sd <= 0 or not 0 < alpha < target < 1:
        raise ValueError("Positive effect and SD and alpha < target power < 1 required")
    high = 2
    while normal_difference_power(high, difference, difference_sd, alpha) < target:
        high *= 2
        if high > 2_000_000:
            raise ValueError("Scenario exceeds the supported planning bound")
    low = 2
    while low < high:
        middle = (low+high)//2
        if normal_difference_power(middle, difference, difference_sd, alpha) >= target:
            high = middle
        else:
            low = middle+1
    return low


def minimum_detectable_difference(repetitions, difference_sd, alpha, target=.8):
    root = optimize.brentq(lambda standardized: normal_difference_power(
        repetitions, standardized*difference_sd, difference_sd, alpha)-target,
        0., 20., xtol=1e-12)
    return float(root*difference_sd)


def main():
    plan = json.loads(PROTOCOL.read_text())
    settings = plan["advanced_planning"]
    freeze = json.loads(FREEZE.read_text())
    baseline = freeze["selection"]["baseline"]
    mean, sd = baseline["cv_rmse_mean"], baseline["cv_rmse_sample_sd"]
    persistence = freeze["selection"]["persistence_cv_rmse"]
    alpha, target = settings["planning_alpha"], settings["power_target"]
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "summary.json").exists():
        raise FileExistsError("Preserve the recorded planning version; do not overwrite it")
    registration = {"registered_before_calculation_at_utc": datetime.now(timezone.utc).isoformat(),
                    "protocol_sha256": digest(PROTOCOL), "code_sha256": digest(__file__),
                    "source_freeze_sha256": digest(FREEZE), "new_model_fits": 0,
                    "new_checkpoint_forward_calls": 0, "raw_archives_opened": 0}
    (OUT / "registration.json").write_text(json.dumps(registration, indent=2)+"\n")
    scenarios, thresholds = [], []
    for fraction in settings["relative_margin_scenarios"]:
        delta = fraction*mean
        thresholds.append({"relative_reduction": fraction, "delta_rmse_vs_neural_CV": delta,
                           "neural_CV_anchor_rmse": mean, "neural_target_rmse": mean-delta,
                           "persistence_CV_reference_rmse": persistence,
                           "persistence_target_at_same_fraction": persistence*(1-fraction),
                           "status": "future operational scenario, not a validated application margin"})
        for ratio in settings["advanced_sd_to_baseline_sd"]:
            for correlation in settings["paired_score_correlation_scenarios"]:
                dispersion = paired_sd(sd, ratio, correlation)
                approximation = (stats.norm.ppf(1-alpha/2)+stats.norm.ppf(target))**2*dispersion**2/delta**2
                exact = minimum_repetitions(delta, dispersion, alpha, target)
                scenarios.append({"relative_reduction": fraction, "delta_rmse": delta,
                    "baseline_seed_sd": sd, "advanced_sd_to_baseline_sd": ratio,
                    "assumed_paired_correlation": correlation, "assumed_difference_sd": dispersion,
                    "planning_alpha_two_sided": alpha, "power_target": target,
                    "null_difference": 0., "planned_alternative_difference": delta,
                    "R_normal_approximation_ceiling": max(2,int(np.ceil(approximation))),
                    "R_noncentral_t_normal_difference": exact,
                    "scenario_power_at_minimum_R": normal_difference_power(exact,delta,dispersion,alpha),
                    "MDE_zero_at_R10_under_scenario": minimum_detectable_difference(10,dispersion,alpha,target),
                    "scope": "Hypothetical normal paired seed differences, fixed CV anchor; not achieved power or a prospective pilot"})
    pd.DataFrame(scenarios).to_csv(OUT / "advanced_margin_scenarios.csv",index=False,lineterminator="\n")
    pd.DataFrame(thresholds).to_csv(OUT / "operational_threshold_scenarios.csv",index=False,lineterminator="\n")
    note = """# Escenarios para un modelo avanzado futuro

La referencia neuronal es MLP/D7, elegida con CV antes del test. Se usan su RMSE media y DE entre semillas de 2024 como anclas históricas. La persistencia se muestra por separado porque tiene menor error observado; superar solo la referencia neuronal no basta para atribuir utilidad frente a esa referencia sencilla.

Los márgenes relativos de 1, 2,5, 5 y 10 % son escenarios operativos, no una tolerancia financiera validada ni un margen seleccionado retroactivamente. Debe documentarse qué diferencia tiene valor para la aplicación. No se ha entrenado aquí un modelo avanzado.

Cada escenario declara la DE de ese modelo como 0,5/1/2 veces la DE histórica del baseline y correlación pareada 0/0,5/0,9. Son supuestos ilustrativos, no estimaciones de un modelo desconocido ni un conjunto exhaustivo de correlaciones posibles. Se calcula R para detectar una diferencia δ frente a cero con prueba bilateral t de diferencias normales, potencia objetivo 0,80 y α de planificación 0,025 (Bonferroni conservador para dos comparaciones futuras). La inversión de t no central añade la corrección de tamaño finito que omite la aproximación normal.

Detectar una diferencia δ frente a cero no equivale a demostrar que la mejora excede δ. Para contrastar una frontera práctica debe definirse una alternativa por encima de esa frontera y planificar con su separación. Tampoco puede deducirse potencia alcanzada de la tabla: faltan la variabilidad/correlación reales, un piloto prospectivo y una evaluación nueva independiente.

Los resultados son condicionales a aleatoriedad de semillas sobre un calendario; no representan automáticamente incertidumbre entre mercados. El margen aplicado permanece pendiente mientras no exista una justificación documentada del objetivo o decisión que lo exige.
"""
    (OUT / "README.md").write_text(note)
    report = {"completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": settings["status"], "baseline_anchor": baseline,
              "persistence_cv_rmse": persistence, "scenarios": len(scenarios),
              "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archives_opened":0,
              "protocol_sha256":digest(PROTOCOL),"source_freeze_sha256":digest(FREEZE),
              "code_sha256":digest(__file__), "applied_margin_closed":False,
              "output_sha256":{p.name:digest(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":"scenario_planning_complete","scenarios":len(scenarios),"applied_margin_closed":False}))


if __name__ == "__main__":
    main()
